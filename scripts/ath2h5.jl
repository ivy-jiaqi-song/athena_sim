#!/usr/bin/env julia
"""Convert one uniform-grid Athena++ .athdf snapshot to analysis HDF5."""

using HDF5

Base.@kwdef struct AthenaLayout
    kind::Symbol
    block_size::NTuple{3,Int}
    nblocks::Int
    nvars::Int
end

function cli_args(args)
    values = Dict{String,String}()
    index = 1
    while index <= length(args)
        args[index] in ("--input", "--output") || error("Unknown argument: $(args[index])")
        index < length(args) || error("Missing value for $(args[index])")
        values[args[index]] = args[index + 1]
        index += 2
    end
    haskey(values, "--input") || error("Required argument: --input")
    haskey(values, "--output") || error("Required argument: --output")
    return values["--input"], values["--output"]
end

function attr_value(object, name)
    attributes = attrs(object)
    haskey(attributes, name) || return nothing
    value = attributes[name]
    try
        return read(value)
    catch
        return value
    end
end

string_vector(value) = value === nothing ? String[] : [strip(String(item)) for item in vec(collect(value))]
int_vector(value) = value === nothing ? Int[] : [Int(item) for item in vec(collect(value))]

function logical_locations(raw)
    ndims(raw) == 2 || error("LogicalLocations must be two-dimensional")
    locations = Int.(raw)
    size(locations, 1) == 3 && return locations
    size(locations, 2) == 3 && return permutedims(locations)
    error("Unexpected LogicalLocations shape $(size(raw))")
end

function infer_layout(data, locations, dataset_name)
    ndims(data) == 5 || error("Expected 5D dataset $dataset_name, got $(size(data))")
    nblocks = size(locations, 2)
    candidates = AthenaLayout[]
    size(data, 4) == nblocks && push!(candidates, AthenaLayout(
        kind=:xyz_block_var, block_size=(size(data, 1), size(data, 2), size(data, 3)),
        nblocks=nblocks, nvars=size(data, 5)))
    size(data, 1) == nblocks && push!(candidates, AthenaLayout(
        kind=:block_var_zyx, block_size=(size(data, 5), size(data, 4), size(data, 3)),
        nblocks=nblocks, nvars=size(data, 2)))
    size(data, 2) == nblocks && push!(candidates, AthenaLayout(
        kind=:var_block_zyx, block_size=(size(data, 5), size(data, 4), size(data, 3)),
        nblocks=nblocks, nvars=size(data, 1)))
    plausible = filter(item -> 1 <= item.nvars <= 64, candidates)
    isempty(plausible) && error("Cannot infer layout for $dataset_name with shape $(size(data))")
    return first(plausible)
end

function variable_names(file, dataset_name, nvars)
    names = string_vector(attr_value(file, "VariableNames"))
    datasets = string_vector(attr_value(file, "DatasetNames"))
    counts = int_vector(attr_value(file, "NumVariables"))
    index = findfirst(==(dataset_name), datasets)
    if index !== nothing && length(counts) == length(datasets)
        first_index = sum(counts[1:index - 1]) + 1
        return names[first_index:first_index + counts[index] - 1]
    end
    length(names) == nvars || error("Cannot map variables for dataset $dataset_name")
    return names
end

normalize(name) = lowercase(replace(String(name), r"[^A-Za-z0-9]" => ""))

function variable_index(names, candidates)
    normalized = normalize.(names)
    for candidate in normalize.(candidates)
        index = findfirst(==(candidate), normalized)
        index === nothing || return index
    end
    error("Missing variable $(join(candidates, "/")); available: $(join(names, ", "))")
end

function block_variable(data, layout, block_index, variable_index)
    layout.kind == :xyz_block_var && return data[:, :, :, block_index, variable_index]
    layout.kind == :block_var_zyx && return permutedims(data[block_index, variable_index, :, :, :], (3, 2, 1))
    layout.kind == :var_block_zyx && return permutedims(data[variable_index, block_index, :, :, :], (3, 2, 1))
    error("Unsupported layout $(layout.kind)")
end

function assemble(data, layout, locations, variable_index, dimensions)
    output = zeros(eltype(data), dimensions)
    for block_index in 1:layout.nblocks
        ranges = ntuple(axis -> begin
            first_cell = locations[axis, block_index] * layout.block_size[axis] + 1
            first_cell:first_cell + layout.block_size[axis] - 1
        end, 3)
        output[ranges...] .= block_variable(data, layout, block_index, variable_index)
    end
    return output
end

function root_bounds(file)
    bounds = Float64[]
    for axis in 1:3
        value = attr_value(file, "RootGridX$axis")
        values = value === nothing ? Float64[] : Float64.(vec(collect(value)))
        length(values) >= 2 || error("Missing RootGridX$axis domain metadata")
        append!(bounds, values[1:2])
    end
    return bounds
end

function convert(input_path, output_path)
    isfile(input_path) || error("Input does not exist: $input_path")
    mkpath(dirname(abspath(output_path)))
    h5open(input_path, "r") do source
        levels = haskey(source, "Levels") ? Int.(read(source, "Levels")) : [0]
        minimum(levels) == maximum(levels) || error("AMR snapshots are not supported")
        locations = logical_locations(read(source, "LogicalLocations"))
        prim = read(source, "prim")
        magnetic = read(source, "B")
        prim_layout = infer_layout(prim, locations, "prim")
        magnetic_layout = infer_layout(magnetic, locations, "B")
        prim_names = variable_names(source, "prim", prim_layout.nvars)
        dimensions = ntuple(axis -> (maximum(locations[axis, :]) + 1) * prim_layout.block_size[axis], 3)
        indices = (
            rho=variable_index(prim_names, ["rho", "density"]),
            vx=variable_index(prim_names, ["vel1", "v1", "vx"]),
            vy=variable_index(prim_names, ["vel2", "v2", "vy"]),
            vz=variable_index(prim_names, ["vel3", "v3", "vz"]),
        )
        fields = (
            gas_density=assemble(prim, prim_layout, locations, indices.rho, dimensions),
            i_velocity=assemble(prim, prim_layout, locations, indices.vx, dimensions),
            j_velocity=assemble(prim, prim_layout, locations, indices.vy, dimensions),
            k_velocity=assemble(prim, prim_layout, locations, indices.vz, dimensions),
            i_mag_field=assemble(magnetic, magnetic_layout, locations, 1, dimensions),
            j_mag_field=assemble(magnetic, magnetic_layout, locations, 2, dimensions),
            k_mag_field=assemble(magnetic, magnetic_layout, locations, 3, dimensions),
        )
        time = Float64(attr_value(source, "Time"))
        cycle = Int(attr_value(source, "NumCycles"))
        bounds = root_bounds(source)
        h5open(output_path, "w") do output
            for (name, field) in pairs(fields)
                write(output, String(name), field)
            end
            write(output, "time", time)
            write(output, "cycle", cycle)
            write(output, "domain_bounds", bounds)
            attrs(output)["source_file"] = abspath(input_path)
            attrs(output)["source_format"] = "athdf"
            attrs(output)["athena_dataset_layout"] = String(prim_layout.kind)
            attrs(output)["athena_grid"] = collect(dimensions)
            attrs(output)["array_axis_order"] = "x1,x2,x3"
        end
    end
    println("[ath2h5] wrote $output_path")
end

input_path, output_path = cli_args(ARGS)
convert(input_path, output_path)
