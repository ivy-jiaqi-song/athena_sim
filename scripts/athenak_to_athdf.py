#!/usr/bin/env python3
"""Stream AthenaK 1.1 binary MeshBlocks into Athena++-compatible ATHDF."""

from __future__ import annotations

import argparse
import os
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterator

import h5py
import numpy as np


SOURCE_NAMES = ("dens", "velx", "vely", "velz", "bcc1", "bcc2", "bcc3")
TARGET_NAMES = ("rho", "vel1", "vel2", "vel3", "Bcc1", "Bcc2", "Bcc3")


@dataclass(frozen=True)
class BinaryHeader:
    time: float
    cycle: int
    location_size: int
    variable_size: int
    variable_names: tuple[str, ...]
    parameters: dict[str, dict[str, str]]
    data_offset: int


@dataclass(frozen=True)
class MeshBlock:
    indices: tuple[int, int, int, int, int, int]
    logical_location: tuple[int, int, int]
    level: int
    geometry: tuple[float, float, float, float, float, float]
    variables: np.ndarray


def _assignment(line: bytes) -> str:
    try:
        return line.decode("utf-8").split("=", 1)[1].strip()
    except (UnicodeDecodeError, IndexError) as exc:
        raise ValueError("Malformed AthenaK binary preheader") from exc


def _parse_parameter_dump(raw: bytes) -> dict[str, dict[str, str]]:
    blocks: dict[str, dict[str, str]] = {}
    current: dict[str, str] | None = None
    for raw_line in raw.decode("utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("<") and line.endswith(">"):
            current = blocks.setdefault(line[1:-1].strip(), {})
        elif current is not None and "=" in line:
            key, value = line.split("=", 1)
            current[key.strip()] = value.strip()
    return blocks


def read_header(stream: BinaryIO) -> BinaryHeader:
    first = stream.readline()
    if first.strip() != b"Athena binary output version=1.1":
        raise ValueError("Only AthenaK binary output version 1.1 is supported")
    preheader_count = int(_assignment(stream.readline()))
    if preheader_count != 5:
        raise ValueError(f"Unsupported AthenaK preheader size: {preheader_count}")
    values: dict[str, str] = {}
    for _ in range(preheader_count - 1):
        line = stream.readline()
        try:
            key, value = line.decode("utf-8").split("=", 1)
        except (UnicodeDecodeError, ValueError) as exc:
            raise ValueError("Malformed AthenaK binary preheader") from exc
        values[key.strip()] = value.strip()
    nvars = int(_assignment(stream.readline()))
    variable_line = stream.readline().decode("utf-8").split()
    if not variable_line or variable_line[0] != "variables:":
        raise ValueError("Malformed AthenaK variable list")
    variable_names = tuple(variable_line[1:])
    if len(variable_names) != nvars:
        raise ValueError("AthenaK variable count does not match variable list")
    header_size = int(_assignment(stream.readline()))
    parameter_dump = stream.read(header_size)
    if len(parameter_dump) != header_size:
        raise ValueError("Truncated AthenaK parameter header")
    location_size = int(values["size of location"])
    variable_size = int(values["size of variable"])
    if location_size not in (4, 8) or variable_size not in (4, 8):
        raise ValueError("AthenaK location and variable sizes must be 4 or 8 bytes")
    return BinaryHeader(
        time=float(values["time"]),
        cycle=int(values["cycle"]),
        location_size=location_size,
        variable_size=variable_size,
        variable_names=variable_names,
        parameters=_parse_parameter_dump(parameter_dump),
        data_offset=stream.tell(),
    )


def _required_int(header: BinaryHeader, block: str, key: str) -> int:
    try:
        return int(header.parameters[block][key])
    except KeyError as exc:
        raise ValueError(f"AthenaK header lacks <{block}>/{key}") from exc


def _required_float(header: BinaryHeader, block: str, key: str) -> float:
    try:
        return float(header.parameters[block][key])
    except KeyError as exc:
        raise ValueError(f"AthenaK header lacks <{block}>/{key}") from exc


def block_shape(header: BinaryHeader) -> tuple[int, int, int]:
    return tuple(_required_int(header, "meshblock", f"nx{i}") for i in (3, 2, 1))


def block_record_size(header: BinaryHeader) -> int:
    cells = int(np.prod(block_shape(header)))
    return 10 * 4 + 6 * header.location_size + cells * len(header.variable_names) * header.variable_size


def count_blocks(path: Path, header: BinaryHeader) -> int:
    payload = path.stat().st_size - header.data_offset
    record_size = block_record_size(header)
    if payload <= 0 or payload % record_size:
        raise ValueError("AthenaK file has a truncated or variable-sized MeshBlock payload")
    return payload // record_size


def iter_blocks(stream: BinaryIO, header: BinaryHeader, count: int) -> Iterator[MeshBlock]:
    nz, ny, nx = block_shape(header)
    cells = nx * ny * nz
    location_dtype = np.dtype("<f8" if header.location_size == 8 else "<f4")
    variable_dtype = np.dtype("<f8" if header.variable_size == 8 else "<f4")
    for _ in range(count):
        integer_raw = stream.read(40)
        if len(integer_raw) != 40:
            raise ValueError("Truncated AthenaK MeshBlock integer metadata")
        integers = struct.unpack("<10i", integer_raw)
        geometry_raw = stream.read(6 * header.location_size)
        if len(geometry_raw) != 6 * header.location_size:
            raise ValueError("Truncated AthenaK MeshBlock geometry")
        geometry = tuple(np.frombuffer(geometry_raw, dtype=location_dtype).astype(float))
        value_count = cells * len(header.variable_names)
        values_raw = stream.read(value_count * header.variable_size)
        if len(values_raw) != value_count * header.variable_size:
            raise ValueError("Truncated AthenaK MeshBlock variables")
        values = np.frombuffer(values_raw, dtype=variable_dtype).reshape(
            len(header.variable_names), nz, ny, nx
        )
        extents = integers[:6]
        actual_shape = (extents[5] - extents[4] + 1,
                        extents[3] - extents[2] + 1,
                        extents[1] - extents[0] + 1)
        if actual_shape != (nz, ny, nx):
            raise ValueError("Sliced and ghost-zone AthenaK outputs are unsupported")
        yield MeshBlock(
            indices=extents,
            logical_location=integers[6:9],
            level=integers[9],
            geometry=geometry,
            variables=values,
        )


def _coordinate_pair(lower: float, upper: float, cells: int, dtype: np.dtype) -> tuple[np.ndarray, np.ndarray]:
    faces = np.linspace(lower, upper, cells + 1, dtype=dtype)
    centers = 0.5 * (faces[:-1] + faces[1:])
    return faces, centers


def convert_file(source: str | Path, destination: str | Path) -> Path:
    source_path = Path(source)
    destination_path = Path(destination)
    if "rank_" in source_path.parts:
        raise ValueError("AthenaK one-file-per-rank output is unsupported")
    with source_path.open("rb") as stream:
        header = read_header(stream)
        count = count_blocks(source_path, header)

    if tuple(header.variable_names) != SOURCE_NAMES:
        raise ValueError(
            f"Expected AthenaK mhd_w_bcc variables {SOURCE_NAMES}, got {header.variable_names}"
        )
    root_size = np.array([
        _required_int(header, "mesh", "nx1"),
        _required_int(header, "mesh", "nx2"),
        _required_int(header, "mesh", "nx3"),
    ], dtype=np.int32)
    meshblock_size = np.array([
        _required_int(header, "meshblock", "nx1"),
        _required_int(header, "meshblock", "nx2"),
        _required_int(header, "meshblock", "nx3"),
    ], dtype=np.int32)
    expected_count = int(np.prod(root_size // meshblock_size))
    if np.any(root_size % meshblock_size) or count != expected_count:
        raise ValueError("Only complete uniform level-zero AthenaK output is supported")

    location_dtype = np.dtype("<f8" if header.location_size == 8 else "<f4")
    variable_dtype = np.dtype("<f8" if header.variable_size == 8 else "<f4")
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination_path.with_name(f".{destination_path.name}.tmp-{os.getpid()}")
    try:
        with h5py.File(temporary, "w") as handle:
            handle.attrs["Coordinates"] = np.bytes_("cartesian")
            handle.attrs["DatasetNames"] = np.asarray([b"prim", b"B"])
            handle.attrs["NumVariables"] = np.asarray([4, 3], dtype=np.int32)
            handle.attrs["VariableNames"] = np.asarray([name.encode("ascii") for name in TARGET_NAMES])
            handle.attrs["NumMeshBlocks"] = np.int32(count)
            handle.attrs["MaxLevel"] = np.int32(0)
            handle.attrs["MeshBlockSize"] = meshblock_size
            handle.attrs["RootGridSize"] = root_size
            handle.attrs["Time"] = header.time
            handle.attrs["NumCycles"] = np.int32(header.cycle)
            for axis in (1, 2, 3):
                lower = _required_float(header, "mesh", f"x{axis}min")
                upper = _required_float(header, "mesh", f"x{axis}max")
                handle.attrs[f"RootGridX{axis}"] = np.asarray([lower, upper, 1.0], dtype=location_dtype)

            nz, ny, nx = block_shape(header)
            primitive = handle.create_dataset("prim", (4, count, nz, ny, nx), dtype=variable_dtype)
            magnetic = handle.create_dataset("B", (3, count, nz, ny, nx), dtype=variable_dtype)
            logical_locations = handle.create_dataset("LogicalLocations", (count, 3), dtype=np.int64)
            levels = handle.create_dataset("Levels", (count,), dtype=np.int32)
            coordinate_datasets = {}
            for axis, cells in zip((1, 2, 3), (nx, ny, nz)):
                coordinate_datasets[f"x{axis}f"] = handle.create_dataset(
                    f"x{axis}f", (count, cells + 1), dtype=location_dtype
                )
                coordinate_datasets[f"x{axis}v"] = handle.create_dataset(
                    f"x{axis}v", (count, cells), dtype=location_dtype
                )

            seen: set[tuple[int, int, int]] = set()
            with source_path.open("rb") as stream:
                read_header(stream)
                for block_index, block in enumerate(iter_blocks(stream, header, count)):
                    if block.level != 0:
                        raise ValueError("AthenaK AMR output is unsupported")
                    location = tuple(int(value) for value in block.logical_location)
                    if location in seen:
                        raise ValueError(f"Duplicate AthenaK logical MeshBlock location {location}")
                    seen.add(location)
                    logical_locations[block_index] = location
                    levels[block_index] = 0
                    primitive[:, block_index] = block.variables[:4]
                    magnetic[:, block_index] = block.variables[4:7]
                    for axis, cells, lower, upper in zip(
                        (1, 2, 3), (nx, ny, nz), block.geometry[::2], block.geometry[1::2]
                    ):
                        faces, centers = _coordinate_pair(lower, upper, cells, location_dtype)
                        coordinate_datasets[f"x{axis}f"][block_index] = faces
                        coordinate_datasets[f"x{axis}v"][block_index] = centers
            expected_locations = {
                (i, j, k)
                for k in range(root_size[2] // meshblock_size[2])
                for j in range(root_size[1] // meshblock_size[1])
                for i in range(root_size[0] // meshblock_size[0])
            }
            if seen != expected_locations:
                raise ValueError("AthenaK binary does not cover the complete uniform root grid")
            handle.flush()
        os.replace(temporary, destination_path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination_path


def output_name(run_name: str, source: str | Path) -> str:
    match = re.search(r"\.(\d+)\.bin$", Path(source).name)
    if match is None:
        raise ValueError(f"Cannot determine AthenaK output number from {Path(source).name}")
    return f"{run_name}.out2.{int(match.group(1)):05d}.athdf"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    convert_file(args.source, args.destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
