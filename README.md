# Athena++ / AthenaK driven MHD turbulence

This repository is a configurable driver for three-dimensional, compressible MHD turbulence simulations with Athena++ or the GPU-ready AthenaK/Kokkos backend. Both solvers initialize the same macroscopic experiment and use their native stochastic turbulence drivers; turbulent trajectories are not expected to match realization by realization.

## Simulation

The supplied problem generators initialize a periodic cube with uniform density, zero velocity, and a divergence-free uniform three-component magnetic guide field. The AthenaK build applies a hash-verified overlay derived from upstream `turb_fix` commit `572f644f3ab3379a32ea2f0bec1658348141dc19`: OU innovations are generated only at `drive_interval`, the final acceleration is momentum-corrected and normalized once, and it is held fixed through RK sub-stages.

## Setup

Requirements are Python 3.11+, NumPy, h5py, Matplotlib, Julia 1.10+, and a Linux C++17 build environment. Athena++ additionally needs FFTW and HDF5. AthenaK needs CMake and a recursive AthenaK checkout at commit `5f1993109bcb2e5d588ba41b4efc897408e9959a`; CUDA builds need the CUDA toolkit, `nvcc`, and a Kokkos architecture matching the local NVIDIA GPU. MPI runs additionally require OpenMPI and an MPI compiler.

Create a machine-local configuration after cloning:

```bash
cp configs/example.toml configs/local.toml
```

`configs/local.toml` is ignored by Git. Update the selected solver source/build paths, `dependency_prefix`, `output_root`, MPI resources, and simulation parameters for the machine before running. Clone AthenaK recursively when using that backend:

```bash
git clone --recursive https://github.com/IAS-Astrophysics/athenak.git /path/to/athenak
git -C /path/to/athenak checkout 5f1993109bcb2e5d588ba41b4efc897408e9959a
git -C /path/to/athenak submodule update --init --recursive
```

Install the Python dependencies and execute all stages:

```bash
python -m pip install -r requirements.txt
julia --project=. -e 'using Pkg; Pkg.instantiate()'
python scripts/pipeline.py all --config configs/local.toml --clean --overwrite
```

Individual stages are also available:

```bash
python scripts/pipeline.py build --config configs/local.toml --clean
python scripts/pipeline.py run --config configs/local.toml --overwrite
python scripts/pipeline.py convert --config configs/local.toml
python scripts/pipeline.py analyze --config configs/local.toml
```

Select a backend in `[execution]` with `solver = "athena++"` or `solver = "athenak"`, or override it per invocation with `--solver`. The Athena++ build remains unchanged: it copies the configured source into `build/athena/`, configures FFTW/HDF5/MHD and MPI/OpenMP, and compiles it. Its narrow x86 `_Float16` compatibility correction is still applied only to that disposable copy.

The AthenaK build verifies the external checkout revision and Kokkos submodule, copies the tree below `athenak_build_dir/source`, installs `solver/athenak_mhd_turbulence.cpp`, applies the repository-owned forcing overlay, verifies its expected SHA256, and builds out of source below `athenak_build_dir/build`. CPU builds enable `Kokkos_ARCH_NATIVE`. CUDA builds enable `Kokkos_ENABLE_CUDA`, select `kokkos_arch`, and use Kokkos' `nvcc_wrapper`. Neither external source checkout is modified.

## Configuration

The tracked [example configuration](configs/example.toml) is a 256^3, 16-rank MPI case targeting moderately supersonic turbulence and $M_A\approx0.9$. Important controls are:

- `resolution` and `meshblock`: global and per-block cube dimensions.
- `tlim`: simulation duration in code units.
- `sound_speed`: sets the sonic Mach-number scale for the isothermal build.
- `guide_field`: mean magnetic field, normally along x1.
- `selection.metric` and `selection.target`: select the saturated snapshot nearest a requested `ms` or magnetic `ma` value.
- `energy_injection_rate`: forcing power.
- `nlow` and `nhigh`: Athena++ drives only modes satisfying `nlow < |k| < nhigh`.
- `solenoidal_fraction = 1.0`: purely solenoidal driving.
- history, snapshot, and restart intervals.
- `mpi_ranks`: number of MPI processes launched with `mpirun`.
- `threads`: OpenMP threads and parallel build jobs. Hybrid runs require both MPI and OpenMP at build time.
- `[athenak].device`: `cpu` or `cuda`; CUDA also requires a nonempty machine-local `kokkos_arch` such as `AMPERE80`.
- `[athenak].integrator = "rk2"` and `reconstruction = "plm"`: the supported second-order counterparts to Athena++ `vl2` and second-order reconstruction.
- AthenaK includes forcing-shell endpoints, so shared exclusive `forcing.nlow`/`nhigh` bounds are translated to `nlow + 1` and `nhigh - 1`; do not duplicate the band under `[athenak]`.
- `execution.simulation_timeout`: optional process-group wall-time limit, capped at two hours. Timeout sends TERM, waits 30 seconds, then sends KILL while preserving partial output.

Athena++ distributes MeshBlocks across ranks and threads, so configure enough MeshBlocks to keep every worker occupied. The example uses 512 MeshBlocks, giving 32 blocks per rank with 16 MPI ranks.

## Outputs and diagnostics

Each run is written below `output_root/run_name/`. The default AthenaK name is suffixed `_athenak` to prevent backend collisions:

- `*.athdf`: primitive-variable snapshots.
- `bin/*.bin`: AthenaK's shared `mhd_w_bcc` snapshots (AthenaK only).
- `*.rst`: restart files.
- `*.hst`: volume-averaged history.
- `analysis/energy_history.png`: kinetic, fluctuating magnetic, turbulent, and total magnetic energy histories.
- `analysis/diagnostics.csv`: scalar diagnostics by snapshot.
- `analysis/diagnostics.json`: formulas, saturation assessment, $M_s$, two $M_A$ estimators, field statistics, plasma beta, and energies.
- `analysis/selected_snapshot/*.athdf`: the selected AthenaK snapshot retained for provenance.
- `analysis/selected_snapshot/*.h5`: only the selected snapshot converted to contiguous cubes.
- `analysis/bfield_slices/*.png`: three orthogonal magnetic slices from the selected snapshot.

The primary definitions are

\[
M_s = \frac{v_{rms,\rho}}{c_s}, \qquad
M_A = \frac{v_{rms,\rho}}{|\langle B\rangle|/\sqrt{\langle\rho\rangle}}, \qquad
M_{A,B}=\frac{\delta B_{rms}}{|\langle B\rangle|}.
\]

The final 25% of usable history samples is the candidate saturation window. The pipeline requires less than 10% fitted turbulent-energy change there and at least `5 × tcorr` elapsed time. It considers only snapshots inside the accepted window and minimizes absolute target error, with later-time tie breaking and no target tolerance. `metric = "ms"` uses $M_s$; `metric = "ma"` uses $M_{A,B}$. If saturation is false or inconclusive, or if no snapshot is inside the window, analysis records the failure in `diagnostics.json` and exits without selected-snapshot conversion or magnetic-slice generation.

AthenaK diagnostics stream directly from every shared `.bin`; Athena++ diagnostics read `.athdf` directly. `run` writes solver output only. After selection, exactly one AthenaK `.bin` is atomically materialized as `.athdf`, retained for provenance, and converted to `.h5`. The explicit `convert` command reuses valid selection metadata or performs diagnostics and selection first. `all` executes `build → run → analyze`. AMR, sliced/ghost-zone output, and one-file-per-rank binaries are rejected.

The AthenaK mapping recorded in `solver_metadata.json` is explicit: `mode → turb_flag`, `energy_injection_rate → dedt`, `correlation_time → tcorr`, `drive_interval → dt_turb_update`, `solenoidal_fraction → sol_fraction`, `random_seed → random_seed`, and `spectrum_exponent → expo` with `spect_form = 2` and isotropic `driving_type = 0`.

Only tune $M_A$ from a run that the saturation diagnostic accepts. A useful next iteration is

\[
B_{0,new} \approx B_{0,old}\frac{M_{A,measured}}{M_{A,target}},
\]

followed by a fresh run because magnetic tension changes the saturated velocity. Repeat at an affordable resolution before committing to $512^3$.

## Production scale

For $512^3$, plan on at least 64 GiB, preferably 128 GiB, distributed across MPI ranks. One uncompressed single-precision primitive snapshot is roughly 3.5-4 GiB, so output cadence must be chosen with storage limits in mind. The streaming analyzer reads one MeshBlock at a time for global diagnostics and only the intersecting blocks for a 2D slice.

## NVIDIA smoke test

Set `execution.solver = "athenak"`, `athenak.device = "cuda"`, and the Kokkos architecture for the machine, then run a reduced configuration:

```bash
python scripts/pipeline.py all --solver athenak --config configs/local.toml --clean --overwrite
```

For an initial smoke test, use `resolution = 16` or `32`, choose a dividing `meshblock`, and shorten `tlim`. CUDA command generation is covered by tests; actual CUDA execution requires `nvcc` and an NVIDIA GPU.
