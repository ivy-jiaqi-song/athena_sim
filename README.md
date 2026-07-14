# Athena++ / AthenaK Harris Sheet MHD

This repository drives a three-dimensional isothermal Harris Sheet simulation with Athena++ or the GPU-ready AthenaK/Kokkos backend. The initialized box is periodic and split into upper/lower magnetic-field domains by a current sheet at `x2=0`: `B1` reverses sign across the sheet, `B2=0`, and `B3` is an optional guide field. A small deterministic velocity perturbation seeds the collapse/tearing dynamics while the run remains in the fluid MHD regime.

## Setup

Requirements are Python 3.11+, NumPy, h5py, Matplotlib, Julia 1.10+, and a Linux C++17 build environment. Athena++ additionally needs FFTW and HDF5. AthenaK needs CMake and a recursive AthenaK checkout; CUDA builds need the CUDA toolkit, `nvcc`, and a Kokkos architecture matching the local NVIDIA GPU. MPI runs additionally require OpenMPI and an MPI compiler.

Run the full pipeline with a machine-local config:

```bash
python scripts/pipeline.py all --config configs/harris-sheet-athenak-gpu.toml --clean --overwrite
```

Individual stages are also available:

```bash
python scripts/pipeline.py build --config configs/harris-sheet-athenak-gpu.toml --clean
python scripts/pipeline.py run --config configs/harris-sheet-athenak-gpu.toml --overwrite
python scripts/pipeline.py convert --config configs/harris-sheet-athenak-gpu.toml
python scripts/pipeline.py analyze --config configs/harris-sheet-athenak-gpu.toml
```

`run` writes simulation outputs only. `analyze` streams AthenaK `.bin` snapshots
directly (Athena++ `.athdf` snapshots directly), selects the absolute peak-current
snapshot, and materializes only that selected AthenaK snapshot. `convert` reuses
existing selection metadata or performs diagnostics and selection first. `all`
runs `build -> run -> analyze` without a bulk conversion stage.

Before production, build and run the tracked N=64 FP32/FP64 fluid preflights and
validate them with `scripts/validate_harris_preflight.py`. The dedicated N=32
particle smoke config contains 1,024 zero-velocity drift particles and validates
allocation, ownership, tracking, and output plumbing only.

## Configuration

Tracked Harris Sheet examples are:

- `configs/harris-sheet-athenak-gpu.toml`: CUDA AthenaK run, writing below `/home/user0001/MHDFlows_replicate/outputs/hs_sim`.
- `configs/harris-sheet-athenak-gpu-preflight-fp32.toml` and `-fp64.toml`: particle-free N=64 parity runs.
- `configs/harris-sheet-athenak-particle-smoke.toml`: N=32 particle plumbing smoke run.
- `configs/harris-sheet-athenapp.toml`: Athena++ parity config for local/CPU environments.

Important controls:

- `simulation.resolution`, `meshblock`, `box_length`, `tlim`, `sound_speed`, `rho0`.
- `harris_sheet.b0`: reversing tangential field amplitude.
- `harris_sheet.guide_b3`: optional guide field.
- `harris_sheet.sheet_width`: physical Harris half-thickness; the periodic double-sheet field is `B1 = b0*tanh(sin(2*pi*x2/L)/(2*pi*sheet_width/L))`.
- `harris_sheet.noise_amplitude`: small initial velocity perturbation.
- `output.snapshot_policy`: `final`, `peak_kinetic`, or `peak_current` for selecting the snapshot converted to `.h5` and B-field slices.
- AthenaK-only `particles.enabled`: center-injected tracked particles for a non-feedback trajectory sanity check.

## Outputs

Each run is written below `output_root/run_name/`. AthenaK appends `_athenak` to the run name.

- `analysis/selected_snapshot/*.athdf`: the one selected AthenaK snapshot retained for provenance (Athena++ sources remain in place).
- `bin/*.bin`: AthenaK shared `mhd_w_bcc` snapshots.
- `*.hst`: volume-averaged history.
- `trk/*.trk`: AthenaK tracked particle trajectory records when particles are enabled.
- `pvtk/*.part.vtk`: AthenaK particle-position VTK snapshots when particles are enabled.
- `analysis/energy_history.png`: kinetic, magnetic-fluctuation, turbulent, and total magnetic energy history.
- `analysis/diagnostics.csv` and `analysis/diagnostics.json`: snapshot diagnostics including reversal contrast and current-sheet proxy.
- `analysis/selected_snapshot/*.h5`: selected `.athdf` converted to contiguous cubes.
- `analysis/bfield_slices/*.png`: three orthogonal magnetic slices from the selected snapshot.

AthenaK's current particle pusher in the remote checkout is `drift`, so these particles are an internal output/plumbing sanity check rather than a full Lorentz-force transport calculation.
