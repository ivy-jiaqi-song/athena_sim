# Athena++ driven MHD turbulence

This repository is a configurable driver for three-dimensional, compressible MHD turbulence simulations with Athena++. It copies an external Athena++ source tree into a disposable local build, installs the custom problem generator, runs the simulation, and analyzes every HDF5 snapshot.

## Simulation

The supplied problem generator initializes a periodic cube with uniform density, zero velocity, and a divergence-free uniform magnetic guide field. Athena++ then drives compressible turbulence over configurable Fourier modes with a configurable mixture of solenoidal and compressive forcing.

## Setup

Requirements are Python 3.11+, NumPy, h5py, Matplotlib, Julia 1.10+, Athena++ source, and a Linux dependency prefix containing a C++ compiler, FFTW, and HDF5. MPI runs additionally require OpenMPI, an MPI compiler, and MPI-enabled FFTW/HDF5.

Create a machine-local configuration after cloning:

```bash
cp configs/example.toml configs/local.toml
```

`configs/local.toml` is ignored by Git. Update `athena_source`, `dependency_prefix`, `output_root`, MPI resources, and simulation parameters for the machine before running.

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
python scripts/pipeline.py analyze --config configs/local.toml
```

The build stage copies the configured Athena++ source into `build/athena/`, installs the custom problem generator, configures FFTW/HDF5/MHD and the selected MPI/OpenMP modes, and compiles it. The original Athena++ checkout is not changed. The expected Athena++ revision has an x86 FP16 detection defect; the build helper applies the narrow `_Float16` compatibility correction only to the disposable copy.

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

Athena++ distributes MeshBlocks across ranks and threads, so configure enough MeshBlocks to keep every worker occupied. The example uses 512 MeshBlocks, giving 32 blocks per rank with 16 MPI ranks.

## Outputs and diagnostics

Each run is written below `output_root/run_name/`:

- `*.athdf`: primitive-variable snapshots.
- `*.rst`: restart files.
- `*.hst`: volume-averaged history.
- `analysis/velocity_slices/*.png`: velocity magnitude plus in-plane arrows for every snapshot.
- `analysis/energy_history.png`: kinetic, fluctuating magnetic, turbulent, and total magnetic energy histories.
- `analysis/diagnostics.csv`: scalar diagnostics by snapshot.
- `analysis/diagnostics.json`: formulas, saturation assessment, $M_s$, two $M_A$ estimators, field statistics, plasma beta, and energies.
- `analysis/selected_snapshot/*.h5`: the selected saturated snapshot converted to contiguous cubes.
- `analysis/bfield_slices/*.png`: three orthogonal magnetic slices from the selected snapshot.

The primary definitions are

\[
M_s = \frac{v_{rms,\rho}}{c_s}, \qquad
M_A = \frac{v_{rms,\rho}}{|\langle B\rangle|/\sqrt{\langle\rho\rangle}}, \qquad
M_{A,B}=\frac{\delta B_{rms}}{|\langle B\rangle|}.
\]

The final 25% of usable history samples is the candidate saturation window. The pipeline first requires less than 10% fitted turbulent-energy change across that window. It then considers only `.athdf` snapshots whose stored Athena time is inside the accepted window and minimizes absolute error against the configured target. `metric = "ms"` uses $M_s$; `metric = "ma"` uses $M_{A,B}$. If saturation is false or inconclusive, or if no snapshot is inside the window, analysis records the failure in `diagnostics.json` and exits without conversion or magnetic-slice generation.

Only tune $M_A$ from a run that the saturation diagnostic accepts. A useful next iteration is

\[
B_{0,new} \approx B_{0,old}\frac{M_{A,measured}}{M_{A,target}},
\]

followed by a fresh run because magnetic tension changes the saturated velocity. Repeat at an affordable resolution before committing to $512^3$.

## Production scale

For $512^3$, plan on at least 64 GiB, preferably 128 GiB, distributed across MPI ranks. One uncompressed single-precision primitive snapshot is roughly 3.5-4 GiB, so output cadence must be chosen with storage limits in mind. The streaming analyzer reads one MeshBlock at a time for global diagnostics and only the intersecting blocks for a 2D slice.
