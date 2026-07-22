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

The AthenaK build verifies the external checkout revision and Kokkos submodule, copies the tree below `athenak_build_dir/source`, installs `solver/athenak_mhd_turbulence.cpp`, applies the repository-owned forcing overlay plus narrow archived-source FP32 `Real` compatibility corrections, verifies the expected SHA256, and builds out of source below `athenak_build_dir/build`. CPU builds enable `Kokkos_ARCH_NATIVE`. CUDA builds enable `Kokkos_ENABLE_CUDA`, select `kokkos_arch`, and use Kokkos' `nvcc_wrapper`. Neither external source checkout is modified.

## Configuration

The tracked [example configuration](configs/example.toml) is a $256^3$, 16-rank MPI case targeting moderately supersonic turbulence and $M_A \approx 0.9$. Important controls are:

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
- `[power_spectra]`: controls the selected-snapshot parallel/perpendicular spectra. Set `enabled = false` to skip the product. Fitting is separate and disabled by default with `fit_enabled = false`. Zero-valued `plot_k_max`, `fit_k_min`, and `fit_k_max` request automatic limits.

Athena++ distributes MeshBlocks across ranks and threads, so configure enough MeshBlocks to keep every worker occupied. The example uses 512 MeshBlocks, giving 32 blocks per rank with 16 MPI ranks.

## Outputs and diagnostics

Each run is written below `output_root/run_name/`. The default AthenaK name is suffixed `_athenak` to prevent backend collisions:

- `*.athdf`: primitive-variable snapshots.
- `bin/*.bin`: AthenaK's shared `mhd_w_bcc` snapshots (AthenaK only).
- `*.rst`: restart files.
- `*.hst`: volume-averaged history.
- `analysis/energy_history.png`: kinetic, fluctuating magnetic, turbulent, and total magnetic energy histories.
- `analysis/diagnostics.csv`: scalar diagnostics by snapshot.
- `analysis/diagnostics.json`: formulas, saturation assessment, $M_s$, two $M_A$ estimators, field statistics, plasma beta, target selection, and derived plot products.
- `analysis/selected_snapshot/*.athdf`: the selected AthenaK snapshot retained for provenance.
- `analysis/selected_snapshot/*.h5`: only the selected snapshot converted to contiguous cubes.
- `analysis/bfield_slices/*.png`: three orthogonal magnetic slices from the selected snapshot.
- `analysis/j_histograms/*.png`: $J = \nabla \times B$ component histograms from the selected snapshot.
- `analysis/power_spectra/power_spectra_parallel_perpendicular.{png,csv,json}`: reduced magnetic and kinetic spectra parallel and perpendicular to the selected snapshot's mean magnetic field.

The primary definitions are

$$
M_s = \frac{v_{\mathrm{rms},\rho}}{c_s}
$$

$$
M_A = \frac{v_{\mathrm{rms},\rho}}{|\langle B \rangle| / \sqrt{\langle \rho \rangle}}
$$

$$
M_{A,B} = \frac{\delta B_{\mathrm{rms}}}{|\langle B \rangle|}
$$

$M_s$ is the density-weighted turbulent rms velocity divided by the isothermal sound speed. $M_A$ compares the same turbulent velocity with the Alfven speed from the mean guide field. $M_{A,B}$ is the magnetic-fluctuation estimator; this is the value used when `selection.metric = "ma"`.

### Target selection

The final 25% of usable history samples is treated as the candidate saturation window. The pipeline accepts that window only when the fitted turbulent-energy drift is below 10% and the elapsed time is at least `5 * tcorr`. It then considers only snapshots whose times fall inside the accepted window.

`selection.metric = "ms"` selects the eligible snapshot nearest the requested sonic Mach number. `selection.metric = "ma"` selects the eligible snapshot nearest the requested magnetic-fluctuation Alfvenic Mach number, $M_{A,B}$. If two snapshots have the same target error, the later snapshot wins. There is intentionally no target tolerance: the nearest accepted snapshot is selected and its absolute and fractional target error are written to `analysis/diagnostics.json`.

If saturation is false or inconclusive, or if no snapshot is inside the window, analysis records the failure in `diagnostics.json` and exits without selected-snapshot conversion, magnetic slices, or J histograms.

### Selected-snapshot plots

After target selection, the pipeline converts exactly one selected snapshot to a contiguous `.h5` cube under `analysis/selected_snapshot/`. The magnetic-slice and J-histogram products are both derived from that same `.h5` file, so they correspond to the target snapshot recorded in `analysis/diagnostics.json`.

`analysis/bfield_slices/` contains three midpoint planes: `bfield_b2b3.png`, `bfield_b1b3.png`, and `bfield_b1b2.png`. Each plot shows magnetic-field magnitude as the image layer and normalized in-plane magnetic-field direction as white arrows. The arrow density is controlled by `output.quiver_stride`.

`analysis/j_histograms/jxyz_histogram.png` computes the current density spectrally as

$$
J = \nabla \times B
$$

and plots centered probability-density histograms for $J_x$, $J_y$, and $J_z$. Each panel subtracts its own mean, reports `mu` and `sigma`, and overlays a Gaussian reference with the same standard deviation. These histograms are intended as a quick intermittency and current-sheet diagnostic for the selected state, not as part of the target-selection criterion.

`analysis/power_spectra/power_spectra_parallel_perpendicular.png` contains two log-log panels: reduced spectra perpendicular and parallel to the selected snapshot's mean guide field. The same selected contiguous `.h5` cube is used for density, velocity, and cell-centered magnetic field. The magnetic fluctuation is

$$
\delta B = B - \langle B \rangle
$$

and the kinetic spectral variable is density weighted,

$$
u_K = \sqrt{\rho}\left(v - \frac{\sum \rho v}{\sum \rho}\right).
$$

Mode powers use NumPy's full complex FFT with explicit normalization `fftn(field) / field.size`, so Parseval is checked as

$$
\sum_k P_B(k) = \frac{1}{2}\langle |\delta B|^2 \rangle,
\quad
\sum_k P_K(k) = \frac{1}{2}\langle \rho |\delta v|^2 \rangle.
$$

The reduced spectra are formed from all 3D Fourier modes, not from a line or slice. For the usual x1-aligned guide field, the parallel spectrum bins all modes by $|k_{x1}|$ and the perpendicular spectrum bins all modes by $\sqrt{k_{x2}^2 + k_{x3}^2}$. Bins are unit-width bins centered on positive integer dimensionless modes $kL/(2\pi)$. The plotted spectral density is summed mode energy per unit k-bin width; it is not mean power per Fourier mode. The CSV also includes `mode_count` and mean mode power diagnostics.

Zero modes are excluded from the logarithmic plot and from power-law fits. By default, spectra are saved through the common Cartesian Nyquist limit `floor(min(Nx, Ny, Nz) / 2)`, while the PNG plots only through `floor(min(Nx, Ny, Nz) / 3)` to avoid the most dissipative high-k range. For $512^3$, these defaults are 256 stored and 170 plotted. When fitting is enabled and fit bounds are left automatic, `fit_k_min = max(1, ceil(0.01 * plot_k_max))` and `fit_k_max = max(fit_k_min + 1, floor(0.2 * plot_k_max))`; for $512^3$ this gives 2 to 34. Vertical lines mark the exact fitted bins. Fit slopes depend on the chosen fit range, and high-k numerical dissipation limits physical interpretation.

AthenaK diagnostics stream directly from every shared `.bin`; Athena++ diagnostics read `.athdf` directly. `run` writes solver output only. After selection, exactly one AthenaK `.bin` is atomically materialized as `.athdf`, retained for provenance, and converted to `.h5`. The explicit `convert` command reuses valid selection metadata or performs diagnostics and selection first. `all` executes `build -> run -> analyze`. AMR, sliced/ghost-zone output, and one-file-per-rank binaries are rejected.

The AthenaK mapping recorded in `solver_metadata.json` is explicit: `mode -> turb_flag`, `energy_injection_rate -> dedt`, `correlation_time -> tcorr`, `drive_interval -> dt_turb_update`, `solenoidal_fraction -> sol_fraction`, `random_seed -> random_seed`, and `spectrum_exponent -> expo` with `spect_form = 2` and isotropic `driving_type = 0`.

Only tune $M_A$ from a run that the saturation diagnostic accepts. A useful next iteration is

$$
B_{0,\mathrm{new}} \approx B_{0,\mathrm{old}} \frac{M_{A,\mathrm{measured}}}{M_{A,\mathrm{target}}}
$$

followed by a fresh run because magnetic tension changes the saturated velocity. Repeat at an affordable resolution before committing to $512^3$.

## Production scale

For $512^3$, plan on at least 64 GiB, preferably 128 GiB, distributed across MPI ranks. One uncompressed single-precision primitive snapshot is roughly 3.5-4 GiB, so output cadence must be chosen with storage limits in mind. The streaming analyzer reads one MeshBlock at a time for global diagnostics and only the intersecting blocks for a 2D slice. The selected-snapshot power-spectrum product is memory intensive because it FFTs the contiguous cube; it processes one vector component at a time and uses one full complex scalar FFT at a time rather than retaining all six component transforms.

## NVIDIA smoke test

Set `execution.solver = "athenak"`, `athenak.device = "cuda"`, and the Kokkos architecture for the machine, then run a reduced configuration:

```bash
python scripts/pipeline.py all --solver athenak --config configs/local.toml --clean --overwrite
```

For an initial smoke test, use `resolution = 16` or `32`, choose a dividing `meshblock`, and shorten `tlim`. CUDA command generation is covered by tests; actual CUDA execution requires `nvcc` and an NVIDIA GPU.
