#!/usr/bin/env python3
"""Build, run, convert, and analyze Athena++/AthenaK MHD turbulence cases."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
ATHENAPP_PGEN_SOURCE = ROOT / "solver" / "mhd_turbulence.cpp"
ATHENAK_PGEN_SOURCE = ROOT / "solver" / "athenak_mhd_turbulence.cpp"
ATHENAK_REVISION = "5f1993109bcb2e5d588ba41b4efc897408e9959a"
SOLVERS = ("athena++", "athenak")


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    with config_path.open("rb") as stream:
        cfg = tomllib.load(stream)
    cfg["_config_path"] = str(config_path)
    for section in ("paths", "execution", "build", "simulation", "forcing", "output", "selection"):
        if section not in cfg:
            raise ValueError(f"Missing [{section}] in {config_path}")
    cfg["execution"].setdefault("solver", "athena++")
    return cfg


def solver_name(cfg: dict[str, Any]) -> str:
    solver = str(cfg["execution"].get("solver", "athena++")).lower()
    if solver not in SOLVERS:
        raise ValueError(f"execution.solver must be one of {SOLVERS}, got {solver!r}")
    return solver


def set_solver(cfg: dict[str, Any], override: str | None) -> dict[str, Any]:
    if override is not None:
        cfg["execution"]["solver"] = override
    solver_name(cfg)
    return cfg


def project_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (ROOT / path).resolve()


def _decode(value: Any) -> str:
    return value.decode("ascii", "replace") if isinstance(value, bytes) else str(value)


def wsl_path(path: Path, distro: str) -> str:
    # Avoid passing backslashes through multiple Windows/WSL quoting layers.
    # Workspace paths are ordinary drive-letter paths mounted below /mnt in WSL.
    resolved = str(path.resolve())
    match = re.match(r"^([A-Za-z]):[\\/](.*)$", resolved)
    if not match:
        raise ValueError(f"Cannot map Windows path to WSL: {resolved}")
    drive, tail = match.groups()
    return f"/mnt/{drive.lower()}/{tail.replace(chr(92), '/')}"


def backend_path(path: Path, cfg: dict[str, Any]) -> str:
    backend = cfg["execution"].get("backend", "linux")
    if platform.system() == "Windows" and backend == "wsl":
        return wsl_path(path.resolve(), cfg["execution"]["wsl_distro"])
    return str(path.resolve())


def run_backend(
    command: list[str],
    cwd: Path,
    cfg: dict[str, Any],
    *,
    log_path: Path | None = None,
) -> None:
    execution = cfg["execution"]
    prefix = execution.get("dependency_prefix", "")
    threads = int(execution.get("threads", 1))
    env_updates = {
        "OMP_NUM_THREADS": str(threads),
        "OMP_PROC_BIND": "spread",
        "OMP_PLACES": "cores",
    }
    if prefix:
        env_updates["PATH"] = f"{prefix}/bin:$PATH"
        env_updates["LD_LIBRARY_PATH"] = (
            f"{prefix}/lib64:{prefix}/lib:${{LD_LIBRARY_PATH:-}}"
        )

    print(f"[pipeline] cwd: {cwd}")
    print("[pipeline] command:", " ".join(shlex.quote(item) for item in command))
    output = log_path.open("w", encoding="utf-8") if log_path else None
    try:
        if platform.system() == "Windows" and execution.get("backend") == "wsl":
            exports = "; ".join(
                f"export {key}={value}" for key, value in env_updates.items()
            )
            shell_command = (
                f"set -euo pipefail; {exports}; "
                f"cd {shlex.quote(backend_path(cwd, cfg))}; exec "
                + " ".join(shlex.quote(item) for item in command)
            )
            subprocess.run(
                [
                    "wsl.exe",
                    "-d",
                    execution["wsl_distro"],
                    "--",
                    "bash",
                    "-lc",
                    shell_command,
                ],
                check=True,
                stdout=output,
                stderr=subprocess.STDOUT if output else None,
            )
        else:
            env = os.environ.copy()
            env.update(env_updates)
            if prefix:
                env["PATH"] = f"{prefix}/bin:{os.environ.get('PATH', '')}"
                old_ld = os.environ.get("LD_LIBRARY_PATH", "")
                env["LD_LIBRARY_PATH"] = f"{prefix}/lib64:{prefix}/lib:{old_ld}"
            subprocess.run(command, check=True, cwd=cwd, env=env, stdout=output,
                           stderr=subprocess.STDOUT if output else None)
    finally:
        if output:
            output.close()


def prepare_athenapp_build_tree(cfg: dict[str, Any], clean: bool) -> Path:
    source = project_path(cfg["paths"]["athena_source"])
    build_dir = project_path(cfg["paths"]["build_dir"])
    if not (source / "configure.py").is_file():
        raise FileNotFoundError(f"Athena++ source not found at {source}")
    if clean and build_dir.exists():
        shutil.rmtree(build_dir)
    if not build_dir.exists():
        print(f"[pipeline] copying pristine Athena++ source to {build_dir}")
        shutil.copytree(
            source,
            build_dir,
            ignore=shutil.ignore_patterns(".git", ".github", "doc", "tst", "vis", "__pycache__"),
        )
    target_pgen = build_dir / "src" / "pgen" / "mhd_turbulence.cpp"
    shutil.copy2(ATHENAPP_PGEN_SOURCE, target_pgen)
    # Athena++ v24.0-145 detects GCC's __FLT16_MAX__ but then aliases the
    # ARM-only __fp16 spelling. On x86 GCC the supported spelling is _Float16.
    # Apply this narrowly to the disposable build copy, never to athena/.
    header = build_dir / "src" / "athena.hpp"
    source_text = header.read_text(encoding="utf-8")
    broken = """#if defined(__fp16) || defined(__FLT16_MAX__) || defined(__ARM_FP16_FORMAT_IEEE)
#define fp16_t __fp16
#elif defined(_Float16)
#define fp16_t _Float16
#endif"""
    fixed = """#if defined(__ARM_FP16_FORMAT_IEEE)
#define fp16_t __fp16
#elif defined(__FLT16_MAX__)
#define fp16_t _Float16
#endif"""
    if broken in source_text:
        header.write_text(source_text.replace(broken, fixed), encoding="utf-8")
    elif fixed not in source_text:
        raise RuntimeError("Downloaded Athena++ FP16 detection differs from the expected revision")
    return build_dir


def build_athenapp(cfg: dict[str, Any], clean: bool = False) -> Path:
    build_dir = prepare_athenapp_build_tree(cfg, clean)
    build_cfg = cfg["build"]
    execution = cfg["execution"]
    prefix = execution.get("dependency_prefix", "")

    configure = [
        "python3",
        "configure.py",
        "--prob=mhd_turbulence",
        "--coord=cartesian",
        f"--eos={build_cfg.get('eos', 'isothermal')}",
        "--flux=hlld",
        "-b",
        "-fft",
        "-hdf5",
    ]
    if prefix:
        configure.extend([f"--fftw_path={prefix}", f"--hdf5_path={prefix}"])
    if build_cfg.get("openmp", True):
        configure.append("-omp")
    if build_cfg.get("mpi", False):
        configure.append("-mpi")
        configure.append(f"--mpiccmd={execution.get('mpi_compiler', 'mpicxx')}")
    if build_cfg.get("single_precision", False):
        configure.append("-float")

    run_backend(configure, build_dir, cfg)
    jobs = max(1, int(execution.get("threads", 1)))
    run_backend(["make", f"-j{jobs}"], build_dir, cfg)
    binary = build_dir / "bin" / "athena"
    if not binary.exists():
        raise RuntimeError(f"Build completed without producing {binary}")
    print(f"[pipeline] built {binary}")
    return binary


def athenak_tree_paths(cfg: dict[str, Any]) -> tuple[Path, Path, Path]:
    root = project_path(cfg["paths"]["athenak_build_dir"])
    return root, root / "source", root / "build"


def _git_revision(source: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(source), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        raise ValueError(f"AthenaK source is not a Git checkout: {source}") from exc
    return result.stdout.strip()


def _validate_recursive_checkout(source: Path) -> None:
    for arguments in (("diff", "--quiet"), ("diff", "--cached", "--quiet")):
        result = subprocess.run(["git", "-C", str(source), *arguments], check=False)
        if result.returncode != 0:
            raise ValueError(f"AthenaK source has tracked local changes: {source}")
    try:
        gitlink = subprocess.run(
            ["git", "-C", str(source), "ls-tree", "HEAD", "kokkos"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.split()[2]
    except (subprocess.CalledProcessError, IndexError) as exc:
        raise ValueError("AthenaK checkout does not contain the pinned Kokkos submodule") from exc
    actual = _git_revision(source / "kokkos")
    if actual != gitlink:
        raise ValueError(f"AthenaK Kokkos submodule is at {actual}; expected {gitlink}")


def prepare_athenak_build_tree(cfg: dict[str, Any], clean: bool) -> tuple[Path, Path]:
    external_source = project_path(cfg["paths"]["athenak_source"])
    expected_revision = str(cfg["athenak"].get("revision", ATHENAK_REVISION)).strip()
    if not (external_source / "CMakeLists.txt").is_file():
        raise FileNotFoundError(f"AthenaK source not found at {external_source}")
    if not (external_source / "kokkos" / "CMakeLists.txt").is_file():
        raise FileNotFoundError("AthenaK Kokkos submodule is missing; update submodules recursively")
    if expected_revision:
        actual_revision = _git_revision(external_source)
        if actual_revision != expected_revision:
            raise ValueError(
                f"AthenaK source revision is {actual_revision}; expected {expected_revision}"
            )
        _validate_recursive_checkout(external_source)

    root, source_copy, cmake_build = athenak_tree_paths(cfg)
    if clean and root.exists():
        shutil.rmtree(root)
    if not source_copy.exists():
        root.mkdir(parents=True, exist_ok=True)
        print(f"[pipeline] copying AthenaK source to {source_copy}")
        shutil.copytree(
            external_source,
            source_copy,
            ignore=shutil.ignore_patterns(".git", ".github", "build", "__pycache__"),
        )
    target_pgen = source_copy / "src" / "pgen" / "mhd_turbulence.cpp"
    shutil.copy2(ATHENAK_PGEN_SOURCE, target_pgen)
    cmake_build.mkdir(parents=True, exist_ok=True)
    return source_copy, cmake_build


def athenak_cmake_command(cfg: dict[str, Any], source: Path, cmake_build: Path) -> list[str]:
    execution = cfg["execution"]
    build_cfg = cfg["build"]
    athenak_cfg = cfg["athenak"]
    device = str(athenak_cfg.get("device", "cpu")).lower()
    command = [
        "cmake",
        "-S", backend_path(source, cfg),
        "-B", backend_path(cmake_build, cfg),
        "-DCMAKE_BUILD_TYPE=Release",
        "-DPROBLEM=mhd_turbulence",
        f"-DAthena_SINGLE_PRECISION={'ON' if build_cfg.get('single_precision', False) else 'OFF'}",
        f"-DAthena_ENABLE_MPI={'ON' if build_cfg.get('mpi', False) else 'OFF'}",
        f"-DAthena_ENABLE_OPENMP={'ON' if build_cfg.get('openmp', False) else 'OFF'}",
    ]
    if build_cfg.get("mpi", False):
        command.append(f"-DMPI_CXX_COMPILER={execution.get('mpi_compiler', 'mpicxx')}")
    if device == "cpu":
        command.append("-DKokkos_ARCH_NATIVE=ON")
    else:
        arch = str(athenak_cfg["kokkos_arch"])
        wrapper = source / "kokkos" / "bin" / "nvcc_wrapper"
        command.extend([
            "-DKokkos_ENABLE_CUDA=ON",
            f"-DKokkos_ARCH_{arch}=ON",
            f"-DCMAKE_CXX_COMPILER={backend_path(wrapper, cfg)}",
        ])
    return command


def build_athenak(cfg: dict[str, Any], clean: bool = False) -> Path:
    validate_simulation_config(cfg)
    source, cmake_build = prepare_athenak_build_tree(cfg, clean)
    run_backend(athenak_cmake_command(cfg, source, cmake_build), ROOT, cfg)
    jobs = max(1, int(cfg["execution"].get("threads", 1)))
    run_backend(
        ["cmake", "--build", backend_path(cmake_build, cfg), "--parallel", str(jobs)],
        ROOT,
        cfg,
    )
    candidates = (cmake_build / "src" / "athena", cmake_build / "athena")
    binary = next((item for item in candidates if item.exists()), candidates[0])
    if not binary.exists():
        raise RuntimeError(f"AthenaK build completed without producing {binary}")
    print(f"[pipeline] built {binary}")
    return binary


def build(cfg: dict[str, Any], clean: bool = False) -> Path:
    return build_athenak(cfg, clean) if solver_name(cfg) == "athenak" else build_athenapp(cfg, clean)


def validate_simulation_config(cfg: dict[str, Any]) -> None:
    solver = solver_name(cfg)
    sim = cfg["simulation"]
    forcing = cfg["forcing"]
    selection = cfg["selection"]
    n = int(sim["resolution"])
    mb = int(sim["meshblock"])
    if n <= 0 or mb <= 0 or n % mb:
        raise ValueError("resolution and meshblock must be positive, and resolution % meshblock == 0")
    if n < 2 * int(forcing["nhigh"]):
        raise ValueError("resolution must be at least twice forcing.nhigh")
    if len(sim["guide_field"]) != 3:
        raise ValueError("simulation.guide_field must contain three components")
    if not 0.0 <= float(forcing["solenoidal_fraction"]) <= 1.0:
        raise ValueError("forcing.solenoidal_fraction must be in [0, 1]")
    if str(selection.get("metric", "")).lower() not in ("ms", "ma"):
        raise ValueError("selection.metric must be 'ms' or 'ma'")
    target = float(selection["target"])
    if not math.isfinite(target) or target <= 0.0:
        raise ValueError("selection.target must be a positive finite number")
    if solver == "athenak":
        if "athenak" not in cfg:
            raise ValueError("Missing [athenak] for the AthenaK solver")
        for key in ("athenak_source", "athenak_build_dir"):
            if key not in cfg["paths"]:
                raise ValueError(f"paths.{key} is required for the AthenaK solver")
        ak = cfg["athenak"]
        revision = str(ak.get("revision", "")).strip()
        if revision and revision != ATHENAK_REVISION:
            raise ValueError(f"athenak.revision must be empty or pinned to {ATHENAK_REVISION}")
        if str(ak.get("device", "cpu")).lower() not in ("cpu", "cuda"):
            raise ValueError("athenak.device must be 'cpu' or 'cuda'")
        if str(ak.get("device", "cpu")).lower() == "cuda" and not str(ak.get("kokkos_arch", "")).strip():
            raise ValueError("athenak.kokkos_arch is required for CUDA builds")
        if str(ak.get("integrator", "rk2")) != "rk2":
            raise ValueError("This workflow requires athenak.integrator = 'rk2'")
        if str(ak.get("reconstruction", "plm")) != "plm":
            raise ValueError("This workflow requires athenak.reconstruction = 'plm'")
        if (int(ak.get("nlow", 2)), int(ak.get("nhigh", 3))) != (2, 3):
            raise ValueError("AthenaK shell mapping must use athenak.nlow=2 and athenak.nhigh=3")
        if not math.isclose(float(forcing["solenoidal_fraction"]), 1.0, abs_tol=1.0e-12):
            raise ValueError("AthenaK supports only solenoidal_fraction = 1.0")


def render_athenapp_input(cfg: dict[str, Any], run_name: str | None = None,
                          guide_field: Iterable[float] | None = None) -> str:
    validate_simulation_config(cfg)
    sim = cfg["simulation"]
    forcing = cfg["forcing"]
    output = cfg["output"]
    build_cfg = cfg["build"]
    name = run_name or str(sim["run_name"])
    n = int(sim["resolution"])
    mb = int(sim["meshblock"])
    length = float(sim["box_length"])
    half = 0.5 * length
    b1, b2, b3 = tuple(guide_field or sim["guide_field"])
    eos = build_cfg.get("eos", "isothermal")
    gamma = float(sim.get("gamma", 5.0 / 3.0))
    sound_speed = float(sim.get("sound_speed", 1.0))

    return f"""<comment>
problem = Homogeneous, driven, compressible MHD turbulence

<job>
problem_id = {name}

<output1>
file_type = hst
dt = {float(output['history_interval']):.16g}

<output2>
file_type = hdf5
variable = prim
dt = {float(output['snapshot_interval']):.16g}

<output3>
file_type = rst
dt = {float(output['restart_interval']):.16g}

<time>
cfl_number = {float(sim['cfl']):.16g}
nlim = -1
tlim = {float(sim['tlim']):.16g}
integrator = {sim.get('integrator', 'vl2')}
xorder = {int(sim.get('reconstruction_order', 2))}
ncycle_out = 10

<mesh>
nx1 = {n}
x1min = {-half:.16g}
x1max = {half:.16g}
ix1_bc = periodic
ox1_bc = periodic
nx2 = {n}
x2min = {-half:.16g}
x2max = {half:.16g}
ix2_bc = periodic
ox2_bc = periodic
nx3 = {n}
x3min = {-half:.16g}
x3max = {half:.16g}
ix3_bc = periodic
ox3_bc = periodic
refinement = none

<meshblock>
nx1 = {mb}
nx2 = {mb}
nx3 = {mb}

<hydro>
gamma = {gamma:.16g}
iso_sound_speed = {sound_speed:.16g}

<problem>
rho0 = {float(sim['rho0']):.16g}
pressure0 = {float(sim.get('pressure0', 1.0)):.16g}
b1 = {float(b1):.16g}
b2 = {float(b2):.16g}
b3 = {float(b3):.16g}
eos_label = {eos}

<turbulence>
turb_flag = {int(forcing['mode'])}
dedt = {float(forcing['energy_injection_rate']):.16g}
nlow = {int(forcing['nlow'])}
nhigh = {int(forcing['nhigh'])}
expo = {float(forcing['spectrum_exponent']):.16g}
tcorr = {float(forcing['correlation_time']):.16g}
dtdrive = {float(forcing['drive_interval']):.16g}
f_shear = {float(forcing['solenoidal_fraction']):.16g}
rseed = {int(forcing['random_seed'])}
"""


def effective_run_name(cfg: dict[str, Any], run_name: str | None = None) -> str:
    if run_name:
        return run_name
    base = str(cfg["simulation"]["run_name"])
    return f"{base}_athenak" if solver_name(cfg) == "athenak" else base


def render_athenak_input(cfg: dict[str, Any], run_name: str | None = None,
                         guide_field: Iterable[float] | None = None) -> str:
    validate_simulation_config(cfg)
    sim = cfg["simulation"]
    forcing = cfg["forcing"]
    output = cfg["output"]
    ak = cfg["athenak"]
    name = effective_run_name(cfg, run_name)
    n = int(sim["resolution"])
    mb = int(sim["meshblock"])
    half = 0.5 * float(sim["box_length"])
    b1, b2, b3 = tuple(guide_field or sim["guide_field"])
    return f"""<comment>
problem = Homogeneous, driven, compressible isothermal MHD turbulence

<job>
basename = {name}

<mesh>
nghost = 2
nx1 = {n}
x1min = {-half:.16g}
x1max = {half:.16g}
ix1_bc = periodic
ox1_bc = periodic
nx2 = {n}
x2min = {-half:.16g}
x2max = {half:.16g}
ix2_bc = periodic
ox2_bc = periodic
nx3 = {n}
x3min = {-half:.16g}
x3max = {half:.16g}
ix3_bc = periodic
ox3_bc = periodic

<meshblock>
nx1 = {mb}
nx2 = {mb}
nx3 = {mb}

<time>
evolution = dynamic
integrator = {ak.get('integrator', 'rk2')}
cfl_number = {float(sim['cfl']):.16g}
nlim = -1
tlim = {float(sim['tlim']):.16g}
ndiag = 10

<mhd>
eos = isothermal
reconstruct = {ak.get('reconstruction', 'plm')}
rsolver = hlld
iso_sound_speed = {float(sim['sound_speed']):.16g}

<problem>
rho0 = {float(sim['rho0']):.16g}
b1 = {float(b1):.16g}
b2 = {float(b2):.16g}
b3 = {float(b3):.16g}

<turb_driving>
type = mhd
driving_type = 0
dedt = {float(forcing['energy_injection_rate']):.16g}
tcorr = {float(forcing['correlation_time']):.16g}
expo = {float(forcing['spectrum_exponent']):.16g}
nlow = {int(ak.get('nlow', 2))}
nhigh = {int(ak.get('nhigh', 3))}

<output1>
file_type = hst
dt = {float(output['history_interval']):.16g}

<output2>
file_type = bin
variable = mhd_w_bcc
id = out2
single_file_per_rank = false
dt = {float(output['snapshot_interval']):.16g}

<output3>
file_type = rst
dt = {float(output['restart_interval']):.16g}
"""


def render_athinput(cfg: dict[str, Any], run_name: str | None = None,
                    guide_field: Iterable[float] | None = None) -> str:
    if solver_name(cfg) == "athenak":
        return render_athenak_input(cfg, run_name, guide_field)
    return render_athenapp_input(cfg, run_name, guide_field)


def run_directory(cfg: dict[str, Any], run_name: str | None = None) -> Path:
    name = effective_run_name(cfg, run_name)
    return project_path(cfg["paths"]["output_root"]) / name


def _prepare_run_directory(cfg: dict[str, Any], overwrite: bool,
                           run_name: str | None) -> Path:
    run_dir = run_directory(cfg, run_name)
    if run_dir.exists() and overwrite:
        if run_dir == run_dir.parent or run_dir.name in ("", ".", ".."):
            raise RuntimeError(f"Refusing unsafe output removal: {run_dir}")
        shutil.rmtree(run_dir)
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {run_dir}; pass --overwrite")
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def run_athenapp(cfg: dict[str, Any], overwrite: bool = False,
                 run_name: str | None = None,
                 guide_field: Iterable[float] | None = None) -> Path:
    build_dir = project_path(cfg["paths"]["build_dir"])
    binary = build_dir / "bin" / "athena"
    if not binary.exists():
        raise FileNotFoundError("Athena++ binary is missing; run the build command first")
    run_dir = _prepare_run_directory(cfg, overwrite, run_name)
    input_path = run_dir / "athinput.generated"
    # Athena++'s parameter parser does not strip a Windows carriage return
    # from string values (for example, it reads "periodic\r"). Force LF.
    input_path.write_text(
        render_athinput(cfg, run_name, guide_field), encoding="utf-8", newline="\n"
    )

    binary_arg = backend_path(binary, cfg)
    command = [binary_arg, "-i", "athinput.generated"]
    mpi_ranks = int(cfg["execution"].get("mpi_ranks", 1))
    if cfg["build"].get("mpi", False) and mpi_ranks > 1:
        command = [cfg["execution"].get("mpi_launcher", "mpirun"), "-np", str(mpi_ranks)] + command
    log_path = run_dir / "run.log"
    run_backend(command, run_dir, cfg, log_path=log_path)
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    if "FATAL ERROR" in log_text:
        raise RuntimeError(f"Athena++ reported a fatal error; inspect {log_path}")
    print(f"[pipeline] simulation log: {log_path}")
    return run_dir


def athenak_binary(cfg: dict[str, Any]) -> Path:
    _, _, cmake_build = athenak_tree_paths(cfg)
    candidates = (cmake_build / "src" / "athena", cmake_build / "athena")
    return next((item for item in candidates if item.exists()), candidates[0])


def run_athenak(cfg: dict[str, Any], overwrite: bool = False,
                run_name: str | None = None,
                guide_field: Iterable[float] | None = None) -> Path:
    validate_simulation_config(cfg)
    binary = athenak_binary(cfg)
    if not binary.exists():
        raise FileNotFoundError("AthenaK binary is missing; run the build command first")
    run_dir = _prepare_run_directory(cfg, overwrite, run_name)
    input_path = run_dir / "athinput.generated"
    input_path.write_text(
        render_athenak_input(cfg, run_name, guide_field), encoding="utf-8", newline="\n"
    )
    command = [backend_path(binary, cfg), "-i", "athinput.generated"]
    mpi_ranks = int(cfg["execution"].get("mpi_ranks", 1))
    if cfg["build"].get("mpi", False) and mpi_ranks > 1:
        command = [cfg["execution"].get("mpi_launcher", "mpirun"), "-np", str(mpi_ranks)] + command
    log_path = run_dir / "run.log"
    run_backend(command, run_dir, cfg, log_path=log_path)
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    if "FATAL ERROR" in log_text:
        raise RuntimeError(f"AthenaK reported a fatal error; inspect {log_path}")
    metadata = {
        "solver": "athenak",
        "revision": str(cfg["athenak"]["revision"]),
        "device": str(cfg["athenak"].get("device", "cpu")),
        "kokkos_arch": str(cfg["athenak"].get("kokkos_arch", "")),
        "backend_controls": {
            "forcing.drive_interval": "not applicable; AthenaK advances OU forcing each timestep",
            "forcing.random_seed": "not applicable; pinned AthenaK uses its native internal RNG initialization",
        },
    }
    (run_dir / "solver_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"[pipeline] simulation log: {log_path}")
    convert_athenak_outputs(cfg, run_name)
    return run_dir


def run_simulation(cfg: dict[str, Any], overwrite: bool = False,
                   run_name: str | None = None,
                   guide_field: Iterable[float] | None = None) -> Path:
    if solver_name(cfg) == "athenak":
        return run_athenak(cfg, overwrite, run_name, guide_field)
    return run_athenapp(cfg, overwrite, run_name, guide_field)


def _load_athenak_converter() -> Any:
    import importlib.util

    path = ROOT / "scripts" / "athenak_to_athdf.py"
    spec = importlib.util.spec_from_file_location("athenak_to_athdf", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load AthenaK converter at {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def convert_athenak_outputs(cfg: dict[str, Any], run_name: str | None = None) -> list[Path]:
    if solver_name(cfg) != "athenak":
        print("[pipeline] Athena++ writes ATHDF directly; no conversion is required")
        return []
    run_dir = run_directory(cfg, run_name)
    binary_dir = run_dir / "bin"
    if not binary_dir.is_dir():
        raise FileNotFoundError(f"No AthenaK binary output directory found at {binary_dir}")
    name = effective_run_name(cfg, run_name)
    converter = _load_athenak_converter()
    sources = sorted(binary_dir.glob(f"{name}.out2.*.bin"))
    if not sources:
        raise FileNotFoundError(f"No shared AthenaK mhd_w_bcc binary snapshots found in {binary_dir}")
    results: list[Path] = []
    for source in sources:
        destination = run_dir / converter.output_name(name, source)
        if destination.exists() and destination.stat().st_mtime_ns >= source.stat().st_mtime_ns:
            print(f"[pipeline] already converted {source.name}")
        else:
            print(f"[pipeline] converting {source.name} -> {destination.name}")
            converter.convert_file(source, destination)
        results.append(destination)
    return results


def convert(cfg: dict[str, Any], run_name: str | None = None) -> list[Path]:
    return convert_athenak_outputs(cfg, run_name)


def _variable_map(handle: Any) -> dict[str, tuple[str, int]]:
    dataset_names = [_decode(item) for item in handle.attrs["DatasetNames"]]
    counts = [int(item) for item in handle.attrs["NumVariables"]]
    variable_names = [_decode(item) for item in handle.attrs["VariableNames"]]
    result: dict[str, tuple[str, int]] = {}
    offset = 0
    for dataset, count in zip(dataset_names, counts):
        for local_index, name in enumerate(variable_names[offset:offset + count]):
            result[name] = (dataset, local_index)
        offset += count
    return result


def _read_block(handle: Any, mapping: dict[str, tuple[str, int]], name: str,
                block: int) -> Any:
    if name not in mapping:
        raise KeyError(f"Variable {name!r} not present; available: {sorted(mapping)}")
    dataset, index = mapping[name]
    return handle[dataset][index, block, ...]


def snapshot_diagnostics(path: Path, sound_speed: float) -> dict[str, Any]:
    import h5py
    import numpy as np

    with h5py.File(path, "r") as handle:
        mapping = _variable_map(handle)
        required = ("rho", "vel1", "vel2", "vel3", "Bcc1", "Bcc2", "Bcc3")
        missing = [name for name in required if name not in mapping]
        if missing:
            raise KeyError(f"Missing variables in {path.name}: {missing}")
        count = 0
        sum_rho = sum_rho2 = sum_rho_v2 = 0.0
        sum_momentum = np.zeros(3, dtype=np.float64)
        sum_b = np.zeros(3, dtype=np.float64)
        sum_b2 = 0.0
        sum_pressure = 0.0
        has_pressure = "press" in mapping
        num_blocks = int(handle.attrs["NumMeshBlocks"])
        for block in range(num_blocks):
            rho = np.asarray(_read_block(handle, mapping, "rho", block), dtype=np.float64)
            velocity = [
                np.asarray(_read_block(handle, mapping, f"vel{i}", block), dtype=np.float64)
                for i in (1, 2, 3)
            ]
            magnetic = [
                np.asarray(_read_block(handle, mapping, f"Bcc{i}", block), dtype=np.float64)
                for i in (1, 2, 3)
            ]
            count += rho.size
            sum_rho += float(rho.sum())
            sum_rho2 += float(np.square(rho).sum())
            speed2 = sum(np.square(component) for component in velocity)
            sum_rho_v2 += float((rho * speed2).sum())
            for component in range(3):
                sum_momentum[component] += float((rho * velocity[component]).sum())
                sum_b[component] += float(magnetic[component].sum())
            sum_b2 += float(sum(np.square(component) for component in magnetic).sum())
            if has_pressure:
                sum_pressure += float(np.asarray(
                    _read_block(handle, mapping, "press", block), dtype=np.float64
                ).sum())

        mean_rho = sum_rho / count
        mean_velocity = sum_momentum / sum_rho
        turbulent_v2 = max(0.0, sum_rho_v2 / sum_rho - float(mean_velocity @ mean_velocity))
        velocity_rms = math.sqrt(turbulent_v2)
        mean_b = sum_b / count
        mean_b_strength = float(np.linalg.norm(mean_b))
        mean_b2 = sum_b2 / count
        delta_b_rms = math.sqrt(max(0.0, mean_b2 - mean_b_strength**2))
        alfven_speed = mean_b_strength / math.sqrt(mean_rho) if mean_rho > 0 else math.nan
        ma_velocity = velocity_rms / alfven_speed if alfven_speed > 0 else math.inf
        ma_magnetic = delta_b_rms / mean_b_strength if mean_b_strength > 0 else math.inf
        pressure = sum_pressure / count if has_pressure else mean_rho * sound_speed**2
        magnetic_energy = 0.5 * mean_b2
        density_variance = max(0.0, sum_rho2 / count - mean_rho**2)

        return {
            "file": path.name,
            "time": float(handle.attrs["Time"]),
            "cycle": int(handle.attrs["NumCycles"]),
            "mean_density": mean_rho,
            "density_std": math.sqrt(density_variance),
            "mean_velocity": mean_velocity.tolist(),
            "velocity_rms_density_weighted": velocity_rms,
            "sonic_mach": velocity_rms / sound_speed,
            "mean_magnetic_field": mean_b.tolist(),
            "mean_magnetic_strength": mean_b_strength,
            "magnetic_fluctuation_rms": delta_b_rms,
            "alfven_speed_mean_field": alfven_speed,
            "alfvenic_mach_velocity": ma_velocity,
            "alfvenic_mach_magnetic": ma_magnetic,
            "kinetic_energy_density": 0.5 * mean_rho * turbulent_v2,
            "magnetic_energy_density": magnetic_energy,
            "magnetic_fluctuation_energy_density": 0.5 * delta_b_rms**2,
            "mean_gas_pressure": pressure,
            "plasma_beta": pressure / magnetic_energy if magnetic_energy > 0 else math.inf,
        }


def extract_velocity_slice(path: Path, axis_name: str, requested_index: int) -> dict[str, Any]:
    import h5py
    import numpy as np

    axis_lookup = {"x1": 0, "x2": 1, "x3": 2}
    if axis_name not in axis_lookup:
        raise ValueError("slice_axis must be x1, x2, or x3")
    axis = axis_lookup[axis_name]
    with h5py.File(path, "r") as handle:
        if not np.all(np.asarray(handle["Levels"]) == 0):
            raise NotImplementedError("Slice extraction currently requires a uniform, level-0 mesh")
        mapping = _variable_map(handle)
        size = np.asarray(handle.attrs["RootGridSize"], dtype=int)
        block_size = np.asarray(handle.attrs["MeshBlockSize"], dtype=int)
        index = int(size[axis] // 2 if requested_index < 0 else requested_index)
        if not 0 <= index < int(size[axis]):
            raise IndexError(f"slice index {index} is outside axis size {size[axis]}")
        if axis == 2:
            shape, plane_axes, components = (size[1], size[0]), ("x1", "x2"), ("vel1", "vel2")
        elif axis == 1:
            shape, plane_axes, components = (size[2], size[0]), ("x1", "x3"), ("vel1", "vel3")
        else:
            shape, plane_axes, components = (size[2], size[1]), ("x2", "x3"), ("vel2", "vel3")
        velocity = [np.full(shape, np.nan, dtype=np.float64) for _ in range(3)]
        logical = np.asarray(handle["LogicalLocations"], dtype=int)
        for block, location in enumerate(logical):
            start = location * block_size
            stop = start + block_size
            if not start[axis] <= index < stop[axis]:
                continue
            local = index - start[axis]
            block_velocity = [
                np.asarray(_read_block(handle, mapping, f"vel{i}", block), dtype=np.float64)
                for i in (1, 2, 3)
            ]
            if axis == 2:
                rows = slice(start[1], stop[1]); cols = slice(start[0], stop[0])
                for component in range(3):
                    velocity[component][rows, cols] = block_velocity[component][local, :, :]
            elif axis == 1:
                rows = slice(start[2], stop[2]); cols = slice(start[0], stop[0])
                for component in range(3):
                    velocity[component][rows, cols] = block_velocity[component][:, local, :]
            else:
                rows = slice(start[2], stop[2]); cols = slice(start[1], stop[1])
                for component in range(3):
                    velocity[component][rows, cols] = block_velocity[component][:, :, local]
        if any(np.isnan(component).any() for component in velocity):
            raise RuntimeError(f"Incomplete {axis_name} slice assembled from {path.name}")
        root_axes = [np.asarray(handle.attrs[f"RootGridX{i}"], dtype=float) for i in (1, 2, 3)]
        horizontal = axis_lookup[plane_axes[0]]
        vertical = axis_lookup[plane_axes[1]]
        return {
            "velocity": velocity,
            "plane_components": components,
            "plane_axes": plane_axes,
            "extent": [root_axes[horizontal][0], root_axes[horizontal][1],
                       root_axes[vertical][0], root_axes[vertical][1]],
            "index": index,
            "time": float(handle.attrs["Time"]),
        }


def plot_velocity_slice(snapshot: Path, output_path: Path, axis: str, index: int,
                        quiver_stride: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    data = extract_velocity_slice(snapshot, axis, index)
    velocity = data["velocity"]
    magnitude = np.sqrt(sum(np.square(component) for component in velocity))
    component_lookup = {f"vel{i}": i - 1 for i in (1, 2, 3)}
    u = velocity[component_lookup[data["plane_components"][0]]]
    v = velocity[component_lookup[data["plane_components"][1]]]
    extent = data["extent"]
    ny, nx = magnitude.shape
    x = np.linspace(extent[0], extent[1], nx, endpoint=False) + (extent[1] - extent[0])/(2*nx)
    y = np.linspace(extent[2], extent[3], ny, endpoint=False) + (extent[3] - extent[2])/(2*ny)
    xx, yy = np.meshgrid(x, y)
    stride = max(1, int(quiver_stride))

    fig, ax = plt.subplots(figsize=(7.2, 6.0), constrained_layout=True)
    image = ax.imshow(magnitude, origin="lower", extent=extent, aspect="equal", cmap="viridis")
    ax.quiver(xx[::stride, ::stride], yy[::stride, ::stride],
              u[::stride, ::stride], v[::stride, ::stride], color="white",
              alpha=0.75, pivot="mid", scale=None)
    ax.set_xlabel(data["plane_axes"][0])
    ax.set_ylabel(data["plane_axes"][1])
    ax.set_title(f"Velocity slice {axis}[{data['index']}], t={data['time']:.4g}")
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label(r"$|\mathbf{v}|$")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def read_history(path: Path) -> dict[str, list[float]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    marker_indices = [index for index, line in enumerate(lines) if line.strip() == "# Athena++ history data"]
    if not marker_indices:
        raise ValueError(f"No Athena++ history header in {path}")
    start = marker_indices[-1]
    if start + 1 >= len(lines):
        raise ValueError(f"Incomplete Athena++ history header in {path}")
    names = re.findall(r"\[\d+\]=(\S+)", lines[start + 1])
    rows: list[list[float]] = []
    for line in lines[start + 2:]:
        if not line or line.startswith("#"):
            continue
        values = [float(value) for value in line.split()]
        if len(values) != len(names):
            continue
        while rows and values[0] <= rows[-1][0]:
            rows.pop()
        rows.append(values)
    return {name: [row[column] for row in rows] for column, name in enumerate(names)}


def saturation_diagnostic(history: dict[str, list[float]], minimum_time: float) -> dict[str, Any]:
    import numpy as np

    time = np.asarray(history.get("time", []), dtype=float)
    energy_names = tuple(f"{axis}-{kind}" for kind in ("KE", "ME") for axis in (1, 2, 3))
    missing = [name for name in energy_names if name not in history]
    if missing:
        return {"saturated": None, "reason": f"missing history columns: {', '.join(missing)}"}
    if any(len(history[name]) != time.size for name in energy_names):
        return {"saturated": None, "reason": "history energy columns have inconsistent lengths"}
    kinetic = sum((np.asarray(history.get(f"{i}-KE", []), dtype=float) for i in (1, 2, 3)),
                  start=np.zeros_like(time))
    magnetic = sum((np.asarray(history.get(f"{i}-ME", []), dtype=float) for i in (1, 2, 3)),
                   start=np.zeros_like(time))
    turbulent = kinetic + magnetic - (magnetic[0] if magnetic.size else 0.0)
    if time.size < 10 or time[-1] <= time[0]:
        return {"saturated": None, "reason": "fewer than 10 usable history samples"}
    if time[-1] - time[0] < minimum_time:
        return {
            "saturated": None,
            "reason": "run is too short for a saturation decision",
            "required_minimum_time": minimum_time,
            "available_time": float(time[-1] - time[0]),
        }
    start = max(0, int(0.75 * time.size))
    x = time[start:]
    y = turbulent[start:]
    mean_energy = float(np.mean(np.abs(y)))
    slope = float(np.polyfit(x, y, 1)[0])
    relative_change = abs(slope) * float(x[-1] - x[0]) / max(mean_energy, 1.0e-30)
    return {
        "saturated": relative_change < 0.10,
        "criterion": "<10% fitted turbulent-energy change over the final 25% of samples",
        "window_start_time": float(x[0]),
        "window_end_time": float(x[-1]),
        "window_sample_count": int(x.size),
        "relative_change_final_quarter": relative_change,
        "fitted_slope": slope,
    }


def select_target_snapshot(
    diagnostics: list[dict[str, Any]],
    saturation: dict[str, Any],
    selection_cfg: dict[str, Any],
) -> dict[str, Any]:
    if saturation.get("saturated") is not True:
        reason = saturation.get("reason", "energy stationarity criterion was not met")
        raise RuntimeError(f"Cannot select a target snapshot: saturation is unconfirmed ({reason})")

    metric = str(selection_cfg.get("metric", "")).lower()
    diagnostic_key = {"ms": "sonic_mach", "ma": "alfvenic_mach_magnetic"}.get(metric)
    if diagnostic_key is None:
        raise ValueError("selection.metric must be 'ms' or 'ma'")
    target = float(selection_cfg["target"])
    if not math.isfinite(target) or target <= 0.0:
        raise ValueError("selection.target must be a positive finite number")

    start = float(saturation["window_start_time"])
    end = float(saturation["window_end_time"])
    candidates = [item for item in diagnostics if start <= float(item["time"]) <= end]
    if not candidates:
        raise RuntimeError(
            f"No Athena snapshot falls inside the accepted saturation window [{start:g}, {end:g}]"
        )
    selected = min(
        candidates,
        key=lambda item: (abs(float(item[diagnostic_key]) - target), -float(item["time"])),
    )
    measured = float(selected[diagnostic_key])
    return {
        "metric": metric,
        "diagnostic": diagnostic_key,
        "target": target,
        "measured": measured,
        "absolute_error": abs(measured - target),
        "fractional_error": (measured - target) / target,
        "eligible_snapshot_count": len(candidates),
        "window_start_time": start,
        "window_end_time": end,
        "snapshot": selected,
    }


def postprocess_selected_snapshot(
    cfg: dict[str, Any], run_dir: Path, analysis_dir: Path, selection: dict[str, Any]
) -> dict[str, str]:
    snapshot = run_dir / str(selection["snapshot"]["file"])
    converted_dir = analysis_dir / "selected_snapshot"
    converted_path = converted_dir / f"{snapshot.stem}.h5"
    bfield_dir = analysis_dir / "bfield_slices"
    converted_dir.mkdir(parents=True, exist_ok=True)

    julia_script = ROOT / "scripts" / "ath2h5.jl"
    bfield_script = ROOT / "scripts" / "make_bfield_slices.py"
    run_backend(
        [
            str(cfg["execution"].get("julia_command", "julia")),
            f"--project={backend_path(ROOT, cfg)}",
            backend_path(julia_script, cfg),
            "--input", backend_path(snapshot, cfg),
            "--output", backend_path(converted_path, cfg),
        ],
        run_dir,
        cfg,
    )
    run_backend(
        [
            str(cfg["execution"].get("python_command", "python3")),
            backend_path(bfield_script, cfg),
            "--input", backend_path(converted_path, cfg),
            "--output-dir", backend_path(bfield_dir, cfg),
            "--quiver-stride", str(int(cfg["output"].get("quiver_stride", 8))),
        ],
        run_dir,
        cfg,
    )
    return {
        "converted_snapshot": str(converted_path),
        "bfield_slice_directory": str(bfield_dir),
    }


def plot_energy_history(history_path: Path, output_path: Path,
                        minimum_saturation_time: float) -> dict[str, Any]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    history = read_history(history_path)
    time = np.asarray(history["time"], dtype=float)
    kinetic_parts = [np.asarray(history.get(f"{i}-KE", np.zeros_like(time)), dtype=float)
                     for i in (1, 2, 3)]
    magnetic_parts = [np.asarray(history.get(f"{i}-ME", np.zeros_like(time)), dtype=float)
                      for i in (1, 2, 3)]
    kinetic = sum(kinetic_parts)
    magnetic = sum(magnetic_parts)
    magnetic_fluctuation = magnetic - (magnetic[0] if magnetic.size else 0.0)
    fig, ax = plt.subplots(figsize=(8.0, 5.0), constrained_layout=True)
    ax.plot(time, kinetic, label="kinetic", linewidth=2.0)
    ax.plot(time, magnetic_fluctuation, label="magnetic fluctuation", linewidth=2.0)
    ax.plot(time, kinetic + magnetic_fluctuation,
            label="turbulent (kinetic + magnetic fluctuation)", linewidth=1.5)
    ax.plot(time, magnetic, label="magnetic (including guide field)", linewidth=1.0,
            linestyle=":", alpha=0.75)
    if "tot-E" in history:
        ax.plot(time, history["tot-E"], label="total", linewidth=1.2, linestyle="--")
    ax.set_xlabel("time")
    ax.set_ylabel("volume-averaged energy density")
    ax.set_title("Athena energy history")
    ax.grid(alpha=0.25)
    ax.legend()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return saturation_diagnostic(history, minimum_saturation_time)


def analyze_common(cfg: dict[str, Any], run_name: str | None = None) -> Path:
    run_dir = run_directory(cfg, run_name)
    snapshots = sorted(run_dir.glob("*.athdf"))
    if solver_name(cfg) == "athenak":
        history_files = [run_dir / f"{effective_run_name(cfg, run_name)}.mhd.hst"]
        history_files = [path for path in history_files if path.is_file()]
    else:
        history_files = sorted(run_dir.glob("*.hst"))
    if not snapshots:
        raise FileNotFoundError(f"No .athdf snapshots found in {run_dir}")
    if not history_files:
        raise FileNotFoundError(f"No .hst history file found in {run_dir}")
    analysis_dir = run_dir / "analysis"
    slice_dir = analysis_dir / "velocity_slices"
    slice_dir.mkdir(parents=True, exist_ok=True)
    sim = cfg["simulation"]
    output_cfg = cfg["output"]
    sound_speed = float(sim["sound_speed"])

    diagnostics: list[dict[str, Any]] = []
    for snapshot in snapshots:
        print(f"[pipeline] analyzing {snapshot.name}")
        diagnostics.append(snapshot_diagnostics(snapshot, sound_speed))
        plot_velocity_slice(
            snapshot,
            slice_dir / f"{snapshot.stem}.png",
            str(output_cfg.get("slice_axis", "x3")),
            int(output_cfg.get("slice_index", -1)),
            int(output_cfg.get("quiver_stride", 8)),
        )

    scalar_keys = [key for key, value in diagnostics[0].items()
                   if not isinstance(value, (list, dict))]
    with (analysis_dir / "diagnostics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=scalar_keys)
        writer.writeheader()
        for item in diagnostics:
            writer.writerow({key: item[key] for key in scalar_keys})

    minimum_saturation_time = 5.0 * float(cfg["forcing"]["correlation_time"])
    saturation = plot_energy_history(
        history_files[-1], analysis_dir / "energy_history.png", minimum_saturation_time
    )
    selection: dict[str, Any] | None = None
    selection_error: str | None = None
    try:
        selection = select_target_snapshot(diagnostics, saturation, cfg["selection"])
    except (RuntimeError, ValueError) as exc:
        selection_error = str(exc)
    summary = {
        "run_directory": str(run_dir),
        "configuration": str(cfg["_config_path"]),
        "solver": solver_metadata(cfg, run_dir),
        "definitions": {
            "sonic_mach": "density-weighted turbulent v_rms / isothermal sound speed",
            "alfvenic_mach_velocity": "density-weighted turbulent v_rms / (|<B>|/sqrt(<rho>))",
            "alfvenic_mach_magnetic": "rms(B-<B>) / |<B>|",
            "athena_units": "magnetic permeability is unity, so magnetic energy density is B^2/2",
        },
        "saturation_heuristic": saturation,
        "target_selection": selection if selection is not None else {
            "status": "failed",
            "error": selection_error,
        },
        "final_snapshot": diagnostics[-1],
        "snapshots": diagnostics,
    }
    diagnostics_path = analysis_dir / "diagnostics.json"
    diagnostics_path.write_text(
        json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
    )
    if selection is None:
        raise RuntimeError(selection_error or "Target snapshot selection failed")

    products = postprocess_selected_snapshot(cfg, run_dir, analysis_dir, selection)
    summary["target_selection"]["products"] = products
    diagnostics_path.write_text(
        json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(f"[pipeline] analysis written to {analysis_dir}")
    return analysis_dir


def analyze_athenapp(cfg: dict[str, Any], run_name: str | None = None) -> Path:
    return analyze_common(cfg, run_name)


def analyze_athenak(cfg: dict[str, Any], run_name: str | None = None) -> Path:
    return analyze_common(cfg, run_name)


def analyze(cfg: dict[str, Any], run_name: str | None = None) -> Path:
    if solver_name(cfg) == "athenak":
        return analyze_athenak(cfg, run_name)
    return analyze_athenapp(cfg, run_name)


def solver_metadata(cfg: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    solver = solver_name(cfg)
    if solver == "athenak":
        path = run_dir / "solver_metadata.json"
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
        return {
            "solver": solver,
            "revision": str(cfg["athenak"]["revision"]),
            "device": str(cfg["athenak"].get("device", "cpu")),
            "kokkos_arch": str(cfg["athenak"].get("kokkos_arch", "")),
        }
    return {"solver": solver, "revision": "not recorded", "device": "cpu"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "run", "convert", "analyze", "all", "input"))
    parser.add_argument("--config", default=str(ROOT / "configs" / "local.toml"))
    parser.add_argument("--solver", choices=SOLVERS, help="override execution.solver")
    parser.add_argument("--clean", action="store_true", help="recreate the disposable Athena build tree")
    parser.add_argument("--overwrite", action="store_true", help="replace the selected run directory")
    parser.add_argument("--run-name", help="override simulation.run_name")
    args = parser.parse_args()

    try:
        cfg = set_solver(load_config(args.config), args.solver)
        if args.command == "input":
            sys.stdout.write(render_athinput(cfg, args.run_name))
        elif args.command == "build":
            build(cfg, clean=args.clean)
        elif args.command == "run":
            run_simulation(cfg, overwrite=args.overwrite, run_name=args.run_name)
        elif args.command == "convert":
            convert(cfg, run_name=args.run_name)
        elif args.command == "analyze":
            analyze(cfg, run_name=args.run_name)
        elif args.command == "all":
            build(cfg, clean=args.clean)
            run_simulation(cfg, overwrite=args.overwrite, run_name=args.run_name)
            convert(cfg, run_name=args.run_name)
            analyze(cfg, run_name=args.run_name)
    except (FileNotFoundError, FileExistsError, KeyError, RuntimeError, ValueError,
            subprocess.CalledProcessError) as exc:
        print(f"[pipeline] ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
