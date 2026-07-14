#!/usr/bin/env python3
"""Validate paired FP32/FP64 AthenaK MA08 preflight runs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pipeline


def advancing_timesteps(history: dict[str, list[float]]) -> list[float]:
    """Return physical timesteps, excluding duplicate terminal history records."""
    times = [float(value) for value in history.get("time", [])]
    timesteps = [float(value) for value in history.get("dt", [])]
    if len(times) != len(timesteps):
        return [value for value in timesteps if value > 0.0]
    scale = max(1.0, *(abs(value) for value in times)) if times else 1.0
    tolerance = 1.0e-7 * scale
    return [
        timestep
        for index, timestep in enumerate(timesteps)
        if timestep > 0.0
        and (index == 0 or times[index] > times[index - 1] + tolerance)
    ]


def summarize(config_path: Path) -> dict[str, float | int | bool | str]:
    cfg = pipeline.load_config(config_path)
    run_dir = pipeline.run_directory(cfg)
    name = pipeline.effective_run_name(cfg)
    snapshots = sorted((run_dir / "bin").glob(f"{name}.out2.*.bin"))
    if len(snapshots) < 2:
        raise RuntimeError(f"Need at least two raw snapshots in {run_dir}")
    diagnostics = [
        pipeline.athenak_snapshot_diagnostics(path, float(cfg["simulation"]["sound_speed"]))
        for path in snapshots
    ]
    history_path = run_dir / f"{name}.mhd.hst"
    history = pipeline.read_history(history_path)
    dt_values = advancing_timesteps(history)
    initial_mass = float(diagnostics[0]["mean_density"])
    final_mass = float(diagnostics[-1]["mean_density"])
    final = diagnostics[-1]
    return {
        "config": str(config_path),
        "run_directory": str(run_dir),
        "finite": all(bool(item["finite_state"]) for item in diagnostics),
        "minimum_density": min(float(item["minimum_density"]) for item in diagnostics),
        "mass_drift": abs(final_mass - initial_mass) / abs(initial_mass),
        "timestep_collapse": (
            max(dt_values) / min(dt_values) if dt_values else math.inf
        ),
        "final_time": float(final["time"]),
        "final_cycle": int(final["cycle"]),
        "kinetic_energy_density": float(final["kinetic_energy_density"]),
        "magnetic_energy_density": float(final["magnetic_energy_density"]),
        "alfvenic_mach_magnetic": float(final["alfvenic_mach_magnetic"]),
        "tlim": float(cfg["simulation"]["tlim"]),
        "nlim": int(cfg["simulation"].get("nlim", -1)),
    }


def relative_difference(left: float, right: float) -> float:
    return abs(left - right) / max(abs(left), abs(right), 1.0e-30)


def validate(fp32: dict, fp64: dict) -> dict:
    errors: list[str] = []
    for label, result in (("fp32", fp32), ("fp64", fp64)):
        if not result["finite"]:
            errors.append(f"{label}: non-finite state")
        if result["minimum_density"] <= 0.0:
            errors.append(f"{label}: non-positive density")
        if result["mass_drift"] > 1.0e-5:
            errors.append(f"{label}: mass drift {result['mass_drift']:.3g} > 1e-5")
        if result["timestep_collapse"] > 10.0:
            errors.append(
                f"{label}: timestep max/min {result['timestep_collapse']:.3g} > 10"
            )
        # AthenaK writes FP32 snapshot times with small representation error.
        tolerance = max(
            64.0 * math.ulp(max(1.0, float(result["tlim"]))),
            1.0e-6 * max(1.0, abs(float(result["tlim"]))),
        )
        if abs(float(result["final_time"]) - float(result["tlim"])) > tolerance:
            errors.append(f"{label}: did not terminate on tlim")
        if int(result["nlim"]) > 0 and int(result["final_cycle"]) >= int(result["nlim"]):
            errors.append(f"{label}: reached nlim")
    differences = {}
    for key in (
        "kinetic_energy_density",
        "magnetic_energy_density",
        "alfvenic_mach_magnetic",
    ):
        difference = relative_difference(float(fp32[key]), float(fp64[key]))
        differences[key] = difference
        if difference > 0.05:
            errors.append(f"FP32/FP64 {key} difference {difference:.3%} > 5%")
    return {"passed": not errors, "errors": errors, "relative_differences": differences}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fp32", type=Path)
    parser.add_argument("fp64", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result32 = summarize(args.fp32)
    result64 = summarize(args.fp64)
    report = {"fp32": result32, "fp64": result64, "validation": validate(result32, result64)}
    text = json.dumps(report, indent=2, allow_nan=False)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    return 0 if report["validation"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
