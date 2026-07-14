#!/usr/bin/env python3
"""Validate paired FP32/FP64 Harris-sheet preflight runs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pipeline


def cfl_timesteps(history: dict[str, list[float]]) -> list[float]:
    values = [float(value) for value in history.get("dt", [])]
    return [value for value in values[:-1] if value > 0.0]


def summarize(config_path: Path) -> dict:
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
    history = pipeline.read_history(run_dir / f"{name}.mhd.hst")
    timesteps = cfl_timesteps(history)
    initial = diagnostics[0]
    final = diagnostics[-1]
    initial_mass = float(initial["mean_density"])
    upper = float(final["mean_b1_upper_half"])
    lower = float(final["mean_b1_lower_half"])
    return {
        "config": str(config_path),
        "run_directory": str(run_dir),
        "finite": all(bool(item["finite_state"]) for item in diagnostics),
        "minimum_density": min(float(item["minimum_density"]) for item in diagnostics),
        "mass_drift": abs(float(final["mean_density"]) - initial_mass) / abs(initial_mass),
        "timestep_collapse": max(timesteps) / min(timesteps) if timesteps else math.inf,
        "field_reversal_preserved": upper > 0.0 and lower < 0.0,
        "finite_current_proxy": all(
            math.isfinite(float(item["max_abs_current_j3_proxy"])) for item in diagnostics
        ),
        "final_time": float(final["time"]),
        "final_cycle": int(final["cycle"]),
        "tlim": float(cfg["simulation"]["tlim"]),
        "nlim": int(cfg["simulation"].get("nlim", -1)),
        "kinetic_energy_density": float(final["kinetic_energy_density"]),
        "magnetic_energy_density": float(final["magnetic_energy_density"]),
        "harris_reversal_contrast": float(final["harris_reversal_contrast"]),
        "max_abs_current_j3_proxy": float(final["max_abs_current_j3_proxy"]),
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
            errors.append(f"{label}: timestep max/min {result['timestep_collapse']:.3g} > 10")
        if not result["field_reversal_preserved"]:
            errors.append(f"{label}: Harris field reversal was not preserved")
        if not result["finite_current_proxy"]:
            errors.append(f"{label}: current proxy is non-finite")
        tolerance = 1.0e-6 * max(1.0, abs(float(result["tlim"])))
        if abs(float(result["final_time"]) - float(result["tlim"])) > tolerance:
            errors.append(f"{label}: did not terminate on tlim")
        if int(result["nlim"]) > 0 and int(result["final_cycle"]) >= int(result["nlim"]):
            errors.append(f"{label}: reached nlim")
    differences = {}
    for key in (
        "kinetic_energy_density",
        "magnetic_energy_density",
        "harris_reversal_contrast",
        "max_abs_current_j3_proxy",
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
    fp32 = summarize(args.fp32)
    fp64 = summarize(args.fp64)
    report = {"fp32": fp32, "fp64": fp64, "validation": validate(fp32, fp64)}
    text = json.dumps(report, indent=2, allow_nan=False)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    return 0 if report["validation"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
