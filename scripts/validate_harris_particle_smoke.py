#!/usr/bin/env python3
"""Validate Harris zero-velocity particle allocation/tracking smoke output."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pipeline


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    cfg = pipeline.load_config(args.config)
    particles = cfg.get("particles", {})
    expected = int(particles.get("nparticles", 0))
    run_dir = pipeline.run_directory(cfg)
    errors: list[str] = []
    if not particles.get("enabled", False) or expected != 1024:
        errors.append("smoke config must enable exactly 1,024 particles")
    if float(particles.get("velocity_scale", -1.0)) != 0.0:
        errors.append("smoke particles must have zero initial drift velocity")
    log_text = (run_dir / "run.log").read_text(encoding="utf-8", errors="replace")
    if "FATAL ERROR" in log_text or "Terminating on time limit" not in log_text:
        errors.append("particle smoke did not complete cleanly on tlim")
    metadata = json.loads((run_dir / "solver_metadata.json").read_text(encoding="utf-8"))
    if not metadata.get("particles", {}).get("enabled", False):
        errors.append("solver metadata does not record enabled particles")
    track_files = sorted((run_dir / "trk").glob("*.trk"))
    vtk_files = sorted((run_dir / "pvtk").glob("*.vtk"))
    tracked_records = 0
    if track_files:
        tracked_records = len(re.findall(
            rb"ntracked_prtcls=\s*" + str(expected).encode(), track_files[-1].read_bytes()
        ))
    if tracked_records < 2:
        errors.append("tracked-particle output lacks repeated 1,024-particle records")
    point_counts = []
    for path in vtk_files:
        match = re.search(rb"POINTS\s+(\d+)\s+float", path.read_bytes())
        point_counts.append(int(match.group(1)) if match else -1)
    if len(point_counts) < 2 or any(count != expected for count in point_counts):
        errors.append("particle VTK output does not contain 1,024 owned particles per dump")
    report = {
        "passed": not errors,
        "errors": errors,
        "run_directory": str(run_dir),
        "expected_particles": expected,
        "track_files": [str(path) for path in track_files],
        "tracked_record_count": tracked_records,
        "vtk_files": [str(path) for path in vtk_files],
        "vtk_point_counts": point_counts,
        "claim": "allocation/ownership/tracking/output smoke only; drift is not Lorentz transport",
    }
    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
