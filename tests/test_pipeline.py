from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


pipeline = load_module("pipeline", ROOT / "scripts" / "pipeline.py")
bfield = load_module("make_bfield_slices", ROOT / "scripts" / "make_bfield_slices.py")


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.saturation = {"saturated": True, "window_start_time": 2.0, "window_end_time": 4.0}
        self.snapshots = [
            {"file": "early.athdf", "time": 1.0, "sonic_mach": 1.0, "alfvenic_mach_magnetic": 0.9},
            {"file": "first.athdf", "time": 2.0, "sonic_mach": 1.2, "alfvenic_mach_magnetic": 0.8},
            {"file": "later.athdf", "time": 3.0, "sonic_mach": 1.4, "alfvenic_mach_magnetic": 1.0},
        ]

    def test_selects_ms_inside_window_and_prefers_later_tie(self):
        result = pipeline.select_target_snapshot(
            self.snapshots, self.saturation, {"metric": "ms", "target": 1.3}
        )
        self.assertEqual(result["snapshot"]["file"], "later.athdf")
        self.assertEqual(result["eligible_snapshot_count"], 2)

    def test_selects_magnetic_ma(self):
        result = pipeline.select_target_snapshot(
            self.snapshots, self.saturation, {"metric": "ma", "target": 0.82}
        )
        self.assertEqual(result["diagnostic"], "alfvenic_mach_magnetic")
        self.assertEqual(result["snapshot"]["file"], "first.athdf")

    def test_rejects_unconfirmed_saturation(self):
        with self.assertRaisesRegex(RuntimeError, "saturation is unconfirmed"):
            pipeline.select_target_snapshot(
                self.snapshots, {"saturated": None, "reason": "too short"},
                {"metric": "ms", "target": 1.0},
            )

    def test_rejects_window_without_snapshots(self):
        with self.assertRaisesRegex(RuntimeError, "No Athena snapshot"):
            pipeline.select_target_snapshot(
                self.snapshots,
                {"saturated": True, "window_start_time": 5.0, "window_end_time": 6.0},
                {"metric": "ms", "target": 1.0},
            )

    def test_saturation_reports_candidate_window(self):
        time = np.linspace(0.0, 1.0, 20)
        history = {"time": time.tolist()}
        for axis in (1, 2, 3):
            history[f"{axis}-KE"] = np.full(time.shape, 1.0 / 3.0).tolist()
            history[f"{axis}-ME"] = np.zeros(time.shape).tolist()
        result = pipeline.saturation_diagnostic(history, minimum_time=0.5)
        self.assertTrue(result["saturated"])
        self.assertEqual(result["window_sample_count"], 5)
        self.assertEqual(result["window_start_time"], time[15])
        self.assertEqual(result["window_end_time"], time[-1])

    def test_saturation_requires_all_energy_columns(self):
        result = pipeline.saturation_diagnostic({"time": list(range(10))}, minimum_time=1.0)
        self.assertIsNone(result["saturated"])
        self.assertIn("missing history columns", result["reason"])


class BFieldSliceTests(unittest.TestCase):
    def test_writes_three_slices(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_path = root / "snapshot.h5"
            with h5py.File(input_path, "w") as handle:
                shape = (4, 6, 8)
                handle["i_mag_field"] = np.ones(shape)
                handle["j_mag_field"] = np.full(shape, 2.0)
                handle["k_mag_field"] = np.full(shape, 3.0)
                handle["time"] = 2.5
                handle["domain_bounds"] = [-0.5, 0.5, -1.0, 1.0, -2.0, 2.0]
            data = bfield.read_snapshot(input_path)
            outputs = [
                bfield.plot_slice(data, name, config, root / "slices", 2)
                for name, config in bfield.SLICES.items()
            ]
            self.assertTrue(all(path.is_file() for path in outputs))


if __name__ == "__main__":
    unittest.main()
