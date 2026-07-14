from __future__ import annotations

import importlib.util
import struct
import sys
import tempfile
import json
import unittest
from unittest import mock
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


pipeline = load_module("pipeline", ROOT / "scripts" / "pipeline.py")
bfield = load_module("make_bfield_slices", ROOT / "scripts" / "make_bfield_slices.py")
converter = load_module("athenak_to_athdf", ROOT / "scripts" / "athenak_to_athdf.py")
harris_preflight = load_module(
    "validate_harris_preflight", ROOT / "scripts" / "validate_harris_preflight.py"
)


def example_config(solver: str = "athena++"):
    return {
        "_config_path": str(ROOT / "configs" / "example.toml"),
        "paths": {
            "athena_source": "athena",
            "build_dir": "build/athena",
            "athenak_source": "athenak",
            "athenak_build_dir": "build/athenak",
            "output_root": "outputs",
        },
        "execution": {"solver": solver, "backend": "linux", "threads": 2},
        "build": {"eos": "isothermal", "mpi": False, "openmp": False},
        "simulation": {
            "run_name": "case", "resolution": 16, "meshblock": 8,
            "box_length": 1.0, "tlim": 0.1, "cfl": 0.3, "rho0": 1.0,
            "sound_speed": 0.75,
        },
        "harris_sheet": {
            "b0": 1.0, "guide_b3": 0.1, "sheet_width": 0.05,
            "noise_amplitude": 0.001,
        },
        "output": {
            "history_interval": 0.05, "snapshot_interval": 0.1,
            "restart_interval": 1.0, "snapshot_policy": "final",
        },
        "athenak": {
            "revision": pipeline.ATHENAK_REVISION, "device": "cpu", "kokkos_arch": "",
            "integrator": "rk2", "reconstruction": "plm", "nlow": 2, "nhigh": 3,
        },
    }


def write_athenak_fixture(path: Path) -> list[np.ndarray]:
    parameters = """<job>\nbasename = fixture\n<mesh>\nnghost = 2\nnx1 = 4\nnx2 = 2\nnx3 = 2\nx1min = -0.5\nx1max = 0.5\nx2min = -0.5\nx2max = 0.5\nx3min = -0.5\nx3max = 0.5\n<meshblock>\nnx1 = 2\nnx2 = 2\nnx3 = 2\n""".encode()
    prefix = (
        b"Athena binary output version=1.1\n"
        b"  size of preheader=5\n"
        b"  time=2.5000000000000000e-01\n"
        b"  cycle=7\n"
        b"  size of location=8\n"
        b"  size of variable=4\n"
        b"  number of variables=7\n"
        b"  variables:  dens  velx  vely  velz  bcc1  bcc2  bcc3  \n"
        + f"  header offset={len(parameters)}\n".encode()
        + parameters
    )
    blocks = []
    with path.open("wb") as stream:
        stream.write(prefix)
        for block_index, (logical_x, xmin, xmax) in enumerate(((0, -0.5, 0.0), (1, 0.0, 0.5))):
            stream.write(struct.pack("<10i", 2, 3, 2, 3, 2, 3, logical_x, 0, 0, 0))
            stream.write(struct.pack("<6d", xmin, xmax, -0.5, 0.5, -0.5, 0.5))
            values = np.empty((7, 2, 2, 2), dtype="<f4")
            values[0] = 1.0 + block_index
            values[1] = 0.1 + block_index
            values[2] = 0.2
            values[3] = 0.3
            values[4] = 1.0
            values[5] = 0.25
            values[6] = -0.5
            stream.write(values.tobytes(order="C"))
            blocks.append(values)
    return blocks


class SolverDispatchTests(unittest.TestCase):
    def test_build_and_analyze_dispatch(self):
        cfg = example_config("athenak")
        with mock.patch.object(pipeline, "build_athenak", return_value=Path("athena")) as build:
            self.assertEqual(pipeline.build(cfg), Path("athena"))
            build.assert_called_once_with(cfg, False)
        with mock.patch.object(pipeline, "analyze_athenak", return_value=Path("analysis")) as analyze:
            self.assertEqual(pipeline.analyze(cfg), Path("analysis"))
            analyze.assert_called_once_with(cfg, None)

    def test_default_solver_preserves_athenapp_name_and_input(self):
        cfg = example_config()
        self.assertEqual(pipeline.solver_name(cfg), "athena++")
        self.assertEqual(pipeline.effective_run_name(cfg), "case")
        text = pipeline.render_athinput(cfg)
        self.assertIn("problem_id = case", text)
        self.assertIn("b0 = 1", text)
        self.assertIn("sheet_width = 0.05", text)

    def test_athenak_input_and_stable_name(self):
        cfg = example_config("athenak")
        self.assertEqual(pipeline.effective_run_name(cfg), "case_athenak")
        text = pipeline.render_athinput(cfg)
        self.assertIn("basename = case_athenak", text)
        self.assertIn("eos = isothermal", text)
        self.assertIn("integrator = rk2", text)
        self.assertIn("reconstruct = plm", text)
        self.assertIn("b0 = 1", text)
        self.assertNotIn("turb_driving", text)

    def test_athenak_particle_input(self):
        cfg = example_config("athenak")
        cfg["athenak"].update(device="cuda", kokkos_arch="AMPERE80")
        cfg["particles"] = {
            "enabled": True, "nparticles": 16, "injection_radius": 0.01,
            "velocity_scale": 0.0, "track_interval": 0.02,
        }
        text = pipeline.render_athinput(cfg)
        self.assertIn("<particles>", text)
        self.assertIn("file_type = trk", text)
        self.assertIn("nparticles = 16", text)

    def test_cuda_cmake_command(self):
        cfg = example_config("athenak")
        cfg["athenak"].update(device="cuda", kokkos_arch="AMPERE80")
        cfg["build"]["mpi"] = True
        command = pipeline.athenak_cmake_command(cfg, Path("/src"), Path("/build"))
        self.assertIn("-DKokkos_ENABLE_CUDA=ON", command)
        self.assertIn("-DKokkos_ARCH_AMPERE80=ON", command)
        self.assertIn("-DAthena_ENABLE_MPI=ON", command)
        self.assertTrue(any("nvcc_wrapper" in item for item in command))

    def test_cpu_cmake_command_uses_native_arch(self):
        cfg = example_config("athenak")
        command = pipeline.athenak_cmake_command(cfg, Path("source"), Path("build"))
        self.assertIn("-DKokkos_ARCH_NATIVE=ON", command)
        self.assertNotIn("-DKokkos_ENABLE_CUDA=ON", command)


class AthenaKConverterTests(unittest.TestCase):
    def test_streaming_multiblock_conversion(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            root = Path(temporary)
            source = root / "fixture.out2.00003.bin"
            expected = write_athenak_fixture(source)
            destination = root / converter.output_name("fixture_athenak", source)
            converter.convert_file(source, destination)
            self.assertEqual(destination.name, "fixture_athenak.out2.00003.athdf")
            self.assertFalse(any(root.glob("*.tmp-*")))
            with h5py.File(destination, "r") as handle:
                self.assertEqual(tuple(x.decode() for x in handle.attrs["DatasetNames"]), ("prim", "B"))
                self.assertEqual(tuple(handle.attrs["NumVariables"]), (4, 3))
                self.assertEqual(
                    tuple(x.decode() for x in handle.attrs["VariableNames"]),
                    converter.TARGET_NAMES,
                )
                np.testing.assert_array_equal(handle.attrs["RootGridSize"], [4, 2, 2])
                np.testing.assert_array_equal(handle.attrs["MeshBlockSize"], [2, 2, 2])
                np.testing.assert_array_equal(handle["LogicalLocations"], [[0, 0, 0], [1, 0, 0]])
                np.testing.assert_array_equal(handle["Levels"], [0, 0])
                self.assertEqual(handle["prim"].dtype, np.dtype("<f4"))
                self.assertEqual(handle["B"].dtype, np.dtype("<f4"))
                np.testing.assert_array_equal(handle["prim"][:, 0], expected[0][:4])
                np.testing.assert_array_equal(handle["B"][:, 1], expected[1][4:])
                np.testing.assert_allclose(handle["x1f"][0], [-0.5, -0.25, 0.0])
                np.testing.assert_allclose(handle["x1v"][1], [0.125, 0.375])
            diagnostics = pipeline.snapshot_diagnostics(destination, 0.75)
            self.assertAlmostEqual(diagnostics["mean_density"], 1.5)
            np.testing.assert_allclose(diagnostics["mean_magnetic_field"], [1.0, 0.25, -0.5])
            velocity_slice = pipeline.extract_velocity_slice(destination, "x3", 0)
            self.assertEqual(velocity_slice["velocity"][0].shape, (2, 4))
            raw = pipeline.athenak_snapshot_diagnostics(source, 0.75)
            self.assertEqual(raw["source_filename"], source.name)
            self.assertAlmostEqual(raw["mean_density"], 1.5)
            self.assertTrue(raw["finite_state"])
            self.assertTrue(np.isfinite(raw["max_abs_current_j3_proxy"]))


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
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
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


class HarrisWorkflowTests(unittest.TestCase):
    def test_physical_half_thickness_and_particle_timestep(self):
        source = (ROOT / "solver" / "athenak_mhd_turbulence.cpp").read_text(
            encoding="utf-8"
        )
        self.assertIn("sheet_scale = two_pi * sheet_width / l2", source)
        self.assertIn("pmbp->ppart->dtnew", source)
        self.assertIn("std::min(mbsize.h_view(0).dx1", source)
        self.assertAlmostEqual(256 * 0.05, 12.8)

    def test_production_config_is_fp32_particle_free_and_bounded(self):
        cfg = pipeline.load_config(ROOT / "configs" / "harris-sheet-athenak-gpu.toml")
        self.assertTrue(cfg["build"]["single_precision"])
        self.assertFalse(cfg["particles"]["enabled"])
        self.assertEqual(cfg["simulation"]["nlim"], 5000)
        self.assertEqual(cfg["execution"]["simulation_timeout"], 6600)

    def test_timeout_reaps_process_and_preserves_clear_error(self):
        cfg = example_config()
        with self.assertRaisesRegex(pipeline.RunTimeoutError, "process group was terminated"):
            pipeline.run_backend(
                [sys.executable, "-c", "import time; time.sleep(30)"],
                ROOT,
                cfg,
                timeout_seconds=0.05,
            )

    def test_convert_reuses_peak_current_selection(self):
        cfg = example_config("athenak")
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            run_dir = Path(temporary)
            analysis = run_dir / "analysis"
            analysis.mkdir()
            summary = {"snapshot_selection": {"snapshot": {"file": "bin/selected.bin"}}}
            (analysis / "diagnostics.json").write_text(json.dumps(summary), encoding="utf-8")
            products = {
                "source_snapshot": str(run_dir / "bin" / "selected.bin"),
                "selected_athdf": str(analysis / "selected_snapshot" / "selected.athdf"),
                "converted_snapshot": str(analysis / "selected_snapshot" / "selected.h5"),
                "bfield_slice_directory": str(analysis / "bfield_slices"),
            }
            with mock.patch.object(pipeline, "run_directory", return_value=run_dir), \
                 mock.patch.object(pipeline, "analyze") as analyze, \
                 mock.patch.object(
                     pipeline, "postprocess_selected_snapshot", return_value=products
                 ) as postprocess:
                outputs = pipeline.convert(cfg)
            analyze.assert_not_called()
            postprocess.assert_called_once()
            self.assertIn(Path(products["selected_athdf"]), outputs)

    def test_harris_preflight_parity_threshold(self):
        common = {
            "finite": True, "minimum_density": 0.9, "mass_drift": 1.0e-7,
            "timestep_collapse": 2.0, "field_reversal_preserved": True,
            "finite_current_proxy": True, "final_time": 0.1, "final_cycle": 20,
            "tlim": 0.1, "nlim": 5000, "kinetic_energy_density": 1.0,
            "magnetic_energy_density": 2.0, "harris_reversal_contrast": 1.9,
            "max_abs_current_j3_proxy": 20.0,
        }
        fp64 = dict(common, kinetic_energy_density=1.04)
        self.assertTrue(harris_preflight.validate(common, fp64)["passed"])
        fp64["kinetic_energy_density"] = 1.10
        self.assertFalse(harris_preflight.validate(common, fp64)["passed"])

    def test_overlay_provenance_is_pinned(self):
        overlay = (ROOT / "scripts" / "apply_athenak_forcing_overlay.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("572f644f3ab3379a32ea2f0bec1658348141dc19", overlay)
        self.assertEqual(
            pipeline.ATHENAK_FORCING_OVERLAY_SHA256,
            "769352f948ec26934db7df8ea4933d99a25fe8bb459909a523aa8923d750ba8f",
        )
        particle_overlay = (ROOT / "scripts" / "apply_athenak_particle_overlay.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("tracked_particle_counter", particle_overlay)
        self.assertIn("6*sizeof(float)*outpart(p).tag", particle_overlay)
        self.assertIn("&(data[6*p])", particle_overlay)


if __name__ == "__main__":
    unittest.main()
