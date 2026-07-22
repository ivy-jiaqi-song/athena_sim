from __future__ import annotations

import importlib.util
import struct
import sys
import tempfile
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
jhist = load_module("plot_jxyz_hist", ROOT / "scripts" / "plot_jxyz_hist.py")
spectra = load_module("plot_power_spectra", ROOT / "scripts" / "plot_power_spectra.py")
converter = load_module("athenak_to_athdf", ROOT / "scripts" / "athenak_to_athdf.py")
preflight = load_module("validate_ma08_preflight", ROOT / "scripts" / "validate_ma08_preflight.py")


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
            "sound_speed": 0.75, "guide_field": [1.0, 0.25, -0.5],
        },
        "selection": {"metric": "ms", "target": 1.0},
        "forcing": {
            "mode": 2, "energy_injection_rate": 1.0, "nlow": 1, "nhigh": 4,
            "spectrum_exponent": 2.0, "correlation_time": 0.1,
            "drive_interval": 0.02, "solenoidal_fraction": 1.0,
            "random_seed": 12345,
        },
        "output": {
            "history_interval": 0.05, "snapshot_interval": 0.1,
            "restart_interval": 1.0,
        },
        "power_spectra": {
            "enabled": True, "plot_k_min": 1, "plot_k_max": 0,
            "fit_enabled": True, "fit_k_min": 0, "fit_k_max": 0,
            "fit_magnetic_only": True, "min_fit_bins": 8,
            "parseval_rtol": 1.0e-5, "guide_alignment_tolerance": 1.0e-8,
            "save_full_nyquist_spectrum": True,
        },
        "athenak": {
            "revision": pipeline.ATHENAK_REVISION, "device": "cpu", "kokkos_arch": "",
            "integrator": "rk2", "reconstruction": "plm",
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


def write_power_spectrum_fixture(path: Path, *, shape=(16, 16, 16), rho=None,
                                 velocity=None, magnetic=None):
    if rho is None:
        rho = np.ones(shape)
    if velocity is None:
        velocity = [np.zeros(shape), np.zeros(shape), np.zeros(shape)]
    if magnetic is None:
        magnetic = [np.ones(shape), np.zeros(shape), np.zeros(shape)]
    with h5py.File(path, "w") as handle:
        handle["gas_density"] = rho
        handle["i_velocity"] = velocity[0]
        handle["j_velocity"] = velocity[1]
        handle["k_velocity"] = velocity[2]
        handle["i_mag_field"] = magnetic[0]
        handle["j_mag_field"] = magnetic[1]
        handle["k_mag_field"] = magnetic[2]
        handle["time"] = 2.5
        handle["cycle"] = 12
        handle["domain_bounds"] = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
        handle.attrs["array_axis_order"] = "x1,x2,x3"


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
        self.assertIn("nlow = 1", text)

    def test_athenak_input_and_stable_name(self):
        cfg = example_config("athenak")
        self.assertEqual(pipeline.effective_run_name(cfg), "case_athenak")
        text = pipeline.render_athinput(cfg)
        self.assertIn("basename = case_athenak", text)
        self.assertIn("eos = isothermal", text)
        self.assertIn("integrator = rk2", text)
        self.assertIn("reconstruct = plm", text)
        self.assertIn("nlow = 2", text)
        self.assertIn("nhigh = 3", text)
        self.assertIn("turb_flag = 2", text)
        self.assertIn("dt_turb_update = 0.02", text)
        self.assertIn("sol_fraction = 1", text)
        self.assertIn("random_seed = 12345", text)
        self.assertIn("spect_form = 2", text)

    def test_athenak_rejects_duplicate_forcing_band(self):
        cfg = example_config("athenak")
        cfg["athenak"]["nlow"] = 2
        with self.assertRaisesRegex(ValueError, "duplicate"):
            pipeline.validate_simulation_config(cfg)

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
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "fixture.out2.00003.bin"
            expected = write_athenak_fixture(source)
            raw_diagnostics = pipeline.athenak_snapshot_diagnostics(source, 0.75)
            self.assertEqual(raw_diagnostics["cycle"], 7)
            self.assertEqual(raw_diagnostics["time"], 0.25)
            self.assertTrue(raw_diagnostics["finite_state"])
            self.assertGreater(raw_diagnostics["minimum_density"], 0.0)
            self.assertAlmostEqual(raw_diagnostics["mean_density"], 1.5)
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


class JHistogramTests(unittest.TestCase):
    def test_writes_histogram(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_path = root / "snapshot.h5"
            x = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)[:, None, None]
            y = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)[None, :, None]
            z = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)[None, None, :]
            with h5py.File(input_path, "w") as handle:
                handle["i_mag_field"] = np.sin(y) + np.zeros((8, 8, 8))
                handle["j_mag_field"] = np.cos(z) + np.zeros((8, 8, 8))
                handle["k_mag_field"] = np.sin(x) + np.zeros((8, 8, 8))
                handle["time"] = 2.5
                handle["cycle"] = 12
                handle["domain_bounds"] = [
                    0.0, 2.0 * np.pi, 0.0, 2.0 * np.pi, 0.0, 2.0 * np.pi,
                ]
            output = jhist.plot_histogram(input_path, root / "j_histograms")
            self.assertTrue(output.is_file())


class PowerSpectrumTests(unittest.TestCase):
    def test_single_parallel_magnetic_mode(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shape = (16, 16, 16)
            x = np.arange(shape[0])[:, None, None] / shape[0]
            magnetic = [1.0 + np.sin(2.0 * np.pi * 2 * x) + np.zeros(shape),
                        np.zeros(shape), np.zeros(shape)]
            path = root / "snapshot.h5"
            write_power_spectrum_fixture(path, shape=shape, magnetic=magnetic)
            result = spectra.calculate_spectra(path)
            self.assertGreater(result["spectral_density"]["parallel_magnetic"][2], 0.24)
            self.assertGreater(result["spectral_density"]["perpendicular_magnetic"][0], 0.24)
            self.assertLess(result["spectral_density"]["perpendicular_magnetic"][1:].sum(), 1.0e-12)
            self.assertLess(result["metadata"]["parseval"]["magnetic_relative_error"], 1.0e-12)

    def test_single_perpendicular_and_oblique_magnetic_modes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shape = (16, 16, 16)
            x = np.arange(shape[0])[:, None, None] / shape[0]
            y = np.arange(shape[1])[None, :, None] / shape[1]
            path = root / "snapshot.h5"

            magnetic = [1.0 + np.sin(2.0 * np.pi * 3 * y) + np.zeros(shape),
                        np.zeros(shape), np.zeros(shape)]
            write_power_spectrum_fixture(path, shape=shape, magnetic=magnetic)
            result = spectra.calculate_spectra(path)
            self.assertGreater(result["spectral_density"]["perpendicular_magnetic"][3], 0.24)
            self.assertGreater(result["spectral_density"]["parallel_magnetic"][0], 0.24)
            self.assertLess(result["spectral_density"]["parallel_magnetic"][1:].sum(), 1.0e-12)

            magnetic = [1.0 + np.sin(2.0 * np.pi * (2 * x + 3 * y)) + np.zeros(shape),
                        np.zeros(shape), np.zeros(shape)]
            write_power_spectrum_fixture(path, shape=shape, magnetic=magnetic)
            result = spectra.calculate_spectra(path)
            self.assertGreater(result["spectral_density"]["parallel_magnetic"][2], 0.24)
            self.assertGreater(result["spectral_density"]["perpendicular_magnetic"][3], 0.24)

    def test_mean_subtraction_velocity_weighting_and_parseval(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shape = (16, 16, 16)
            x = np.arange(shape[0])[:, None, None] / shape[0]
            rho = np.full(shape, 4.0)
            velocity = [1.0 + np.sin(2.0 * np.pi * 2 * x) + np.zeros(shape),
                        np.zeros(shape), np.zeros(shape)]
            magnetic = [np.ones(shape), np.zeros(shape), np.zeros(shape)]
            path = root / "snapshot.h5"
            write_power_spectrum_fixture(path, shape=shape, rho=rho, velocity=velocity, magnetic=magnetic)
            result = spectra.calculate_spectra(path)
            self.assertLess(result["spectral_density"]["parallel_magnetic"].sum(), 1.0e-12)
            self.assertAlmostEqual(result["spectral_density"]["parallel_kinetic"][2], 1.0)
            self.assertAlmostEqual(result["metadata"]["parseval"]["kinetic_real_space_energy"], 1.0)
            self.assertLess(result["metadata"]["parseval"]["kinetic_relative_error"], 1.0e-12)

    def test_projected_oblique_guide_field_bins_modes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shape = (16, 16, 16)
            x = np.arange(shape[0])[:, None, None] / shape[0]
            y = np.arange(shape[1])[None, :, None] / shape[1]
            magnetic = [1.0 + np.sin(2.0 * np.pi * (2 * x + 2 * y)) + np.zeros(shape),
                        np.ones(shape), np.zeros(shape)]
            path = root / "snapshot.h5"
            write_power_spectrum_fixture(path, shape=shape, magnetic=magnetic)
            result = spectra.calculate_spectra(path)
            self.assertEqual(result["metadata"]["guide_alignment_method"], "projected_mean_field")
            self.assertGreater(result["spectral_density"]["parallel_magnetic"][3], 0.24)
            self.assertGreater(result["spectral_density"]["perpendicular_magnetic"][0], 0.24)

    def test_spectral_density_is_not_divided_by_mode_count(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shape = (16, 16, 16)
            y = np.arange(shape[1])[None, :, None] / shape[1]
            z = np.arange(shape[2])[None, None, :] / shape[2]
            magnetic = [1.0 + np.sin(2.0 * np.pi * 3 * y) + np.sin(2.0 * np.pi * 3 * z) + np.zeros(shape),
                        np.zeros(shape), np.zeros(shape)]
            path = root / "snapshot.h5"
            write_power_spectrum_fixture(path, shape=shape, magnetic=magnetic)
            result = spectra.calculate_spectra(path)
            density = result["spectral_density"]["perpendicular_magnetic"][3]
            count = result["mode_count"]["perpendicular"][3]
            mean = result["mean_mode_power"]["perpendicular_magnetic"][3]
            self.assertGreater(count, 1)
            self.assertAlmostEqual(density, mean * count)

    def test_power_law_fit_and_resolution_limits(self):
        k = np.arange(0, 16)
        spectrum = np.zeros_like(k, dtype=float)
        spectrum[2:10] = 3.0 * np.power(k[2:10], -5.0 / 3.0)
        counts = np.ones_like(k)
        fit = spectra.fit_power_law(k, spectrum, counts, 2, 9, 5)
        self.assertEqual(fit["status"], "ok")
        self.assertAlmostEqual(fit["slope"], -5.0 / 3.0)
        self.assertEqual(fit["fit_k_min"], 2)
        self.assertEqual(fit["fit_k_max"], 9)
        limits = spectra.resolve_limits((512, 512, 512), 1, 0, 0, 0, 4)
        self.assertEqual(limits["plot_k_max"], 170)
        self.assertEqual(limits["common_cartesian_nyquist"], 256)

    def test_invalid_fit_range_skips_fit(self):
        k = np.arange(0, 8)
        spectrum = np.zeros_like(k, dtype=float)
        spectrum[2] = 1.0
        counts = np.ones_like(k)
        fit = spectra.fit_power_law(k, spectrum, counts, 2, 5, 3)
        self.assertEqual(fit["status"], "skipped")
        self.assertIn("too few", fit["warning"])

    def test_writes_power_spectrum_products(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shape = (16, 16, 16)
            x = np.arange(shape[0])[:, None, None] / shape[0]
            y = np.arange(shape[1])[None, :, None] / shape[1]
            magnetic = [1.0 + np.sin(2.0 * np.pi * 2 * x) + np.zeros(shape),
                        np.zeros(shape), np.zeros(shape)]
            velocity = [np.sin(2.0 * np.pi * 3 * y) + np.zeros(shape),
                        np.zeros(shape), np.zeros(shape)]
            path = root / "snapshot.h5"
            write_power_spectrum_fixture(path, shape=shape, velocity=velocity, magnetic=magnetic)
            outputs = spectra.write_outputs(path, root / "power_spectra", min_fit_bins=2)
            self.assertTrue(outputs["png"].is_file())
            self.assertTrue(outputs["csv"].is_file())
            self.assertTrue(outputs["json"].is_file())


class WorkflowControlTests(unittest.TestCase):
    def test_convert_reuses_selected_metadata(self):
        cfg = example_config("athenak")
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            analysis = run_dir / "analysis"
            analysis.mkdir()
            summary = {
                "target_selection": {
                    "snapshot": {"file": "bin/selected.bin", "time": 1.0}
                }
            }
            (analysis / "diagnostics.json").write_text(
                __import__("json").dumps(summary), encoding="utf-8"
            )
            products = {
                "source_snapshot": str(run_dir / "bin" / "selected.bin"),
                "selected_athdf": str(analysis / "selected_snapshot" / "selected.athdf"),
                "converted_snapshot": str(analysis / "selected_snapshot" / "selected.h5"),
                "bfield_slice_directory": str(analysis / "bfield_slices"),
                "j_histogram_directory": str(analysis / "j_histograms"),
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

    def test_postprocess_runs_power_spectra_on_selected_cube_once(self):
        cfg = example_config()
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            source = run_dir / "selected.athdf"
            source.write_text("placeholder", encoding="utf-8")
            analysis = run_dir / "analysis"
            selection = {"snapshot": {"file": "selected.athdf", "time": 1.0}}
            with mock.patch.object(pipeline, "run_backend") as run_backend:
                products = pipeline.postprocess_selected_snapshot(cfg, run_dir, analysis, selection)
            commands = [call.args[0] for call in run_backend.call_args_list]
            spectra_commands = [command for command in commands if "plot_power_spectra.py" in command[1]]
            self.assertEqual(len(spectra_commands), 1)
            self.assertIn(str(analysis / "selected_snapshot" / "selected.h5"), spectra_commands[0])
            self.assertEqual(products["power_spectra_directory"], str(analysis / "power_spectra"))
            self.assertEqual(products["bfield_slice_directory"], str(analysis / "bfield_slices"))
            self.assertEqual(products["j_histogram_directory"], str(analysis / "j_histograms"))

    def test_timeout_reaps_process_and_preserves_clear_error(self):
        cfg = example_config()
        with self.assertRaisesRegex(pipeline.RunTimeoutError, "process group was terminated"):
            pipeline.run_backend(
                [sys.executable, "-c", "import time; time.sleep(30)"],
                ROOT,
                cfg,
                timeout_seconds=0.05,
            )

    def test_forcing_overlay_provenance_is_pinned(self):
        overlay = (ROOT / "scripts" / "apply_athenak_forcing_overlay.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("572f644f3ab3379a32ea2f0bec1658348141dc19", overlay)
        self.assertIn("beta[stage-1] * dt", overlay)
        self.assertIn("dt_turb_update", overlay)
        self.assertIn("double * table_", overlay)
        self.assertIn("static_cast<Real>", overlay)

    def test_preflight_parity_threshold(self):
        common = {
            "finite": True, "minimum_density": 0.9, "mass_drift": 1.0e-7,
            "timestep_collapse": 2.0, "final_time": 0.1, "final_cycle": 20,
            "tlim": 0.1, "nlim": 5000, "kinetic_energy_density": 1.0,
            "magnetic_energy_density": 2.0, "alfvenic_mach_magnetic": 0.5,
        }
        fp32 = dict(common)
        fp64 = dict(common, kinetic_energy_density=1.04)
        self.assertTrue(preflight.validate(fp32, fp64)["passed"])
        fp64["kinetic_energy_density"] = 1.10
        self.assertFalse(preflight.validate(fp32, fp64)["passed"])

    def test_preflight_ignores_duplicate_terminal_timestep(self):
        history = {
            "time": [0.08, 0.10, 0.10],
            "dt": [0.002, 0.001, 1.0e-8],
        }
        self.assertEqual(preflight.advancing_timesteps(history), [0.002, 0.001])

        # read_history keeps the later record when terminal times compare equal.
        collapsed = {"time": [0.08, 0.10], "dt": [0.002, 1.0e-8]}
        self.assertEqual(preflight.advancing_timesteps(collapsed), [0.002])

    def test_preflight_accepts_fp32_tlim_roundoff(self):
        common = {
            "finite": True, "minimum_density": 0.9, "mass_drift": 1.0e-7,
            "timestep_collapse": 2.0, "final_time": 0.100000001,
            "final_cycle": 20, "tlim": 0.1, "nlim": 5000,
            "kinetic_energy_density": 1.0, "magnetic_energy_density": 2.0,
            "alfvenic_mach_magnetic": 0.5,
        }
        self.assertTrue(preflight.validate(common, common)["passed"])


if __name__ == "__main__":
    unittest.main()
