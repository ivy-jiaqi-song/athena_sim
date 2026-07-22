#!/usr/bin/env python3
"""Draw selected-snapshot parallel and perpendicular MHD power spectra."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
from pathlib import Path
from typing import Any

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


OUTPUT_BASENAME = "power_spectra_parallel_perpendicular"
FIELD_DEFINITIONS = {
    "magnetic": "delta_B = B - volume_mean(B)",
    "kinetic": "u_K = sqrt(rho) * (v - sum(rho*v)/sum(rho))",
    "mode_power": "P(k) = 0.5 * sum_i |FFT_normalized(component_i)|^2",
}
FFT_NORMALIZATION = "numpy.fft.fftn(field) / field.size; sum_k |F_k|^2 = mean_x |field_x|^2"


def _as_float(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _read_scalar(handle: h5py.File, name: str) -> np.ndarray:
    if name not in handle:
        raise KeyError(f"Missing dataset {name!r}; available: {sorted(handle.keys())}")
    data = np.asarray(handle[name][...])
    if data.ndim != 3:
        raise ValueError(f"{name} must be a 3D array, got shape {data.shape}")
    if not np.isfinite(data).all():
        raise ValueError(f"{name} contains non-finite values")
    return data


def load_snapshot_metadata(snapshot_path: Path) -> dict[str, Any]:
    with h5py.File(snapshot_path, "r") as handle:
        bx = handle["i_mag_field"]
        shape = tuple(int(item) for item in bx.shape)
        required = (
            "gas_density", "i_velocity", "j_velocity", "k_velocity",
            "i_mag_field", "j_mag_field", "k_mag_field", "domain_bounds", "time",
        )
        missing = [name for name in required if name not in handle]
        if missing:
            raise KeyError(f"Missing datasets in {snapshot_path}: {missing}")
        for name in required[:7]:
            if tuple(handle[name].shape) != shape:
                raise ValueError(f"{name} shape {handle[name].shape} does not match {shape}")
        bounds = np.asarray(handle["domain_bounds"], dtype=np.float64)
        if bounds.shape != (6,):
            raise ValueError("domain_bounds must contain [x1min,x1max,x2min,x2max,x3min,x3max]")
        lengths = np.asarray([bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4]], dtype=np.float64)
        if np.any(lengths <= 0.0):
            raise ValueError(f"invalid domain bounds: {bounds.tolist()}")
        return {
            "shape": shape,
            "bounds": bounds,
            "lengths": lengths,
            "time": float(handle["time"][()]),
            "cycle": int(handle["cycle"][()]) if "cycle" in handle else 0,
            "source_file": str(handle.attrs.get("source_file", snapshot_path)),
            "array_axis_order": str(handle.attrs.get("array_axis_order", "x1,x2,x3")),
            "dtypes": {name: str(handle[name].dtype) for name in required[:7]},
        }


def positive_int_modes(n: int) -> np.ndarray:
    return np.rint(np.fft.fftfreq(n) * n).astype(np.int64)


def aligned_axis(b0_hat: np.ndarray, tolerance: float) -> int | None:
    for axis in range(3):
        target = np.zeros(3, dtype=np.float64)
        target[axis] = math.copysign(1.0, b0_hat[axis] if b0_hat[axis] != 0.0 else 1.0)
        if np.linalg.norm(b0_hat - target, ord=np.inf) <= tolerance:
            return axis
    return None


def mode_bin_indices(
    shape: tuple[int, int, int],
    b0_hat: np.ndarray,
    max_bin: int,
    tolerance: float,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, str, int | None]:
    modes = [positive_int_modes(n) for n in shape]
    abs_modes = [np.abs(item) for item in modes]
    axis = aligned_axis(b0_hat, tolerance)
    if axis is not None:
        parallel = abs_modes[axis]
        perpendicular_axes = [index for index in range(3) if index != axis]
        a = modes[perpendicular_axes[0]]
        b = modes[perpendicular_axes[1]]
        perp = np.floor(np.sqrt(a[:, None] * a[:, None] + b[None, :] * b[None, :]) + 0.5).astype(np.int64)
        perp = np.where(perp <= max_bin, perp, -1)
        return modes, parallel, perp, f"coordinate_aligned_x{axis + 1}", axis

    return modes, np.empty(0, dtype=np.int64), np.empty((0, 0), dtype=np.int64), "projected_mean_field", None


def accumulate_aligned(
    power: np.ndarray,
    parallel_bins: np.ndarray,
    perpendicular_bins: np.ndarray,
    aligned: int,
    max_bin: int,
    parallel_sum: np.ndarray,
    perpendicular_sum: np.ndarray,
) -> None:
    moved = np.moveaxis(power, aligned, 0)
    plane_power = moved.reshape(moved.shape[0], -1).sum(axis=1)
    parallel_sum += np.bincount(parallel_bins, weights=plane_power, minlength=max_bin + 1)[:max_bin + 1]

    collapsed = moved.sum(axis=0)
    weights = collapsed.ravel()
    bins = perpendicular_bins.ravel()
    valid = bins >= 0
    perpendicular_sum += np.bincount(bins[valid], weights=weights[valid], minlength=max_bin + 1)[:max_bin + 1]


def accumulate_projected(
    power: np.ndarray,
    modes: list[np.ndarray],
    b0_hat: np.ndarray,
    max_bin: int,
    parallel_sum: np.ndarray,
    perpendicular_sum: np.ndarray,
) -> None:
    ky = modes[1][:, None]
    kz = modes[2][None, :]
    yz2 = ky * ky + kz * kz
    for ix, kx in enumerate(modes[0]):
        k_parallel = np.abs(kx * b0_hat[0] + ky * b0_hat[1] + kz * b0_hat[2])
        k2 = float(kx * kx) + yz2
        k_perp = np.sqrt(np.maximum(k2 - np.square(k_parallel), 0.0))
        parallel_bins = np.floor(k_parallel + 0.5).astype(np.int64)
        perpendicular_bins = np.floor(k_perp + 0.5).astype(np.int64)
        slab = power[ix, :, :]
        for target, bins in ((parallel_sum, parallel_bins), (perpendicular_sum, perpendicular_bins)):
            valid = bins <= max_bin
            target += np.bincount(
                bins[valid].ravel(), weights=slab[valid].ravel(), minlength=max_bin + 1
            )[:max_bin + 1]


def count_modes(
    shape: tuple[int, int, int],
    b0_hat: np.ndarray,
    max_bin: int,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray, str, int | None]:
    modes, parallel_bins, perpendicular_bins, method, axis = mode_bin_indices(shape, b0_hat, max_bin, tolerance)
    parallel_count = np.zeros(max_bin + 1, dtype=np.int64)
    perpendicular_count = np.zeros(max_bin + 1, dtype=np.int64)
    if axis is not None:
        other = int(np.prod([shape[index] for index in range(3) if index != axis]))
        parallel_count += np.bincount(parallel_bins, minlength=max_bin + 1)[:max_bin + 1] * other
        values = np.bincount(
            perpendicular_bins[perpendicular_bins >= 0].ravel(), minlength=max_bin + 1
        )[:max_bin + 1]
        perpendicular_count += values * shape[axis]
    else:
        parallel_float = np.zeros(max_bin + 1, dtype=np.float64)
        perpendicular_float = np.zeros(max_bin + 1, dtype=np.float64)
        ones = np.ones(shape, dtype=np.float32)
        accumulate_projected(ones, modes, b0_hat, max_bin, parallel_float, perpendicular_float)
        parallel_count = np.rint(parallel_float).astype(np.int64)
        perpendicular_count = np.rint(perpendicular_float).astype(np.int64)
    return perpendicular_count, parallel_count, method, axis


def _component_power(field: np.ndarray) -> np.ndarray:
    transformed = np.fft.fftn(field, axes=(0, 1, 2)) / field.size
    return 0.5 * np.square(np.abs(transformed))


def accumulate_component(
    field: np.ndarray,
    modes: list[np.ndarray],
    parallel_bins: np.ndarray,
    perpendicular_bins: np.ndarray,
    method_axis: int | None,
    b0_hat: np.ndarray,
    max_bin: int,
    parallel_sum: np.ndarray,
    perpendicular_sum: np.ndarray,
) -> float:
    power = _component_power(field)
    total = float(power.sum())
    if method_axis is None:
        accumulate_projected(power, modes, b0_hat, max_bin, parallel_sum, perpendicular_sum)
    else:
        accumulate_aligned(
            power, parallel_bins, perpendicular_bins, method_axis,
            max_bin, parallel_sum, perpendicular_sum,
        )
    del power
    return total


def calculate_spectra(
    snapshot_path: Path,
    *,
    parseval_rtol: float = 1.0e-5,
    guide_alignment_tolerance: float = 1.0e-8,
) -> dict[str, Any]:
    metadata = load_snapshot_metadata(snapshot_path)
    shape = tuple(metadata["shape"])
    max_bin = min(shape) // 2

    with h5py.File(snapshot_path, "r") as handle:
        b_means = np.asarray([
            float(np.asarray(handle[name][...], dtype=np.float64).mean())
            for name in ("i_mag_field", "j_mag_field", "k_mag_field")
        ], dtype=np.float64)
        b0_magnitude = float(np.linalg.norm(b_means))
        if b0_magnitude <= np.finfo(np.float64).tiny:
            raise ValueError("Mean guide field magnitude is numerically zero; cannot define parallel direction")
        b0_hat = b_means / b0_magnitude

        rho = _read_scalar(handle, "gas_density")
        if np.any(rho <= 0.0):
            raise ValueError("gas_density must be strictly positive for sqrt(rho) kinetic spectra")
        rho64 = np.asarray(rho, dtype=np.float64)
        rho_sum = float(rho64.sum())
        velocity_means = []
        for name in ("i_velocity", "j_velocity", "k_velocity"):
            velocity = _read_scalar(handle, name)
            velocity_means.append(float((rho64 * velocity).sum() / rho_sum))
            del velocity
        velocity_means_array = np.asarray(velocity_means, dtype=np.float64)

        perpendicular_count, parallel_count, method, axis = count_modes(
            shape, b0_hat, max_bin, guide_alignment_tolerance
        )
        modes, parallel_bins, perpendicular_bins, _, _ = mode_bin_indices(
            shape, b0_hat, max_bin, guide_alignment_tolerance
        )
        spectra = {
            "perpendicular_magnetic": np.zeros(max_bin + 1, dtype=np.float64),
            "perpendicular_kinetic": np.zeros(max_bin + 1, dtype=np.float64),
            "parallel_magnetic": np.zeros(max_bin + 1, dtype=np.float64),
            "parallel_kinetic": np.zeros(max_bin + 1, dtype=np.float64),
        }

        magnetic_real = 0.0
        magnetic_fourier = 0.0
        for component, name in enumerate(("i_mag_field", "j_mag_field", "k_mag_field")):
            field = _read_scalar(handle, name).astype(np.float64, copy=False)
            fluctuation = field - b_means[component]
            magnetic_real += float(0.5 * np.mean(np.square(fluctuation)))
            magnetic_fourier += accumulate_component(
                fluctuation, modes, parallel_bins, perpendicular_bins, axis, b0_hat, max_bin,
                spectra["parallel_magnetic"], spectra["perpendicular_magnetic"],
            )
            del field, fluctuation
            gc.collect()

        sqrt_rho = np.sqrt(rho64)
        kinetic_real = 0.0
        kinetic_fourier = 0.0
        for component, name in enumerate(("i_velocity", "j_velocity", "k_velocity")):
            velocity = _read_scalar(handle, name).astype(np.float64, copy=False)
            delta_v = velocity - velocity_means_array[component]
            kinetic_real += float(0.5 * np.mean(rho64 * np.square(delta_v)))
            weighted = sqrt_rho * delta_v
            kinetic_fourier += accumulate_component(
                weighted, modes, parallel_bins, perpendicular_bins, axis, b0_hat, max_bin,
                spectra["parallel_kinetic"], spectra["perpendicular_kinetic"],
            )
            del velocity, delta_v, weighted
            gc.collect()

    rel_b = abs(magnetic_fourier - magnetic_real) / max(abs(magnetic_real), 1.0e-300)
    rel_k = abs(kinetic_fourier - kinetic_real) / max(abs(kinetic_real), 1.0e-300)
    if rel_b > parseval_rtol or rel_k > parseval_rtol:
        raise RuntimeError(
            "Parseval check failed: "
            f"magnetic relerr={rel_b:.3e}, kinetic relerr={rel_k:.3e}, rtol={parseval_rtol:.3e}"
        )

    with np.errstate(divide="ignore", invalid="ignore"):
        mean_power = {
            "perpendicular_magnetic": np.divide(
                spectra["perpendicular_magnetic"], perpendicular_count,
                out=np.zeros_like(spectra["perpendicular_magnetic"]), where=perpendicular_count > 0,
            ),
            "perpendicular_kinetic": np.divide(
                spectra["perpendicular_kinetic"], perpendicular_count,
                out=np.zeros_like(spectra["perpendicular_kinetic"]), where=perpendicular_count > 0,
            ),
            "parallel_magnetic": np.divide(
                spectra["parallel_magnetic"], parallel_count,
                out=np.zeros_like(spectra["parallel_magnetic"]), where=parallel_count > 0,
            ),
            "parallel_kinetic": np.divide(
                spectra["parallel_kinetic"], parallel_count,
                out=np.zeros_like(spectra["parallel_kinetic"]), where=parallel_count > 0,
            ),
        }

    return {
        "k_mode": np.arange(max_bin + 1, dtype=np.int64),
        "spectral_density": spectra,
        "mean_mode_power": mean_power,
        "mode_count": {"perpendicular": perpendicular_count, "parallel": parallel_count},
        "metadata": metadata | {
            "guide_field_vector": b_means.tolist(),
            "guide_field_magnitude": b0_magnitude,
            "guide_field_unit_direction": b0_hat.tolist(),
            "guide_alignment_method": method,
            "guide_alignment_axis": None if axis is None else f"x{axis + 1}",
            "stored_k_min": 0,
            "stored_k_max": max_bin,
            "fft_implementation": "numpy.fft.fftn",
            "fft_normalization": FFT_NORMALIZATION,
            "field_definitions": FIELD_DEFINITIONS,
            "binning_definition": "unit-width linear bins centered on integer dimensionless modes; bin n covers [n-0.5,n+0.5)",
            "parseval": {
                "magnetic_real_space_energy": magnetic_real,
                "magnetic_fourier_space_energy": magnetic_fourier,
                "magnetic_relative_error": rel_b,
                "kinetic_real_space_energy": kinetic_real,
                "kinetic_fourier_space_energy": kinetic_fourier,
                "kinetic_relative_error": rel_k,
                "rtol": parseval_rtol,
            },
            "mass_weighted_velocity_mean": velocity_means_array.tolist(),
        },
    }


def automatic_fit_min(forcing_nhigh: int) -> int:
    return max(4, int(forcing_nhigh) + 2)


def resolve_limits(
    shape: tuple[int, int, int],
    plot_k_min: int,
    plot_k_max: int,
    fit_k_min: int,
    fit_k_max: int,
    forcing_nhigh: int,
) -> dict[str, int]:
    nyquist = min(shape) // 2
    effective_plot_max = min(shape) // 3
    resolved_plot_max = effective_plot_max if int(plot_k_max) == 0 else int(plot_k_max)
    resolved_fit_min = automatic_fit_min(forcing_nhigh) if int(fit_k_min) == 0 else int(fit_k_min)
    resolved_fit_max = resolved_plot_max if int(fit_k_max) == 0 else int(fit_k_max)
    result = {
        "plot_k_min": int(plot_k_min),
        "plot_k_max": int(resolved_plot_max),
        "fit_k_min": int(resolved_fit_min),
        "fit_k_max": int(resolved_fit_max),
        "effective_plot_k_max": int(effective_plot_max),
        "common_cartesian_nyquist": int(nyquist),
    }
    if not 1 <= result["plot_k_min"] < result["plot_k_max"]:
        raise ValueError("power_spectra requires 1 <= plot_k_min < plot_k_max")
    if result["plot_k_max"] > nyquist:
        raise ValueError("power_spectra.plot_k_max must be <= common Cartesian Nyquist")
    return result


def fit_power_law(
    k_mode: np.ndarray,
    spectrum: np.ndarray,
    counts: np.ndarray,
    fit_k_min: int,
    fit_k_max: int,
    min_fit_bins: int,
) -> dict[str, Any]:
    if int(fit_k_min) >= int(fit_k_max):
        return {
            "status": "skipped", "warning": "invalid fit range",
            "n_bins": 0, "fit_k_min": int(fit_k_min), "fit_k_max": int(fit_k_max),
        }
    mask = (
        (k_mode > 0) & (k_mode >= fit_k_min) & (k_mode <= fit_k_max) &
        np.isfinite(spectrum) & (spectrum > 0.0) & (counts > 0)
    )
    x_values = np.log10(k_mode[mask].astype(np.float64))
    y_values = np.log10(spectrum[mask].astype(np.float64))
    if x_values.size < int(min_fit_bins):
        return {
            "status": "skipped", "warning": "too few finite positive populated bins",
            "n_bins": int(x_values.size), "fit_k_min": int(fit_k_min), "fit_k_max": int(fit_k_max),
        }
    slope, intercept = np.polyfit(x_values, y_values, 1)
    predicted = intercept + slope * x_values
    residual = y_values - predicted
    ss_res = float(np.sum(np.square(residual)))
    ss_tot = float(np.sum(np.square(y_values - y_values.mean())))
    dof = x_values.size - 2
    sxx = float(np.sum(np.square(x_values - x_values.mean())))
    stderr = math.sqrt(ss_res / dof / sxx) if dof > 0 and sxx > 0.0 else math.nan
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 1.0
    if not all(math.isfinite(float(value)) for value in (slope, intercept, stderr, r2)):
        return {
            "status": "skipped", "warning": "fit produced non-finite coefficients",
            "n_bins": int(x_values.size), "fit_k_min": int(fit_k_min), "fit_k_max": int(fit_k_max),
        }
    fitted_k = k_mode[mask]
    return {
        "status": "ok", "warning": "", "slope": float(slope), "intercept": float(intercept),
        "slope_standard_error": float(stderr), "r_squared": float(r2),
        "n_bins": int(x_values.size), "fit_k_min": int(fitted_k.min()), "fit_k_max": int(fitted_k.max()),
    }


def write_csv(result: dict[str, Any], output_path: Path) -> None:
    fields = [
        "k_mode",
        "perpendicular_magnetic_spectral_density",
        "perpendicular_kinetic_spectral_density",
        "perpendicular_mode_count",
        "perpendicular_magnetic_mean_mode_power",
        "perpendicular_kinetic_mean_mode_power",
        "parallel_magnetic_spectral_density",
        "parallel_kinetic_spectral_density",
        "parallel_mode_count",
        "parallel_magnetic_mean_mode_power",
        "parallel_kinetic_mean_mode_power",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for i, k in enumerate(result["k_mode"]):
            writer.writerow({
                "k_mode": int(k),
                "perpendicular_magnetic_spectral_density": result["spectral_density"]["perpendicular_magnetic"][i],
                "perpendicular_kinetic_spectral_density": result["spectral_density"]["perpendicular_kinetic"][i],
                "perpendicular_mode_count": int(result["mode_count"]["perpendicular"][i]),
                "perpendicular_magnetic_mean_mode_power": result["mean_mode_power"]["perpendicular_magnetic"][i],
                "perpendicular_kinetic_mean_mode_power": result["mean_mode_power"]["perpendicular_kinetic"][i],
                "parallel_magnetic_spectral_density": result["spectral_density"]["parallel_magnetic"][i],
                "parallel_kinetic_spectral_density": result["spectral_density"]["parallel_kinetic"][i],
                "parallel_mode_count": int(result["mode_count"]["parallel"][i]),
                "parallel_magnetic_mean_mode_power": result["mean_mode_power"]["parallel_magnetic"][i],
                "parallel_kinetic_mean_mode_power": result["mean_mode_power"]["parallel_kinetic"][i],
            })


def plot_spectra(result: dict[str, Any], limits: dict[str, int], fits: dict[str, Any], output_path: Path) -> None:
    k = result["k_mode"]
    plot_mask = (k >= limits["plot_k_min"]) & (k <= limits["plot_k_max"])
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.2), constrained_layout=True)
    panels = (
        (axes[0], "perpendicular", "Perpendicular to mean field", r"$k_\perp L / (2\pi)$"),
        (axes[1], "parallel", "Parallel to mean field", r"$k_\parallel L / (2\pi)$"),
    )
    for ax, direction, title, xlabel in panels:
        magnetic = result["spectral_density"][f"{direction}_magnetic"]
        kinetic = result["spectral_density"][f"{direction}_kinetic"]
        mag_mask = plot_mask & np.isfinite(magnetic) & (magnetic > 0.0)
        kin_mask = plot_mask & np.isfinite(kinetic) & (kinetic > 0.0)
        ax.loglog(k[mag_mask], magnetic[mag_mask], label="Magnetic", linewidth=2.0)
        ax.loglog(k[kin_mask], kinetic[kin_mask], label="Kinetic", linewidth=2.0, linestyle="--")
        fit = fits.get(direction, {})
        if fit.get("status") == "ok":
            fit_mask = (k >= fit["fit_k_min"]) & (k <= fit["fit_k_max"])
            fit_k = k[fit_mask].astype(np.float64)
            fit_y = np.power(10.0, fit["intercept"] + fit["slope"] * np.log10(fit_k))
            ax.loglog(
                fit_k, fit_y, color="black", linewidth=1.5, linestyle=":",
                label=(
                    f"Magnetic fit: alpha_{'perp' if direction == 'perpendicular' else 'parallel'} "
                    f"= {fit['slope']:.2f} +/- {fit['slope_standard_error']:.2f}"
                ),
            )
            for bound in (fit["fit_k_min"], fit["fit_k_max"]):
                ax.axvline(bound, color="0.35", linewidth=0.9, alpha=0.45)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("spectral energy density E(k)")
        ax.set_xlim(limits["plot_k_min"], limits["plot_k_max"])
        ax.grid(True, which="both", alpha=0.22)
        ax.legend(fontsize="small")
    meta = result["metadata"]
    resolution = "x".join(str(item) for item in meta["shape"])
    fig.suptitle(
        f"Selected snapshot spectra | t={meta['time']:.5g}, N={resolution} | summed mode energy per unit k-bin",
        y=1.03,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def write_outputs(
    snapshot_path: Path,
    output_dir: Path,
    *,
    plot_k_min: int = 1,
    plot_k_max: int = 0,
    fit_enabled: bool = True,
    fit_k_min: int = 0,
    fit_k_max: int = 0,
    min_fit_bins: int = 8,
    parseval_rtol: float = 1.0e-5,
    guide_alignment_tolerance: float = 1.0e-8,
    forcing_nlow: int = 1,
    forcing_nhigh: int = 4,
) -> dict[str, Path]:
    result = calculate_spectra(
        snapshot_path,
        parseval_rtol=parseval_rtol,
        guide_alignment_tolerance=guide_alignment_tolerance,
    )
    limits = resolve_limits(
        tuple(result["metadata"]["shape"]), plot_k_min, plot_k_max, fit_k_min, fit_k_max, forcing_nhigh
    )
    fits = {
        "perpendicular": {"status": "disabled", "warning": "fitting disabled"},
        "parallel": {"status": "disabled", "warning": "fitting disabled"},
    }
    if fit_enabled:
        fits = {
            "perpendicular": fit_power_law(
                result["k_mode"], result["spectral_density"]["perpendicular_magnetic"],
                result["mode_count"]["perpendicular"], limits["fit_k_min"], limits["fit_k_max"], min_fit_bins,
            ),
            "parallel": fit_power_law(
                result["k_mode"], result["spectral_density"]["parallel_magnetic"],
                result["mode_count"]["parallel"], limits["fit_k_min"], limits["fit_k_max"], min_fit_bins,
            ),
        }
    result["metadata"].update({
        "source_snapshot_path": str(snapshot_path),
        "plotted_k_range": [limits["plot_k_min"], limits["plot_k_max"]],
        "stored_k_range": [0, result["metadata"]["stored_k_max"]],
        "effective_plot_k_max": limits["effective_plot_k_max"],
        "common_cartesian_nyquist": limits["common_cartesian_nyquist"],
        "forcing_range_for_automatic_fit": {
            "nlow": int(forcing_nlow), "nhigh": int(forcing_nhigh),
            "interpretation": "shared Athena++ exclusive bounds nlow < |k| < nhigh; AthenaK input translates to inclusive nlow+1, nhigh-1",
        },
        "fit_configuration": {
            "fit_enabled": bool(fit_enabled), "fit_k_min": limits["fit_k_min"],
            "fit_k_max": limits["fit_k_max"], "min_fit_bins": int(min_fit_bins),
        },
        "fits": {"magnetic": fits},
        "warnings": [fit["warning"] for fit in fits.values() if fit.get("warning")],
    })

    png_path = output_dir / f"{OUTPUT_BASENAME}.png"
    csv_path = output_dir / f"{OUTPUT_BASENAME}.csv"
    json_path = output_dir / f"{OUTPUT_BASENAME}.json"
    write_csv(result, csv_path)
    plot_spectra(result, limits, fits, png_path)
    json_path.write_text(json.dumps(json_ready(result["metadata"]), indent=2, allow_nan=False), encoding="utf-8")
    print(f"[spectra] wrote {png_path}")
    print(f"[spectra] wrote {csv_path}")
    print(f"[spectra] wrote {json_path}")
    return {"png": png_path, "csv": csv_path, "json": json_path}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--plot-k-min", type=int, default=1)
    parser.add_argument("--plot-k-max", type=int, default=0)
    parser.add_argument("--fit-enabled", choices=("true", "false"), default="true")
    parser.add_argument("--fit-k-min", type=int, default=0)
    parser.add_argument("--fit-k-max", type=int, default=0)
    parser.add_argument("--min-fit-bins", type=int, default=8)
    parser.add_argument("--parseval-rtol", type=float, default=1.0e-5)
    parser.add_argument("--guide-alignment-tolerance", type=float, default=1.0e-8)
    parser.add_argument("--forcing-nlow", type=int, default=1)
    parser.add_argument("--forcing-nhigh", type=int, default=4)
    args = parser.parse_args()
    if not args.input.is_file():
        raise SystemExit(f"File not found: {args.input}")
    write_outputs(
        args.input, args.output_dir,
        plot_k_min=args.plot_k_min,
        plot_k_max=args.plot_k_max,
        fit_enabled=args.fit_enabled == "true",
        fit_k_min=args.fit_k_min,
        fit_k_max=args.fit_k_max,
        min_fit_bins=args.min_fit_bins,
        parseval_rtol=args.parseval_rtol,
        guide_alignment_tolerance=args.guide_alignment_tolerance,
        forcing_nlow=args.forcing_nlow,
        forcing_nhigh=args.forcing_nhigh,
    )


if __name__ == "__main__":
    main()
