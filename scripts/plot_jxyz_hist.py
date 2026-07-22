#!/usr/bin/env python3
"""Draw J=curl(B) component histograms from a converted Athena snapshot."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def wave_numbers(n: int, lower: float, upper: float) -> np.ndarray:
    length = upper - lower
    if length <= 0.0:
        raise ValueError(f"invalid domain bounds: lower={lower}, upper={upper}")
    dx = length / n
    return 2.0 * np.pi * np.fft.fftfreq(n, d=dx)


def curl_b(
    bx: np.ndarray, by: np.ndarray, bz: np.ndarray, bounds: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nx, ny, nz = bx.shape
    kx = wave_numbers(nx, bounds[0], bounds[1])[:, None, None]
    ky = wave_numbers(ny, bounds[2], bounds[3])[None, :, None]
    kz = wave_numbers(nz, bounds[4], bounds[5])[None, None, :]

    bxh = np.fft.fftn(bx, axes=(0, 1, 2))
    byh = np.fft.fftn(by, axes=(0, 1, 2))
    bzh = np.fft.fftn(bz, axes=(0, 1, 2))

    jx = np.fft.ifftn(1j * ky * bzh - 1j * kz * byh, axes=(0, 1, 2)).real
    jy = np.fft.ifftn(1j * kz * bxh - 1j * kx * bzh, axes=(0, 1, 2)).real
    jz = np.fft.ifftn(1j * kx * byh - 1j * ky * bxh, axes=(0, 1, 2)).real
    return jx, jy, jz


def gaussian_pdf(x: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0.0:
        return np.zeros_like(x)
    return np.exp(-0.5 * (x / sigma) ** 2) / (sigma * np.sqrt(2.0 * np.pi))


def load_current(
    snapshot_path: Path,
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray, float, int]:
    with h5py.File(snapshot_path, "r") as handle:
        bx = handle["i_mag_field"][...]
        by = handle["j_mag_field"][...]
        bz = handle["k_mag_field"][...]
        bounds = np.asarray(handle["domain_bounds"], dtype=float)
        time = float(handle["time"][()])
        cycle = int(handle["cycle"][()]) if "cycle" in handle else 0

    if bx.shape != by.shape or bx.shape != bz.shape:
        raise ValueError(f"magnetic-field shape mismatch in {snapshot_path}")
    if bounds.shape != (6,):
        raise ValueError(
            "domain_bounds must contain [x1min,x1max,x2min,x2max,x3min,x3max]"
        )

    return curl_b(bx, by, bz, bounds), bounds, time, cycle


def plot_histogram(snapshot_path: Path, output_dir: Path) -> Path:
    currents, bounds, time, cycle = load_current(snapshot_path)
    labels = (("$J_x$", currents[0]), ("$J_y$", currents[1]), ("$J_z$", currents[2]))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
    for ax, (label, values) in zip(axes, labels):
        flat = values.ravel()
        mu = float(flat.mean())
        sigma = float(flat.std())
        centered = flat - mu
        ax.hist(centered, bins=200, density=True, histtype="step", linewidth=1.6)
        if sigma > 0.0:
            xx = np.linspace(-5.0 * sigma, 5.0 * sigma, 2000)
            ax.plot(xx, gaussian_pdf(xx, sigma), "k--", linewidth=1.4, alpha=0.85)

        ax.set_xlabel(f"{label} - mu")
        ax.set_ylabel(f"P({label} - mu)")
        ax.set_title(f"{label}: mu={mu:.3e}, sigma={sigma:.3e}")
        ax.set_yscale("log")
        ax.set_ylim(bottom=1e-6)
        ax.grid(True, alpha=0.25)

    domain = (
        f"x=[{bounds[0]:.3g},{bounds[1]:.3g}], "
        f"y=[{bounds[2]:.3g},{bounds[3]:.3g}], "
        f"z=[{bounds[4]:.3g},{bounds[5]:.3g}]"
    )
    cycle_text = f", cycle={cycle}" if cycle else ""
    fig.suptitle(
        f"J = curl(B) histogram | {snapshot_path.name} | t={time:.5g}{cycle_text} | {domain}",
        y=1.02,
    )
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "jxyz_histogram.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[jhist] wrote {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if not args.input.is_file():
        raise SystemExit(f"File not found: {args.input}")
    plot_histogram(args.input, args.output_dir)


if __name__ == "__main__":
    main()
