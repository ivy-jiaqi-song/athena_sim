#!/usr/bin/env python3
"""Draw orthogonal magnetic-field slices from a converted Athena snapshot."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SLICES = {
    "b2b3": {"axis": 0, "axes": ("x2", "x3"), "comps": ("j_mag_field", "i_mag_field")},
    "b1b3": {"axis": 1, "axes": ("x1", "x3"), "comps": ("k_mag_field", "i_mag_field")},
    "b1b2": {"axis": 2, "axes": ("x1", "x2"), "comps": ("k_mag_field", "j_mag_field")},
}


def read_snapshot(path: Path) -> dict:
    with h5py.File(path, "r") as handle:
        fields = {
            "i_mag_field": handle["i_mag_field"][...],
            "j_mag_field": handle["j_mag_field"][...],
            "k_mag_field": handle["k_mag_field"][...],
            "time": float(handle["time"][()]),
            "domain_bounds": np.asarray(handle["domain_bounds"], dtype=float),
        }
    if fields["domain_bounds"].shape != (6,):
        raise ValueError(
            "domain_bounds must contain [x1min,x1max,x2min,x2max,x3min,x3max]"
        )
    return fields


def display_quiver_vectors(u: np.ndarray, v: np.ndarray, length: float) -> tuple[np.ndarray, np.ndarray]:
    norm = np.sqrt(np.square(u) + np.square(v))
    safe_norm = np.where(norm > 0.0, norm, 1.0)
    return u / safe_norm * length, v / safe_norm * length


def plot_slice(
    data: dict, name: str, config: dict, output_dir: Path, quiver_stride: int = 8
) -> Path:
    components = {key: data[key] for key in ("i_mag_field", "j_mag_field", "k_mag_field")}
    magnitude = np.sqrt(sum(np.square(component) for component in components.values()))
    axis = config["axis"]
    index = components["i_mag_field"].shape[axis] // 2
    selection = [slice(None)] * 3
    selection[axis] = index
    selection = tuple(selection)

    # Converted arrays are [x1, x2, x3]; plotting arrays are [vertical, horizontal].
    plane_magnitude = magnitude[selection].T
    u = components[config["comps"][0]][selection].T
    v = components[config["comps"][1]][selection].T

    bounds = data["domain_bounds"]
    axis_indices = {"x1": 0, "x2": 1, "x3": 2}
    horizontal = axis_indices[config["axes"][0]]
    vertical = axis_indices[config["axes"][1]]
    extent = [
        bounds[2 * horizontal], bounds[2 * horizontal + 1],
        bounds[2 * vertical], bounds[2 * vertical + 1],
    ]
    ny, nx = plane_magnitude.shape
    x = np.linspace(extent[0], extent[1], nx, endpoint=False)
    y = np.linspace(extent[2], extent[3], ny, endpoint=False)
    x += (extent[1] - extent[0]) / (2 * nx)
    y += (extent[3] - extent[2]) / (2 * ny)
    xx, yy = np.meshgrid(x, y)
    stride = max(1, int(quiver_stride))
    cell_width = (extent[1] - extent[0]) / nx
    cell_height = (extent[3] - extent[2]) / ny
    arrow_length = 0.75 * stride * min(cell_width, cell_height)
    u_display, v_display = display_quiver_vectors(u, v, arrow_length)

    figure, axes = plt.subplots(figsize=(7.2, 6.0), constrained_layout=True)
    image = axes.imshow(plane_magnitude, origin="lower", extent=extent, aspect="equal", cmap="magma")
    axes.quiver(
        xx[::stride, ::stride], yy[::stride, ::stride],
        u_display[::stride, ::stride], v_display[::stride, ::stride],
        color="white", alpha=0.85, pivot="mid", angles="xy", scale_units="xy", scale=1,
        width=0.0032, headwidth=4.2, headlength=5.0,
    )
    axes.set_xlabel(config["axes"][0])
    axes.set_ylabel(config["axes"][1])
    axes.set_title(
        f"B-field slice {name}, {config['axes'][0]}-{config['axes'][1]} plane "
        f"at mid-{['x1', 'x2', 'x3'][axis]}[{index}], t={data['time']:.4g}"
    )
    figure.colorbar(image, ax=axes).set_label(r"$|\mathbf{B}|$")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"bfield_{name}.png"
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    print(f"[bfield] wrote {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--quiver-stride", type=int, default=8)
    args = parser.parse_args()
    if not args.input.is_file():
        raise SystemExit(f"File not found: {args.input}")
    data = read_snapshot(args.input)
    for name, config in SLICES.items():
        plot_slice(data, name, config, args.output_dir, args.quiver_stride)


if __name__ == "__main__":
    main()
