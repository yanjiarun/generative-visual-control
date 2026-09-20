from __future__ import annotations

import argparse
import os

import numpy as np


A, C = 6.0, 18.0
SCALE_FACTOR = 200.0
X_RANGE, Y_RANGE, N_POINTS = (-0.62, 0.62), (0.10, 0.82), 60
BRANCH_X = (-0.48, 0.48)
BRANCH_AMPLITUDE = 0.20
FIELD_GAIN = 5.0
# The -0.32 exit lets a direct-heading lower mode cross the spatial separator
# while retaining 36 m before the obstacle's left edge.
DECISION_X = (BRANCH_X[0], -0.32)  # width = 0.16 in normalized x
OBSTACLE_CENTER = (0.0, (1.0 / C) ** 0.25)
OBSTACLE_RADII = (0.14, 0.10)  # ellipse semi-axes in normalized coordinates
# The ellipse is a virtual planning/evaluation constraint.  Unity need not
# contain an obstacle; obstacle_world_geometry() is only an optional placement
# helper when a visible obstacle is desired.


def obstacle_world_geometry(base=(576.0, 275.0)):
    """Return ``(center_xy, radii_xy)`` for placing the ellipse in Unity."""
    return (np.asarray(base) + SCALE_FACTOR * np.asarray(OBSTACLE_CENTER),
            SCALE_FACTOR * np.asarray(OBSTACLE_RADII))


def quartic_upper(x, a=A, c=C):
    """Upper branch of ``a*x**4 + c*y**4 = 1``."""
    x = np.asarray(x, dtype=float)
    q = np.maximum((1.0 - a * x**4) / c, 1e-10)
    return q**0.25


def _bump(x, x0=BRANCH_X[0], x1=BRANCH_X[1]):
    """C2 compact bump: zero value/first/second derivative at both ends."""
    x = np.asarray(x, dtype=float)
    s = np.clip((x - x0) / (x1 - x0), 0.0, 1.0)
    inside = (x > x0) & (x < x1)
    b = 64.0 * s**3 * (1.0 - s)**3
    db = 192.0 * s**2 * (1.0 - s)**2 * (1.0 - 2.0 * s) / (x1 - x0)
    return np.where(inside, b, 0.0), np.where(inside, db, 0.0)


def route(x, mode=0):
    """Return route ordinate and slope for base (0), upper (+1), or lower (-1)."""
    x = np.asarray(x, dtype=float)
    q = quartic_upper(x)
    # 4*q^3*q' = -4*A*x^3.
    qp = -A * x**3 / (C * q**3)
    b, db = _bump(x)
    return q + float(mode) * BRANCH_AMPLITUDE * b, qp + float(mode) * BRANCH_AMPLITUDE * db


def _field(x, y, mode=0, gain=FIELD_GAIN):
    """Smooth normalized graph-GVF; analytic path, no numerical lookup table."""
    f, fp = route(x, mode)
    alpha = np.asarray(y, dtype=float) - f
    # tangent minus a normal error correction
    vx = 1.0 + gain * alpha * fp
    vy = fp - gain * alpha
    norm = np.maximum(np.hypot(vx, vy), 1e-8)
    return vx / norm, vy / norm, alpha


def compute_quartic_obstacle_field(X, Y, mode=0, gradients=True, h=1e-4):
    """Return ``ux, uy, alpha, grad_ux, grad_uy`` for one route.

    The spatial derivatives are central differences of the short closed-form
    field.  This keeps the implementation compact while remaining smooth and
    numerically stable for the training grid and the online controller.
    """
    X, Y = np.asarray(X, dtype=float), np.asarray(Y, dtype=float)
    ux, uy, alpha = _field(X, Y, mode)
    if not gradients:
        return ux, uy, alpha
    ux_x, uy_x, _ = _field(X + h, Y, mode)
    ux_mx, uy_mx, _ = _field(X - h, Y, mode)
    ux_y, uy_y, _ = _field(X, Y + h, mode)
    ux_my, uy_my, _ = _field(X, Y - h, mode)
    grad_ux = np.array(((ux_x - ux_mx) / (2 * h), (ux_y - ux_my) / (2 * h)))
    grad_uy = np.array(((uy_x - uy_mx) / (2 * h), (uy_y - uy_my) / (2 * h)))
    return ux, uy, alpha, grad_ux, grad_uy


def obstacle_mask(x, y):
    """True inside the physical elliptical obstacle."""
    cx, cy = OBSTACLE_CENTER
    rx, ry = OBSTACLE_RADII
    return ((np.asarray(x) - cx) / rx) ** 2 + ((np.asarray(y) - cy) / ry) ** 2 <= 1.0


def modes_for_point(x, y):
    """Return the training labels for one point under the obstacle split.

    The decision strip is rectangular, as in ``RectangularMaskedDataset``.
    Between the front/rear junctions, the original Quartic graph is the
    separator: its upper half follows ``mode=+1`` and its lower half follows
    ``mode=-1``.  All remaining points use the deterministic ``mode=0``
    label.
    """
    x, y = float(x), float(y)
    if DECISION_X[0] <= x < DECISION_X[1]:
        return (1, -1)

    if BRANCH_X[0] <= x <= BRANCH_X[1]:
        q = float(quartic_upper(x))
        return (1,) if y >= q else (-1,)

    return (0,)


def write_dataset(csv_path, n_points=N_POINTS, x_range=X_RANGE, y_range=Y_RANGE):
    """Write a standard-compatible GVF CSV and return its unique grid arrays."""
    import pandas as pd

    x = np.linspace(*x_range, n_points + 1)
    ny = int(round(n_points * (y_range[1] - y_range[0]) /
                   (x_range[1] - x_range[0]))) + 1
    y = np.linspace(*y_range, ny)
    X, Y = np.meshgrid(x, y)
    rows, frame_id = [], 0
    for xi, yi in zip(X.ravel(), Y.ravel()):
        frame_id += 1
        for mode in modes_for_point(xi, yi):
            ux, uy, _, gux, guy = compute_quartic_obstacle_field(xi, yi, mode)
            rows.append({
                "frame_id": frame_id, "x": xi, "y": yi,
                "ux": ux, "uy": uy,
                "grad_ux_dx": gux[0], "grad_ux_dy": gux[1],
                "grad_uy_dx": guy[0], "grad_uy_dy": guy[1],
                "mode": mode, "obstacle": int(obstacle_mask(xi, yi)),
            })
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return X, Y


def plot_obstacle_field(save_path=None, csv_path=None, n=180):
    """Plot exactly the vector labels used for training.

    If ``csv_path`` is supplied, every row in that CSV is drawn, including
    duplicated rows in the bimodal region.  This makes the figure a label
    visualization rather than a schematic background field.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    if csv_path and os.path.exists(csv_path):
        import pandas as pd
        data = pd.read_csv(csv_path)
    else:
        # Keep the function useful when called without first writing a CSV.
        x = np.linspace(*X_RANGE, n + 1)
        ny = int(round(n * (Y_RANGE[1] - Y_RANGE[0]) /
                       (X_RANGE[1] - X_RANGE[0]))) + 1
        y = np.linspace(*Y_RANGE, ny)
        rows = []
        for xi in x:
            for yi in y:
                for label in modes_for_point(xi, yi):
                    ux, uy, _ = compute_quartic_obstacle_field(xi, yi, label,
                                                                  gradients=False)
                    rows.append((xi, yi, float(ux), float(uy), label))
        data = np.asarray(rows, dtype=float)

    if hasattr(data, "columns"):
        x, y, ux, uy, labels = (data[c].to_numpy() for c in
                                 ("x", "y", "ux", "uy", "mode"))
        frame_ids = data["frame_id"].to_numpy() if "frame_id" in data else None
    else:
        x, y, ux, uy, labels = data.T
        frame_ids = None

    labels = labels.astype(int)
    colors = {0: "0.45", 1: "tab:blue", -1: "tab:orange"}
    names = {0: "mode 0: original quartic", 1: "mode +1: upper bypass",
             -1: "mode -1: lower bypass"}

    fig, ax = plt.subplots(figsize=(11, 6.5))
    # Draw each training label separately.  No unlabelled background quiver is
    # added, so every arrow in the figure corresponds to a training target.
    for label in (0, 1, -1):
        mask = labels == label
        if not np.any(mask):
            continue
        ax.quiver(x[mask], y[mask], ux[mask], uy[mask],
                  color=colors[label], alpha=0.62 if label == 0 else 0.78,
                  angles="xy", scale_units="xy", scale=15,
                  width=0.0022, zorder=3 + (label != 0),
                  label=f"{names[label]} ({int(mask.sum())} labels)")

    # Mark positions for which one image/frame has two training targets.
    if frame_ids is not None:
        counts = {}
        for frame, label in zip(frame_ids, labels):
            counts.setdefault(int(frame), set()).add(int(label))
        bimodal_frames = {frame for frame, modes in counts.items() if len(modes) > 1}
        bimodal = np.isin(frame_ids, list(bimodal_frames))
        if np.any(bimodal):
            # One marker per unique frame, avoiding a double-sized marker for
            # the two rows that share the same image.
            points = {}
            for frame, xi, yi in zip(frame_ids[bimodal], x[bimodal], y[bimodal]):
                points[int(frame)] = (xi, yi)
            px, py = np.asarray(list(points.values())).T
            ax.scatter(px, py, s=22, facecolors="none", edgecolors="tab:purple",
                       linewidths=0.9, zorder=6,
                       label=f"bimodal image locations ({len(points)})")

    # Route and obstacle geometry are references, not additional training
    # vectors, so they use line/patch legend entries below.
    xx = np.linspace(BRANCH_X[0], BRANCH_X[1], 500)
    ax.plot(xx, route(xx, 1)[0], color="tab:blue", lw=2.0,
            label="upper feasible route")
    ax.plot(xx, route(xx, -1)[0], color="tab:orange", lw=2.0,
            label="lower feasible route")
    ax.plot(xx, route(xx, 0)[0], "k--", lw=1.2, alpha=0.7,
            label="original quartic reference")
    t = np.linspace(0, 2 * np.pi, 240)
    ax.fill(OBSTACLE_CENTER[0] + OBSTACLE_RADII[0] * np.cos(t),
            OBSTACLE_CENTER[1] + OBSTACLE_RADII[1] * np.sin(t),
            color="tab:red", alpha=0.25, zorder=1)
    ax.axvspan(BRANCH_X[0], BRANCH_X[1], color="tab:green", alpha=0.035,
               zorder=0)
    ax.axvspan(*DECISION_X, color="tab:purple", alpha=0.08, zorder=0)
    ax.axvline(BRANCH_X[0], color="tab:purple", ls=":", lw=1.2,
               label="front path junction")
    ax.axvline(BRANCH_X[1], color="tab:purple", ls=":", lw=1.2,
               label="rear path junction")

    # Keep labels explicit even when a mode is absent in a custom CSV.
    handles, _ = ax.get_legend_handles_labels()
    handles.extend([
        Patch(facecolor="tab:red", alpha=0.25, label="virtual elliptical obstacle"),
        Patch(facecolor="tab:purple", alpha=0.08,
              label="decision strip: both +1/-1 labels"),
        Patch(facecolor="tab:green", alpha=0.08,
              label="between junctions: split above/below quartic"),
    ])
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1),
              fontsize=8, frameon=True)
    unique_frames = len(set(frame_ids)) if frame_ids is not None else len(x)
    ax.set_title(f"Training vector labels: {len(x)} rows, {unique_frames} images")
    ax.set(xlim=X_RANGE, ylim=Y_RANGE, aspect="equal", xlabel="x", ylabel="y")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=220, bbox_inches="tight")
    return fig, ax


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="./src/path_following/vector_field_quartic_obstacle.csv")
    parser.add_argument("--plot", default="./src/path_following/quartic_obstacle_field.png")
    parser.add_argument("--n-points", type=int, default=N_POINTS)
    args = parser.parse_args()
    X, Y = write_dataset(args.csv, args.n_points)
    plot_obstacle_field(args.plot, csv_path=args.csv)
    extra = sum(len(modes_for_point(x, y)) - 1
                for x, y in zip(X.ravel(), Y.ravel()))
    print(f"wrote {args.csv} ({X.size} unique images, {X.size + extra} labels)")
    center, radii = obstacle_world_geometry()
    print(f"Virtual XY ellipse (optional Unity placement): center={center.tolist()}, "
          f"radii={radii.tolist()} m")
