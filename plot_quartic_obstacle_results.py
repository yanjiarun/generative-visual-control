from pathlib import Path
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, Patch
from torch.utils.data import DataLoader

from src.evaluate_obstacle_models import GridImages, load_model
from src.generative_model.core.generative_controller import sample_action
from src.path_following.create_path_quartic_obstacle import (
    BRANCH_X, DECISION_X, OBSTACLE_CENTER, OBSTACLE_RADII, X_RANGE, Y_RANGE,
    obstacle_mask, route,
)


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "comparison_plots_quartic_obstacle"
CSV = ROOT / "src/path_following/vector_field_quartic_obstacle.csv"
IMAGES = ROOT / "src/path_following/rectified_panorama_quartic_obstacle"
WEIGHTS = {
    "cnn": ROOT / "src/baseline/models/cnn_quartic_obstacle_models/cnn_final.pth",
    "dp": ROOT / "src/generative_model/models/diffusion_policy_quartic_obstacle_models/diffusion_policy_final.pth",
    "gvc": ROOT / "src/generative_model/models/flow_matching_quartic_obstacle_models/flow_matching_policy_final.pth",
}
TRAJECTORIES = {
    "Traditional upper": (ROOT / "data_quartic_obstacle/traditional_mode1", "#0072B2", "-"),
    "Traditional lower": (ROOT / "data_quartic_obstacle/traditional_mode-1", "#0072B2", "--"),
    "CNN": (ROOT / "data_quartic_obstacle/cnn_mode1", "#E69F00", "-."),
    "DP": (ROOT / "data_quartic_obstacle/dp_mode1", "#009E73", ":"),
    "GVC": (ROOT / "data_quartic_obstacle/generative_mode1", "#CC79A7", "-"),
}
TABLE_METHODS = {
    "Classical GVF": ("traditional_mode1", "traditional_mode-1"),
    "CNN-based": ("cnn_mode1",),
    "Diffusion Policy": ("dp_mode1",),
    "Proposed GVC": ("generative_mode1",),
}

FONT = "Times New Roman"
MODE_COLORS = {0: "#777777", 1: "#0072B2", -1: "#E69F00"}
MODE_NAMES = {0: "Mode 0 / Quartic", 1: "Upper", -1: "Lower"}
DECISION_COLOR = "#B7C9D6"
JUNCTION_COLOR = "#59636E"
OBSTACLE_COLOR = "#F4B183"
OBSTACLE_ALPHA = 0.5
# Larger values shorten the displayed vectors.
VECTOR_SCALE = 22
LEGEND_COLUMN_SPACING = 0.8
STRIDE, SAMPLES, ODE_STEPS, DP_STEPS = 2, 8, 50, 20

plt.rcParams.update({
    "font.family": FONT, "font.size": 24, "axes.labelsize": 24,
    "axes.titlesize": 24, "xtick.labelsize": 24, "ytick.labelsize": 24,
    "legend.fontsize": 24,
})


def geometry(ax, colored_routes=True):
    """Draw common reference geometry and return legend handles."""
    xx = np.linspace(*BRANCH_X, 500)
    ax.plot(xx, route(xx, 0)[0], "--", color=MODE_COLORS[0], lw=1.7)
    route_colors = (MODE_COLORS[1], MODE_COLORS[-1]) if colored_routes else ("0.45",) * 2
    route_style = "-" if colored_routes else (0, (2, 2))
    ax.plot(xx, route(xx, 1)[0], ls=route_style, color=route_colors[0], lw=1.8)
    ax.plot(xx, route(xx, -1)[0], ls=route_style, color=route_colors[1], lw=1.8)
    ax.axvspan(*DECISION_X, color=DECISION_COLOR, alpha=.35, zorder=0)
    for x in BRANCH_X:
        ax.axvline(x, color=JUNCTION_COLOR, ls=":", lw=1.05, zorder=2)
        ax.plot(x, route(x, 0)[0], "o", color=JUNCTION_COLOR, ms=4, zorder=6)
    ax.add_patch(Ellipse(OBSTACLE_CENTER, 2 * OBSTACLE_RADII[0],
                         2 * OBSTACLE_RADII[1], facecolor=OBSTACLE_COLOR,
                         edgecolor=OBSTACLE_COLOR, alpha=OBSTACLE_ALPHA,
                         linewidth=.8, zorder=20))
    ax.set(xlim=X_RANGE, ylim=Y_RANGE, aspect="equal", xlabel="x", ylabel="y")
    ax.grid(alpha=.22)
    handles = [
        Line2D([], [], color=MODE_COLORS[0], ls="--", lw=1.7,
               label="Quartic"),
    ]
    handles += ([
        Line2D([], [], color=route_colors[0], lw=1.8, label="Upper Path"),
        Line2D([], [], color=route_colors[1], lw=1.8, label="Lower Path"),
    ] if colored_routes else [
        Line2D([], [], color="0.45", ls=route_style, lw=1.8,
               label="Bypass Paths"),
    ])
    handles += [
        Patch(facecolor=OBSTACLE_COLOR, alpha=OBSTACLE_ALPHA, label="Obstacle"),
        Patch(facecolor=DECISION_COLOR, alpha=.35, label="Decision Region"),
        Line2D([], [], color=JUNCTION_COLOR, ls=":", marker="o", lw=1.05,
               label="Junctions"),
    ]
    return handles


def plot_labels(table):
    fig, ax = plt.subplots(figsize=(12, 8))
    for mode in (0, 1, -1):
        rows = table[table["mode"] == mode]
        if len(rows):
            visible = ~obstacle_mask(rows.x.to_numpy(), rows.y.to_numpy())
            rows = rows.loc[visible]
            ax.quiver(rows.x, rows.y, rows.ux, rows.uy, color=MODE_COLORS[mode],
                      angles="xy", scale_units="xy", scale=VECTOR_SCALE, width=.0021,
                      alpha=.55 if mode == 0 else .78, zorder=4,
                      label=MODE_NAMES[mode])
    handles = geometry(ax)
    ax.legend(handles=handles, loc="upper right", ncol=2, fontsize=18,
              columnspacing=LEGEND_COLUMN_SPACING,
              frameon=True, framealpha=.88, borderpad=.45, handlelength=1.7)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "quartic_obstacle_field.png", dpi=300,
                bbox_inches="tight")
    fig.savefig(OUT_DIR / "quartic_obstacle_field.pdf",
                bbox_inches="tight")
    plt.close(fig)


def unit(values):
    return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-8)


def predict(method, model, loader, device):
    outputs = []
    with torch.no_grad():
        for images in loader:
            images = images.to(device)
            batch = []
            for _ in range(SAMPLES if method != "cnn" else 1):
                if method == "gvc":
                    out = sample_action(model, images, ODE_STEPS, device)
                elif method == "dp":
                    out = model.predict_action(images, DP_STEPS,
                                               deterministic=False)[:, 0]
                else:
                    out = model(images[:, 0])
                batch.append(out)
            outputs.append(torch.stack(batch).cpu().numpy())
    return np.concatenate(outputs, axis=1)


def assignments(predictions, points, table):
    groups = [table[table.frame_id == int(frame)] for frame in points.frame_id]
    result = []
    for group, pred in zip(groups, predictions.transpose(1, 0, 2)):
        truth = unit(group[["ux", "uy"]].to_numpy(float))
        nearest = np.argmax(unit(pred[:, :2]) @ truth.T, axis=1)
        result.append(group["mode"].to_numpy(int)[nearest])
    return np.asarray(result).T


def plot_model_fields(table):
    metrics = evaluate_models(table, ("cnn", "dp", "gvc"), draw=True)
    return metrics


def model_metrics(predictions, points, table):
    groups = [table[table.frame_id == int(frame)] for frame in points.frame_id]
    angles, assigned, bimodal = [], [], []
    pred = unit(predictions[..., :2])
    for i, group in enumerate(groups):
        truth = unit(group[["ux", "uy"]].to_numpy(float))
        nearest = np.argmax(pred[:, i] @ truth.T, axis=1)
        assigned.append(group["mode"].to_numpy(int)[nearest])
        angles.append(np.degrees(np.arccos(np.clip(
            np.sum(pred[:, i] * truth[nearest], axis=1), -1, 1))))
        bimodal.append(len(group) > 1)
    angles, assigned, bimodal = np.asarray(angles).T, np.asarray(assigned).T, np.asarray(bimodal)
    recover = [len(np.unique(assigned[:, i])) == 2 for i in np.flatnonzero(bimodal)]
    return {
        "decision": float(angles[:, bimodal].mean()),
        "single": float(angles[:, ~bimodal].mean()),
        "recovery": 100.0 * float(np.mean(recover)) if recover else np.nan,
    }


def evaluate_models(table, methods=("cnn", "dp", "gvc"), draw=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = GridImages(table, IMAGES, stride=STRIDE)
    points = dataset.points.reset_index(drop=True)
    selected = table[table.frame_id.isin(points.frame_id)]
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    predictions, labels = {}, {}
    metrics = {}
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    for method in methods:
        print(f"Evaluating {method} on {len(dataset)} positions...")
        model = load_model(method, WEIGHTS[method], device)
        predictions[method] = predict(method, model, loader, device)
        labels[method] = assignments(predictions[method], points, selected)
        metrics[method] = model_metrics(predictions[method], points, selected)

    if not draw:
        return metrics
    for method in ("cnn", "dp", "gvc"):
        fig, ax = plt.subplots(figsize=(12, 10))
        pred = unit(predictions[method][..., :2])
        assign = labels[method]
        for sample in range(len(pred)):
            for mode in (0, 1, -1):
                mask = assign[sample] == mode
                mask &= ~obstacle_mask(points.x.to_numpy(), points.y.to_numpy())
                if np.any(mask):
                    ax.quiver(
                        points.x.to_numpy()[mask], points.y.to_numpy()[mask],
                        pred[sample, mask, 0], pred[sample, mask, 1],
                        color=MODE_COLORS[mode], angles="xy",
                        scale_units="xy", scale=VECTOR_SCALE,
                        alpha=.20 if len(pred) > 1 else .78,
                        width=.0021, zorder=4)
        geometry(ax)
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"model_vector_field_{method}.png", dpi=300,
                    bbox_inches="tight")
        fig.savefig(OUT_DIR / f"model_vector_field_{method}.pdf",
                    bbox_inches="tight")
        plt.close(fig)
    return metrics


def plot_trajectories():
    fig, ax = plt.subplots(figsize=(12, 8))
    handles = geometry(ax, colored_routes=False)
    for label, (folder, color, linestyle) in TRAJECTORIES.items():
        files = sorted(folder.glob("run_*/poses.npy"))
        for file in files:
            poses = np.load(file)
            xy = (poses[:, :2] - np.array([576., 275.])) / 200.
            ax.plot(xy[:, 0], xy[:, 1], color=color, ls=linestyle,
                    lw=1.25, alpha=.34, zorder=3)
        handles.append(Line2D([], [], color=color, ls=linestyle, lw=2,
                               label=f"{label} ({len(files)} runs)"))
    ax.legend(handles=handles, loc="upper right", ncol=2, fontsize=18,
              columnspacing=LEGEND_COLUMN_SPACING,
              frameon=True, framealpha=.88, borderpad=.45, handlelength=1.7)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "trajectories.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / "trajectories.pdf", bbox_inches="tight")
    plt.close(fig)


def trajectory_metrics():
    """Return closed-loop route, safety and upper/lower selection statistics."""
    result = {}
    for method, folders in TABLE_METHODS.items():
        runs, selections = [], []
        for name in folders:
            folder = ROOT / "data_quartic_obstacle" / name
            for file in sorted(folder.glob("run_*/poses.npy")):
                poses = np.load(file)
                x = (poses[:, 0] - 576.0) / 200.0
                y = (poses[:, 1] - 275.0) / 200.0
                q, qp = route(x, 0)
                branch = (x >= BRANCH_X[0]) & (x <= BRANCH_X[1])
                choose = (x >= DECISION_X[1]) & (x <= BRANCH_X[1])
                if not np.any(choose):
                    choose = branch
                mode = 1 if np.mean(y[choose] - q[choose]) >= 0.0 else -1
                f, fp = route(x, mode)
                target, slope = np.where(branch, f, q), np.where(branch, fp, qp)
                error = (y - target) / np.sqrt(1.0 + slope * slope) * 200.0
                hit = (((x - OBSTACLE_CENTER[0]) / OBSTACLE_RADII[0]) ** 2
                        + ((y - OBSTACLE_CENTER[1]) / OBSTACLE_RADII[1]) ** 2 <= 1.0).any()
                finished = np.isfinite(x[-1]) and x[-1] >= X_RANGE[1] - 0.05
                runs.append((float(np.sqrt(np.mean(error ** 2))),
                             float(np.max(np.abs(error))), not hit and finished))
                selections.append(mode)
        if not runs:
            raise FileNotFoundError(f"No trajectories found for {method}")
        result[method] = {
            "rmse": float(np.mean([r[0] for r in runs])),
            "max": float(np.max([r[1] for r in runs])),
            "success": 100.0 * float(np.mean([r[2] for r in runs])),
            "upper": int(sum(m == 1 for m in selections)),
            "lower": int(sum(m == -1 for m in selections)),
            "runs": len(runs),
        }
    return result


def write_table(model, closed_loop, path):
    """Write the paper-ready quantitative comparison and a CSV companion."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in TABLE_METHODS:
        key = {"CNN-based": "cnn", "Proposed GVC": "gvc"}.get(name, name)
        m, t = model.get(key, {}), closed_loop[name]
        if name == "Classical GVF":
            decision = recovery = single = "--"
            selection = "Prescribed"
        elif name == "Diffusion Policy":
            decision = recovery = single = "--"
            selection = f"{100*t['upper']/t['runs']:.0f}/{100*t['lower']/t['runs']:.0f}"
        else:
            decision = f"{m['decision']:.2f}"
            recovery = f"{m['recovery']:.1f}"
            single = f"{m['single']:.2f}"
            selection = f"{100*t['upper']/t['runs']:.0f}/{100*t['lower']/t['runs']:.0f}"
        rows.append((name, decision, recovery, single, selection,
                     f"{t['success']:.1f}", f"{t['rmse']:.2f}", f"{t['max']:.2f}"))
    tex = [
        r"\begin{table*}[t]", r"\centering",
        r"\caption{Quantitative comparison in the multimodal obstacle-avoidance experiment.}",
        r"\label{tab:multimodal_obstacle}", r"\small", r"\setlength{\tabcolsep}{5pt}",
        r"\begin{tabular}{lccccccc}", r"\toprule",
        r"Method & \shortstack{Decision error\\($^\circ$)$\downarrow$} & \shortstack{Mode recovery\\(\%)$\uparrow$} & \shortstack{Single-mode error\\($^\circ$)$\downarrow$} & \shortstack{U/L selection\\(\%)} & \shortstack{Success\\(\%)$\uparrow$} & \shortstack{Route RMSE\\(m)$\downarrow$} & \shortstack{Max. deviation\\(m)$\downarrow$} \\",
        r"\midrule",
    ]
    tex += [("{} & {} & {} & {} & {} & {} & {} & {} " + r"\\").format(*r)
            for r in rows]
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    path.write_text("\n".join(tex), encoding="utf-8")
    csv = path.with_suffix(".csv")
    pd.DataFrame(rows, columns=["Method", "Decision error (deg)", "Mode recovery (%)",
                                "Single-mode error (deg)", "U/L selection (%)",
                                "Success (%)", "Route RMSE (m)", "Max. deviation (m)"]).to_csv(csv, index=False)
    print(f"Saved table: {path}\nSaved table data: {csv}")
    print("\n".join(tex))


def main(table_only=False, table_path=None):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(CSV)
    model = evaluate_models(table, ("cnn", "gvc") if table_only else ("cnn", "dp", "gvc"),
                            draw=not table_only)
    closed_loop = trajectory_metrics()
    write_table(model, closed_loop, table_path or OUT_DIR / "multimodal_obstacle_table.tex")
    if not table_only:
        plot_labels(table)
        plot_trajectories()
        print(f"Saved three experiment-2 figures to {OUT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--table-only", action="store_true",
                        help="only evaluate and write the quantitative table")
    parser.add_argument("--table-out", type=Path, default=None,
                        help="optional .tex output path")
    args = parser.parse_args()
    main(args.table_only, args.table_out)
