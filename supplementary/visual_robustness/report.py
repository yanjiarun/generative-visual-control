from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pandas as pd

from supplementary.visual_robustness.common import (
    COLORS, CONDITIONS, DT, LABELS, METHODS, OFFLINE_DIR, REPORT_DIR, ROOT, RUN_DIR, RETRY_DIR,
    corrupt_image, corruption_seed,
)

CONDITION_LABELS = ["Clean", "Illumination", "Noise", "Blur", "Occlusion"]


def mean_sd(values):
    values = pd.Series(values, dtype=float).dropna()
    if not len(values):
        return "—"
    sd = f"{values.std(ddof=1):.3f}" if len(values) > 1 else "—"
    return f"{values.mean():.3f} ± {sd}"


def save_figure(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(REPORT_DIR / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_table(frame, name, title):
    frame.to_csv(REPORT_DIR / f"{name}.csv", index=False)
    fig, ax = plt.subplots(figsize=(17, max(3.5, 0.38 * (len(frame) + 2))))
    ax.axis("off")
    table = ax.table(cellText=frame.fillna("—").to_numpy(), colLabels=frame.columns,
                     cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.6)
    table.auto_set_column_width(list(range(len(frame.columns))))
    ax.set_title(title, pad=25)
    save_figure(fig, name)


def closed_loop():
    path = RETRY_DIR / "manifest_resolved.csv"
    if not path.exists():
        path = RUN_DIR / "manifest.csv"
    if not path.exists():
        print("Closed-loop manifest absent; skipping Table 1.")
        return
    runs = pd.read_csv(path)
    for column in ("rmse_m", "max_error_m"):
        if column not in runs:
            runs[column] = np.nan
    runs.to_csv(REPORT_DIR / "closed_loop_metrics_per_run.csv", index=False)
    runs["completed"] = runs.status.eq("completed")
    completed = runs[runs.completed]
    numeric, formatted, supplement = [], [], []
    for method in METHODS:
        cells = {"Method": LABELS[method]}
        for condition, heading in zip(CONDITIONS, CONDITION_LABELS):
            group = runs[(runs.method == method) & (runs.condition == condition)]
            good = group[group.completed]
            cells[heading] = mean_sd(good.rmse_m)
            numeric.append({"method": method, "condition": condition, "n": len(good),
                            "rmse_mean_m": good.rmse_m.mean(), "rmse_sd_m": good.rmse_m.std(ddof=1)})
            supplement.append({
                "Method": LABELS[method], "Condition": condition,
                "Max error (m)": mean_sd(good.max_error_m),
                "Completed/Planned": f"{len(good)}/{len(group)}",
                "Pending": int(group.status.eq("pending").sum()),
                "Completion (%)": 100 * len(good) / len(group) if len(group) else np.nan,
                "Tracking success (%)": 100 * group.get("tracking_success", pd.Series(False, index=group.index)).eq(True).sum() / len(group) if len(group) else np.nan,
            })
        formatted.append(cells)
    save_table(pd.DataFrame(formatted), "table1_closed_loop_rmse",
               "Closed-loop path-following performance under visual corruptions (m; mean ± SD)")
    pd.DataFrame(supplement).to_csv(REPORT_DIR / "supplement_max_completion.csv", index=False)
    pd.DataFrame(numeric).to_csv(REPORT_DIR / "closed_loop_rmse_numeric.csv", index=False)

    clean = completed[completed.condition == "clean"][["method", "repetition", "rmse_m"]]
    paired = runs[runs.condition != "clean"].merge(
        clean.rename(columns={"rmse_m": "clean_rmse_m"}), on=["method", "repetition"], how="left")
    paired["delta_percent"] = np.where(
        paired.completed & (paired.clean_rmse_m > 0),
        100 * (paired.rmse_m / paired.clean_rmse_m - 1), np.nan)
    paired.to_csv(REPORT_DIR / "paired_rmse_change_per_run.csv", index=False)
    paired.groupby(["method", "condition"], sort=False).delta_percent.agg(
        mean_percent="mean", sd_percent="std", paired_trials="count").reset_index().to_csv(
            REPORT_DIR / "supplement_rmse_change_percent.csv", index=False)

    pivot = pd.DataFrame(numeric).pivot(index="method", columns="condition", values="rmse_mean_m")
    matrix = pivot.reindex(index=METHODS, columns=CONDITIONS).to_numpy()
    if np.isfinite(matrix).any():
        fig, ax = plt.subplots(figsize=(11, 5))
        im = ax.imshow(matrix, aspect="auto", cmap="viridis")
        ax.set_xticks(range(5), CONDITION_LABELS)
        ax.set_yticks(range(4), [LABELS[m] for m in METHODS])
        for (i, j), value in np.ndenumerate(matrix):
            if np.isfinite(value):
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", color="white")
        fig.colorbar(im, ax=ax, label="Geometric RMSE (m)")
        save_figure(fig, "closed_loop_rmse_heatmap")

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    for ax, method in zip(axes.ravel(), METHODS):
        for condition, style in (("clean", "--"), ("motion_blur", "-")):
            pair = runs[(runs.method == method) & (runs.condition == condition) & (runs.repetition == 0)]
            if pair.empty:
                continue
            path = Path(pair.folder.iloc[0]) / "geometric_errors_m.npy"
            if path.exists():
                values = np.load(path)[:2000]
                label = ("Clean" if condition == "clean" else "Motion blur") + (
                    "" if bool(pair.completed.iloc[0]) else " (incomplete)")
                ax.plot(np.arange(len(values)) * DT, values, style, color=COLORS[method], label=label, linewidth=2)
        ax.set_title(LABELS[method])
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Geometric tracking error (m)")
        ax.grid(True)
        if ax.lines:
            ax.legend()
    fig.tight_layout()
    save_figure(fig, "figure1_motion_blur_rep00")


def offline_table():
    files = sorted((RUN_DIR / "offline").glob("errors_rep_*.csv"))
    if not files:
        print("Offline errors absent; skipping Table 2.")
        return
    errors = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    by_repeat = errors.groupby(["method", "condition", "repetition"], sort=False).agg(
        mean_deg=("angular_error_deg", "mean"),
        p95_deg=("angular_error_deg", lambda v: v.quantile(0.95)),
        samples=("angular_error_deg", "count"),
        invalid_predictions=("valid_prediction", lambda v: int((~v.astype(bool)).sum())),
    ).reset_index()
    by_repeat.to_csv(REPORT_DIR / "offline_gvf_metrics_per_repeat.csv", index=False)
    rows = []
    for method in METHODS[:3]:
        for condition in CONDITIONS:
            group = by_repeat[(by_repeat.method == method) & (by_repeat.condition == condition)]
            rows.append({"Method": LABELS[method], "Condition": condition,
                         "Mean angular error (°)": mean_sd(group.mean_deg),
                         "P95 angular error (°)": mean_sd(group.p95_deg),
                         "N images/repeat": int(group.samples.min()) if len(group) else 0,
                         "Invalid predictions (total)": int(group.invalid_predictions.sum())})
    save_table(pd.DataFrame(rows), "table2_gvf_direction",
               "GVF prediction accuracy under visual corruptions (mean ± SD across repetitions)")


def input_preview():
    candidates = sorted((OFFLINE_DIR / "images").glob("*.png"))
    path = candidates[0] if candidates else ROOT / "src/path_following/rectified_panorama_cassini/rectified_color_panorama_0001.jpg"
    image = cv2.imread(str(path))
    if image is None:
        return
    fig, axes = plt.subplots(5, 1, figsize=(14, 7))
    for ax, condition in zip(axes, CONDITIONS):
        corrupted = corrupt_image(image, condition, corruption_seed(0, 1, condition))
        ax.imshow(cv2.cvtColor(corrupted, cv2.COLOR_BGR2RGB))
        ax.set_ylabel(condition.replace("_", " "), fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    save_figure(fig, "visual_corruptions_preview")


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 14})
    closed_loop()
    offline_table()
    input_preview()
    print(f"Reports: {REPORT_DIR}")


if __name__ == "__main__":
    main()
