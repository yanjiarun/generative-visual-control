import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.baseline.position_localization import (
    BATCH_SIZE, CSV_PATH, IMAGE_DIR, METERS_PER_UNIT, CNNPositionRegressor,
    UnityPositionDataset,
)
from src.path_following.create_path_cassini_oval import compute_cassini_field_with_gradients


MODEL_VARIANT = "cnn256"
MODEL_PATH = ROOT / "src/baseline/models/position_cassini_cnn/position_best.pth"
RUN_DIR = ROOT / "data/cassini_path_following_data/position_gvf"
OUTPUT_DIR = RUN_DIR / "sensitivity_diagnostics"
VELOCITY_MPS, SCALE, KP, TAU_THETA, FIELD_EPS = 23.0, 100.0, 0.18, 6.0, 1e-8


def field_terms(x, y):
    ux, uy, _, grad_ux, grad_uy = compute_cassini_field_with_gradients(float(x), float(y))
    vector = np.array([ux, uy], dtype=float)
    mix = grad_uy * vector[0] - grad_ux * vector[1]
# Retain atan2(0, 0) for control-law replay.
    angle = np.arctan2(vector[1], vector[0])
    return vector, mix, angle


def wrap(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


def summarize(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"count": 0}
    return {"count": int(len(values)), "mean": float(values.mean()),
            "median": float(np.median(values)), "p95_abs": float(np.percentile(np.abs(values), 95)),
            "max_abs": float(np.max(np.abs(values)))}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    dataset = UnityPositionDataset(CSV_PATH, IMAGE_DIR)
    model = CNNPositionRegressor().to(device)
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
    model.eval()

    predictions = []
    with torch.inference_mode():
        for images, _ in DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False):
            predictions.extend(model(images.to(device)).cpu().numpy())
    predictions = np.asarray(predictions)
    table = dataset.rows.copy()
    truth = table[["x", "y"]].to_numpy(dtype=float)
    errors_m = np.linalg.norm(predictions - truth, axis=1) * METERS_PER_UNIT
    radius = np.linalg.norm(truth, axis=1)
    true_vectors, true_mixes, true_angles = map(np.asarray, zip(*(field_terms(*xy) for xy in truth)))
    pred_vectors, pred_mixes, pred_angles = map(np.asarray, zip(*(field_terms(*xy) for xy in predictions)))
    angle_error_deg = np.degrees(wrap(pred_angles - true_angles))
    angle_valid = ((np.linalg.norm(true_vectors, axis=1) > FIELD_EPS) &
                   (np.linalg.norm(pred_vectors, axis=1) > FIELD_EPS))
    angle_error_deg[~angle_valid] = np.nan
    table["pred_x"], table["pred_y"] = predictions[:, 0], predictions[:, 1]
    table["position_error_m"], table["radius"] = errors_m, radius
    table["true_field_angle_deg"], table["pred_field_angle_deg"] = np.degrees(true_angles), np.degrees(pred_angles)
    table["field_angle_error_deg"] = angle_error_deg
    table["true_field_norm"] = np.linalg.norm(true_vectors, axis=1)
    table["pred_field_norm"] = np.linalg.norm(pred_vectors, axis=1)
    table["gradient_mix_error_norm"] = np.linalg.norm(pred_mixes - true_mixes, axis=1)
    training_samples = checkpoint.get("training_samples")
    trained_on_all = training_samples == len(table) if training_samples is not None else None
    metrics = {"model_variant": MODEL_VARIANT, "model_path": str(MODEL_PATH),
               "device": str(device), "samples": len(table),
               "training_samples": training_samples,
               "trained_on_all_grid_images": trained_on_all,
               "meters_per_unit": METERS_PER_UNIT,
               "metric_scope": ("full-grid training fit; not held-out validation"
                                if trained_on_all else
                                "full-grid evaluation; checkpoint training coverage is unconfirmed"),
               "full_grid_position_error_m": {}, "full_grid_field_angle_error_deg": {},
               "full_grid_gradient_mix_error_norm": {}}
    regions = [("r_lt_0_2", radius < 0.2), ("r_0_2_to_0_5", (radius >= 0.2) & (radius < 0.5)),
               ("r_ge_0_5", radius >= 0.5)]
    for name, mask in [("all", np.ones(len(table), dtype=bool)), *regions]:
        metrics["full_grid_position_error_m"][name] = summarize(errors_m[mask])
        metrics["full_grid_field_angle_error_deg"][name] = summarize(angle_error_deg[mask])
        metrics["full_grid_gradient_mix_error_norm"][name] = summarize(
            np.linalg.norm(pred_mixes - true_mixes, axis=1)[mask])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT_DIR / "dataset_position_gvf.csv", index=False)
    run_path = RUN_DIR / "poses.npy"
    if run_path.exists():
        poses = np.load(run_path)
        estimates = np.load(RUN_DIR / "position_estimates_xy.npy")
        if len(poses) != len(estimates):
            raise ValueError("saved poses and position estimates are not aligned")
        run_errors = np.linalg.norm(estimates - (poses[:, :2] - [576, 275]) / SCALE, axis=1) * SCALE
        commands = np.load(RUN_DIR / "heading_commands.npy")
        if len(commands) != len(poses):
            raise ValueError("saved poses and heading commands are not aligned")
        actual_xy = (poses[:, :2] - [576, 275]) / SCALE
        true_vectors, true_mixes, run_true_angles = map(np.asarray, zip(*(field_terms(*xy) for xy in actual_xy)))
        pred_vectors, pred_mixes, run_pred_angles = map(np.asarray, zip(*(field_terms(*xy) for xy in estimates)))
        pdot = (VELOCITY_MPS / SCALE) * np.column_stack((np.cos(poses[:, 2]), np.sin(poses[:, 2])))
        run_true_mix, run_pred_mix = (np.sum(mix * pdot, axis=1) for mix in (true_mixes, pred_mixes))
        run_fb_true, run_fb_pred = (KP * np.sin(angles - poses[:, 2]) for angles in (run_true_angles, run_pred_angles))
        run_omega_true, run_omega_pred = run_fb_true + run_true_mix, run_fb_pred + run_pred_mix
        run_angle_error = np.degrees(wrap(run_pred_angles - run_true_angles))
        true_run_norm, pred_run_norm = (np.linalg.norm(vectors, axis=1) for vectors in (true_vectors, pred_vectors))
        run_angle_error[(true_run_norm <= FIELD_EPS) | (pred_run_norm <= FIELD_EPS)] = np.nan
        gradient_error = run_pred_mix - run_true_mix
        omega_error = run_omega_pred - run_omega_true
        offline_commands = poses[:, 2] + TAU_THETA * run_omega_pred
        run_table = pd.DataFrame({
            "frame": np.arange(1, len(poses) + 1), "true_x": poses[:, 0], "true_y": poses[:, 1],
            "true_x_norm": (poses[:, 0] - 576) / SCALE, "true_y_norm": (poses[:, 1] - 275) / SCALE,
            "pred_x_norm": estimates[:, 0], "pred_y_norm": estimates[:, 1], "position_error_m": run_errors,
            "actual_heading_deg": np.degrees(poses[:, 2]),
            "true_field_angle_deg": np.degrees(run_true_angles), "pred_field_angle_deg": np.degrees(run_pred_angles),
            "field_angle_error_deg": run_angle_error,
            "true_feedback_rad_s": run_fb_true, "pred_feedback_rad_s": run_fb_pred,
            "true_gradient_rad_s": run_true_mix, "pred_gradient_rad_s": run_pred_mix,
            "gradient_error_rad_s": gradient_error,
            "true_omega_rad_s": run_omega_true, "pred_omega_rad_s": run_omega_pred,
            "yaw_rate_error_rad_s": omega_error,
            "saved_heading_command_deg": np.degrees(commands),
            "offline_pred_heading_command_deg": np.degrees(offline_commands),
            "pred_vs_true_command_delta_deg": np.degrees(TAU_THETA * (run_omega_pred - run_omega_true)),
        })
        run_table.to_csv(OUTPUT_DIR / "run_position_gvf.csv", index=False)
        frames = run_table["frame"].to_numpy()
        fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
        axes[0].plot(frames, run_errors)
        axes[0].set_ylabel("Position error (m)")
        axes[1].plot(frames, run_table["field_angle_error_deg"])
        axes[1].set_ylabel("GVF angle error (deg)")
        axes[2].plot(frames, run_pred_mix, label="Estimated position")
        axes[2].plot(frames, run_true_mix, label="True position", alpha=0.75)
        axes[2].set_ylabel("Gradient term (rad/s)")
        axes[2].set_xlabel("Frame")
        axes[2].legend()
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "online_sensitivity.png", dpi=160)
        plt.close(fig)
        metrics["online_run"] = {
            "frames": len(poses), "position_error_m": summarize(run_errors),
            "field_angle_error_deg": summarize(run_angle_error),
            "true_gradient_rad_s": summarize(run_true_mix),
            "pred_gradient_rad_s": summarize(run_pred_mix),
            "gradient_error_rad_s": summarize(gradient_error),
            "true_omega_rad_s": summarize(run_omega_true), "pred_omega_rad_s": summarize(run_omega_pred),
            "yaw_rate_error_rad_s": summarize(omega_error),
            "frames_abs_gradient_exceeds_feedback": int(np.sum(np.abs(run_pred_mix) > np.abs(run_fb_pred))),
            "saved_command_recompute_max_abs_error_rad": float(np.max(np.abs(offline_commands - commands))),
        }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Offline diagnostic outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
