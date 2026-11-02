import json
import time

import os
import numpy as np
import torch

from src.cnn_visual_control import CNNVisualControl
from src.baseline.position_localization import METERS_PER_UNIT, CNNPositionRegressor
from src.path_following.create_path_cassini_oval import compute_cassini_field_with_gradients


class PositionGVFVisualControl(CNNVisualControl):
    def __init__(self, config):
        if config.get("path_type", "cassini") != "cassini":
            raise ValueError("VisLoc-GVF currently supports Cassini only")
        if config.get("seq_length", 1) != 1:
            raise ValueError("Position baseline is trained for one image frame")
        super().__init__({**config, "cnn_model_path": config["position_model_path"]})
        self.position_predictions = []
        self.inference_times_ms = []
        self.pipeline_times_ms = []

    def _load_cnn_model(self, model_path):
        model = CNNPositionRegressor().to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
        return model.eval()

    def generate_vector_field(self, image_tensor, _current_position=None):

        image = image_tensor.squeeze(1)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        start = time.perf_counter()
        with torch.inference_mode():
            position = self.cnn_model(image).squeeze(0).cpu().numpy()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.inference_times_ms.append((time.perf_counter() - start) * 1000)
        x_hat, y_hat = map(float, position)
        self.position_predictions.append([x_hat, y_hat])

        ux, uy, _, delta_ux, delta_uy = compute_cassini_field_with_gradients(x_hat, y_hat)
        command = np.r_[ux, uy, delta_uy * ux - delta_ux * uy]
        if not np.isfinite(command).all():
            raise ValueError("Non-finite position/GVF prediction")
        self.pipeline_times_ms.append((time.perf_counter() - start) * 1000)
        return command

    def control_loop(self, max_frames=1000, save_data=False):
        super().control_loop(max_frames=max_frames, save_data=save_data)
        save_folder = self.config.get("save_folder")
        if not save_data or not save_folder:
            return

        os.makedirs(save_folder, exist_ok=True)
        estimates = np.asarray(self.position_predictions, dtype=float).reshape(-1, 2)
        inference = np.asarray(self.inference_times_ms, dtype=float)
        pipeline = np.asarray(self.pipeline_times_ms, dtype=float)
        np.save(os.path.join(save_folder, "position_estimates_xy.npy"), estimates)
        world_estimates = np.array([576.0, 275.0]) + METERS_PER_UNIT * estimates
        np.save(os.path.join(save_folder, "position_estimates_world_xy.npy"), world_estimates)
        np.save(os.path.join(save_folder, "cnn_inference_times_ms.npy"), inference)
        np.save(os.path.join(save_folder, "position_gvf_times_ms.npy"), pipeline)
        with open(os.path.join(save_folder, "run_config.json"), "w", encoding="utf-8") as f:
            json.dump({**self.config, "max_frames": max_frames, "kp": 0.18,
                       "init_pose": np.asarray(self.config.get("init_pose", [0, 0, 0])).tolist(),
                       "meters_per_unit": METERS_PER_UNIT}, f, indent=2)

        comparison_frames = min(2000, len(estimates))
        metrics = {"frames": len(estimates), "comparison_frames": comparison_frames}
        if len(inference):
            warmup = 20 if len(inference) > 20 else 0
            common_inference = inference[warmup:comparison_frames]
            common_pipeline = pipeline[warmup:comparison_frames]
            full_inference, full_pipeline = inference[warmup:], pipeline[warmup:]
            metrics.update(timing_warmup_frames=warmup,
                           cnn_inference_mean_ms=float(common_inference.mean()),
                           cnn_inference_p95_ms=float(np.percentile(common_inference, 95)),
                           cnn_inference_full_run_mean_ms=float(full_inference.mean()),
                           position_gvf_mean_ms=float(common_pipeline.mean()),
                           position_gvf_full_run_mean_ms=float(full_pipeline.mean()))
        pose_path = os.path.join(save_folder, "poses.npy")
        if os.path.exists(pose_path):
            from plot_data_cassini import calculate_geometric_path_errors
            poses = np.load(pose_path)
            if len(poses) != len(estimates):
                raise ValueError("Saved poses and position predictions are not aligned")
# Ground truth is used only for scoring.
            loc_errors = np.linalg.norm(world_estimates - poses[:, :2], axis=1) if len(poses) else np.array([])
            np.save(os.path.join(save_folder, "position_errors_m.npy"), loc_errors)
            if len(loc_errors):
                common_loc_errors = loc_errors[:comparison_frames]
                metrics.update(localization_mae_m=float(common_loc_errors.mean()),
                               localization_rmse_m=float(np.sqrt(np.mean(common_loc_errors ** 2))),
                               localization_max_error_m=float(common_loc_errors.max()),
                               localization_mae_full_run_m=float(loc_errors.mean()),
                               localization_rmse_full_run_m=float(np.sqrt(np.mean(loc_errors ** 2))),
                               localization_max_error_full_run_m=float(loc_errors.max()))
            tracking_all = calculate_geometric_path_errors(poses) if len(poses) else np.array([])
            if len(tracking_all):
                start_frame = 500 if len(tracking_all) > 500 else 0
                common_tracking = tracking_all[start_frame:comparison_frames]
                full_tracking = tracking_all[start_frame:]
                np.save(os.path.join(save_folder, "geometric_errors_m.npy"), tracking_all)
                metrics.update(geometric_tracking_rmse_m=float(np.sqrt(np.mean(common_tracking ** 2))),
                               geometric_tracking_max_error_m=float(common_tracking.max()),
                               geometric_tracking_start_index=start_frame,
                               geometric_tracking_frames=len(common_tracking),
                               geometric_tracking_mean_m=float(common_tracking.mean()),
                               geometric_tracking_std_m=float(common_tracking.std()),
                               geometric_tracking_rmse_full_run_m=float(np.sqrt(np.mean(full_tracking ** 2))),
                               geometric_tracking_max_error_full_run_m=float(full_tracking.max()),
                               geometric_tracking_rmse_all_m=float(np.sqrt(np.mean(tracking_all ** 2))),
                               geometric_tracking_max_error_all_m=float(tracking_all.max()))
        with open(os.path.join(save_folder, "position_gvf_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"VisLoc-GVF metrics: {metrics}")
