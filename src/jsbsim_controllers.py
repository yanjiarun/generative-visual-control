import json
import math
import os
import time

import numpy as np
import torch

from src.baseline.cnn_model import BaselineCNN
from src.baseline.position_localization import CNNPositionRegressor, METERS_PER_UNIT
from src.generative_model.core.diffusion_policy import DiffusionPolicy
from src.generative_model.core.generative_controller import (
    FlowMatchingController, sample_action)
from src.path_following.collect_imgs_path_following import (
    compute_path_field_with_gradients)
from src.visual_control_base_jsbsim import VisualControlJSBSim, ned_to_enu_heading


def _state(path, device):
    state = torch.load(path, map_location=device)
    return state.get("model_state_dict", state)


class CNNVisualControlJSBSim(VisualControlJSBSim):
    def __init__(self, config):
        super().__init__(config)
        self.model = BaselineCNN().to(self.device)
        self.model.load_state_dict(_state(config["cnn_model_path"], self.device))
        self.model.eval()

    def guidance(self, image_sequence, _position, _state):
        with torch.no_grad():
            return self.model(image_sequence[:, -1]).cpu().numpy()[0]


class DiffusionPolicyVisualControlJSBSim(VisualControlJSBSim):
    def __init__(self, config):
        super().__init__(config)
        self.inference_steps = int(config.get("num_inference_steps", 20))
        self.heading_gain = float(config.get("heading_gain", 1.0))
        self.model = DiffusionPolicy(
            history_length=self.seq_length,
            pred_horizon=int(config.get("pred_horizon", 2)),
            inference_steps=self.inference_steps).to(self.device)
        self.model.load_state_dict(_state(config["dp_model_path"], self.device))
        self.model.eval()

    def guidance(self, image_sequence, _position, _state):
        with torch.no_grad():
            return self.model.predict_action(
                image_sequence, self.inference_steps)[0, 0].cpu().numpy()

    def heading_command(self, output, state):
        desired = math.atan2(output[1], output[0])
        current = ned_to_enu_heading(state.get_attitude()[2])
        error = math.atan2(math.sin(desired - current), math.cos(desired - current))
        rate = np.clip(self.heading_gain * error, -self.max_turn_rate, self.max_turn_rate)
        return current + self.heading_lookahead * rate


class GenerativeVisualControlJSBSim(VisualControlJSBSim):
    def __init__(self, config):
        super().__init__(config)
        self.num_steps = int(config.get("num_steps", 50))
        self.model = FlowMatchingController(history_length=self.seq_length).to(self.device)
        self.model.load_state_dict(_state(config["model_path"], self.device))
        self.model.eval()

    def guidance(self, image_sequence, _position, _state):
        with torch.no_grad():
            return sample_action(self.model, image_sequence, self.num_steps,
                                 self.device)[0].cpu().numpy()


class TraditionalVectorFieldControlJSBSim(VisualControlJSBSim):
    requires_image = False

    def guidance(self, _image_sequence, position, _state):
        xy = (position - np.array([576.0, 275.0])) / self.scale_factor
        if np.linalg.norm(xy) < 1e-4:
            return np.array([1.0, 0.0, 0.0, 0.0])
        ux, uy, _alpha, grad_ux, grad_uy = compute_path_field_with_gradients(
            self.path_type, xy[0], xy[1])
        gradient = grad_uy * ux - grad_ux * uy
        return np.r_[ux, uy, gradient]


class PositionGVFVisualControlJSBSim(VisualControlJSBSim):
    def __init__(self, config):
        if config.get("path_type", "cassini") != "cassini":
            raise ValueError("VisLoc-GVF currently supports Cassini only")
        if int(config.get("seq_length", 1)) != 1:
            raise ValueError("The position model was trained with a single image frame")
        super().__init__(config)
        self.position_model = CNNPositionRegressor().to(self.device)
        self.position_model.load_state_dict(_state(config["position_model_path"], self.device))
        self.position_model.eval()
        self.position_predictions = []
        self.inference_times_ms = []
        self.pipeline_times_ms = []

    def guidance(self, image_sequence, _position, _state):
        start = time.perf_counter()
        with torch.inference_mode():
            estimate = self.position_model(image_sequence[:, -1]).squeeze(0).cpu().numpy()
        self.inference_times_ms.append((time.perf_counter() - start) * 1000.0)
        x_hat, y_hat = map(float, estimate)
        self.position_predictions.append([x_hat, y_hat])
        ux, uy, _, grad_ux, grad_uy = compute_path_field_with_gradients(
            "cassini", x_hat, y_hat)
        gradient = grad_uy * ux - grad_ux * uy
        output = np.r_[ux, uy, gradient]
        if not np.isfinite(output).all():
            raise ValueError("Non-finite estimated-position GVF output")
        self.pipeline_times_ms.append((time.perf_counter() - start) * 1000.0)
        return output

    def control_loop(self, max_frames=4000, save_data=True):
        super().control_loop(max_frames=max_frames, save_data=save_data)
        folder = self.config.get("save_folder")
        if not save_data or not folder:
            return
        estimates = np.asarray(self.position_predictions, dtype=float).reshape(-1, 2)
        inference = np.asarray(self.inference_times_ms, dtype=float)
        pipeline = np.asarray(self.pipeline_times_ms, dtype=float)
        self._save(folder, {
            "position_estimates_xy": estimates,
            "position_estimates_world_xy": np.array([576.0, 275.0]) + METERS_PER_UNIT * estimates,
            "cnn_inference_times_ms": inference,
            "position_gvf_times_ms": pipeline,
        })
        positions_path = os.path.join(folder, "positions.npy")
        metrics = {"frames": len(estimates), "comparison_frames": min(2000, len(estimates))}
        if os.path.isfile(positions_path) and len(estimates):
            positions = np.load(positions_path)
            if len(positions) != len(estimates):
                raise ValueError("Saved positions and CNN estimates are not aligned")
            world_estimates = np.array([576.0, 275.0]) + METERS_PER_UNIT * estimates
            localization = np.linalg.norm(world_estimates - positions[:, :2], axis=1)
            self._save(folder, {"position_errors_m": localization})
            count = min(2000, len(localization))
            comparison_localization = localization[:count]
            metrics.update(
                localization_mae_m=float(comparison_localization.mean()),
                localization_rmse_m=float(np.sqrt(np.mean(comparison_localization ** 2))),
                localization_max_error_m=float(comparison_localization.max()),
                localization_rmse_full_run_m=float(np.sqrt(np.mean(localization ** 2))),
                localization_max_error_full_run_m=float(localization.max()),
            )
            from plot_data_cassini import calculate_geometric_path_errors
            geometric = calculate_geometric_path_errors(positions)
            self._save(folder, {"geometric_errors_m": geometric})
            start_frame = 500 if len(geometric) > 500 else 0
            end_frame = min(2000, len(geometric))
            comparison = geometric[start_frame:end_frame]
            settled = geometric[start_frame:]
            metrics.update(
                geometric_tracking_start_index=start_frame,
                geometric_tracking_end_index_exclusive=end_frame,
                geometric_tracking_frames=len(comparison),
                geometric_tracking_rmse_m=float(np.sqrt(np.mean(comparison ** 2))),
                geometric_tracking_max_error_m=float(comparison.max()),
                geometric_tracking_rmse_full_run_m=float(np.sqrt(np.mean(settled ** 2))),
                geometric_tracking_max_error_full_run_m=float(settled.max()),
                geometric_tracking_rmse_all_m=float(np.sqrt(np.mean(geometric ** 2))),
                geometric_tracking_max_error_all_m=float(geometric.max()),
            )
        warmup, timing_end = (20 if len(inference) > 20 else 0), min(2000, len(inference))
        if timing_end > warmup:
            metrics.update(
                timing_warmup_frames=warmup,
                cnn_inference_mean_ms=float(inference[warmup:timing_end].mean()),
                cnn_inference_p95_ms=float(np.percentile(inference[warmup:timing_end], 95)),
                position_gvf_mean_ms=float(pipeline[warmup:timing_end].mean()),
            )
        with open(os.path.join(folder, "position_gvf_metrics.json"), "w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)
        print(f"VisLoc-GVF metrics: {metrics}")


TraditionalVectorFieldControl = TraditionalVectorFieldControlJSBSim
GenerativeVisualControl = GenerativeVisualControlJSBSim
