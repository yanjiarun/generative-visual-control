import math
from collections import deque

import numpy as np
import torch

from src.cnn_visual_control import CNNVisualControl
from src.generative_model.core.diffusion_policy import DiffusionPolicy
from src.path_following.quartic_mode_memory import QuarticModeMemory


class DiffusionPolicyVisualControl(CNNVisualControl):
    """Reuse the panorama/kinematic loop, replacing GVF by DDPM."""

    def __init__(self, config):
        self.seq_length = int(config.get("seq_length", 1))
        self.pred_horizon = int(config.get("pred_horizon", 2))
        self.inference_steps = int(config.get("num_inference_steps", 20))
        self.heading_gain = float(config.get("heading_gain", 0.18))
        self.dp_deterministic = bool(config.get("dp_deterministic", False))
        self.max_turn_rate = math.radians(config.get("dp_max_turn_rate_deg", 15.0))
        config = dict(config)
        config.setdefault("cnn_model_path", config.get("dp_model_path"))
        super().__init__(config)
        self.image_history = deque(maxlen=self.seq_length)
        self.mode_memory = QuarticModeMemory(config)
        self.mode_latent = None
        seed = config.get("mode_seed")
        self.mode_generator = None if seed is None else torch.Generator(
            device=self.device).manual_seed(int(seed))

    def _load_cnn_model(self, _unused_path):
        model = DiffusionPolicy(
            history_length=self.seq_length, pred_horizon=self.pred_horizon,
            inference_steps=self.inference_steps).to(self.device)
        path = self.config.get("dp_model_path")
        if not path:
            raise ValueError("dp_model_path is required for CONTROL_METHOD='dp'")
        checkpoint = torch.load(path, map_location=self.device)
        model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
        model.eval()
        return model

    def generate_vector_field(self, image_sequence, current_position=None):
        position = current_position if current_position is not None else (0.0, 0.0)
        if self.mode_memory.reset_if_finished(position):
            self.mode_latent = None
        count = self.mode_memory.candidate_count(position)
        if self.mode_memory.locked and self.mode_latent is not None:
            latent = self.mode_latent
        else:
            latent = torch.randn(
                count, self.pred_horizon, self.cnn_model.action_dim,
                device=self.device, generator=self.mode_generator)
        images = image_sequence.expand(latent.shape[0], *image_sequence.shape[1:])
        with torch.no_grad():
            actions = self.cnn_model.predict_action(
                images, self.inference_steps, initial_noise=latent,
                deterministic=self.dp_deterministic)[:, 0]
        index = self.mode_memory.select(actions.cpu().numpy(), position)
        if self.mode_memory.locked and self.mode_latent is None:
            self.mode_latent = latent[index:index + 1].clone()
            mode = self.mode_memory.mode
            print("Locked DP obstacle latent" +
                  (f" (mode {mode:+d})" if mode is not None else ""))
        action = actions[index]
        return np.array([action[0].item(), action[1].item(), 0.0, 0.0],
                        dtype=np.float32)

    def calculate_heading_cmd(self, vector, _delta_mix, current_heading,
                              _current_velocity, tao_heading=28.0, **_kwargs):
        desired = math.atan2(float(vector[1]), float(vector[0]))
        error = math.atan2(math.sin(desired - current_heading),
                           math.cos(desired - current_heading))
        turn_rate = np.clip(self.heading_gain * error,
                            -self.max_turn_rate, self.max_turn_rate)
        return current_heading + tao_heading * turn_rate
