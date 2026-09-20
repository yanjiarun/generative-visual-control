import math

import numpy as np
import torch

from src.baseline.cnn_model import BaselineCNN
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


TraditionalVectorFieldControl = TraditionalVectorFieldControlJSBSim
GenerativeVisualControl = GenerativeVisualControlJSBSim
