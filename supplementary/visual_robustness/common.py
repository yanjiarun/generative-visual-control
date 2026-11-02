import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "data/visual_robustness"
RUN_TAG = "run01"
RUN_DIR = RESULTS / RUN_TAG
RETRY_DIR = RUN_DIR / "unity_timeout_retry_01"
OFFLINE_DIR = RESULTS / "offline_heldout"
REPORT_DIR = RUN_DIR / "reports"
CONDITIONS = ("clean", "low_illumination", "gaussian_noise", "motion_blur", "partial_occlusion")
METHODS = ("gvc", "cnn_based", "cnn_localization_gvf", "diffusion_policy")
LABELS = {
    "gvc": "Proposed GVC",
    "cnn_based": "CNN-based GVF Regression",
    "cnn_localization_gvf": "VisLoc-GVF",
    "diffusion_policy": "Diffusion Policy",
}
COLORS = {"gvc": "darkred", "cnn_based": "darkblue",
          "cnn_localization_gvf": "purple", "diffusion_policy": "darkorange"}

INIT_POSE = [576.0, 275.0, np.pi / 3.0]
VELOCITY_MPS, DT, TAU_THETA = 23.0, 0.05, 6.0
MAX_FRAMES, REPETITIONS = 2000, 5
TRACKING_LIMIT_M = 50.0  # Maximum error on frames [500:2000].
OFFLINE_COUNT, BATCH_SIZE = 300, 16
BASE_SEED = 20261025
BLUR_KERNEL = 7  # Horizontal pixels.
OCCLUSION_FRACTION = 0.10
CORRUPTION_SPEC = {"stage": "rectified panorama before resize/crop",
                   "illumination_factor": 0.8, "gaussian_sigma_0_1": 0.02,
                   "blur_horizontal_pixels": BLUR_KERNEL,
                   "occlusion_fraction": OCCLUSION_FRACTION,
                   "occlusion_fill": 0, "occlusion_position": "per-frame paired RNG"}

MODEL_PATHS = {
    "gvc": "src/generative_model/models/flow_matching_cassini_models/flow_matching_policy_final.pth",
    "cnn_based": "src/baseline/models/cnn_cassini_models/cnn_final.pth",
    "cnn_localization_gvf": "src/baseline/models/position_cassini_cnn/position_best.pth",
    "diffusion_policy": "src/generative_model/models/diffusion_policy_cassini_models/diffusion_policy_final.pth",
}


def corrupt_image(image, condition, seed):
    if condition == "clean":
        return image
    rng = np.random.default_rng(seed)
    if condition == "low_illumination":
        return np.clip(image.astype(np.float32) * 0.8, 0, 255).astype(np.uint8)
    if condition == "gaussian_noise":
        noise = rng.normal(0.0, 0.02 * 255.0, image.shape)
        return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if condition == "motion_blur":
        kernel = np.ones((1, BLUR_KERNEL), np.float32) / BLUR_KERNEL
        return cv2.filter2D(image, -1, kernel)
    if condition == "partial_occlusion":
        h, w = image.shape[:2]
        box_w = max(1, round(w * np.sqrt(OCCLUSION_FRACTION)))
        box_h = max(1, round(h * OCCLUSION_FRACTION * w / box_w))
        x, y = int(rng.integers(w - box_w + 1)), int(rng.integers(h - box_h + 1))
        output = image.copy()
        output[y:y + box_h, x:x + box_w] = 0
        return output
    raise ValueError(f"Unknown visual condition: {condition}")


def corruption_seed(repetition, frame_id, condition):
    return int(BASE_SEED + repetition * 1_000_000 + frame_id * 10 + CONDITIONS.index(condition))


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def model_fingerprints():
    return {method: {"path": path, "sha256": hashlib.sha256((ROOT / path).read_bytes()).hexdigest()}
            for method, path in MODEL_PATHS.items()}


class PanoramaMixin:
    def _initialize_unity_bridge(self, camera_config):
        from supplementary.visual_robustness.bridge import PanoramaBridge
        return PanoramaBridge(camera_config)

    def rectify_panorama(self, panorama, color_mode=True):
        if not hasattr(self, "_rectify_maps"):
            radius = np.arange(242, 512)[:, None]
            theta = np.linspace(-np.pi, np.pi, 1800, endpoint=False)[None]
            self._rectify_maps = tuple((radius * f(theta) + 512).astype(np.int32).astype(np.float32)
                                       for f in (np.cos, np.sin))
        return cv2.remap(cv2.rotate(panorama, cv2.ROTATE_90_COUNTERCLOCKWISE),
                         *self._rectify_maps, cv2.INTER_NEAREST)

    def get_panoramic_image(self, frame_id, current_position):
        from AFPV_Kit.common.transfer import (
            convert_position_to_unity, convert_quaternion_to_unity, convert_ned_quaternion_to_enu)
        if not np.isfinite(current_position).all():
            raise ValueError("Non-finite simulation position")
# Avoid stale frames from earlier trials.
        if not hasattr(self, "_wire_offset"):
            self._wire_offset = int(time.time_ns() // 1_000_000 % 1_000_000_000)
        wire_id = self._wire_offset + frame_id
        pose = dict(frame_id=wire_id, vehicle_id="FPV",
                    position=convert_position_to_unity([*current_position[:2], 75.0]),
                    rotation=convert_quaternion_to_unity(convert_ned_quaternion_to_enu([0, 0, 0, 1])),
                    cameras=self._get_default_camera_config())
        images = self.unity_bridge.capture(pose)
        return self.rectify_panorama(self._create_color_panorama(images))


def make_corrupted_controller(controller_class):
    class CorruptedImageMixin(PanoramaMixin):
        def __init__(self, config):
            self.visual_condition = config["visual_condition"]
            self.visual_repetition = int(config["visual_repetition"])
            self.prediction_log = []
            super().__init__(config)

        def get_panoramic_image(self, frame_id, current_position):
            image = super().get_panoramic_image(frame_id, current_position)
            return corrupt_image(
                image, self.visual_condition,
                corruption_seed(self.visual_repetition, frame_id, self.visual_condition),
            )

        def generate_vector_field(self, image_tensor, _current_position=None):
# Policies receive images only.
            output = np.asarray(super().generate_vector_field(image_tensor, None), dtype=float)
            if output.shape != (4,) or not np.isfinite(output).all():
                raise ValueError("Invalid network/GVF prediction")
            self.prediction_log.append(output.copy())
            return output

        def calculate_heading_cmd(self, *args, **kwargs):
            command = super().calculate_heading_cmd(*args, **kwargs)
            if not np.isfinite(command):
                raise ValueError("Non-finite heading command")
            return command

    return type(f"Corrupted{controller_class.__name__}",
                (CorruptedImageMixin, controller_class), {})
