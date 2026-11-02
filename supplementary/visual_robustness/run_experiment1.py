import json
import random
import time

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import torch

from supplementary.visual_robustness.common import (
    BASE_SEED, CONDITIONS, CORRUPTION_SPEC, DT, INIT_POSE, MAX_FRAMES, METHODS,
    MODEL_PATHS, REPETITIONS, ROOT, RUN_DIR, TAU_THETA, VELOCITY_MPS,
    TRACKING_LIMIT_M, make_corrupted_controller, model_fingerprints, write_json,
)
from src.cnn_visual_control import CNNVisualControl
from src.diffusion_policy_control import DiffusionPolicyVisualControl
from src.generative_visual_control import GenerativeVisualControl
from plot_data_cassini import build_cassini_zero_contour, calculate_geometric_path_errors
from src.position_gvf_control import PositionGVFVisualControl

CONTROLLERS = dict(zip(METHODS, (GenerativeVisualControl, CNNVisualControl,
                               PositionGVFVisualControl, DiffusionPolicyVisualControl)))


def config_for(condition, repetition, folder):
    return {
        "path_type": "cassini", "noise_type": None,
        "model_path": str(ROOT / MODEL_PATHS["gvc"]),
        "cnn_model_path": str(ROOT / MODEL_PATHS["cnn_based"]),
        "position_model_path": str(ROOT / MODEL_PATHS["cnn_localization_gvf"]),
        "dp_model_path": str(ROOT / MODEL_PATHS["diffusion_policy"]),
        "init_pose": INIT_POSE, "v0": VELOCITY_MPS, "dt": DT, "tau_theta": TAU_THETA,
        "seq_length": 1, "pred_horizon": 2, "num_inference_steps": 20, "num_steps": 50,
        "heading_gain": 1.0, "gradient_gain": 1.0, "gradient_clip": None,
        "dp_max_turn_rate_deg": 90.0, "dp_deterministic": False, "mode_persistence": False,
        "visual_condition": condition, "visual_repetition": repetition,
        "mode_seed": BASE_SEED + repetition, "save_folder": str(folder),
    }


def score(folder, contour):
    path = folder / "poses.npy"
    poses = np.load(path) if path.exists() else np.empty((0, 3))
    commands_path = folder / "heading_commands.npy"
    commands = np.load(commands_path) if commands_path.exists() else np.array([])
    finite = np.isfinite(poses).all() and np.isfinite(commands).all()
    errors = np.concatenate([calculate_geometric_path_errors(poses[i:i + 256], contour)
                             for i in range(0, len(poses), 256)]) if len(poses) and finite else np.array([])
    np.save(folder / "geometric_errors_m.npy", errors)
    metrics = {"frames": len(poses), "completed": bool(
        finite and len(poses) == len(commands) == len(errors) == MAX_FRAMES)}
    values = errors[500:2000]
    metrics["tracking_success"] = bool(metrics["completed"] and len(errors) > 500
                                       and values.max() <= TRACKING_LIMIT_M)
    metrics.update(rmse_m=float(np.sqrt(np.mean(values ** 2))) if len(values) else None,
                   max_error_m=float(values.max()) if len(values) else None)
    return metrics


def save_manifest():
    rows = []
    for method in METHODS:
        for condition in CONDITIONS:
            for repetition in range(REPETITIONS):
                folder = RUN_DIR / method / condition / f"rep_{repetition:02d}"
                path = folder / "trial_status.json"
                row = {"status": "pending", "rmse_m": None, "max_error_m": None}
                if path.exists():
                    row.update(json.loads(path.read_text(encoding="utf-8")))
                rows.append({**row, "method": method, "condition": condition,
                             "repetition": repetition, "folder": str(folder)})
    pd.DataFrame(rows).to_csv(RUN_DIR / "manifest.csv", index=False)


def run_trial(method, condition, repetition, folder, contour):
    folder.mkdir(parents=True, exist_ok=False)
    cfg = config_for(condition, repetition, folder)
    write_json(folder / "run_config.json", cfg)
    status = {"status": "interrupted", "completed": False, "frames": 0,
              "policy_seed": BASE_SEED + repetition, "failure_reason": "interrupted before completion"}
    write_json(folder / "trial_status.json", status)
    controller, stop, start = None, False, time.perf_counter()
    try:
        controller_type = make_corrupted_controller(CONTROLLERS[method])
        controller = controller_type.__new__(controller_type)
        controller.__init__(cfg)
        seed = BASE_SEED + repetition
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        controller.control_loop(max_frames=MAX_FRAMES, save_data=True)
        status.update(score(folder, contour))
        status.update(status="completed" if status["completed"] else "failed",
                      failure_reason=None if status["completed"] else "incomplete/non-finite arrays")
        stop = not status["completed"]
    except KeyboardInterrupt:
        stop = True
    except Exception as exc:
        status.update(status="failed", failure_reason=f"{type(exc).__name__}: {exc}")
        status.update(score(folder, contour))
        status["completed"] = False
        status["tracking_success"] = False
    finally:
        if controller is not None:
            np.save(folder / "predicted_vectors.npy", np.asarray(controller.prediction_log))
            bridge, sender = getattr(controller, "unity_bridge", None), getattr(controller, "data_sender", None)
            if bridge is not None:
                bridge.close()
            if sender is not None and not sender.zmq_context.closed:
                sender.close()
        status["elapsed_seconds"] = time.perf_counter() - start
        write_json(folder / "trial_status.json", status)
    return status, stop


def main():
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    protocol = {"config": config_for("clean", 0, ""), "max_frames": MAX_FRAMES,
                "base_seed": BASE_SEED, "conditions": list(CONDITIONS), "methods": list(METHODS),
                "corruption": CORRUPTION_SPEC, "models": model_fingerprints(),
                "completion": "all requested frames and aligned finite positions/commands",
                "tracking_success": f"complete run and every geometric error on [500:2000] <= {TRACKING_LIMIT_M} m",
                "metric_window": [500, 2000], "representative_repetition": 0}
    path = RUN_DIR / "protocol.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != protocol:
        raise ValueError("Protocol changed: choose a new RUN_TAG in common.py")
    write_json(path, protocol)
    write_json(RUN_DIR / "environment.json", {"torch": str(torch.__version__),
               "numpy": np.__version__, "device": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu"})
    contour = build_cassini_zero_contour()
    save_manifest()
    for method in METHODS:
        for condition in CONDITIONS:
            for repetition in range(REPETITIONS):
                folder = RUN_DIR / method / condition / f"rep_{repetition:02d}"
                status_path = folder / "trial_status.json"
                if status_path.exists():
                    continue
                status, stop = run_trial(method, condition, repetition, folder, contour)
                save_manifest()
                print(f"{method} / {condition} / rep_{repetition:02d}: {status['status']}")
                if stop:
                    return
    print(f"Results: {RUN_DIR}")


if __name__ == "__main__":
    main()
