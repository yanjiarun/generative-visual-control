import argparse

import numpy as np

from src.path_following.create_path_quartic_obstacle import route
from src.traditional_vector_field_control_quartic_obstacle import QuarticObstacleVectorFieldControl


def make_config(method, init_pose, run_id, route_mode=1, ode_steps=50,
                mode_hint=None, mode_samples=8, dp_heading_gain=2.0,
                gvc_gradient_gain=1.0, gvc_gradient_clip=None):
    root = "./src/generative_model/models"
    save_name = (f"{method}_hint{mode_hint}" if
                 method in ("generative", "dp") and mode_hint else
                 f"{method}_mode{route_mode}")
    if method == "generative" and (gvc_gradient_gain != 1.0 or
                                    gvc_gradient_clip is not None):
        clip = "none" if gvc_gradient_clip is None else f"{gvc_gradient_clip:g}"
        save_name += f"_grad{gvc_gradient_gain:g}_clip{clip}"
    cfg = {
        "path_type": "quartic",
        "model_path": f"{root}/flow_matching_quartic_obstacle_models/flow_matching_policy_final.pth",
        "cnn_model_path": "./src/baseline/models/cnn_quartic_obstacle_models/cnn_final.pth",
        "dp_model_path": f"{root}/diffusion_policy_quartic_obstacle_models/diffusion_policy_final.pth",
        "init_pose": np.asarray(init_pose, dtype=float), "v0": 23.0, "dt": 0.05,
        "tau_theta": 6.0, "seq_length": 1,
        "pred_horizon": 2, "num_inference_steps": 20, "num_steps": ode_steps,
        "heading_gain": dp_heading_gain, "dp_max_turn_rate_deg": 90.0,
        "gradient_gain": gvc_gradient_gain,
        "gradient_clip": gvc_gradient_clip,
        "dp_deterministic": method == "dp",
        "mode_persistence": method in ("generative", "dp"),
        "mode_samples": mode_samples, "mode_lock_x": -0.44,
        "mode_hint": mode_hint, "mode_seed": run_id,
        "save_folder": f"./data_quartic_obstacle/{save_name}/run_{run_id}",
        "route_mode": route_mode,
    }
    return cfg


def run(method, cfg, max_frames):
    if method == "traditional":
        controller = QuarticObstacleVectorFieldControl(cfg)
    elif method == "generative":
        from src.generative_visual_control import GenerativeVisualControl
        controller = GenerativeVisualControl(cfg)
    elif method == "cnn":
        from src.cnn_visual_control import CNNVisualControl
        controller = CNNVisualControl(cfg)
    elif method == "dp":
        from src.diffusion_policy_control import DiffusionPolicyVisualControl
        controller = DiffusionPolicyVisualControl(cfg)
    else:
        raise ValueError(f"unknown method: {method}")
    controller.control_loop(max_frames=max_frames, save_data=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--methods", nargs="+", default=["traditional", "generative", "cnn", "dp"])
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--max-frames", type=int, default=1800)
    p.add_argument("--route-mode", type=int, choices=(-1, 1), default=1)
    p.add_argument("--ode-steps", type=int, default=50)
    p.add_argument("--mode-hint", type=int, choices=(-1, 1))
    p.add_argument("--mode-samples", type=int, default=8)
    p.add_argument("--dp-heading-gain", type=float, default=2.0)
    p.add_argument("--gvc-gradient-gain", type=float, default=1.0)
    p.add_argument("--gvc-gradient-clip", type=float)
    a = p.parse_args()

    x0 = -0.56
    f, fp = route(x0, 0)
    init = [576.0 + 200.0 * x0, 275.0 + 200.0 * float(f), np.arctan(float(fp))]
    for method in a.methods:
        for run_id in range(a.runs):
            print(f"\n=== quartic obstacle: {method}, run {run_id} ===")
            try:
                cfg = make_config(method, init, run_id, a.route_mode,
                                  a.ode_steps, a.mode_hint, a.mode_samples,
                                  a.dp_heading_gain, a.gvc_gradient_gain,
                                  a.gvc_gradient_clip)
                run(method, cfg, a.max_frames)
            except Exception as exc:
                print(f"{method} failed: {exc}")
