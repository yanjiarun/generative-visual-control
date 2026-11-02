import math

from src.jsbsim_config import X8_AUTOPILOT_PID

from src.jsbsim_controllers import (
    CNNVisualControlJSBSim,
    DiffusionPolicyVisualControlJSBSim,
    GenerativeVisualControlJSBSim,
    PositionGVFVisualControlJSBSim,
    TraditionalVectorFieldControlJSBSim,
)


# Shared guidance parameters keep the traditional, CNN, and GVC comparison
# independent of method-specific heading-controller tuning.
SHARED_HEADING_PARAMS = {
    "heading_lookahead": 3.5,
    "heading_gain": 1.0,
    "max_turn_rate_deg": 30.0,
    "max_bank_deg": 60.0,
}

METHOD_HEADING_PARAMS = {
    "traditional": SHARED_HEADING_PARAMS,
    "cnn": SHARED_HEADING_PARAMS,
    "generative": SHARED_HEADING_PARAMS,
    "position_gvf": SHARED_HEADING_PARAMS,
    # DP uses a separate heading gain because it directly controls heading rate.
    "dp": {
        "heading_lookahead": 6.0,
        "heading_gain": 0.2,
        "max_turn_rate_deg": 25.0,
        "max_bank_deg": 55.0,
    },
}


if __name__ == "__main__":
    PATH_TYPE = "cassini"
    base_config = {
        "path_type": PATH_TYPE,
        "model_path": f"./src/generative_model/models/flow_matching_{PATH_TYPE}_models/flow_matching_policy_final.pth",
        "cnn_model_path": f"./src/baseline/models/cnn_{PATH_TYPE}_models/cnn_final.pth",
        "position_model_path": "./src/baseline/models/position_cassini_cnn/position_best.pth",
        "autopilot_pid": X8_AUTOPILOT_PID,
        "dp_model_path": f"./src/generative_model/models/diffusion_policy_{PATH_TYPE}_models/diffusion_policy_final.pth",
        "init_pose": [576.0, 275.0, math.pi / 3.0],
        "target_altitude": 75.0,
        "target_speed": 23.0,
        "visual_altitude": 75.0,
        "control_dt": 0.05,
        "jsbsim_dt": 0.001,
        "seq_length": 1,
        "num_steps": 50,
        "pred_horizon": 2,
        "num_inference_steps": 20,
        "world_aligned_panorama": True,
        "visualize": True,
        "realtime": False,
    }

    controllers = [
        ("traditional", TraditionalVectorFieldControlJSBSim),
        ("cnn", CNNVisualControlJSBSim),
        ("dp", DiffusionPolicyVisualControlJSBSim),
        ("generative", GenerativeVisualControlJSBSim),
        ("position_gvf", PositionGVFVisualControlJSBSim),
    ]
    for method, controller_class in controllers:
        config = {**base_config, **METHOD_HEADING_PARAMS[method], "save_folder":
                  f"./jsbsim_data/{PATH_TYPE}_path_following_data/{method}"}
        print(f"\n=== {method} ===")
        controller = controller_class(config)
        print(f"{controller.__class__.__name__} initialized")
        controller.control_loop(max_frames=4000, save_data=True)
        del controller
