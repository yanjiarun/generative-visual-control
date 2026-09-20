import numpy as np
from src.generative_visual_control import GenerativeVisualControl
from src.traditional_vector_field_control import TraditionalVectorFieldControl
from src.cnn_visual_control import CNNVisualControl
from src.diffusion_policy_control import DiffusionPolicyVisualControl

if __name__ == "__main__":
    CONTROL_METHOD = "dp"
    PATH_TYPE = "cassini"  # "quartic" or "cassini"

    config = {
        "path_type": PATH_TYPE,
        "model_path": f"./src/generative_model/models/flow_matching_{PATH_TYPE}_models/flow_matching_policy_final.pth",
        "cnn_model_path": f"./src/baseline/models/cnn_{PATH_TYPE}_models/cnn_final.pth",
        "dp_model_path": f"./src/generative_model/models/diffusion_policy_{PATH_TYPE}_models/diffusion_policy_final.pth",
        "init_pose": [576, 275, np.pi/3],
        "v0": 23.0,
        "dt": 0.05,
        "tau_theta": 6.0,
        "seq_length": 1,
        "pred_horizon": 2,
        "num_inference_steps": 20,
        "heading_gain": 1.0,
        "dp_max_turn_rate_deg": 90.0,
        "save_folder": f"./data/{PATH_TYPE}_path_following_data/{CONTROL_METHOD}",
    }
    
    try:
        if CONTROL_METHOD == "generative":
            controller = GenerativeVisualControl(config)
            print("GenerativeVisualControl initialized successfully")
        elif CONTROL_METHOD == "traditional":
            controller = TraditionalVectorFieldControl(config)
            print("TraditionalVectorFieldControl initialized successfully")
        elif CONTROL_METHOD == "cnn":
            controller = CNNVisualControl(config)
            print("CNNVisualControl initialized successfully")
        elif CONTROL_METHOD == "dp":
            controller = DiffusionPolicyVisualControl(config)
            print("DiffusionPolicyVisualControl initialized successfully")
        else:
            raise ValueError(f"Unknown control method: {CONTROL_METHOD}")
        
        controller.control_loop(
            max_frames=4000,
            save_data=True
        )
        
    except Exception as e:
        print(f"Error: {e}")
