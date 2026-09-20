from AFPV_Kit.common.state import PoseState
from AFPV_Kit.common.transfer import convert_ned_quaternion_to_enu
from src.utils.collect_imgs import ImageCollector
from src.path_following.create_path_quartic import compute_quartic_field, plot_quartic_field, compute_quartic_field_with_gradients
from src.path_following.create_path_cassini_oval import compute_cassini_field, plot_cassini_field, compute_cassini_field_with_gradients
import numpy as np
import pandas as pd
import os


PATH_CONFIGS = {
    "quartic": {"x_range": (-1.2, 1.2), "y_range": (-0.8, 0.8), "n_points": 60, "scale_factor": 200},
    "cassini": {"x_range": (-2.8, 2.8), "y_range": (-1.4, 1.4), "n_points": 70, "scale_factor": 100},
}

compute_path_field = lambda path_type, X, Y, **kwargs: {
    "quartic": compute_quartic_field, "cassini": compute_cassini_field
}[path_type](X, Y, **kwargs)

plot_path_field = lambda path_type, X, Y, u_x, u_y, alpha, **kwargs: {
    "quartic": plot_quartic_field, "cassini": plot_cassini_field
}[path_type](X, Y, u_x, u_y, alpha, **kwargs)

compute_path_field_with_gradients = lambda path_type, X, Y, **kwargs: {
    "quartic": compute_quartic_field_with_gradients, "cassini": compute_cassini_field_with_gradients
}[path_type](X, Y, **kwargs)

if __name__ == "__main__":
    # Configuration
    PATH_TYPE = "cassini"  # "quartic" or "cassini"
    
    image_collector = ImageCollector(f"./panorama_{PATH_TYPE}", debug=True)
    config = PATH_CONFIGS[PATH_TYPE]

    # Create vector field grid
    x = np.linspace(config["x_range"][0], config["x_range"][1], config["n_points"]+1)
    y = np.linspace(config["y_range"][0], config["y_range"][1], int(config["n_points"] * (config["y_range"][1]-config["y_range"][0])/(config["x_range"][1]-config["x_range"][0]))+1)
    X, Y = np.meshgrid(x, y)
    
    # Compute vector field for visualization
    u_x, u_y, alpha, grad_u_x, grad_u_y = compute_path_field_with_gradients(PATH_TYPE, X, Y)
    plot_path_field(PATH_TYPE, X, Y, u_x, u_y, alpha, sample_rate=1, scale=50)
    
    # Save vector field data to CSV
    csv_file_path = f"./vector_field_with_gradients_{PATH_TYPE}.csv"
    os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)
    
    frame_ids = np.arange(1, X.size + 1)
    pd.DataFrame({'frame_id': frame_ids, 'x': X.flatten(), 'y': Y.flatten(), 'ux': u_x.flatten(), 'uy': u_y.flatten(), 
                  'grad_ux_dx': grad_u_x[0].flatten(), 'grad_ux_dy': grad_u_x[1].flatten(),
                  'grad_uy_dx': grad_u_y[0].flatten(), 'grad_uy_dy': grad_u_y[1].flatten()}).to_csv(csv_file_path, index=False)
    print(f"Vector field data saved to {csv_file_path}")

    # Collect panorama images at each grid position
    # frame_id = 0
    # base_x, base_y = 576, 275  # Base coordinates in Unity
    # quaternion_enu = convert_ned_quaternion_to_enu([0, 0, 0, 1])
    # total_points = X.shape[0] * X.shape[1]
    # scale_factor = config["scale_factor"]
    
    # for i in range(X.shape[0]):
    #     for j in range(X.shape[1]):
    #         frame_id += 1

    #         # Convert grid coordinates to Unity world coordinates
    #         # Scale factor to convert from [-1.2,1.2] to appropriate Unity scale
    #         unity_x = base_x + X[i,j] * scale_factor  # Adjust scale factor as needed
    #         unity_y = base_y + Y[i,j] * scale_factor
            
    #         current_pose = PoseState([unity_x, unity_y, 75], quaternion_enu)  # Keep z=75
            
    #         print(f"[{frame_id}/{total_points}] Collecting panorama at position ({X[i,j]:.2f}, {Y[i,j]:.2f}) -> Unity ({unity_x:.1f}, {unity_y:.1f})")
    #         image_collector.get_panorama(frame_id, current_pose)
    
    # print(f"Total {frame_id} panorama images collected")
