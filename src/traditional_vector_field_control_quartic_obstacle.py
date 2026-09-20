import numpy as np

from src.traditional_vector_field_control import TraditionalVectorFieldControl
from src.path_following.create_path_quartic_obstacle import compute_quartic_obstacle_field


class QuarticObstacleVectorFieldControl(TraditionalVectorFieldControl):
    """The existing kinematic loop with an explicit upper/lower route mode."""

    def __init__(self, config):
        config = dict(config)
        config["path_type"] = "quartic"  # retain the repository's 200 m scaling
        super().__init__(config)
        self.route_mode = int(config.get("route_mode", 1))

    def generate_vector_field(self, uav_position):
        p = np.asarray(uav_position, dtype=float)
        x, y = (p[:2] - np.array([576.0, 275.0])) / 200.0
        ux, uy, _, grad_ux, grad_uy = compute_quartic_obstacle_field(
            x, y, mode=self.route_mode)
        return np.array([float(ux), float(uy)]), grad_ux, grad_uy
