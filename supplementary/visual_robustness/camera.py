from supplementary.visual_robustness.common import PanoramaMixin
from src.cnn_visual_control import CNNVisualControl


class Camera(PanoramaMixin, CNNVisualControl):
    def __init__(self):
        self.img_index, self.p_x, self.p_y = self._load_panorama_mapping()
        self.unity_bridge = self._initialize_unity_bridge(self._get_default_camera_config())

    def close(self):
        self.unity_bridge.close()
