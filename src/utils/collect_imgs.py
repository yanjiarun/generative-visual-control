import os, time
import cv2
import numpy as np
from AFPV_Kit.bridges.unity_bridge import UnityBridge
from AFPV_Kit.common.state import PoseState
from AFPV_Kit.common.transfer import convert_position_to_unity, convert_quaternion_to_unity, calculate_camera_matrix

class ImageCollector:
    # Default camera configuration
    BASE_CONFIG = {
        "width": 1024, "height": 1024, "channels": 3, "fov": 90.0,
        "nearClipPlane": [0.01, 0.5, 0.5], "farClipPlane": [1000.0, 550.0, 1000.0],
        "enabledLayers": [False, False]
    }
    
    CAMERA_CONFIG = [
        {"ID": "Camera_Front", "T_BC": calculate_camera_matrix(x_offset=0., y_offset=-0.20)},
        {"ID": "Camera_Back", "T_BC": calculate_camera_matrix(yaw_deg=180, x_offset=0., y_offset=-0.20)},
        {"ID": "Camera_Left", "T_BC": calculate_camera_matrix(yaw_deg=-90, x_offset=0., y_offset=-0.20)},
        {"ID": "Camera_Right", "T_BC": calculate_camera_matrix(yaw_deg=90, x_offset=0., y_offset=-0.20)},
        {"ID": "Camera_Top", "T_BC": calculate_camera_matrix(pitch_deg=90, x_offset=0., y_offset=-0.20)},
        {"ID": "Camera_Bottom", "T_BC": calculate_camera_matrix(pitch_deg=-90, x_offset=0., y_offset=-0.20)}
    ]
    
    def __init__(self, save_folder, camera_config=None, debug=False):
        self.save_folder, self.debug = save_folder, debug
        self.camera_config = camera_config or self._get_default_camera_config()
        os.makedirs(self.save_folder, exist_ok=True)
        self.initialize_unity_bridge()
        self.img_index, self.p_x, self.p_y = self.load_panorama_mapping()
    
    def _get_default_camera_config(self):
        """Generate default camera config with base config applied"""
        return [{**self.BASE_CONFIG, **cam} for cam in self.CAMERA_CONFIG]

    def initialize_unity_bridge(self):
        vehicles_info = {"vehicles": [{
            "ID": "FPV", "position": [0.0, 0.0, 0.0], "rotation": [0.0, 0, 0, 1],
            "scale": [0.5, 0.5, 0.5], "cameras": self.camera_config
        }]}
        self.unity_bridge = UnityBridge(vehicles_info=vehicles_info)
        self.unity_bridge.start()

    def load_panorama_mapping(self):
        class_dir = os.path.dirname(os.path.abspath(__file__))
        mapping_folder = os.path.join(class_dir, "required_mats")
        return tuple(np.load(os.path.join(mapping_folder, name)) for name in
                     ("img_index.npy", "p_x_RGB.npy", "p_y_RGB.npy"))

    def create_panorama(self, images):
        h, w = 1024, 1024
        panorama = np.zeros((h,w), dtype=np.uint8)
        img_index_flat = self.img_index.flatten().astype(int)
        p_x_flat = self.p_x.flatten().astype(int)
        p_y_flat = self.p_y.flatten().astype(int)

        for cam_idx in range(6):
            mask = (img_index_flat == cam_idx)
            if np.any(mask):
                x_coords = p_x_flat[mask]
                y_coords = p_y_flat[mask]
                valid_mask = (x_coords >= 0) & (x_coords < images[cam_idx].shape[1]) & \
                           (y_coords >= 0) & (y_coords < images[cam_idx].shape[0])
                if np.any(valid_mask):
                    valid_indices = np.where(mask)[0][valid_mask]
                    panorama_flat = panorama.flatten()
                    panorama_flat[valid_indices] = images[cam_idx][y_coords[valid_mask], x_coords[valid_mask]]
                    panorama = panorama_flat.reshape((h, w))
        return panorama

    def create_color_panorama(self, images):
        h, w = 1024, 1024
        panorama = np.zeros((h, w, 3), dtype=np.uint8)
        img_index_flat = self.img_index.flatten().astype(int)
        p_x_flat = self.p_x.flatten().astype(int)
        p_y_flat = self.p_y.flatten().astype(int)

        for cam_idx in range(6):
            mask = (img_index_flat == cam_idx)
            if np.any(mask):
                x_coords = p_x_flat[mask]
                y_coords = p_y_flat[mask]
                valid_mask = (x_coords >= 0) & (x_coords < images[cam_idx].shape[1]) & \
                           (y_coords >= 0) & (y_coords < images[cam_idx].shape[0])
                if np.any(valid_mask):
                    valid_indices = np.where(mask)[0][valid_mask]
                    rgb_pixels = images[cam_idx][y_coords[valid_mask], x_coords[valid_mask], :]
                    panorama.reshape(-1, 3)[valid_indices] = rgb_pixels
        return panorama

    def get_panorama(self, frame_id, pose: PoseState, save_color=True):
        current_pos = pose.get_position()
        current_quaternion = pose.get_quaternion()
        
        position_unity = convert_position_to_unity(current_pos)
        quaternion_unity = convert_quaternion_to_unity(current_quaternion)
        unity_pose_kwargs = {
            'frame_id': frame_id, 'vehicle_id': 'FPV', 'position': position_unity,
            'rotation': quaternion_unity, 'scale': [0.2, 0.2, 0.2], 'cameras': self.camera_config
        }
        
        if frame_id == 1:
            for _ in range(100):
                self.unity_bridge.send_pose(**unity_pose_kwargs)
                time.sleep(0.01)
            self.debug and print(f"Send frame_id: {frame_id}")
        
        while True:
            self.unity_bridge.send_pose(**unity_pose_kwargs)
            time.sleep(0.1)
            img_data_list = [self.unity_bridge.get_latest_image(0, i) for i in range(6)]
            if all(img is not None and img["frame_id"] == frame_id for img in img_data_list):
                gray_images, color_images = [], []
                for img_data in img_data_list:
                    color_img = cv2.cvtColor(img_data["rgb"], cv2.COLOR_RGB2BGR)
                    gray_images.append(cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY))
                    color_images.append(color_img)
                
                if save_color:
                    color_panorama = self.create_color_panorama(color_images)
                    color_panorama_path = f"{self.save_folder}/color_panorama_{frame_id:04d}.jpg"
                    cv2.imwrite(color_panorama_path, color_panorama)
                    print(f"Color panorama saved to: {color_panorama_path}")
                else:
                    panorama = self.create_panorama(gray_images)
                    panorama_path = f"{self.save_folder}/panorama_{frame_id:04d}.jpg"
                    cv2.imwrite(panorama_path, panorama)
                    print(f"Panorama saved to: {panorama_path}")
                break

    def get_one_test(self):
        frame_id = 1
        current_pose = PoseState([137, 577, 30])
        self.get_panorama(frame_id, current_pose)


if __name__ == "__main__":
    image_collector = ImageCollector("../data/panorama", debug=True)
    image_collector.get_one_test()
