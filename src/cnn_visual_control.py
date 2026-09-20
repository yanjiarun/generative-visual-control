import os
import math
import time
import cv2
import numpy as np
import torch
from collections import deque
from torchvision import transforms
from PIL import Image
from AFPV_Kit.bridges.unity_bridge import UnityBridge
from AFPV_Kit.common.transfer import convert_position_to_unity, convert_quaternion_to_unity, calculate_camera_matrix, convert_ned_quaternion_to_enu
from src.baseline.cnn_model import BaselineCNN
from src.online_visualizer import ZMQDataSender
from src.fixed_wing_model.fixed_wing import FixedWingUAV


class CNNVisualControl:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.path_type = self.config.get("path_type", "quartic")
        self.scale_factor = 200 if self.config.get("path_type", "quartic") == "quartic" else 100

        self.cnn_model = self._load_cnn_model(config["cnn_model_path"])
        self.unity_bridge = self._initialize_unity_bridge(config.get("camera_config", self._get_default_camera_config()))
        self.tau_theta = config.get("tau_theta", 28)
        self.fixed_wing = FixedWingUAV(config.get("init_pose", [0, 0, 0]), config.get("v0", 18), config.get("dt", 0.05), self.tau_theta)
        self.preprocess = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop((224, 224))
        ])
        
        self.image_history = deque(maxlen=1)
        self.predicted_vector_seq = None
        
        self.img_index, self.p_x, self.p_y = self._load_panorama_mapping()

        self.data_sender = ZMQDataSender()
        
    def _load_cnn_model(self, model_path):
        """Load the CNN model for vector field prediction"""
        try:
            model = BaselineCNN().to(self.device)
            checkpoint = torch.load(model_path)
            
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            model.eval()
            return model
        except ImportError:
            print("Warning: BaselineCNN not found, using dummy model")
            return None
        
    def _get_default_camera_config(self):
        """Get default camera configuration for panoramic imaging"""
        base_config = {
            "width": 1024,
            "height": 1024,
            "channels": 3,
            "fov": 90.0,
            "nearClipPlane": [0.01, 0.5, 0.5],
            "farClipPlane": [1000.0, 550.0, 1000.0],
            "enabledLayers": [False, False]
        }
        
        camera_configs = [
            {"ID": "Camera_Front", "T_BC": calculate_camera_matrix(x_offset=0., y_offset=-0.20)},
            {"ID": "Camera_Back", "T_BC": calculate_camera_matrix(yaw_deg=180, x_offset=0., y_offset=-0.20)},
            {"ID": "Camera_Left", "T_BC": calculate_camera_matrix(yaw_deg=-90, x_offset=0., y_offset=-0.20)},
            {"ID": "Camera_Right", "T_BC": calculate_camera_matrix(yaw_deg=90, x_offset=0., y_offset=-0.20)},
            {"ID": "Camera_Top", "T_BC": calculate_camera_matrix(pitch_deg=90, x_offset=0., y_offset=-0.20)},
            {"ID": "Camera_Bottom", "T_BC": calculate_camera_matrix(pitch_deg=-90, x_offset=0., y_offset=-0.20)}
        ]
        
        return [{**base_config, **cam} for cam in camera_configs]
        
    def _initialize_unity_bridge(self, camera_config):
        """Initialize Unity bridge for simulation"""
        vehicles_info = {
            "vehicles": [{
                "ID": "FPV",
                "position": [0.0, 0.0, 0.0],
                "rotation": [0.0, 0, 0, 1],
                "scale": [0.05, 0.05, 0.05],
                "cameras": camera_config
            }]
        }
        bridge = UnityBridge(vehicles_info=vehicles_info)
        bridge.start()
        return bridge
        
    def _load_panorama_mapping(self):
        """Load panorama mapping data for image stitching"""
        class_dir = os.path.dirname(os.path.abspath(__file__))
        mapping_folder = os.path.join(class_dir, "utils", "required_mats")
        
        img_index = np.load(os.path.join(mapping_folder, "img_index.npy"))
        p_x = np.load(os.path.join(mapping_folder, "p_x_RGB.npy"))
        p_y = np.load(os.path.join(mapping_folder, "p_y_RGB.npy"))
        
        return img_index, p_x, p_y
        
    def _create_color_panorama(self, images):
        """Create color panorama from 6 camera images"""
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

    def rectify_panorama(self, panorama, color_mode=True):            
        panorama = cv2.rotate(panorama, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        r1, r2, w, h = 242, 511, 1024, 1024
        angle_res = 1800
        
        if color_mode:
            rectified_image = np.zeros((r2-r1+1, angle_res, 3), dtype=np.uint8)
        else:
            rectified_image = np.zeros((r2-r1+1, angle_res), dtype=np.uint8)
        
        theta = np.linspace(-np.pi, np.pi, angle_res, endpoint=False)
        
        for radius in range(r1, r2+1):
            x_centered = radius * np.cos(theta)
            y_centered = radius * np.sin(theta)
            x = x_centered + w/2
            y = y_centered + h/2
            
            for i in range(len(y)):
                x_coord = int(x[i])
                y_coord = int(y[i])
                if color_mode:
                    rectified_image[radius-r1][i] = panorama[y_coord][x_coord]
                else:
                    rectified_image[radius-r1][i] = panorama[y_coord][x_coord]

        return rectified_image
        
    def get_panoramic_image(self, frame_id, current_position):
        """Capture panoramic image from Unity simulation"""
        current_pos = np.array([current_position[0], current_position[1], 75.0])  # Keep altitude fixed
        
        position_unity = convert_position_to_unity(current_pos)
        quaternion_unity = convert_quaternion_to_unity(convert_ned_quaternion_to_enu([0, 0, 0, 1]))  # Keep camera level for better panorama stitching
        
        unity_pose_kwargs = {
            'frame_id': frame_id,
            'vehicle_id': 'FPV',
            'position': position_unity,
            'rotation': quaternion_unity,
            'cameras': self._get_default_camera_config()
        }
        
        if frame_id == 1:
            for _ in range(10):
                self.unity_bridge.send_pose(**unity_pose_kwargs)
                time.sleep(0.01)
        while True:
            self.unity_bridge.send_pose(**unity_pose_kwargs)
            time.sleep(0.033)   # ~30fps
            
            img_data_list = [self.unity_bridge.get_latest_image(0, i) for i in range(6)]
            if all(img is not None and img["frame_id"] == frame_id for img in img_data_list):
                color_images = []
                for img_data in img_data_list:
                    color_img = cv2.cvtColor(img_data["rgb"], cv2.COLOR_RGB2BGR)
                    color_images.append(color_img)
                
                panorama = self._create_color_panorama(color_images)
                rectified_panorama = self.rectify_panorama(panorama)
                return rectified_panorama
      
    def preprocess_image(self, image):
        """Preprocess image for CNN model input"""
        if image is None:
            return None
            
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(image_rgb.astype('uint8'), 'RGB')
        img_pil = self.preprocess(img_pil)
        img_tensor = torch.FloatTensor(np.array(img_pil)).permute(2, 0, 1) / 255.0

        return img_tensor.unsqueeze(0).to(self.device)  # Add batch dimension
        
    def generate_vector_field(self, image_tensor, _current_position=None):
        """Generate vector field from panoramic image using CNN model"""
        if self.cnn_model is None:
            return np.array([1.0, 0.0, 0.0, 0.0])
            
        with torch.no_grad():
            image_tensor = image_tensor.squeeze(1)  # Remove sequence dimension
            predicted_vector = self.cnn_model(image_tensor)
            predicted_vector = predicted_vector.cpu().numpy().squeeze(0)  # Remove batch dimension
            return predicted_vector
    
    def calculate_heading_cmd(self, vector, delta_mix, current_heading, current_velocity, tao_heading=28.0, kp=0.18):
        current_velocity = current_velocity/self.scale_factor
        p_dot = current_velocity * np.array([np.cos(current_heading), np.sin(current_heading)])
        delta_desired_heading = delta_mix @ p_dot
        omega_cmd = kp * np.sin(np.arctan2(vector[1], vector[0]) - current_heading) + delta_desired_heading
        heading_cmd = current_heading + tao_heading * omega_cmd
        return heading_cmd
    
    def control_loop(self, max_frames=1000, save_data=False):
        frame_id = 1
        simulation_dt = 0.05
        
        if save_data:
            if self.config.get('save_folder'):
                save_folder = self.config['save_folder']
                os.makedirs(save_folder, exist_ok=True)
                save_poses = []
                save_heading_commands = []
            else:
                save_data = False
                print("save_folder not specified in config. Data saving disabled.") 

        try:
            for frame_id in range(1, max_frames + 1):
                curr_position = self.fixed_wing.get_position()  
                curr_heading = self.fixed_wing.get_heading() 

                image = self.get_panoramic_image(frame_id, curr_position)
                if image is None:
                    continue
                    
                self.image_history.append(image)
                while len(self.image_history) < self.config.get("seq_length", 1):  # Ensure history is filled
                    self.image_history.append(image)
                    
                image_sequence = []
                for img in self.image_history:
                    img_tensor = self.preprocess_image(img)
                    if img_tensor is not None:
                        image_sequence.append(img_tensor)
                        
                if not image_sequence:
                    continue
                    
                image_sequence = torch.cat(image_sequence, dim=0).unsqueeze(0)
                if len(img_tensor.shape) == 4 and img_tensor.shape[0] == 1:
                    img_tensor = img_tensor.squeeze(0)  # Remove batch dimension for visualization
                self.predicted_vector_seq = self.generate_vector_field(
                    image_sequence, curr_position)
                current_vector = self.predicted_vector_seq[:2]
                heading_command = self.calculate_heading_cmd(current_vector, self.predicted_vector_seq[2:4], curr_heading, self.fixed_wing.get_velocity(), self.tau_theta)
                data = {
                    'uav_position': curr_position,
                    'uav_heading_vector': current_vector,
                    'uav_actual_heading_vector': np.array([
                        np.cos(curr_heading), np.sin(curr_heading)])
                }
                self.data_sender.send_data(data)

                if save_data:
                    save_poses.append(np.array([curr_position[0], curr_position[1], curr_heading]))
                    save_heading_commands.append(heading_command)
                
                self.fixed_wing.run(simulation_dt, heading_command)

                if frame_id % 20 == 0:
                    print(f"Frame {frame_id}: Position {curr_position}, Heading: {math.degrees(curr_heading):.1f}°")
                time.sleep(simulation_dt)
                    
        except KeyboardInterrupt:
            print("Control loop interrupted")
        finally:
            self.unity_bridge.close()
            if save_data:
                np.save(os.path.join(save_folder, "poses.npy"), np.array(save_poses))
                np.save(os.path.join(save_folder, "heading_commands.npy"), np.array(save_heading_commands))
                print(f"Data saved to {save_folder}")
            self.data_sender.close()
