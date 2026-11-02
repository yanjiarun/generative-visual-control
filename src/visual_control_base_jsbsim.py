import json
import math
import os
import time
from collections import deque
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import cv2
import jsbsim
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from AFPV_Kit.bridges.unity_bridge import UnityBridge
from AFPV_Kit.common.transfer import (
    calculate_camera_matrix,
    convert_euler_to_quaternion,
    convert_ned_quaternion_to_enu,
    convert_position_to_unity,
    convert_quaternion_to_unity,
)
from AFPV_Kit.objects.fixedwing import FixedWing
from AFPV_Kit.utils.autopilot import X8Autopilot
from src.jsbsim_config import X8_AUTOPILOT_PID
from src.online_visualizer import ZMQDataSender
from src.path_following.collect_imgs_path_following import PATH_CONFIGS


def enu_to_ned_heading(theta):
    return (90.0 - math.degrees(theta)) % 360.0


def ned_to_enu_heading(psi):
    return math.pi / 2.0 - psi


class X8AutopilotJSBSim(X8Autopilot):
    def __init__(self, pid_params=None, max_bank_deg=35.0):
        super().__init__(pid_params)
        self.max_bank = math.radians(max_bank_deg)

    def heading_hold(self, desired_heading):
        error = (desired_heading - math.degrees(
            self.current_state.get_attitude()[2]) + 180.0) % 360.0 - 180.0
        self.desired_roll = np.clip(
            self.heading_controller(error), -self.max_bank, self.max_bank)
        self.roll_hold(self.desired_roll)


class VisualControlJSBSim:
    requires_image = True

    def __init__(self, config):
        self.config = dict(config)
        overrides = config.get("autopilot_pid") or {}
        self.config["autopilot_pid"] = {
            loop: {**gains, **overrides.get(loop, {})}
            for loop, gains in X8_AUTOPILOT_PID.items()
        }
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.path_type = config.get("path_type", "cassini")
        self.scale_factor = PATH_CONFIGS[self.path_type]["scale_factor"]
        self.control_dt = float(config.get("control_dt", 0.05))
        self.target_speed = float(config.get("target_speed", 23.0))
        self.target_altitude = float(config.get("target_altitude", 75.0))
        self.visual_altitude = float(config.get("visual_altitude", 75.0))
        self.heading_lookahead = float(config.get("heading_lookahead", 6.0))
        self.heading_gain = float(config.get("heading_gain", 0.18))
        self.max_turn_rate = math.radians(config.get("max_turn_rate_deg", 15.0))
        self.seq_length = int(config.get("seq_length", 1))
        self.image_history = deque(maxlen=self.seq_length)

        pose = config.get("init_pose", [576.0, 275.0, math.pi / 3.0])
        position = config.get("init_position", [pose[0], pose[1], self.target_altitude])
        self.fixed_wing, self.autopilot = self._create_aircraft(
            position, config.get("init_heading_deg", enu_to_ned_heading(pose[2])))
        self.sender = ZMQDataSender() if config.get("visualize", True) else None
        self.unity_bridge = None
        if self.requires_image:
            self.camera_config = config.get("camera_config", self._camera_config())
            self.unity_bridge = self._create_unity_bridge()
            self._prepare_panorama_maps()
            self.preprocess = transforms.Compose([
                transforms.Resize((256, 256)), transforms.CenterCrop((224, 224))])

    @staticmethod
    def _camera_config():
        base = dict(width=1024, height=1024, channels=3, fov=90.0,
                    nearClipPlane=[0.01, 0.5, 0.5],
                    farClipPlane=[1000.0, 550.0, 1000.0],
                    enabledLayers=[False, False])
        poses = [
            ("Camera_Front", {}), ("Camera_Back", {"yaw_deg": 180}),
            ("Camera_Left", {"yaw_deg": -90}), ("Camera_Right", {"yaw_deg": 90}),
            ("Camera_Top", {"pitch_deg": 90}), ("Camera_Bottom", {"pitch_deg": -90}),
        ]
        return [{**base, "ID": name,
                 "T_BC": calculate_camera_matrix(x_offset=0.0, y_offset=-0.2, **rot)}
                for name, rot in poses]

    def _create_aircraft(self, position, heading_deg):
        jsb_config = {
            "jsbsim_root": str(Path(jsbsim.__file__).parent),
            "dt": float(self.config.get("jsbsim_dt", 0.001)),
            "aircraft": self.config.get("aircraft", "x8"),
            "init_conditions": {
                "lat_deg": self.config.get("lat_deg", 35.6541666666666),
                "lon_deg": self.config.get("lon_deg", 139.75625),
                "alt_ft": position[2] / 0.3048,
                "airspeed_fps": self.target_speed / 0.3048,
                "psi_deg": heading_deg,
            },
        }
        return FixedWing(init_position=np.array([position[0], position[1], 0.0]),
                         jsb_config=jsb_config), X8AutopilotJSBSim(
                             self.config.get("autopilot_pid"),
                             self.config.get("max_bank_deg", 35.0))

    def _create_unity_bridge(self):
        bridge = UnityBridge(vehicles_info={"vehicles": [{
            "ID": "FPV", "position": [0.0, 0.0, 0.0],
            "rotation": [0.0, 0.0, 0.0, 1.0], "scale": [0.05] * 3,
            "cameras": self.camera_config,
        }]})
        bridge.start()
        return bridge

    def _prepare_panorama_maps(self):
        root = os.path.dirname(os.path.abspath(__file__))
        folder = os.path.join(root, "utils", "required_mats")
        index = np.load(os.path.join(folder, "img_index.npy")).ravel().astype(int)
        px = np.load(os.path.join(folder, "p_x_RGB.npy")).ravel().astype(int)
        py = np.load(os.path.join(folder, "p_y_RGB.npy")).ravel().astype(int)
        self.stitch_maps = []
        for camera in range(6):
            mask = index == camera
            valid = mask & (px >= 0) & (px < 1024) & (py >= 0) & (py < 1024)
            self.stitch_maps.append((np.flatnonzero(valid), px[valid], py[valid]))
        radius = np.arange(242, 512)[:, None]
        theta = np.linspace(-np.pi, np.pi, 1800, endpoint=False)[None]
        self.rectify_x = (radius * np.cos(theta) + 512.0).astype(np.int32).astype(np.float32)
        self.rectify_y = (radius * np.sin(theta) + 512.0).astype(np.int32).astype(np.float32)

    def _panorama(self, images):
        panorama = np.zeros((1024, 1024, 3), dtype=np.uint8)
        flat = panorama.reshape(-1, 3)
        for image, (out, x, y) in zip(images, self.stitch_maps):
            flat[out] = image[y, x]
        rotated = cv2.rotate(panorama, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return cv2.remap(rotated, self.rectify_x, self.rectify_y,
                         cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)

    def _unity_rotation(self):
        if self.config.get("world_aligned_panorama", True):
            q_ned = np.array([0.0, 0.0, 0.0, 1.0])
        else:
            yaw = self.fixed_wing.get_state().get_attitude()[2]
            q_ned = convert_euler_to_quaternion(0.0, 0.0, yaw)
        return convert_quaternion_to_unity(convert_ned_quaternion_to_enu(q_ned))

    def get_image(self, frame_id, position):
        pose = dict(frame_id=frame_id, vehicle_id="FPV",
                    position=convert_position_to_unity(
                        [position[0], position[1], self.visual_altitude]),
                    rotation=self._unity_rotation(), cameras=self.camera_config)
        attempts = int(self.config.get("image_attempts", 300))
        for attempt in range(attempts):
            self.unity_bridge.send_pose(**pose)
            time.sleep(0.01 if frame_id == 1 and attempt < 10 else 0.033)
            data = [self.unity_bridge.get_latest_image(0, i) for i in range(6)]
            if all(item is not None and item["frame_id"] == frame_id for item in data):
                images = [cv2.cvtColor(item["rgb"], cv2.COLOR_RGB2BGR) for item in data]
                return self._panorama(images)
        raise TimeoutError(f"Unity did not return panorama frame {frame_id}")

    def image_tensor(self, image):
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        tensor = torch.from_numpy(np.array(self.preprocess(image), copy=True))
        tensor = tensor.permute(2, 0, 1).float() / 255.0
        return tensor

    def guidance(self, image_sequence, position, state):
        raise NotImplementedError

    def gvf_heading(self, output, state):
        current = ned_to_enu_heading(state.get_attitude()[2])
        output = np.nan_to_num(np.asarray(output, dtype=float),
                               nan=0.0, posinf=0.0, neginf=0.0)
        vector = output[:2]
        gradient = output[2:4] if output.size >= 4 else np.zeros(2)
        if np.linalg.norm(vector) < 1e-6:
            return current
        velocity = state.get_velocity()  # north, east, down
        p_dot = np.array([velocity[1], velocity[0]]) / self.scale_factor
        desired = math.atan2(vector[1], vector[0])
        omega = self.heading_gain * math.sin(desired - current) + gradient @ p_dot
        omega = np.clip(omega, -self.max_turn_rate, self.max_turn_rate)
        return current + self.heading_lookahead * omega

    def heading_command(self, output, state):
        return self.gvf_heading(output, state)

    def _save(self, folder, data):
        os.makedirs(folder, exist_ok=True)
        for name, values in data.items():
            np.save(os.path.join(folder, f"{name}.npy"), np.asarray(values))

    def control_loop(self, max_frames=4000, save_data=True):
        names = ("poses", "heading_commands", "positions", "quaternions",
                 "vah_commands", "commands", "lla_data", "vned_data",
                 "vair_data", "atti_data")
        log = {name: [] for name in names}
        try:
            for frame_id in range(1, max_frames + 1):
                state = self.fixed_wing.get_state()
                pose = self.fixed_wing.get_state_trans()
                xy = np.asarray(pose.get_position()[:2])
                images = None
                if self.requires_image:
                    image = self.get_image(frame_id, xy)
                    self.image_history.append(self.image_tensor(image))
                    while len(self.image_history) < self.seq_length:
                        self.image_history.append(self.image_history[-1])
                    images = torch.stack(tuple(self.image_history)).unsqueeze(0).to(self.device)
                output = np.nan_to_num(np.asarray(
                    self.guidance(images, xy, state), dtype=float).reshape(-1),
                    nan=0.0, posinf=0.0, neginf=0.0)
                if output.size < 2:
                    raise ValueError("guidance output must contain at least two values")
                command_enu = self.heading_command(output, state)
                command_ned = enu_to_ned_heading(command_enu)
                control = self.autopilot.heading_and_altitude_hold(
                    self.target_speed, command_ned, self.target_altitude, state)
                current_enu = ned_to_enu_heading(state.get_attitude()[2])

                if self.sender:
                    self.sender.send_data({
                        "uav_position": xy,
                        "uav_heading_vector": output[:2],
                        "uav_actual_heading_vector": np.array([
                            math.cos(current_enu), math.sin(current_enu)]),
                    })
                if save_data:
                    actual_position = np.r_[xy, state.get_position()[2]]
                    log["poses"].append([xy[0], xy[1], current_enu])
                    log["heading_commands"].append(command_enu)
                    log["positions"].append(actual_position)
                    log["quaternions"].append(pose.get_quaternion())
                    log["vah_commands"].append([
                        self.target_speed, command_ned, self.target_altitude])
                    log["commands"].append([control.get_throttle(), control.get_aileron(),
                                             control.get_elevator(), control.get_rudder()])
                    log["lla_data"].append(state.get_position())
                    log["vned_data"].append(state.get_velocity())
                    log["vair_data"].append(state.get_airspeed())
                    log["atti_data"].append(state.get_attitude())
                self.fixed_wing.run(self.control_dt, control)
                if frame_id % 20 == 0:
                    print(f"frame {frame_id}: xy={xy.round(2)}, altitude={state.get_position()[2]:.2f} m, "
                          f"airspeed={state.get_airspeed():.2f} m/s")
                if self.config.get("realtime", False):
                    time.sleep(self.control_dt)
        except KeyboardInterrupt:
            print("Control loop interrupted")
        finally:
            if save_data:
                folder = self.config["save_folder"]
                self._save(folder, log)
                effective = {**self.config, "control_dt": self.control_dt,
                             "jsbsim_dt": float(self.config.get("jsbsim_dt", 0.001)),
                             "target_speed": self.target_speed,
                             "target_altitude": self.target_altitude,
                             "heading_gain": self.heading_gain,
                             "heading_lookahead": self.heading_lookahead,
                             "max_turn_rate_deg": math.degrees(self.max_turn_rate),
                             "max_bank_deg": math.degrees(self.autopilot.max_bank),
                             "max_frames": max_frames, "saved_frames": len(log["poses"])}
                with open(os.path.join(folder, "run_config.json"), "w", encoding="utf-8") as handle:
                    json.dump(effective, handle, indent=2,
                              default=lambda value: value.tolist() if hasattr(value, "tolist") else str(value))
            if self.unity_bridge:
                self.unity_bridge.close()
            if self.sender:
                self.sender.close()
