import json
import time

import cv2
import numpy as np
import zmq

from AFPV_Kit.bridges.pose_sender import PoseSender


class PanoramaBridge:
    def __init__(self, cameras):
        self.cameras = cameras
        self.sender = PoseSender()
        self.sender.pose_socket.setsockopt(zmq.LINGER, 0)
        self.sender.pose_socket.setsockopt(zmq.SNDHWM, 1)
        self.context = zmq.Context()
        self.receiver = self.context.socket(zmq.SUB)
        self.receiver.setsockopt(zmq.SUBSCRIBE, b"")
        self.receiver.setsockopt(zmq.RCVHWM, 2)
        self.receiver.setsockopt(zmq.LINGER, 0)
        self.receiver.connect("tcp://127.0.0.1:5001")
        self.closed = False

    def capture(self, pose):
        deadline, next_send = time.monotonic() + 60.0, 0.0
        last_frame, sends, send_ok = None, 0, None
        while time.monotonic() < deadline:
            now = time.monotonic()
            if now >= next_send:
                send_ok = self.sender.send_pose(**pose)
                sends += 1
                next_send = now + 5.0  # Avoid a render backlog.
            if not self.receiver.poll(50, zmq.POLLIN):
                continue
            parts = self.receiver.recv_multipart()
            if not parts:
                continue
            meta = json.loads(parts[0].decode("utf-8"))
            last_frame = meta.get("frame_id")
            if last_frame != pose["frame_id"]:
                continue
            if len(parts) != 1 + len(self.cameras):
                raise ValueError(f"Incomplete Unity multipart frame: {len(parts)} parts")
            images = []
            for raw, camera in zip(parts[1:], self.cameras):
                rgb = np.frombuffer(raw, np.uint8).reshape(camera["height"], camera["width"], camera["channels"])
                images.append(cv2.cvtColor(cv2.flip(rgb, 0), cv2.COLOR_RGB2BGR))
            return images
        raise TimeoutError(f"Unity timeout: requested={pose['frame_id']}, last received={last_frame}, "
                           f"pose sends={sends}, last send succeeded={send_ok}")

    def close(self):
        if not self.closed:
            self.closed = True
            self.receiver.close()
            self.context.term()
            self.sender.close()
