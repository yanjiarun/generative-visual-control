import cv2
import numpy as np
import zmq
import json
import pickle
import time
from typing import Dict, Any, Optional
import matplotlib.pyplot as plt
from src.path_following.collect_imgs_path_following import PATH_CONFIGS, compute_path_field, plot_path_field


class ZMQDataSender:
    def __init__(self, host: str = "localhost", port: int = 5555):
        self.zmq_context = zmq.Context()
        self.zmq_sender = self.zmq_context.socket(zmq.PUB)
        self.zmq_sender.bind(f"tcp://{host}:{port}")
        
    def send_data(self, data: Dict[str, Any]):
        try:
            # Convert image to bytes if present
            if 'panorama_image' in data and data['panorama_image'] is not None:
                # Convert image to JPEG bytes
                success, encoded_image = cv2.imencode('.jpg', data['panorama_image'])
                if success:
                    data['panorama_image_bytes'] = encoded_image.tobytes()
                    del data['panorama_image']  # Remove the image object
            
            # Serialize and send data
            message = pickle.dumps(data)
            self.zmq_sender.send(message)
        except Exception as e:
            print(f"Error sending data: {e}")
    
    def close(self):
        if self.zmq_sender:
            self.zmq_sender.close()
        if self.zmq_context:
            self.zmq_context.term()

class Visualizer:
    def __init__(self, path_type: str = "quartic"):
        self.path_type = path_type
        self.config = PATH_CONFIGS[path_type]
        
        # Base coordinates and scaling for real-world position
        self.base_x, self.base_y = 576, 275
        self.scale_factor = self.config["scale_factor"]

        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        
        self.visualize_data = {
            'uav_position': (500.0, 300.0),
            'uav_heading_vector': (1.0, 0.0),
            'uav_actual_heading_vector': (1.0, 0.0),
        }
        self.is_running = False

        self._setup_plot()
        
    def _setup_plot(self):
        # Create vector field grid
        x = np.linspace(self.config["x_range"][0], self.config["x_range"][1], self.config["n_points"]+1)
        y = np.linspace(self.config["y_range"][0], self.config["y_range"][1], int(self.config["n_points"] * (self.config["y_range"][1]-self.config["y_range"][0])/(self.config["x_range"][1]-self.config["x_range"][0]))+1)
        X, Y = np.meshgrid(x, y)
        u_x, u_y, alpha = compute_path_field(self.path_type, X, Y)

        # Convert grid coordinates to real-world positions
        X_real = self.base_x + X * self.scale_factor
        Y_real = self.base_y + Y * self.scale_factor

        # Plot vector field using quiver with real-world coordinates
        sample_rate = 1  # Adjust sample rate for quiver density
        scale = 50  # Adjust scale for quiver arrows
        self.ax.contour(X_real, Y_real, alpha, levels=[0], colors='red', linewidths=2)
        self.ax.quiver(X_real[::sample_rate, ::sample_rate], Y_real[::sample_rate, ::sample_rate], 
                       u_x[::sample_rate, ::sample_rate], u_y[::sample_rate, ::sample_rate], 
                       color='blue', alpha=0.5, scale=scale)
        
        # Plot UAV position as red dot, heading as arrow
        uav_position = self.visualize_data['uav_position']
        guidance_vector = self.visualize_data['uav_heading_vector']
        actual_heading_vector = self.visualize_data['uav_actual_heading_vector']
        self.uav_point, = self.ax.plot([uav_position[0]], 
                                        [uav_position[1]], 
                                        'ro', markersize=5, label='UAV')
        self.uav_arrow = self.ax.quiver(uav_position[0], uav_position[1],
                                        actual_heading_vector[0], actual_heading_vector[1],
                                        color='green', alpha=0.8, scale=30,
                                        width=0.005, headwidth=3, headlength=4,
                                        headaxislength=3.5, label='Actual heading')
        self.guidance_arrow = self.ax.quiver(
            uav_position[0], uav_position[1], guidance_vector[0], guidance_vector[1],
            color='darkorange', alpha=0.8, scale=30, width=0.005,
            headwidth=3, headlength=4, headaxislength=3.5,
            label='Guidance command')

        # Setup plot properties with real-world coordinate ranges
        x_min_real = self.base_x + self.config["x_range"][0] * self.scale_factor
        x_max_real = self.base_x + self.config["x_range"][1] * self.scale_factor
        y_min_real = self.base_y + self.config["y_range"][0] * self.scale_factor
        y_max_real = self.base_y + self.config["y_range"][1] * self.scale_factor
        
        self.ax.set_xlim([x_min_real, x_max_real])
        self.ax.set_ylim([y_min_real, y_max_real])
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.set_title(f'Vector Field for {self.path_type.title()} Path Following')
        self.ax.grid(True, alpha=0.3)
        self.ax.legend(loc='upper right', framealpha=0.8)
        
    def update_data(self, data):
        self.visualize_data.update(data)

    def _visualize(self):
        if not plt.fignum_exists(self.fig.number):
            self.is_running = False
            return
        
        uav_position = self.visualize_data['uav_position']
        guidance_vector = self.visualize_data['uav_heading_vector']
        actual_heading_vector = self.visualize_data.get(
            'uav_actual_heading_vector', guidance_vector)
        
        self.uav_point.set_data([uav_position[0]], [uav_position[1]])
        self.uav_arrow.remove()
        self.uav_arrow = self.ax.quiver(uav_position[0], uav_position[1],
                                       actual_heading_vector[0], actual_heading_vector[1],
                                       color='green', alpha=0.8, scale=30,
                                       width=0.005, headwidth=3, headlength=4,
                                       headaxislength=3.5)
        self.guidance_arrow.remove()
        self.guidance_arrow = self.ax.quiver(
            uav_position[0], uav_position[1], guidance_vector[0], guidance_vector[1],
            color='darkorange', alpha=0.8, scale=30, width=0.005,
            headwidth=3, headlength=4, headaxislength=3.5)
        
        self.fig.canvas.draw_idle()
        
    def run(self):
        self.is_running = True
        plt.ion()
        plt.show(block=False)
        
        # Connect key press event
        self.fig.canvas.mpl_connect('key_press_event', self._on_key_press)
        
        try:
            while self.is_running:
                # Check if window is still open
                if not plt.fignum_exists(self.fig.number):
                    break
                
                # Update visualization
                self._visualize()
                plt.pause(0.01)
        finally:
            self.stop()
    
    def _on_key_press(self, event):
        if event.key == 'q':
            self.is_running = False

    def stop(self):
        self.is_running = False
        try:
            plt.close(self.fig)
        except:
            pass


class ZMQReceiverVisualizer(Visualizer):    
    def __init__(self, host: str = "localhost", port: int = 5555, path_type: str = "quartic"):
        super().__init__(path_type=path_type)
        self._setup_zmq(host, port)
        self.is_received = False
    
    def _setup_zmq(self, host, port):
        self.zmq_context = zmq.Context()
        self.zmq_receiver = self.zmq_context.socket(zmq.SUB)
        self.zmq_receiver.connect(f"tcp://{host}:{port}")
        self.zmq_receiver.setsockopt_string(zmq.SUBSCRIBE, '')
        print(f"ZMQ connected to {host}:{port}")

    def run(self, timeout_ms: int = 100):
        self.is_running = True
        plt.ion()
        plt.show(block=False)
        
        # Connect key press event
        self.fig.canvas.mpl_connect('key_press_event', self._on_key_press)
        
        try:
            while self.is_running:
                if self.zmq_receiver.poll(timeout_ms):
                    message = self.zmq_receiver.recv()
                    data = pickle.loads(message)
                    self.update_data(data)
                    self.is_received = True
                
                # Update visualization
                if self.is_received:
                    self._visualize()
                plt.pause(0.001)
        except Exception as e:
            print(f"Error receiving data: {e}")
        except KeyboardInterrupt:
            print("ZMQ receiver interrupted")
        finally:
            self.stop()

    
    def stop(self):
        super().stop()
        if self.zmq_receiver:
            self.zmq_receiver.close()
        if self.zmq_context:
            self.zmq_context.term()


if __name__ == "__main__":
    visualizer = ZMQReceiverVisualizer(path_type="cassini")  # "quartic" for quartic path or "cassini" for Cassini oval path
    visualizer.run()
