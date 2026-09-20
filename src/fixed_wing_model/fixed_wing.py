import numpy as np


class FixedWingUAV:
    def __init__(self, init_pose=[-300, 0.0, np.pi/6], v0=23, dt=0.001, tau_theta=28):
        self.x = init_pose[0]
        self.y = init_pose[1]
        self.theta = init_pose[2]
        self.v = v0
        self.dt = dt
        self.tau_theta = tau_theta
    
    def run(self, run_time, theta_cmd):
        steps = int(run_time / self.dt)
        for _ in range(steps):
            self.update(theta_cmd)

    def update(self, theta_cmd):
        self.x += self.v * np.cos(self.theta) * self.dt
        self.y += self.v * np.sin(self.theta) * self.dt
        self.theta += (theta_cmd - self.theta) / self.tau_theta * self.dt
        return self.x, self.y, self.theta
    
    def get_position(self):
        return np.array([self.x, self.y])
    
    def get_heading(self):
        return self.theta
    
    def get_velocity(self):
        return self.v