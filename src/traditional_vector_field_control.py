import os
import math
import time
import numpy as np
from src.online_visualizer import ZMQDataSender
from src.fixed_wing_model.fixed_wing import FixedWingUAV


class TraditionalVectorFieldControl:
    def __init__(self, config):
        self.config = config

        self.path_type = self.config.get("path_type", "quartic")
        self.scale_factor = 200 if self.config.get("path_type", "quartic") == "quartic" else 100
        
        self.tau_theta = config.get("tau_theta", 28)
        self.fixed_wing = FixedWingUAV(config.get("init_pose", [0, 0, 0]), config.get("v0", 18), config.get("dt", 0.05), self.tau_theta)

        self.data_sender = ZMQDataSender()
        
    def generate_vector_field(self, uav_position):
        """Generate vector field command based on current UAV position and path type"""
        current_position = np.array(uav_position)
        base_x, base_y = 576, 275
        x_norm = (current_position[0] - base_x) / self.scale_factor
        y_norm = (current_position[1] - base_y) / self.scale_factor
        
        if self.path_type == "quartic":
            ux, uy, delta_ux, delta_uy = self._compute_quartic_field_with_gradients(x_norm, y_norm)
        elif self.path_type == "cassini":
            ux, uy, delta_ux, delta_uy = self._compute_cassini_field_with_gradients(x_norm, y_norm)
        else:
            ux, uy = -x_norm, -y_norm
            norm = np.linalg.norm([ux, uy])
            if norm > 1e-6:
                ux, uy = ux/norm, uy/norm
        
        return np.array([ux, uy]), delta_ux, delta_uy
    
    def _compute_quartic_field_with_gradients(self, x, y, a=6.0, b=0.0, c=18.0):
        """Compute vector field with gradients for quartic curve: α = ax⁴ + bx²y² + cy⁴ - 1 = 0"""
        alpha = a*x**4 + b*x**2*y**2 + c*y**4 - 1
        dalpha_dx = 4*a*x**3 + 2*b*x*y**2
        dalpha_dy = 2*b*x**2*y + 4*c*y**3
        
        P = 0.5 * alpha**2
        dP_dx = alpha * dalpha_dx
        dP_dy = alpha * dalpha_dy
        
        G = -(2/np.pi) * np.arctan(2*np.sqrt(P))
        H = np.sqrt(1 - G**2)
        
        norm_dP = np.sqrt(dP_dx**2 + dP_dy**2)
        norm_Rdalpha = np.sqrt(dalpha_dy**2 + dalpha_dx**2)
        
        epsilon = 1e-8
        norm_dP = norm_dP if norm_dP > epsilon else epsilon
        norm_Rdalpha = norm_Rdalpha if norm_Rdalpha > epsilon else epsilon
        
        u_x = G * (dP_dx/norm_dP) + H * (-dalpha_dy/norm_Rdalpha)
        u_y = G * (dP_dy/norm_dP) + H * (dalpha_dx/norm_Rdalpha)
        
        dG_dP = -(4/np.pi) / (1 + 4*P) * (0.5/np.sqrt(P + epsilon))
        dG_dx = dG_dP * dP_dx
        dG_dy = dG_dP * dP_dy
        
        dH_dG = -G / (H + epsilon)
        dH_dx = dH_dG * dG_dx
        dH_dy = dH_dG * dG_dy
        
        d2alpha_dx2 = 12*a*x**2 + 2*b*y**2
        d2alpha_dy2 = 2*b*x**2 + 12*c*y**2
        d2alpha_dxdy = 4*b*x*y
        
        d2P_dx2 = dalpha_dx**2 + alpha * d2alpha_dx2
        d2P_dy2 = dalpha_dy**2 + alpha * d2alpha_dy2
        d2P_dxdy = dalpha_dx * dalpha_dy + alpha * d2alpha_dxdy
        
        dnorm_dP_dx = (dP_dx*d2P_dx2 + dP_dy*d2P_dxdy) / norm_dP
        dnorm_dP_dy = (dP_dx*d2P_dxdy + dP_dy*d2P_dy2) / norm_dP
        
        de1x_dx = (d2P_dx2*norm_dP - dP_dx*dnorm_dP_dx) / (norm_dP**2)
        de1x_dy = (d2P_dxdy*norm_dP - dP_dx*dnorm_dP_dy) / (norm_dP**2)
        de1y_dx = (d2P_dxdy*norm_dP - dP_dy*dnorm_dP_dx) / (norm_dP**2)
        de1y_dy = (d2P_dy2*norm_dP - dP_dy*dnorm_dP_dy) / (norm_dP**2)
        
        dnorm_Rdalpha_dx = (dalpha_dy*d2alpha_dxdy + dalpha_dx*d2alpha_dx2) / norm_Rdalpha
        dnorm_Rdalpha_dy = (dalpha_dy*d2alpha_dy2 + dalpha_dx*d2alpha_dxdy) / norm_Rdalpha
        
        de2x_dx = (-d2alpha_dxdy*norm_Rdalpha + dalpha_dy*dnorm_Rdalpha_dx) / (norm_Rdalpha**2)
        de2x_dy = (-d2alpha_dy2*norm_Rdalpha + dalpha_dy*dnorm_Rdalpha_dy) / (norm_Rdalpha**2)
        de2y_dx = (d2alpha_dx2*norm_Rdalpha - dalpha_dx*dnorm_Rdalpha_dx) / (norm_Rdalpha**2)
        de2y_dy = (d2alpha_dxdy*norm_Rdalpha - dalpha_dx*dnorm_Rdalpha_dy) / (norm_Rdalpha**2)
        
        du_x_dx = dG_dx*(dP_dx/norm_dP) + G*de1x_dx + dH_dx*(-dalpha_dy/norm_Rdalpha) + H*de2x_dx
        du_x_dy = dG_dy*(dP_dx/norm_dP) + G*de1x_dy + dH_dy*(-dalpha_dy/norm_Rdalpha) + H*de2x_dy
        du_y_dx = dG_dx*(dP_dy/norm_dP) + G*de1y_dx + dH_dx*(dalpha_dx/norm_Rdalpha) + H*de2y_dx
        du_y_dy = dG_dy*(dP_dy/norm_dP) + G*de1y_dy + dH_dy*(dalpha_dx/norm_Rdalpha) + H*de2y_dy
        
        delta_ux = np.array([du_x_dx, du_x_dy])
        delta_uy = np.array([du_y_dx, du_y_dy])
        
        return u_x, u_y, delta_ux, delta_uy
    
    def _compute_cassini_field_with_gradients(self, x, y, a=1.5, b=0.9):
        """Compute vector field for Cassini oval: α = (x²+y²)² - 2a²(x²-y²) - b⁴ = 0"""
        alpha = (x**2 + y**2)**2 - 2*a**2*(x**2 - y**2) - b**4
        dalpha_dx = 4*x*(x**2 + y**2) - 4*a**2*x
        dalpha_dy = 4*y*(x**2 + y**2) + 4*a**2*y
        
        P = 0.5 * alpha**2
        dP_dx = alpha * dalpha_dx
        dP_dy = alpha * dalpha_dy
        
        G = -(2/np.pi) * np.arctan(2*np.sqrt(np.abs(P)))
        H = np.sqrt(1 - G**2)
        
        norm_dP = np.sqrt(dP_dx**2 + dP_dy**2)
        norm_Rdalpha = np.sqrt(dalpha_dy**2 + dalpha_dx**2)
        
        epsilon = 1e-8
        norm_dP = norm_dP if norm_dP > epsilon else epsilon
        norm_Rdalpha = norm_Rdalpha if norm_Rdalpha > epsilon else epsilon
        
        u_x = G * (dP_dx/norm_dP) + H * (-dalpha_dy/norm_Rdalpha)
        u_y = G * (dP_dy/norm_dP) + H * (dalpha_dx/norm_Rdalpha)
        
        dG_dP = -(4/np.pi) / (1 + 4*P) * (0.5/np.sqrt(np.abs(P) + epsilon))
        dG_dx = dG_dP * dP_dx
        dG_dy = dG_dP * dP_dy
        
        dH_dG = -G / (H + epsilon)
        dH_dx = dH_dG * dG_dx
        dH_dy = dH_dG * dG_dy
        
        # Compute second derivatives of α
        d2alpha_dx2 = 4*(x**2 + y**2) + 8*x**2 - 4*a**2
        d2alpha_dy2 = 4*(x**2 + y**2) + 8*y**2 + 4*a**2
        d2alpha_dxdy = 8*x*y
        
        # Compute second derivatives of P
        d2P_dx2 = dalpha_dx**2 + alpha * d2alpha_dx2
        d2P_dy2 = dalpha_dy**2 + alpha * d2alpha_dy2
        d2P_dxdy = dalpha_dx * dalpha_dy + alpha * d2alpha_dxdy
        
        # Compute gradients of norm_dP
        dnorm_dP_dx = (dP_dx*d2P_dx2 + dP_dy*d2P_dxdy) / norm_dP
        dnorm_dP_dy = (dP_dx*d2P_dxdy + dP_dy*d2P_dy2) / norm_dP
        
        # Compute gradients of unit vector e1 = (dP_dx/norm_dP, dP_dy/norm_dP)
        de1x_dx = (d2P_dx2*norm_dP - dP_dx*dnorm_dP_dx) / (norm_dP**2)
        de1x_dy = (d2P_dxdy*norm_dP - dP_dx*dnorm_dP_dy) / (norm_dP**2)
        de1y_dx = (d2P_dxdy*norm_dP - dP_dy*dnorm_dP_dx) / (norm_dP**2)
        de1y_dy = (d2P_dy2*norm_dP - dP_dy*dnorm_dP_dy) / (norm_dP**2)
        
        # Compute gradients of norm_Rdalpha
        dnorm_Rdalpha_dx = (dalpha_dy*d2alpha_dxdy + dalpha_dx*d2alpha_dx2) / norm_Rdalpha
        dnorm_Rdalpha_dy = (dalpha_dy*d2alpha_dy2 + dalpha_dx*d2alpha_dxdy) / norm_Rdalpha
        
        # Compute gradients of unit vector e2 = (-dalpha_dy/norm_Rdalpha, dalpha_dx/norm_Rdalpha)
        de2x_dx = (-d2alpha_dxdy*norm_Rdalpha + dalpha_dy*dnorm_Rdalpha_dx) / (norm_Rdalpha**2)
        de2x_dy = (-d2alpha_dy2*norm_Rdalpha + dalpha_dy*dnorm_Rdalpha_dy) / (norm_Rdalpha**2)
        de2y_dx = (d2alpha_dx2*norm_Rdalpha - dalpha_dx*dnorm_Rdalpha_dx) / (norm_Rdalpha**2)
        de2y_dy = (d2alpha_dxdy*norm_Rdalpha - dalpha_dx*dnorm_Rdalpha_dy) / (norm_Rdalpha**2)
        
        # Compute gradients of vector field components
        du_x_dx = dG_dx*(dP_dx/norm_dP) + G*de1x_dx + dH_dx*(-dalpha_dy/norm_Rdalpha) + H*de2x_dx
        du_x_dy = dG_dy*(dP_dx/norm_dP) + G*de1x_dy + dH_dy*(-dalpha_dy/norm_Rdalpha) + H*de2x_dy
        du_y_dx = dG_dx*(dP_dy/norm_dP) + G*de1y_dx + dH_dx*(dalpha_dx/norm_Rdalpha) + H*de2y_dx
        du_y_dy = dG_dy*(dP_dy/norm_dP) + G*de1y_dy + dH_dy*(dalpha_dx/norm_Rdalpha) + H*de2y_dy
        
        delta_ux = np.array([du_x_dx, du_x_dy])
        delta_uy = np.array([du_y_dx, du_y_dy])
        
        return u_x, u_y, delta_ux, delta_uy
    
    def calculate_heading_cmd(self, vector, delta_ux, delta_uy, current_heading, current_velocity, tao_heading=28.0, kp=0.18):
        current_velocity = current_velocity/self.scale_factor
        p_dot = current_velocity * np.array([np.cos(current_heading), np.sin(current_heading)])
        delta_desired_heading = (delta_uy * vector[0] - delta_ux * vector[1]) @ p_dot
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
                                    
                current_vector, delta_ux, delta_uy = self.generate_vector_field(curr_position)
                heading_command = self.calculate_heading_cmd(current_vector, delta_ux, delta_uy, curr_heading, self.fixed_wing.get_velocity(), self.tau_theta)
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
            if save_data:
                np.save(os.path.join(save_folder, "poses.npy"), np.array(save_poses))
                np.save(os.path.join(save_folder, "heading_commands.npy"), np.array(save_heading_commands))
                print(f"Data saved to {save_folder}")
            self.data_sender.close()
