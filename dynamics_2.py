import numpy as np

class UnderactuatedQuarterRoboatDynamics:
    def __init__(self, cfg) -> None:
        self.aa = 0.45  # Distance between thrusters (half-width)
        self.bb = 0.90  # Not used in differential thrust

        # Inertia parameters
        self.m11 = 12
        self.m22 = 24
        self.m33 = 1.5

        # Damping parameters
        self.d11 = 6
        self.d22 = 8
        self.d33 = 1.35

        self.cfg = cfg
        self.dt = cfg["dt"]

        # Dynamics matrices
        self.D = np.array([
            [self.d11, 0, 0],
            [0, self.d22, 0],
            [0, 0, self.d33]
        ])

        self.M = np.array([
            [self.m11, 0, 0],
            [0, self.m22, 0],
            [0, 0, self.m33]
        ])

        # Modified B matrix for differential thrust (2 inputs)
        self.B = np.array([
            [1, 1],
            [0, 0],
            [self.aa / 2, -self.aa / 2]
        ])

        # Inverse of inertia matrix
        self.Minv = np.linalg.inv(self.M)

    def rot_matrix(self, heading):
        cos_h = np.cos(heading)
        sin_h = np.sin(heading)
        return np.array([
            [cos_h, -sin_h, 0],
            [sin_h,  cos_h, 0],
            [0,      0,     1]
        ])

    def coriolis(self, vel):
        u, v, r = vel
        C = np.array([
            [0, 0, -self.m22 * v],
            [0, 0,  self.m11 * u],
            [self.m22 * v, -self.m11 * u, 0]
        ])
        return C

    def step(self, state: np.ndarray, action: np.ndarray, t: int = -1):
        # Extract pose and velocities (ENU convention)
        x, y, psi = state[0:3]
        u, v, r = state[3:6]

        # Convert to NED and body frame conventions
        pose = np.array([y, x, np.pi / 2 - psi])
        vel = np.array([v, u, -r])

        # Transform velocity to body frame
        R = self.rot_matrix(-pose[2])
        vel_body = R @ vel

        # Apply safety clipping to prevent numerical issues
        vel_body = np.clip(vel_body, -100, 100)
        
        # Compute coriolis and damping forces
        C = self.coriolis(vel_body)
        damping = self.D @ vel_body

        # Control input
        thrust = self.B @ action
        
        # Clip thrust to reasonable values
        thrust = np.clip(thrust, -1000, 1000)
        
        # Compute net force with safety checks
        coriolis_term = C @ vel_body
        coriolis_term = np.clip(coriolis_term, -1000, 1000)
        
        damping_term = np.clip(damping, -1000, 1000)
        
        net_force = thrust - coriolis_term - damping_term
        net_force = np.clip(net_force, -1000, 1000)
        
        # Dynamics update with safety checks
        acc_body = self.Minv @ net_force
        acc_body = np.clip(acc_body, -50, 50)  # Limit acceleration to reasonable values
        
        new_vel_body = vel_body + acc_body * self.dt
        new_vel_body = np.clip(new_vel_body, -100, 100)  # Prevent velocity explosion

        # Transform back to inertial frame
        R_world = self.rot_matrix(pose[2])
        vel_world = R_world @ new_vel_body

        # Pose update
        pose_update = vel_world * self.dt
        pose_update = np.clip(pose_update, -1, 1)  # Limit position change per step
        pose += pose_update

        # Convert back to ENU
        new_pose = np.array([pose[1], pose[0], np.pi / 2 - pose[2]])
        new_vel = np.array([vel_world[1], vel_world[0], -vel_world[2]])

        # New state
        new_state = np.concatenate([new_pose, new_vel])

        return new_state

import numpy as np

class LearnableRoboatDynamicsNumpy:
    def __init__(self, cfg):
        self.dt = cfg["dt"]
        self.m_rb = 18.6029
        self.Iz_rb = 10.6863
        self.X_du = -0.2183
        self.Y_dv = -1.2848
        self.N_dr = -0.0358
        self.d11 = 1.0786
        self.d22 = 2.7058
        self.d33 = 0.0772

        self.aa = 0.4500

        self.thrust_coef = 1.7862
        self.steer_coef = 0.2521

    def step(self, state: np.ndarray, action: np.ndarray, t: int = -1) -> tuple:
        """
        Args:
            state: np.ndarray of shape (6,) -> [x, y, theta, u, v, r]
            action: np.ndarray of shape (2,) -> [left_force, right_force]
        Returns:
            new_state: np.ndarray of shape (6,)
            action: unchanged
        """
        x, y, theta, u, v, r = state
        # left_force, right_force = action
        steering = action

        # throttle = throttle * self.thrust_coef
        # steering = steering * self.steer_coef

        # Zero action decay
        # if abs(throttle) < 1e-6 and abs(steering) < 1e-6:
        #     decay = np.exp(-10.0 * self.dt)
        #     u *= decay
        #     v *= decay
        #     r *= decay

        #     if abs(u) < 1e-3: u = 0.0
        #     if abs(v) < 1e-3: v = 0.0
        #     if abs(r) < 1e-3: r = 0.0
        # else:
        # throttle = 0.5
        # steering = (steering - throttle) / self.aa

        # u_dot = (throttle * self.thrust_coef - self.d11 * u) / (self.m_rb - self.X_du)
        # v_dot = (-self.d22 * v) / (self.m_rb - self.Y_dv)
        # r_dot = (steering * self.steer_coef - self.d33 * r) / (self.Iz_rb - self.N_dr)
        T = 0.7
        K = 0.7
        # steering = 1.0
        r_dot = -r/T + K*steering/T

        # new_u = u + u_dot * self.dt
        # new_v = v + v_dot * self.dt
        # new_r = r + r_dot * self.dt
        U = 1
        # new_u = u + U * np.cos(theta) * self.dt
        # new_v = v + U * np.sin(theta) * self.dt
        new_r = r + r_dot * self.dt


        # Cap surge/sway velocity
        # vel_mag = np.sqrt(u ** 2 + v ** 2)
        # max_vel = 2.0
        # if vel_mag > max_vel:
        #     scale = max_vel / vel_mag
        #     u *= scale
        #     v *= scale

        
        # max_u = 0.7
        # if new_u > max_u:
        #     new_u = max_u
        # max_v = 0.05
        # if new_v > max_v:
        #     v = max_v
        # max_r = 0.01
        # if new_r > max_r:
        #     new_r = max_r

        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        # vx = new_u * cos_theta - new_v * sin_theta
        # vy = new_u * sin_theta + new_v * cos_theta

        # new_x = x + vx * self.dt
        # new_y = y + vy * self.dt  
        new_x = x + U * np.cos(theta) * self.dt
        new_y = y + U * np.sin(theta) * self.dt
        v_x = (new_x - x) / self.dt
        v_y = (new_y - y) / self.dt

        new_u = U * np.cos(theta) * self.dt #v_x * cos_theta + v_y * sin_theta
        new_v = U * np.sin(theta) * self.dt  #-v_x * sin_theta + v_y * cos_theta

        new_theta = theta + new_r * self.dt
        new_theta = (new_theta + np.pi) % (2 * np.pi) - np.pi

      

        new_state = np.array([new_x, new_y, new_theta, new_u, new_v, new_r])
        return new_state
