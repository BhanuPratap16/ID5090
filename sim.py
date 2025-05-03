import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Arrow, Polygon
from dynamics_2 import LearnableRoboatDynamicsNumpy

# Import RVO functions
from RVO import RVO_update, in_between, intersect, distance


class Agent:
    def __init__(self, device, agent_cfg):
        self.device = device
        self.initial_pose = agent_cfg['initial_pose']
        self.initial_goal = agent_cfg['initial_goal']
        self.height = 0.075
        self.state = np.array([self.initial_pose[0], self.initial_pose[1], self.initial_pose[2], 0, 0, 0])
        self.pos = np.array([self.state[0], self.state[1], 0.075])
        self.rot = np.array([np.pi, 0, self.state[2]])
        self.lin_vel = np.array([self.state[3], self.state[4], 0])
        self.ang_vel = np.array([0, 0, self.state[5]])
        self.steering = 0.0
        self.psi_des = 0.0
        self.length = 1.0 
        self.y_p_int = 0.0
        
    def get_state(self):
        return self.state

    def update_state(self, state):
        self.state = state
        self.pos = np.array([state[0], state[1], 0.075])
        self.rot = np.array([np.pi, 0, state[2]])
        self.lin_vel = np.array([state[3], state[4], 0])
        self.ang_vel = np.array([0, 0, state[5]])


        
class Simulator:
    def __init__(self, cfg, use_matplotlib=True):
        self.use_matplotlib = use_matplotlib

        self.cfg = cfg
        self.dt = 0.1
        self.dynamics = LearnableRoboatDynamicsNumpy(cfg)
        self._agents = {}
        self.terminateds = {}
        self.collisions = {}

        if self.use_matplotlib:
                self.fig, self.ax = plt.subplots(figsize=(10, 10))
                self.ax.set_xlim(-5, 60)
                self.ax.set_ylim(0, 60)
                self.ax.set_aspect('equal')
                self.ax.grid(True)
                self.ax.set_title('Multi-Agent MPPI Collision Avoidance Simulation')
                self.ax.set_xlabel('X (m)')
                self.ax.set_ylabel('Y (m)')
                
                # Store visualization objects
                self.agent_circles = {}
                self.agent_arrows = {}
                self.agent_labels = {}
                self.trajectory_lines = {}
                self.goal_markers = {}
                self.start_markers = {}
                
                # Generate colors using HSV color space for better distinction
                def generate_colors(n):
                    colors = {}
                    for i in range(n):
                        hue = i / n
                        # Convert HSV to RGB (using saturation=0.7, value=0.9 for good visibility)
                        rgb = plt.cm.hsv(hue)
                        colors[f'agent{i}'] = rgb
                    return colors
                
                # Get number of agents from config
                num_agents = len(cfg['agents'])
                self.agent_colors = generate_colors(num_agents)
                
                # Define marker styles
                self.goal_marker = '*'    # Star for goals
                self.start_marker = 'o'   # Circle for start positions

        if cfg is not None:
            # Load agents from config
            # if 'agents' in cfg:
            #     for agent_id, agent_info in cfg['agents'].items():
            #         self._agents[agent_id] = self.load_agent(agent_id, agent_info, 'agent')
            
            # Load pursuers from config
            for agent_id, agent_cfg in cfg['agents'].items():
                self._agents[agent_id] =  Agent(cfg['device'], agent_cfg)
                self.load_agent(agent_id, agent_cfg)
                self.terminateds[agent_id] = False
                self.collisions[agent_id] = False
  

    def eul_to_rotm(self, eul, order='ZYX', deg=False):
        if deg:
            eul = np.radians(eul)
        
        phi, theta, psi = eul  # Roll, Pitch, Yaw

        # Rotation about Z-axis (Yaw)
        R_z = np.array([
            [np.cos(psi), -np.sin(psi), 0],
            [np.sin(psi), np.cos(psi), 0],
            [0, 0, 1]
        ])

        # Rotation about Y-axis (Pitch)
        R_y = np.array([
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)]
        ])

        # Rotation about X-axis (Roll)
        R_x = np.array([
            [1, 0, 0],
            [0, np.cos(phi), -np.sin(phi)],
            [0, np.sin(phi), np.cos(phi)]
        ])

        # Combined rotation matrix: ZYX order
        rotm = R_z @ R_y @ R_x

        return rotm

    def ilos_guidance(self, agent_id):
        agent = self._agents[agent_id]
        
        # Guidance mechanism
        # gwp_ned = self.eul_to_rotm([
        #         np.pi, 0, -np.pi/2
        #         ])@np.array([agent.initial_goal[0], agent.initial_goal[1], 0])

        # cwp_ned = self.eul_to_rotm([
        #         np.pi, 0, -np.pi/2
        #         ])@np.array([agent.initial_pose[0], agent.initial_pose[1], 0])
        
        # curr_pos_ned = self.eul_to_rotm([
        #         np.pi, 0, -np.pi/2
        #         ])@np.array([agent.state[0], agent.state[1], 0])
        
        # current_waypoint = cwp_ned
        # goal_waypoint = gwp_ned
        length = agent.length

        current_waypoint = np.array([agent.state[0], agent.state[1]])
        goal_waypoint = np.array([agent.initial_goal[0], agent.initial_goal[1]])
        current_pose = np.array([agent.state[0], agent.state[1], agent.state[2]])
        
        # Calculate path angle
        pi_p = np.arctan2(goal_waypoint[1] - current_waypoint[1], 
                        goal_waypoint[0] - current_waypoint[0])

        # Rotation matrix from NED to path-tangential reference frame
        R_n_p = np.array([[np.cos(pi_p), -np.sin(pi_p)], [np.sin(pi_p), np.cos(pi_p)]])
        
        # Normalize positions by vessel length
        x_n_i = np.array(current_waypoint[0:2]) / length
        x_g_i = np.array(goal_waypoint[0:2]) / length
        x_n = np.array(current_pose[0:2])

        # Along track distance of goal
        xy_g_e = R_n_p.T @ (x_g_i - x_n_i)
        x_g_e = xy_g_e[0]

        # Along and cross track distance of ship
        xy_p_e = R_n_p.T @ (x_n - x_n_i)
        x_p_e = xy_p_e[0]
        y_p_e = xy_p_e[1]

        # Look ahead distance and integral gain
        Delta = 3.0
        kappa = 0.05

        # Integration time step
        dt = 0.1

        # Update the integral term (for ILOS guidance)
        yd_p_int = Delta * y_p_e / (Delta**2 + (y_p_e + kappa * agent.y_p_int)**2)
        agent.y_p_int = agent.y_p_int + dt * yd_p_int

        # Compute the desired heading angle
        if x_p_e < x_g_e:
            psi_des = pi_p - np.arctan(y_p_e / Delta + kappa * agent.y_p_int / Delta)
        else:
            psi_des = pi_p - np.pi + np.arctan(y_p_e / Delta + kappa * agent.y_p_int / Delta)

        
        # Calculate heading error
        current_heading = agent.state[2]
        heading_error = psi_des - current_heading
        heading_error = 5*np.arctan2(np.sin(heading_error), np.cos(heading_error))  # Normalize to [-pi, pi]
        
        # Convert heading error to steering command (range -1 to 1)
        steering = np.clip(heading_error / np.pi, -1.0, 1.0)
        
        return steering
    
    def rvo_guidance(self, current_agent, agents_dict, goal_position,Rf=0.5,max_speed=1.0):
        """
        Compute steering command using Reciprocal Velocity Obstacles (RVO)
        
        Args:
            current_agent: The agent for which to compute guidance
            agents_dict: Dictionary of agents to avoid
            goal_position: Target position [x, y]
            max_speed: Maximum speed of the agent
            
        Returns:
            steering: Steering command in range [-1, 1]
        """
        # Create workspace model
        ws_model = {
            'robot_radius': Rf,  # Robot radius
            'circular_obstacles': []  # No static obstacles for now
        }
        
        # Extract current agent state
        agent_state = current_agent.get_state()
        x, y, theta = agent_state[0], agent_state[1], agent_state[2]
        u, v = agent_state[3], agent_state[4]
        
        # Convert to RVO format
        X = [[x, y]]  # Position of current agent
        V = [[u, v]]  # Velocity of current agent
        V_max = [max_speed]  # Max velocity
        goal_list = [[goal_position[0], goal_position[1]]]  # Goal position
        
        # Add other agents to the simulation
        for other_id, other_agent in agents_dict.items():
            if other_agent is current_agent:
                continue  # Skip self
            
            # Other agent position and velocity
            other_state = other_agent.get_state()
            X.append([other_state[0], other_state[1]])
            V.append([other_state[3], other_state[4]])
        
        # Compute desired velocity (towards goal)
        V_des = []
        for i in range(len(X)):
            if i == 0:  # Only compute for current agent
                dif_x = [goal_list[0][k] - X[0][k] for k in range(2)]
                norm = distance(dif_x, [0, 0])
                if norm < 0.1:  # Very close to goal
                    V_des.append([0, 0])
                else:
                    norm_dif_x = [dif_x[k] * V_max[0] / norm for k in range(2)]
                    V_des.append(norm_dif_x[:])
            else:
                # For other agents, just use their current velocity as desired
                V_des.append(V[i][:])
        
        # Compute optimal velocity using RVO with COLREGS rules
        V_opt = RVO_update(X, V_des, V, ws_model, time_horizon=4.0, use_colregs=True)
        
        # Extract optimal velocity for current agent
        v_opt = V_opt[0]
        
        # Convert optimal velocity to steering command
        desired_heading = np.arctan2(v_opt[1], v_opt[0])
        heading_error = desired_heading - theta
        
        # Normalize to [-pi, pi]
        heading_error = 5*np.arctan2(np.sin(heading_error), np.cos(heading_error))
        
        # Convert to steering command [-1, 1]
        steering = np.clip(heading_error, -1.0, 1.0)
        
        return steering

    def find_best_velocity(self, pA, v_des, v_current, RVO_BA_all, max_speed):
        """Find the best velocity that avoids collisions"""
        # Try the desired velocity first
        if self.is_valid_velocity(pA, v_des, RVO_BA_all):
            return v_des
        
        # Sample velocities
        best_vel = v_current
        min_cost = float('inf')
        
        # Sample velocities in a grid
        for speed in np.linspace(0.1, max_speed, 5):
            for angle in np.linspace(0, 2*np.pi, 16):
                v_sample = [speed * np.cos(angle), speed * np.sin(angle)]
                
                if self.is_valid_velocity(pA, v_sample, RVO_BA_all):
                    # Valid velocity - compute cost (distance from desired velocity)
                    cost = np.sqrt((v_sample[0] - v_des[0])**2 + (v_sample[1] - v_des[1])**2)
                    
                    if cost < min_cost:
                        min_cost = cost
                        best_vel = v_sample
        
        return best_vel

    def is_valid_velocity(self, pA, v, RVO_BA_all):
        """Check if velocity v is valid (outside all RVO cones)"""
        for RVO_BA in RVO_BA_all:
            p_0 = RVO_BA[0]  # Translated center point
            left = RVO_BA[1]  # Left leg of cone
            right = RVO_BA[2]  # Right leg of cone
            
            # Vector from translated center to new position
            dif = [v[0] + pA[0] - p_0[0], v[1] + pA[1] - p_0[1]]
            
            # Angles
            theta_dif = np.arctan2(dif[1], dif[0])
            theta_right = np.arctan2(right[1], right[0])
            theta_left = np.arctan2(left[1], left[0])
            
            # Check if velocity is inside the RVO cone
            if self.in_between_angles(theta_right, theta_dif, theta_left):
                return False
        
        return True

    def in_between_angles(self, theta_right, theta_dif, theta_left):
        """Check if theta_dif is between theta_right and theta_left"""
        if abs(theta_right - theta_left) <= np.pi:
            return theta_right <= theta_dif <= theta_left
        else:
            if (theta_left < 0) and (theta_right > 0):
                theta_left += 2*np.pi
                if theta_dif < 0:
                    theta_dif += 2*np.pi
                return theta_right <= theta_dif <= theta_left
            if (theta_left > 0) and (theta_right < 0):
                theta_right += 2*np.pi
                if theta_dif < 0:
                    theta_dif += 2*np.pi
                return theta_left <= theta_dif <= theta_right
            return False

    def step(self,noise,epsilon,Rf):
        observation = []
        for agent_id in self._agents:
            agent = self._agents[agent_id]
            # action = np.array([1.0])
            # steering = self.vo_guidance(agent_id, self._agents)
            steering_val = 0.0
            vo_flag = False
            steering = self.rvo_guidance(agent, self._agents, agent.initial_goal,Rf) + np.random.uniform(-noise,noise) +epsilon

            # for other_agent_id, other_agent in self._agents.items():
            #     if agent_id == other_agent_id:
            #         continue
            #     else:
            #         other_state = other_agent.get_state()
            #         other_x, other_y = other_state[0], other_state[1]
            #         x, y = agent.state[0], agent.state[1]
                    
            #         # Vector from current agent to other agent
            #         relative_pos = np.array([other_x - x, other_y - y])
            #         distance = np.linalg.norm(relative_pos)
                    
            #         # If agents are close enough to consider collision avoidance
            #         if distance < 9.0:
            #             steering_val += self.rvo_guidance(agent, self._agents, agent.initial_goal)
            #             vo_flag = True
            #         # else:
            #         #     steering = self.ilos_guidance(agent_id)
            # if vo_flag:
            #     goal_weight = 0.1
            #     avoidance_weight = 5.0
                
            #     # Combine steering commands
            #     steering = avoidance_weight * steering_val  + (np.random.normal(0, 0.01))
            #     steering = np.clip(steering, -1.0, 1.0)
            # else:
            #     steering = self.ilos_guidance(agent_id)

            new_state = self.dynamics.step(agent.state, steering)
            if not self.terminateds[agent_id]:
                agent.update_state(new_state)
            observation.append(new_state)
        self._update_matplotlib_visualization()

        for i, agent_name in enumerate(self._agents.keys()):
            if not self.terminateds[agent_name]:
                terminated = np.linalg.norm(observation[i][0:2] - np.array(self._agents[agent_name].initial_goal[0:2])) < 1.0
                self.terminateds[agent_name] = terminated
                for j, other_agent_name in enumerate(self._agents.keys()):
                    if j != i:
                        collision = np.linalg.norm(observation[i][0:2] - observation[j][0:2] ) < 0.5
                        if collision:
                            print(f"collision between {agent_name} and {other_agent_name}")
                            self.collisions[agent_name] = True
                            break

        return np.array(observation)

    def load_agent(self, agent_id, agent_info, agent_type='agent', agent_status='virtual'):
        """Load an agent into the simulation"""
        if self.use_matplotlib:
            initial_pose = agent_info['initial_pose']
            initial_goal = agent_info['initial_goal']
            
            # Get color for this specific agent
            color = self.agent_colors.get(agent_id, 'blue')
            
            # Add agent circle with type-based label
            label_text = f"{agent_type.capitalize()}_{agent_id}_{agent_status.capitalize()}"
            circle = Circle((initial_pose[0], initial_pose[1]), 0.01, color=color, alpha=0.1)
            self.ax.add_patch(circle)
            self.agent_circles[agent_id] = circle
            
            # Add direction arrow
            dx = np.cos(initial_pose[2])
            dy = np.sin(initial_pose[2])
            arrow = Arrow(initial_pose[0], initial_pose[1], dx, dy, width=0.3, color=color)
            # self.ax.add_patch(arrow)
            self.agent_arrows[agent_id] = arrow
            
            # # Add agent label
            # label = self.ax.text(initial_pose[0], initial_pose[1] + 0.7, label_text, 
            #             horizontalalignment='center', verticalalignment='center',
            #             color=color)
            # # self.agent_labels[agent_id] = label
            
            # Add goal marker with same color as agent
            goal_marker = self.ax.plot(initial_goal[0], initial_goal[1], 
                                     marker=self.goal_marker, color=color, 
                                     markersize=15, alpha=0.7, 
                                     label=f'{label_text} Goal')[0]
            self.goal_markers[agent_id] = goal_marker
            
            # Add start position marker with same color as agent
            start_marker = self.ax.plot(initial_pose[0], initial_pose[1], 
                                      marker=self.start_marker, color=color, 
                                      markersize=10, alpha=0.5, 
                                      label=f'{label_text} Start')[0]
            self.start_markers[agent_id] = start_marker
            
            # Update legend with grouped items
            handles = []
            labels = []
            for a_id in self.agent_circles.keys():
                agent_color = self.agent_colors[a_id]
                agent_type = self.cfg['agents'][a_id]['type']
                # Add agent circle
                handles.append(Circle((0, 0), 0.5, color=agent_color, alpha=0.7))
                # labels.append(f"{agent_type.capitalize()} {a_id}")
                # Add goal marker
                handles.append(plt.Line2D([0], [0], marker=self.goal_marker, color=agent_color, 
                                       linestyle='None', markersize=10, alpha=0.7))
                labels.append(f"{agent_type.capitalize()} {a_id} Goal")
                # Add start marker
                handles.append(plt.Line2D([0], [0], marker=self.start_marker, color=agent_color, 
                                       linestyle='None', markersize=8, alpha=0.5))
                labels.append(f"{agent_type.capitalize()} {a_id} Start")
            
            self.ax.legend(handles, labels, loc='center left', bbox_to_anchor=(1.05, 0.5))


    def _update_matplotlib_visualization(self):
        '''
        Update the matplotlib visualization with current agent states and obstacles
        '''
        # Import required modules
        from matplotlib.patches import Polygon
        
        # Import ShipPatch if not already imported
        if not hasattr(self, 'ShipPatch'):
            try:
                from ship_geom import ShipPatch
                self.ShipPatch = ShipPatch
            except ImportError:
                # Create a simple fallback if import fails
                class FallbackShipPatch(Polygon):
                    def __init__(self, xy=(0, 0), angle=0, size=1.0, **kwargs):
                        # Fallback ship geometry
                        x_ship = np.array([-0.5, -0.5, 0.25, 0.5, 0.25, -0.5, -0.5, 0.5, 0.25, 0, 0])
                        y_ship = 0.25 * np.array([-1, 1, 1, 0, -1, -1, 0, 0, 1, 1, -1])
                        
                        # Scale, rotate, and translate
                        x_ship = x_ship * size
                        y_ship = y_ship * size
                        x_rotated = x_ship * np.cos(angle) - y_ship * np.sin(angle)
                        y_rotated = x_ship * np.sin(angle) + y_ship * np.cos(angle)
                        x_final = x_rotated + xy[0]
                        y_final = y_rotated + xy[1]
                        
                        vertices = np.column_stack([x_final, y_final])
                        super().__init__(vertices, closed=True, **kwargs)
                        self.center = xy
                    
                    def update_position(self, xy, angle=None):
                        if angle is not None:
                            self.angle = angle
                        
                        # Fallback ship geometry
                        x_ship = np.array([-0.5, -0.5, 0.25, 0.5, 0.25, -0.5, -0.5, 0.5, 0.25, 0, 0])
                        y_ship = 0.25 * np.array([-1, 1, 1, 0, -1, -1, 0, 0, 1, 1, -1])
                        
                        # Scale, rotate, and translate
                        x_ship = x_ship * self.size
                        y_ship = y_ship * self.size
                        x_rotated = x_ship * np.cos(self.angle) - y_ship * np.sin(self.angle)
                        y_rotated = x_ship * np.sin(self.angle) + y_ship * np.cos(self.angle)
                        x_final = x_rotated + xy[0]
                        y_final = y_rotated + xy[1]
                        
                        vertices = np.column_stack([x_final, y_final])
                        self.set_xy(vertices)
                        self.center = xy
                
                self.ShipPatch = FallbackShipPatch
                print("Using fallback ship patch")
        
        # Draw obstacle polygons if they don't exist yet
        if not hasattr(self, 'obstacle_patches'):
            self.obstacle_patches = []
            if 'static_obstacles' in self.cfg:
                for obstacle_name, obstacle_info in self.cfg['static_obstacles'].items():
                    # Convert the flat list of coordinates to a list of (x,y) points
                    vertices_flat = obstacle_info['vertices']
                    vertices = []
                    for i in range(0, len(vertices_flat), 2):
                        if i+1 < len(vertices_flat):  # Make sure we have both x and y
                            vertices.append([vertices_flat[i], vertices_flat[i+1]])
                    
                    # Create polygon patch
                    obstacle_patch = Polygon(vertices, closed=True, 
                                            facecolor='gray', alpha=0.7, 
                                            edgecolor='black', linewidth=2)
                    self.ax.add_patch(obstacle_patch)
                    self.obstacle_patches.append(obstacle_patch)
        
        # Initialize path history dictionary if it doesn't exist
        if not hasattr(self, 'path_history'):
            self.path_history = {}
            self.path_lines = {}
            self.start_goal_lines = {}
            self.ship_patches = {}  # Store ship patches
        
        # Update agent visualizations
        for agent_id, agent in self._agents.items():
            state = agent.get_state()
            x, y, theta = state[0].item(), state[1].item(), state[2].item()
            
            # Get color for this specific agent
            color = self.agent_colors[agent_id]
            agent_type = self.cfg['agents'][agent_id]['type']
            label_text = f"{agent_type.capitalize()} {agent_id}"
            
            # Update or create ship patch
            if agent_id in self.ship_patches:
                # Update existing ship patch
                self.ship_patches[agent_id].update_position((x, y), angle=theta)
            else:
                # Create new ship patch
                ship = self.ShipPatch(
                    xy=(x, y),
                    angle=theta,
                    size=1.0,  # Adjust size as needed
                    facecolor=color,
                    alpha=0.7,
                    edgecolor='black',
                    linewidth=1
                )
                self.ax.add_patch(ship)
                self.ship_patches[agent_id] = ship
            
            # # Update text label position
            # if agent_id in self.agent_labels:
            #     self.agent_labels[agent_id].set_position((x, y + 0.7))
            # else:
            #     label = self.ax.text(x, y + 0.7, label_text, 
            #                 horizontalalignment='center', verticalalignment='center',
            #                 color=color)
            #     self.agent_labels[agent_id] = label
            
            # Draw straight line from start to goal (if not already drawn)
            if agent_id not in self.start_goal_lines:
                start_x = self.cfg['agents'][agent_id]['initial_pose'][0]
                start_y = self.cfg['agents'][agent_id]['initial_pose'][1]
                goal_x = self.cfg['agents'][agent_id]['initial_goal'][0]
                goal_y = self.cfg['agents'][agent_id]['initial_goal'][1]
                
                line, = self.ax.plot([start_x, goal_x], [start_y, goal_y], '--', 
                                   color=color, alpha=0.3, linewidth=1)
                self.start_goal_lines[agent_id] = line
            
            # Update path history
            if agent_id not in self.path_history:
                self.path_history[agent_id] = []
            
            self.path_history[agent_id].append((x, y))
            
            # Plot the path traced by the agent
            xs = [point[0] for point in self.path_history[agent_id]]
            ys = [point[1] for point in self.path_history[agent_id]]
            
            if agent_id in self.path_lines:
                self.path_lines[agent_id].set_data(xs, ys)
            else:
                line, = self.ax.plot(xs, ys, '-', color=color, linewidth=1.5, alpha=0.7)
                self.path_lines[agent_id] = line
        
        # Refresh the plot
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
