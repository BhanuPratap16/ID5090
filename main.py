import numpy as np 
import matplotlib.pyplot as plt
from sim import Simulator
import yaml
import os
from tqdm import tqdm
import math
import traceback
import pandas as pd
import copy
from datetime import datetime
import io
from PIL import Image

config_path = f"scenerios/scenerio_20.yml"
noise=float(input("Enter the sigma: "))
epsilon=float(input("Enter the epsilon:"))
count=0
Rf=float(input("Enter the radius of influence:"))
def arrange_agents_in_polygon(config):
    """
    Arrange agents in an n-sided polygon with diagonally opposite points as goals.
    """
    num_agents = len(config['agents'])
    radius = 30.0  # Radius of the polygon
    center = [30.0, 30.0]  # Center of the polygon
    
    # Calculate positions around the polygon
    for i, agent_id in enumerate(config['agents']):
        angle = 2 * 3.14159 * i / num_agents  # Using regular float math
        x = center[0] + radius * math.cos(angle)
        y = center[1] + radius * math.sin(angle)
        
        # Set initial position
        config['agents'][agent_id]['initial_pose'][0] = float(x)
        config['agents'][agent_id]['initial_pose'][1] = float(y)
        config['agents'][agent_id]['initial_pose'][2] = float(-angle) + np.random.rand()*0.1 # Face toward center
        
        # Set goal as the diagonally opposite point
        opposite_angle = angle + 3.14159
        goal_x = center[0] + radius * math.cos(opposite_angle)
        goal_y = center[1] + radius * math.sin(opposite_angle)
        
        config['agents'][agent_id]['initial_goal'][0] = float(goal_x)
        config['agents'][agent_id]['initial_goal'][1] = float(goal_y)
        config['agents'][agent_id]['initial_goal'][2] = 0.0

def arrange_agents_randomly(config):
    """
    Arrange agents randomly in the arena, avoiding static obstacles and maintaining
    minimum clearance between agents and obstacles.
    """
    # Arena dimensions
    arena_min_x, arena_max_x = 0.0, 30.0
    arena_min_y, arena_max_y = 0.0, 30.0
    
    # Minimum clearance from obstacles and other agents
    min_obstacle_clearance = 3.0
    min_agent_clearance = 2.5
    
    # Extract obstacle vertices
    obstacles = []
    if 'static_obstacles' in config:
        for obstacle_name, obstacle_info in config['static_obstacles'].items():
            vertices_flat = obstacle_info['vertices']
            vertices = []
            for i in range(0, len(vertices_flat), 2):
                if i+1 < len(vertices_flat):
                    vertices.append([vertices_flat[i], vertices_flat[i+1]])
            obstacles.append(vertices)
    
    # Function to check if a point is too close to obstacles
    def is_too_close_to_obstacles(x, y):
        for obstacle_vertices in obstacles:
            # Simple check: compute distance to each vertex and to each edge
            for i in range(len(obstacle_vertices)):
                # Distance to vertex
                vx, vy = obstacle_vertices[i]
                dist_to_vertex = np.sqrt((x - vx)**2 + (y - vy)**2)
                if dist_to_vertex < min_obstacle_clearance:
                    return True
                
                # Distance to edge (simplified - just checking a few points along the edge)
                next_i = (i + 1) % len(obstacle_vertices)
                nx, ny = obstacle_vertices[next_i]
                
                # Check 5 points along the edge
                for t in np.linspace(0, 1, 5):
                    edge_x = vx + t * (nx - vx)
                    edge_y = vy + t * (ny - vy)
                    dist_to_edge = np.sqrt((x - edge_x)**2 + (y - edge_y)**2)
                    if dist_to_edge < min_obstacle_clearance:
                        return True
        
        return False
    
    # Function to check if a point is too close to other agents
    def is_too_close_to_agents(x, y, placed_agents):
        for agent_pos in placed_agents:
            dist = np.sqrt((x - agent_pos[0])**2 + (y - agent_pos[1])**2)
            if dist < min_agent_clearance:
                return True
        return False
    
    # Place agents randomly
    placed_agents = []
    
    for agent_id in config['agents']:
        # Try to find a valid position (max 100 attempts)
        valid_position_found = False
        for _ in range(100):
            # Generate random position
            x = np.random.uniform(arena_min_x + min_obstacle_clearance, 
                                 arena_max_x - min_obstacle_clearance)
            y = np.random.uniform(arena_min_y + min_obstacle_clearance, 
                                 arena_max_y - min_obstacle_clearance)
            
            # Check if position is valid
            if not is_too_close_to_obstacles(x, y) and not is_too_close_to_agents(x, y, placed_agents):
                valid_position_found = True
                placed_agents.append((x, y))
                
                # Set agent position
                config['agents'][agent_id]['initial_pose'][0] = float(x)
                config['agents'][agent_id]['initial_pose'][1] = float(y)
                
                # Random orientation
                theta = np.random.uniform(0, 2 * np.pi)
                config['agents'][agent_id]['initial_pose'][2] = float(theta)
                
                # Set goal position (also randomly, but far from start)
                while True:
                    goal_x = np.random.uniform(arena_min_x + min_obstacle_clearance, 
                                             arena_max_x - min_obstacle_clearance)
                    goal_y = np.random.uniform(arena_min_y + min_obstacle_clearance, 
                                             arena_max_y - min_obstacle_clearance)
                    
                    # Ensure goal is far enough from start
                    dist_to_start = np.sqrt((goal_x - x)**2 + (goal_y - y)**2)
                    if (dist_to_start > 10.0 and 
                        not is_too_close_to_obstacles(goal_x, goal_y)):
                        break
                
                config['agents'][agent_id]['initial_goal'][0] = float(goal_x)
                config['agents'][agent_id]['initial_goal'][1] = float(goal_y)
                config['agents'][agent_id]['initial_goal'][2] = 0.0
                
                break
        
        if not valid_position_found:
            print(f"Warning: Could not find valid position for {agent_id} after 100 attempts")
            # Place at default position as fallback
            config['agents'][agent_id]['initial_pose'][0] = float(arena_min_x + 5.0)
            config['agents'][agent_id]['initial_pose'][1] = float(arena_min_y + 5.0)
            config['agents'][agent_id]['initial_goal'][0] = float(arena_max_x - 5.0)
            config['agents'][agent_id]['initial_goal'][1] = float(arena_max_y - 5.0)
    
    return config

def scale_agents(config, num_agents):
    """
    Scale the number of agents in the configuration to the specified number.
    If there are fewer agents than requested, add new ones with default settings.
    If there are more agents than requested, remove excess agents.
    
    Args:
        config: The configuration dictionary
        num_agents: The desired number of agents
    
    Returns:
        The updated configuration dictionary
    """
    current_agents = list(config['agents'].keys())
    current_count = len(current_agents)
    
    # If we already have the right number of agents, do nothing
    if current_count == num_agents:
        return config
    
    # If we need to remove agents
    if current_count > num_agents:
        # Remove excess agents (from the end of the list)
        for i in range(num_agents, current_count):
            del config['agents'][current_agents[i]]
        return config
    
    # If we need to add agents
    # Use the first agent as a template for new agents
    template_agent = current_agents[0]
    template_config = config['agents'][template_agent].copy()
    
    # Add new agents
    for i in range(current_count, num_agents):
        new_agent_id = f"agent{i}"
        config['agents'][new_agent_id] = template_config.copy()
        
        # Initialize with default values - these will be properly set by arrange_agents_randomly
        config['agents'][new_agent_id]['initial_pose'] = [0.0, 0.0, 0.0]
        config['agents'][new_agent_id]['initial_goal'] = [0.0, 0.0, 0.0]
    
    return config

# Load configuration
CONFIG = yaml.safe_load(open(config_path))

# Scale to desired number of agents (e.g., 12)
num_agents = int(input("Enter number of agents:"))  # Change this to your desired number
CONFIG = scale_agents(CONFIG, num_agents)


enable_plotting = True  # Set to False to disable plotting and speed up computation

# Arrange agents randomly

# arrange_agents_randomly(CONFIG)
arrange_agents_in_polygon(CONFIG)

sim = Simulator(CONFIG)

# Create a directory for saving data if it doesn't exist
os.makedirs('data', exist_ok=True)

# Initialize a list to store observations at each time step
observations_history = []
time_steps = []
current_time = 0
time_step = 0.1  # Assuming this is your simulation time step

date = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
# Flags for visualization and recording
record_gif = False  # Set to True to save a GIF of the simulation
gif_filename = "simulation.gif"  # Name of the output GIF file
gif_fps = 10  # Frames per second in the output GIF

# Initialize GIF recording if enabled
frames = []
fig = None

# Initialize metrics tracking
metrics = {
    'collision_count': 0,
    'total_collisions': 0,
    'completion_time': 0,
    'completion_rate': 0,
    'avg_path_length': 0,
    'communication_range': 0,
    'num_agents': num_agents,
    'simulation_date': date,
    'agent_metrics': {}
}

# Initialize per-agent metrics
for agent_id in CONFIG['agents']:
    metrics['agent_metrics'][agent_id] = {
        'collisions': 0,
        'path_length': 0,
        'completion_time': None,
        'goal_reached': False
    }

collision_status = {}
agent_paths = {agent_id: [] for agent_id in sim._agents.keys()}
agent_completion_times = {}

end_sim = False
for step in tqdm(range(CONFIG['simulator']['steps'])):
    try:
        observation = sim.step(noise,epsilon,Rf) # shape: (num_agents, num_states) (num_states = 6)
        print(observation.shape)
        
        # Only create and update plots if plotting is enabled
        if enable_plotting or record_gif:
            # Create a figure only once and update it
         
        
            # Save frame for GIF if recording is enabled
            if record_gif:
                buf = io.BytesIO()
                plt.savefig(buf, format='png', dpi=100)
                buf.seek(0)
                # Create a copy of the image before closing the buffer
                img = Image.open(buf)
                img_copy = img.copy()  # Create a copy that doesn't depend on the buffer
                frames.append(img_copy)
                buf.close()
            
            # Update display if plotting is enabled
            if enable_plotting:
                plt.draw()
                plt.pause(0.001)
                
                # Check if window was closed
                if not plt.get_fignums():  # If figure was closed
                    print("Plot window closed by user")
                    end_sim = True
                    break
        
        # Track collisions and update metrics
        for agent_name in sim._agents.keys():
            # Track agent path
            agent_state = sim._agents[agent_name].state
            agent_pos = (agent_state[0], agent_state[1])
            agent_paths[agent_name].append(agent_pos)
            
            # Check for collisions
            if sim.collisions[agent_name]:
                print(f"collision between {agent_name}")
              
                metrics['agent_metrics'][agent_name]['collisions'] = 1
                
                if agent_name in collision_status:
                    collision_status[agent_name].append(1)
                else:
                    collision_status[agent_name] = [1]
            else:
                if agent_name in collision_status:
                    collision_status[agent_name].append(0)
                else:
                    collision_status[agent_name] = [0]
        
        # Check for goal completion
        for agent_name, agent in sim._agents.items():
            if sim.terminateds[agent_name] and not metrics['agent_metrics'][agent_name]['goal_reached'] and not sim.collisions[agent_name]:
                metrics['agent_metrics'][agent_name]['goal_reached'] = True
                metrics['agent_metrics'][agent_name]['completion_time'] = current_time
                agent_completion_times[agent_name] = current_time
                
                # Calculate path length
                path = agent_paths[agent_name]
                path_length = 0
                for i in range(1, len(path)):
                    path_length += np.sqrt((path[i][0] - path[i-1][0])**2 + 
                                          (path[i][1] - path[i-1][1])**2)
                metrics['agent_metrics'][agent_name]['path_length'] = path_length
        
        print(f"collisions: {sim.collisions}")
        
        # Store the current time and observation
        time_steps.append(current_time)
        observations_history.append(observation.copy())
        current_time += time_step

        terminated = sim.terminateds
        if all(terminated.values()):
            print("all agents terminated")
            end_sim = True
            break
        
        print(f"terminated: {terminated}")
        if end_sim:
            break
      
    except Exception as e:
        traceback.print_exc()
        print(e)
        break
    finally:
        # Calculate final metrics
        if len(observations_history) > 0:
            # Calculate completion rate
            completed_agents = sum(1 for agent_id in metrics['agent_metrics'] 
                                 if metrics['agent_metrics'][agent_id]['goal_reached'])
            metrics['completion_rate'] = completed_agents / num_agents
            
            # Calculate average path length for completed agents
            completed_path_lengths = [metrics['agent_metrics'][agent_id]['path_length'] 
                                    for agent_id in metrics['agent_metrics'] 
                                    if metrics['agent_metrics'][agent_id]['goal_reached']]
            if completed_path_lengths:
                metrics['avg_path_length'] = sum(completed_path_lengths) / len(completed_path_lengths)
            metrics['collision_count'] = sum(sim.collisions.values())
            # Calculate total collisions
            metrics['total_collisions'] = metrics['collision_count']/num_agents
            
            # Calculate overall completion time (max of all agents)
            completion_times = [metrics['agent_metrics'][agent_id]['completion_time'] 
                              for agent_id in metrics['agent_metrics'] 
                              if metrics['agent_metrics'][agent_id]['completion_time'] is not None]
            if completion_times:
                metrics['completion_time'] = max(completion_times)
            
            # Get dimensions from the first observation
            num_agents = observations_history[0].shape[0]
            num_states = observations_history[0].shape[1]
            state_names = ['x', 'y', 'theta', 'v_x', 'v_y', 'omega']
            
            # Create column names
            column_names = []
            for agent_id in range(num_agents):
                for state_name in state_names:
                    column_names.append(f'agent_{agent_id}_{state_name}')
                column_names.append(f'agent_{agent_id}_collision')  # Add collision column for each agent
            
            # Convert observations to a DataFrame
            data = np.zeros((len(time_steps), num_agents * (len(state_names) + 1)))  # +1 for collision status
            
            # Get agent names to ensure consistent ordering
            agent_names = list(sim._agents.keys())
            
            for t, obs in enumerate(observations_history):
                # Fill in state data
                for i, agent_name in enumerate(agent_names):
                    # States for this agent (x, y, theta, etc.)
                    start_idx = i * (len(state_names) + 1)
                    data[t, start_idx:start_idx + len(state_names)] = obs[i, :]
                    
                    # Collision status for this agent
                    if agent_name in collision_status and t < len(collision_status[agent_name]):
                        data[t, start_idx + len(state_names)] = collision_status[agent_name][t]
            
            # Create DataFrame with time as first column
            df = pd.DataFrame(data, columns=column_names)
            df.insert(0, 'time', time_steps)
            
            # # Save to CSV
            csv_filename = f'data/simulation_results_{date}.csv'
            df.to_csv(csv_filename, index=False)
            print(f"Saved time-series data to {csv_filename}")
            
            # Convert NumPy values to native Python types
            def convert_to_native_types(obj):
                if isinstance(obj, dict):
                    return {k: convert_to_native_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_to_native_types(item) for item in obj]
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.number):
                    return float(obj) if isinstance(obj, np.floating) else int(obj)
                else:
                    return obj

            # Convert metrics to native Python types
            metrics_native = convert_to_native_types(metrics)

            # Save metrics to YAML
            yaml_filename = f'data/simulation_metrics_{date}.yaml'
            with open(yaml_filename, 'w') as yaml_file:
                yaml.dump(metrics_native, yaml_file, default_flow_style=False)
            print(f"Saved metrics to {yaml_filename}")

# Save the GIF if recording was enabled
if record_gif and frames:
    print(f"Saving GIF to {gif_filename}...")
    try:
        if frames:  # Check if we have any frames
            frames[0].save(
                gif_filename,
                save_all=True,
                append_images=frames[1:],
                optimize=False,
                duration=1000/gif_fps,
                loop=0
            )
            print(f"GIF saved successfully!")
        else:
            print("No frames were captured for the GIF.")
    except Exception as e:
        print(f"Error saving GIF: {e}")

