import yaml
import numpy as np
import argparse
import os

def generate_scenario(num_agents, radius=25, output_file="scenerios/scenario.yml"):
    """
    Generate a YAML scenario file with agents positioned in a circle.
    
    Parameters:
    -----------
    num_agents : int
        Number of agents to position in a circle
    radius : float
        Radius of the circle where agents are positioned
    output_file : str
        Output YAML file name
    """
    # Calculate angular spacing between agents
    angle_increment = 2 * np.pi / num_agents
    
    agents = {}
    
    for i in range(num_agents):
        # Calculate angle for this agent
        angle = i * angle_increment
        
        # Calculate position on circle
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        
        # Calculate goal (opposite side of circle)
        goal_x = -x
        goal_y = -y
        
        # Calculate heading (pointing toward center)
        heading = angle + np.pi  # Pointing toward center
        
        # Add agent to dictionary
        agents[f"agent{i}"] = {
            "initial_pose": [float(x), float(y), float(heading)],
            "initial_goal": [float(goal_x), float(goal_y), 0.0],
            "type": "agent",
            "urdf": "aritra.urdf"
        }
    
    # Create the full scenario dictionary
    scenario = {
        "agents": agents,
        "control": {
            "u_max": [6, 6],
            "u_min": [-6, -6]
        },
        "device": "cuda",
        "dt": 0.5,
        "objective": {
            "max_speed": 0.5
        },
        "simulator": {
            "mode": "thrust",
            "render": True,
            "steps": 700,
            "urdf": "quarter_roboat.urdf"
        }
    }
    
    # Write to YAML file
    with open(output_file, 'w') as file:
        yaml.dump(scenario, file, default_flow_style=False, sort_keys=False)
    
    print(f"Scenario with {num_agents} agents generated and saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Generate a scenario YAML file with agents in a circle')
    parser.add_argument('-n', '--num_agents', type=int, default=8, help='Number of agents (default: 8)')
    parser.add_argument('-r', '--radius', type=float, default=15.0, help='Circle radius (default: 15.0)')
    parser.add_argument('-o', '--output', type=str, default='scenario.yml', help='Output file name (default: scenario.yml)')
    args = parser.parse_args()
    
    generate_scenario(args.num_agents, args.radius, args.output)

if __name__ == "__main__":
    main()