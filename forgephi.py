import pandas as pd
import math

# Load the full CSV
df = pd.read_csv("data/simulation_results_2025_05_03_13_13_03.csv")

# Select a specific row (e.g., time = 0.0)
row_index = 24
row = df.iloc[row_index]

# Extract agent count
num_agents = 12
Referen
# Prepare node and edge lists
nodes = []
edges = []

for i in range(num_agents):
    x = row[f"agent_{i}_x"]
    y = row[f"agent_{i}_y"]
    vx = row[f"agent_{i}_v_x"]
    vy = row[f"agent_{i}_v_y"]
    
    # Node ID is agent_i
    nodes.append({
        "Id": f"agent_{i}",
        "Label": f"agent_{i}",
        "x": x,
        "y": y
    })
    
    # Optional: add a velocity arrow as an edge from position to next (x+vx, y+vy)
    edges.append({
        "Source": f"agent_{i}",
        "Target": f"vel_{i}",
        "Type": "Directed",
        "Weight": math.sqrt(vx**2 + vy**2)
    })
    
    # Add the velocity endpoint as a node
    nodes.append({
        "Id": f"vel_{i}",
        "Label": f"vel_{i}",
        "x": x + vx,
        "y": y + vy
    })

# Save nodes
nodes_df = pd.DataFrame(nodes)
nodes_df.to_csv("nodes.csv", index=False)

# Save edges
edges_df = pd.DataFrame(edges)
edges_df.to_csv("edges.csv", index=False)
