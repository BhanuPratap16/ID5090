# import numpy as np
# import matplotlib.pyplot as plt


# x_ship = np.array([-0.5, -0.5, 0.25, 0.5, 0.25, -0.5, -0.5, 0.5, 0.25, 0, 0])
# y_ship = 0.25 * np.array([-1, 1, 1, 0, -1, -1, 0, 0, 1, 1, -1])



# psi_new =0
# x_new_ship = x_ship * np.cos(psi_new) - y_ship * np.sin(psi_new)
# y_new_ship = x_ship * np.sin(psi_new) + y_ship * np.cos(psi_new)
# plt.plot(x_new_ship, y_new_ship, 'r')
# plt.show()


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import matplotlib.transforms as transforms

class ShipPatch(Polygon):
    """
    A custom matplotlib patch for ship visualization.
    
    This patch uses the ship geometry defined in ship_geom.py and
    allows for easy positioning and rotation.
    """
    
    def __init__(self, xy=(0, 0), angle=0, size=1.0, **kwargs):
        """
        Initialize a ship patch.
        
        Parameters:
        -----------
        xy : tuple
            The (x, y) coordinates of the ship's center.
        angle : float
            The rotation angle in radians.
        size : float
            Scaling factor for the ship size.
        **kwargs : dict
            Additional arguments passed to Polygon.
        """
        # Define the ship geometry
        try:
            from ship_geom import x_ship, y_ship
            self.x_ship = x_ship
            self.y_ship = y_ship
        except ImportError:
            # Fallback ship geometry if import fails
            self.x_ship = np.array([-0.5, -0.5, 0.25, 0.5, 0.25, -0.5, -0.5, 0.5, 0.25, 0, 0])
            self.y_ship = 0.25 * np.array([-1, 1, 1, 0, -1, -1, 0, 0, 1, 1, -1])
        
        # Scale the ship
        x_ship_scaled = self.x_ship * size
        y_ship_scaled = self.y_ship * size
        
        # Rotate the ship
        x_rotated = x_ship_scaled * np.cos(angle) - y_ship_scaled * np.sin(angle)
        y_rotated = x_ship_scaled * np.sin(angle) + y_ship_scaled * np.cos(angle)
        
        # Translate to the specified position
        x_final = x_rotated + xy[0]
        y_final = y_rotated + xy[1]
        
        # Create the polygon vertices
        vertices = np.column_stack([x_final, y_final])
        
        # Initialize the Polygon with the computed vertices
        super().__init__(vertices, closed=True, **kwargs)
        
        # Store parameters for later updates
        self.center = xy
        self.angle = angle
        self.size = size
    
    def update_position(self, xy, angle=None):
        """
        Update the ship's position and orientation.
        
        Parameters:
        -----------
        xy : tuple
            The new (x, y) coordinates of the ship's center.
        angle : float, optional
            The new rotation angle in radians. If None, the current angle is kept.
        """
        if angle is not None:
            self.angle = angle
        
        # Scale the ship
        x_ship_scaled = self.x_ship * self.size
        y_ship_scaled = self.y_ship * self.size
        
        # Rotate the ship
        x_rotated = x_ship_scaled * np.cos(self.angle) - y_ship_scaled * np.sin(self.angle)
        y_rotated = x_ship_scaled * np.sin(self.angle) + y_ship_scaled * np.cos(self.angle)
        
        # Translate to the specified position
        x_final = x_rotated + xy[0]
        y_final = y_rotated + xy[1]
        
        # Update the polygon vertices
        vertices = np.column_stack([x_final, y_final])
        self.set_xy(vertices)
        
        # Update stored center
        self.center = xy


# Example usage
if __name__ == "__main__":
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect('equal')
    ax.grid(True)
    
    # Create a ship at the origin with 0 rotation
    ship1 = ShipPatch(xy=(0, 0), angle=0, facecolor='red', alpha=0.7)
    ax.add_patch(ship1)
    
    # Create another ship at (1, 1) with 45-degree rotation
    ship2 = ShipPatch(xy=(1, 1), angle=np.pi/4, facecolor='blue', alpha=0.7)
    ax.add_patch(ship2)
    
    plt.title('Ship Patch Example')
    plt.show()