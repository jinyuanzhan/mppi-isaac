from envs.base_env import BaseEnv
from models.primative_objects.puck import HockeyPuck
import numpy as np
import pybullet as p
import os

class PuckToGoalEnv(BaseEnv):
    def __init__(
        self, 
        **kwargs,
    ):
        # Define puck start position in front of the robot
        self.puck_start_pos = np.array([0.8, 0.6, 0.012])  # 0.0127 is half the puck height

        # Define goal position in front of the puck
        self.goal_pos = np.array([2.0, 0.0, 0.0])
        self.goal_ori = p.getQuaternionFromEuler([0, 0, np.pi/2])

        # Goal dimensions from hockey_goal.urdf
        self.goal_width = 0.4572  # Width between uprights (x-axis)
        self.goal_height = 0.3048  # Height of the goal (z-axis)
        self.goal_depth = 0.254   # Depth of the goal (y-axis)
        self.frame_bar_radius = 0.01905  # Radius of the goal frame bars

        # Define goal box region (x_min, x_max, y_min, y_max, z_min, z_max)
        # Center the box on goal position
        half_width = self.goal_width/2
        self.goal_box = np.array([
            self.goal_pos[0],  # x_min
            self.goal_pos[0] + half_width,  # x_max
            self.goal_pos[1] - (self.goal_depth - 2*self.frame_bar_radius),  # y_min (behind goal position)
            self.goal_pos[1] + (self.goal_depth - 2*self.frame_bar_radius),  # y_max (slightly in front for better detection)
            0.0,  # z_min (ground level)
            self.goal_height  # z_max
        ])
        
        super().__init__(
            camera_distance=0.65,
            camera_yaw=-12,
            camera_pitch=-35,
            camera_target=[0.8, -0.4, 0.5],
            **kwargs,
        )
        
        # # Get the end effector pose of the Franka robot
        # ee_state = self.get_robot_ee_state()
        # self.ee_pos = ee_state[0]  # Position (x, y, z)
        # print("self.ee_pos: ", self.ee_pos)
        # self.ee_ori = ee_state[1]  # Orientation (quaternion)
    
    def _setup_environment(self):
        # Load plane for the hockey rink surface
        planePos = [0, 0, -0.01]
        planeOri = p.getQuaternionFromEuler([0, 0, 0])
        self.plane = p.loadURDF("plane/plane.urdf", planePos, planeOri, useFixedBase=True)
        p.changeDynamics(self.plane, -1, lateralFriction=0.1)  # Simulate ice surface with low friction
        
        # Load hockey puck
        self.puck = HockeyPuck(position=self.puck_start_pos).get_body_id()
        
        # Load hockey goal
        goal_urdf_path = os.path.join("hockey_goal", "hockey_goal.urdf")
        self.goal = p.loadURDF(goal_urdf_path, self.goal_pos, self.goal_ori, useFixedBase=True)
        
        # Optional: Create a visual marker for the goal box (for debugging)
        # self._create_goal_box_visual()
    
    def _create_goal_box_visual(self):
        """Optional method to create a visual marker for the goal box"""
        box_pos = [
            (self.goal_box[0] + self.goal_box[1])/2,  # x center
            (self.goal_box[2] + self.goal_box[3])/2,  # y center
            (self.goal_box[4] + self.goal_box[5])/2   # z center
        ]
        box_size = [
            (self.goal_box[1] - self.goal_box[0]),  # x size
            (self.goal_box[3] - self.goal_box[2]),  # y size
            (self.goal_box[5] - self.goal_box[4])   # z size
        ]
        visual_id = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[s/2 for s in box_size],
            rgbaColor=[0, 1, 0, 0.3]  # Semi-transparent green
        )
        self.goal_box_visual = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=visual_id,
            basePosition=box_pos
        )
        
    def reward(self):
        # Get current puck position
        puck_pos, _ = p.getBasePositionAndOrientation(self.puck)
        puck_pos = np.array(puck_pos)
        
        # Check if puck is in the goal box region
        in_goal_box = (
            self.goal_box[0] <= puck_pos[0] <= self.goal_box[1] and  # x bounds
            self.goal_box[2] <= puck_pos[1] <= self.goal_box[3] and  # y bounds
            self.goal_box[4] <= puck_pos[2] <= self.goal_box[5]      # z bounds
        )
        
        # Calculate distance to goal center (for partial reward)
        goal_center = np.array([
            (self.goal_box[0] + self.goal_box[1])/2,
            (self.goal_box[2] + self.goal_box[3])/2,
            puck_pos[2]  # Keep same z-height as puck for distance calculation
        ])
        
        if in_goal_box:
            reward = 1.0  # Maximum reward when the puck is in the goal
        else:
            # Calculate current distance to goal center
            current_puck_to_goal_dist = np.linalg.norm(puck_pos[:2] - goal_center[:2])
            initial_puck_to_goal_dist = np.linalg.norm(self.puck_start_pos[:2] - goal_center[:2])
            
            # Calculate normalized distance ratio (1.0 when at start, approaches 0 as puck gets closer to goal)
            distance_ratio = current_puck_to_goal_dist / initial_puck_to_goal_dist
            
            # Base reward is 1.0 - distance_ratio, so it's normalized between 0 and 1
            reward = max(0.0, 1.0 - distance_ratio)
        
        return reward
    
    def reset(self):
        super().reset()
        # Reset puck position
        p.resetBasePositionAndOrientation(
            self.puck, 
            self.puck_start_pos, 
            p.getQuaternionFromEuler([0, 0, 0])
        )
        
    def is_success(self):
        """Check if the puck is in the goal box"""
        puck_pos, _ = p.getBasePositionAndOrientation(self.puck)
        puck_pos = np.array(puck_pos)
        
        # Check if puck is in the goal box region
        in_goal_box = (
            self.goal_box[0] <= puck_pos[0] <= self.goal_box[1] and  # x bounds
            self.goal_box[2] <= puck_pos[1] <= self.goal_box[3] and  # y bounds
            self.goal_box[4] <= puck_pos[2] <= self.goal_box[5]      # z bounds
        )
        
        return in_goal_box
