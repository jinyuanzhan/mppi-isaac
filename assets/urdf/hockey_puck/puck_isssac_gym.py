from isaacgym import gymapi
import numpy as np

# ------------------------------
# Regulation-size hockey puck
# ------------------------------
PUCK_RADIUS = 0.0500      # [m] 3 in → 76.2 mm  ⇒ radius 38.1 mm
PUCK_HEIGHT = 0.0254      # [m] 1 in  → 25.4 mm
PUCK_MASS = 0.17        # [kg]  ≈ 6 oz
PUCK_COLOR = [0, 0, 0, 1]  # opaque black


class HockeyPuckIsaacGym:
    def __init__(self, gym, sim, env, position=(0, 0, PUCK_HEIGHT/2), name="hockey_puck"):
        """
        create a puck in isaac gym
        
        Args:
            gym: gymapi instance
            sim: simulation instance
            env: environment instance
            position: initial position (x, y, z), unit is meter
            name: name of the puck
        """
        self.gym = gym
        self.sim = sim
        self.env = env
        self.name = name
        
        # create asset options
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = False
        asset_options.density = PUCK_MASS / (np.pi * PUCK_RADIUS**2 * PUCK_HEIGHT)
        asset_options.disable_gravity = False
        asset_options.use_physx_armature = True
        
        # create a cylinder as the puck
        self.puck_asset = self.gym.create_asset(
            sim=self.sim,
            cylinder_props=gymapi.CylinderProperties(
                radius=PUCK_RADIUS,
                halfLength=PUCK_HEIGHT/2
            ),
            asset_options=asset_options
        )
        
        # set the pose of the puck
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(position[0], position[1], position[2])
        pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        
        # create the puck actor
        self.puck_handle = self.gym.create_actor(
            env=self.env,
            asset=self.puck_asset,
            pose=pose,
            name=self.name,
            group=0,
            filter=1,
            segmentation_id=0
        )
        
        # set the physical properties of the puck
        self.gym.set_rigid_body_color(
            env=self.env,
            actor_handle=self.puck_handle,
            rigid_body_index=0,
            target=gymapi.MESH_VISUAL,
            color=gymapi.Vec3(PUCK_COLOR[0], PUCK_COLOR[1], PUCK_COLOR[2])
        )
        
        # set the interaction parameters between the puck and the ice
        self.gym.set_actor_dof_properties(
            env=self.env,
            actor_handle=self.puck_handle,
            properties=gymapi.DofProperties()
        )
        
        # set the friction and elasticity
        props = self.gym.get_actor_rigid_body_properties(self.env, self.puck_handle)
        props[0].friction = 0.4      # ≈ ice + rubber
        props[0].rolling_friction = 1e-4  # small but non-zero
        props[0].torsion_friction = 1e-4  # spinning friction
        props[0].restitution = 0.1   # slight bounce
        self.gym.set_actor_rigid_body_properties(self.env, self.puck_handle, props)
    
    def get_handle(self):
        """return the handle of the puck"""
        return self.puck_handle
    
    def get_state(self):
        """get the position and rotation state of the puck"""
        state = self.gym.get_actor_rigid_body_states(
            self.env, self.puck_handle, gymapi.STATE_POS | gymapi.STATE_ROT | gymapi.STATE_VEL
        )
        return state
    
    def set_position(self, position):
        """set the position of the puck"""
        state = self.gym.get_actor_rigid_body_states(
            self.env, self.puck_handle, gymapi.STATE_POS
        )
        # state['pose']['p']['x'] = position[0]
        # state['pose']['p']['y'] = position[1]
        # state['pose']['p']['z'] = position[2]
        state['pose']['p'][:] = position
        self.gym.set_actor_rigid_body_states(
            self.env, self.puck_handle, state, gymapi.STATE_POS
        )

    def apply_force(self, force, position=None):
        """apply a force to the puck"""
        if position is None:
            # if no position is specified, apply the force at the center of the puck
            state = self.gym.get_actor_rigid_body_states(
                self.env, self.puck_handle, gymapi.STATE_POS
            )
            position = [state['pose']['p']['x'], state['pose']['p']['y'], state['pose']['p']['z']]
        
        self.gym.apply_body_force(
            self.env, 
            self.gym.get_actor_rigid_body_index(self.env, self.puck_handle, 0),
            gymapi.Vec3(force[0], force[1], force[2]),
            gymapi.Vec3(position[0], position[1], position[2]),
            gymapi.GLOBAL_SPACE
        )

# example usage
if __name__ == "__main__":
    # initialize isaac gym
    gym = gymapi.acquire_gym()
    
    # create simulation
    sim_params = gymapi.SimParams()
    sim_params.dt = 1.0 / 60.0
    sim_params.substeps = 2
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
    
    # set the collision properties
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 6
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.contact_offset = 0.01
    sim_params.physx.rest_offset = 0.0
    
    # create simulation
    sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
    
    # create ground
    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0, 0, 1)  # z-up
    plane_params.distance = 0
    plane_params.static_friction = 0.1  # 冰面摩擦小
    plane_params.dynamic_friction = 0.1
    plane_params.restitution = 0.1  # 略微的弹性
    gym.add_ground(sim, plane_params)
    
    # create viewer
    cam_props = gymapi.CameraProperties()
    viewer = gym.create_viewer(sim, cam_props)
    if viewer is None:
        print("Failed to create viewer")
        quit()
    
    # create environment
    env_spacing = 1.5
    env_lower = gymapi.Vec3(-env_spacing, -env_spacing, 0.0)
    env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)
    env = gym.create_env(sim, env_lower, env_upper, 1)
    
    # create puck
    puck = HockeyPuckIsaacGym(gym, sim, env, position=(0, 0, PUCK_HEIGHT/2 + 0.001))
    
    # apply an initial force to the puck to move it
    puck.apply_force([10.0, 5.0, 0.0])
    
    # run simulation
    while not gym.query_viewer_has_closed(viewer):
        # step simulation
        gym.simulate(sim)
        gym.fetch_results(sim, True)
        
        # update viewer
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, True)
        
        # wait for a short time
        gym.sync_frame_time(sim)
    
    # clean up
    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)
