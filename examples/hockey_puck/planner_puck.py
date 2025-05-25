from mppiisaac.planner.mppi_isaac import MPPIisaacPlanner
from mppiisaac.utils.config_store import ExampleConfig
import hydra
import torch
import torch.nn.functional as F
import pytorch3d.transforms
import zerorpc
from enum import Enum


class ControlPhase(Enum):
    """Control phases for hockey puck hitting"""
    APPROACH = 1  # Approaching and aligning phase
    HIT = 2       # Hitting phase


class Objective(object):
    """Two-phase cost function for hitting a hockey puck into the goal.
    
    Phase 1 (APPROACH): Move to optimal position behind puck and align orientation
    Phase 2 (HIT): Execute swift hitting motion with desired velocity
    
    The controller switches from APPROACH to HIT when position and orientation
    errors are below thresholds
    """

    def __init__(self, cfg):
        # Target hitting velocity magnitude
        self.v_hit = 20.0  # target hitting velocity magnitude
        
        # Strike positioning parameters
        self.strike_distance = 0.2  #  strike distance
        self.approach_zone_width = 0.15  # acceptable width for approach corridor
        
        # Phase transition thresholds
        self.position_threshold_enter_hit = 0.1  # 
        self.position_threshold_exit_hit = 0.2   # 
        self.orientation_threshold_enter_hit = 0.2  # 
        self.orientation_threshold_exit_hit = 0.6   # 
        
        # Cost weights for APPROACH phase
        self.w1_approach = 800.0  # position cost
        self.w2_approach = 300.0  # orientation cost
        self.w3_approach = 0.0    # velocity cost
        self.w4_approach = 0.0    # swing cost
        self.w5_approach = 200.0  # 
        self.w_progress_approach = 300.0  # progress reward
        self.w_ee_velocity_approach = 0.0  # ee velocity cost
        
        # Cost weights for HIT phase
        self.w1_hit = 50.0    # position cost
        self.w2_hit = 30.0    # orientation cost
        self.w3_hit = 0.0     # velocity cost
        self.w4_hit = 5000.0   # swing cost
        self.w5_hit = 0.0     # 
        self.w_progress_hit = 0.0  # No progress reward during hit
        self.w_ee_velocity_hit = 5000.0  #
        
        # Additional weights
        self.w_collision = 0.0  # collision cost
        self.w_joint_limits_approach = 0.001  # joint velocity limit
        self.w_joint_limits_hit = 0.0  # joint velocity limit
        self.w_comfy_pose_approach = 50.0  # approach phase comfortable pose
        self.w_comfy_pose_hit = 10.0  # hit phase comfortable pose
        
        # self.comfy_arm_pose = torch.tensor([0.3, 0.8, 0.0, -2.0, 0.0, 1.8, 0.8], device=cfg.mppi.device)
        self.comfy_arm_pose = torch.tensor([0.0, 0.6, 0.0, -1.2, 0.0, 1.6, 0.0], device=cfg.mppi.device)
        
        # hit phase joint velocity
        self.hit_joint_velocities = torch.tensor([20, 1.0, 0.0, 1.5, 20.0, 0.5, 0.5], device=cfg.mppi.device)
        
        # State tracking
        self.current_phase = ControlPhase.APPROACH
        self.phase_timer = 0  #
        self.min_hit_duration = 10  
        self.max_hit_duration = 30  # 
        self.max_approach_duration = 200  
        
        self.hit_executed = False
        self.max_ee_velocity_achieved = 0.0
        
        self.prev_ee_pos = None
        self.stuck_counter = 0
        
        # Debugging
        self.debug = True
        self.debug_frequency = 20 
        self.step_count = 0
        
        self.device = cfg.mppi.device
        self.reset()

    def reset(self):
        """Reset controller state."""
        self.current_phase = ControlPhase.APPROACH
        self.phase_timer = 0
        self.step_count = 0
        self.prev_ee_pos = None
        self.stuck_counter = 0
        self.hit_executed = False
        self.max_ee_velocity_achieved = 0.0

    def update_phase(self, position_error, orientation_error, ee_pos, ee_velocity_magnitude):
        """Update control phase based on current errors with hysteresis and timeout protection."""
        self.phase_timer += 1
        
        # Check if robot is stuck (not moving much) stuck detection
        if self.prev_ee_pos is not None:
            movement = torch.norm(ee_pos - self.prev_ee_pos).item()
            if movement < 0.001:
                self.stuck_counter += 1
            else:
                self.stuck_counter = 0
        
        self.prev_ee_pos = ee_pos.clone()
        
        if self.current_phase == ControlPhase.APPROACH:
            # Check if ready to switch to HIT phase
            ready_to_hit = (position_error < self.position_threshold_enter_hit and 
                           orientation_error < self.orientation_threshold_enter_hit)
            
            # Force transition if stuck too long in approach
            stuck_too_long = self.phase_timer > self.max_approach_duration
            robot_stuck = self.stuck_counter > 50
            
            if ready_to_hit or stuck_too_long or robot_stuck:
                self.current_phase = ControlPhase.HIT
                self.phase_timer = 0
                self.stuck_counter = 0
                self.hit_executed = False
                self.max_ee_velocity_achieved = 0.0
                if self.debug:
                    reason = "ready" if ready_to_hit else ("timeout" if stuck_too_long else "stuck")
                    print(f"\n[PHASE CHANGE] APPROACH -> HIT ({reason}: pos_err={position_error:.3f}, ori_err={orientation_error:.3f})")
                    
        elif self.current_phase == ControlPhase.HIT:
            # track max ee velocity
            if ee_velocity_magnitude > self.max_ee_velocity_achieved:
                self.max_ee_velocity_achieved = ee_velocity_magnitude
            
            # hit success detection
            if ee_velocity_magnitude > self.v_hit * 0.7:  # 
                self.hit_executed = True
            
            # check if should return to approach phase
            should_exit = False
            exit_reason = ""
            
            if self.hit_executed and ee_velocity_magnitude < self.max_ee_velocity_achieved * 0.5:
                should_exit = True
                exit_reason = "hit_completed"
            
            # max hit duration
            if self.phase_timer > self.max_hit_duration:
                should_exit = True
                exit_reason = "timeout"
            
            # position error
            if self.phase_timer > self.min_hit_duration:
                if position_error > self.position_threshold_exit_hit:
                    should_exit = True
                    exit_reason = "position_error"
            
            if should_exit:
                self.current_phase = ControlPhase.APPROACH
                self.phase_timer = 0
                self.stuck_counter = 0
                if self.debug:
                    print(f"\n[PHASE CHANGE] HIT -> APPROACH ({exit_reason}: max_vel={self.max_ee_velocity_achieved:.3f}, current_vel={ee_velocity_magnitude:.3f})")


    def compute_cost(self, sim):
        """Compute cost for each trajectory with phase-dependent weights."""
        
        ee_pose = sim.get_actor_link_by_name("omnipanda_effort_fixed_base", "hockey_tip")
        p_ee = ee_pose[:, :3]
        q_ee = ee_pose[:, 3:7]
        v_ee = ee_pose[:, 7:10]
        
        p_puck = sim.get_actor_position_by_name("hockey_puck")[:, :3]
        p_goal = sim.get_actor_position_by_name("hockey_goal")[:, :3]
        
        contact_forces = sim.get_actor_contact_forces_by_name("omnipanda_effort_fixed_base", "hockey_tip")
        collision_magnitude = torch.sum(torch.abs(contact_forces), dim=1)
        
        dof_state = sim.get_dof_state()
        joint_velocities = dof_state[:, 1::2]
        joint_positions = dof_state[:, 0::2]
        arm_positions = joint_positions[:, :7]
        d = F.normalize(p_goal - p_puck, dim=1)
        
        # ============= Core cost components =============
        # C1: Position to optimal strike point
        optimal_strike_pos = p_puck - self.strike_distance * d
        position_error = torch.norm(p_ee - optimal_strike_pos, dim=1)
        c1 = position_error ** 2
        
        # C2: Orientation alignment
        R_ee = pytorch3d.transforms.quaternion_to_matrix(q_ee)
        n_ee = R_ee[:, :, 1]  # y axis as hitting direction
        orientation_error = torch.norm(n_ee - d, dim=1)
        c2 = orientation_error ** 2
        
        # C3: Velocity alignment (deprecated in favor of ee velocity)
        v_target = self.v_hit * d
        c3 = torch.norm(v_ee - v_target, dim=1) ** 2
        
        # C4: Multi-joint swing encouragement
        joint_vel_diff = torch.abs(joint_velocities - self.hit_joint_velocities)
        c4 = -torch.sum(torch.abs(joint_velocities) * torch.tensor([5.0, 0, 0, 0, 5.0, 0, 0], device=self.device), dim=1)
        
        # C5: Behind puck constraint
        puck_to_ee = p_ee - p_puck
        projection_onto_goal_direction = torch.sum(puck_to_ee * d, dim=1)
        c5 = torch.clamp(projection_onto_goal_direction, min=0.0) ** 2
        
        # C6: ee velocity towards puck
        v_ee_magnitude = torch.norm(v_ee, dim=1)
        v_ee_normalized = F.normalize(v_ee + 1e-6, dim=1)  
        velocity_alignment = torch.sum(v_ee_normalized * d, dim=1) 
        # reward: large velocity and correct direction
        c6_ee_velocity = -(v_ee_magnitude * torch.clamp(velocity_alignment, min=0.0))
        
        # Additional costs
        c_collision = collision_magnitude
        
        # comfy pose cost
        arm_pose_cost = torch.sum(torch.square(arm_positions - self.comfy_arm_pose), dim=1)
        
        # Progress reward
        current_distance_to_target = torch.norm(p_ee - optimal_strike_pos, dim=1)
        c_progress = current_distance_to_target
        
        # ============= Phase-dependent cost computation =============
        # Update phase for the first trajectory
        if ee_pose.shape[0] > 0:
            self.update_phase(
                position_error[0].item(), 
                orientation_error[0].item(), 
                p_ee[0],
                v_ee_magnitude[0].item()
            )
        
        # Select weights based on current phase
        if self.current_phase == ControlPhase.APPROACH:
            w1, w2, w3, w4, w5 = (self.w1_approach, self.w2_approach, 
                                   self.w3_approach, self.w4_approach, self.w5_approach)
            w_progress = self.w_progress_approach
            w_ee_velocity = self.w_ee_velocity_approach
            w_joint_limits = self.w_joint_limits_approach
            w_comfy_pose = self.w_comfy_pose_approach
        else:  # HIT phase
            w1, w2, w3, w4, w5 = (self.w1_hit, self.w2_hit, 
                                   self.w3_hit, self.w4_hit, self.w5_hit)
            w_progress = self.w_progress_hit
            w_ee_velocity = self.w_ee_velocity_hit
            w_joint_limits = self.w_joint_limits_hit
            w_comfy_pose = self.w_comfy_pose_hit
        
        # joint velocity limit cost
        c_joint_limits = torch.sum(joint_velocities ** 2, dim=1) * w_joint_limits
        
        # Total cost
        total_cost = (
            w1 * c1 +
            w2 * c2 +
            w3 * c3 +
            w4 * c4 +
            w5 * c5 +
            w_ee_velocity * c6_ee_velocity +
            w_progress * c_progress +
            self.w_collision * c_collision +
            c_joint_limits +
            w_comfy_pose * arm_pose_cost
        )
        
        # ============= Debugging output =============
        if self.debug and self.step_count % self.debug_frequency == 0:
            print(f"\n[Step {self.step_count}] Phase: {self.current_phase.name} (timer={self.phase_timer})")
            print(f"  Position error:    {position_error[0].item():.3f} m")
            print(f"  Orientation error: {orientation_error[0].item():.3f}")
            print(f"  EE velocity mag:   {v_ee_magnitude[0].item():.3f} m/s")
            print(f"  Velocity alignment: {velocity_alignment[0].item():.3f}")
            print(f"  Max velocity achieved: {self.max_ee_velocity_achieved:.3f} m/s")
            
            print(f"\nCost Components (trajectory 0):")
            print(f"  c1 (position):     {w1 * c1[0].item():.2f} (w={w1})")
            print(f"  c2 (orientation):  {w2 * c2[0].item():.2f} (w={w2})")
            print(f"  c4 (swing):        {w4 * c4[0].item():.2f} (w={w4})")
            print(f"  c6 (ee_velocity):  {w_ee_velocity * c6_ee_velocity[0].item():.2f} (w={w_ee_velocity})")
            if self.current_phase == ControlPhase.APPROACH:
                print(f"  c5 (behind_puck):  {w5 * c5[0].item():.2f} (w={w5})")
                print(f"  c_progress:        {w_progress * c_progress[0].item():.2f} (w={w_progress})")
            print(f"  arm_pose:          {w_comfy_pose * arm_pose_cost[0].item():.2f}")
            print(f"  joint_limits:      {c_joint_limits[0].item():.2f}")
            print(f"  TOTAL:             {total_cost[0].item():.2f}")
            
            # Joint velocities
            print(f"\nJoint velocities: {joint_velocities[0].cpu().numpy()}")
            
            # Phase transition status
            if self.current_phase == ControlPhase.APPROACH:
                print(f"\nApproach Phase Status:")
                print(f"  Ready to hit: pos={position_error[0].item():.3f}<{self.position_threshold_enter_hit}, ori={orientation_error[0].item():.3f}<{self.orientation_threshold_enter_hit}")
            else:
                print(f"\nHit Phase Status:")
                print(f"  Hit executed: {self.hit_executed}")
        
        self.step_count += 1
        
        return total_cost

    def set_weights(self, **kwargs):
        """Update cost function weights dynamically."""
        # Update approach phase weights
        if 'w1_approach' in kwargs:
            self.w1_approach = kwargs['w1_approach']
        if 'w2_approach' in kwargs:
            self.w2_approach = kwargs['w2_approach']
        if 'w_progress_approach' in kwargs:
            self.w_progress_approach = kwargs['w_progress_approach']
            
        # Update hit phase weights
        if 'w1_hit' in kwargs:
            self.w1_hit = kwargs['w1_hit']
        if 'w2_hit' in kwargs:
            self.w2_hit = kwargs['w2_hit']
        if 'w4_hit' in kwargs:
            self.w4_hit = kwargs['w4_hit']
        if 'w_ee_velocity_hit' in kwargs:
            self.w_ee_velocity_hit = kwargs['w_ee_velocity_hit']
            
        # Update thresholds
        if 'position_threshold_enter_hit' in kwargs:
            self.position_threshold_enter_hit = kwargs['position_threshold_enter_hit']
        if 'position_threshold_exit_hit' in kwargs:
            self.position_threshold_exit_hit = kwargs['position_threshold_exit_hit']
        if 'orientation_threshold_enter_hit' in kwargs:
            self.orientation_threshold_enter_hit = kwargs['orientation_threshold_enter_hit']
            
        # Update other parameters
        if 'v_hit' in kwargs:
            self.v_hit = kwargs['v_hit']
        if 'strike_distance' in kwargs:
            self.strike_distance = kwargs['strike_distance']
        if 'debug' in kwargs:
            self.debug = kwargs['debug']
            
        print(f"Updated weights for phase-based control")


# ------------------------- runner -----------------------------
@hydra.main(version_base=None, config_path=".", config_name="hockey_puck")
def run_hockey_planner(cfg: ExampleConfig):
    objective = Objective(cfg)
    planner = zerorpc.Server(MPPIisaacPlanner(cfg, objective, prior=None))
    planner.bind("tcp://0.0.0.0:4242")
    planner.run()


if __name__ == "__main__":
    run_hockey_planner()
