from mppiisaac.planner.isaacgym_wrapper import IsaacGymWrapper
from isaacgym import gymapi
import hydra
from mppiisaac.utils.config_store import ExampleConfig
import time

@hydra.main(version_base=None, config_path=".", config_name="hockey_puck")
def run_hockey_puck(cfg: ExampleConfig):
    # create IsaacGymWrapper instance, load URDF file
    sim = IsaacGymWrapper(
        cfg.isaacgym,
        actors=cfg.actors,
        init_positions=cfg.initial_actor_positions,
        num_envs=1,
        viewer=True,
        device=cfg.mppi.device,
    )

    # set camera view
    sim._gym.viewer_camera_look_at(
        sim.viewer,
        None,
        gymapi.Vec3(1.0, 6.5, 4),  # camera position
        gymapi.Vec3(1.0, 0, 0),     # camera focus point
    )
    
    print("URDF loaded, showing GUI...")
    
    # main loop, keep GUI running
    while True:
        # step simulation
        sim.step()
        
        # control simulation time
        time.sleep(cfg.isaacgym.dt)

if __name__ == "__main__":
    run_hockey_puck()