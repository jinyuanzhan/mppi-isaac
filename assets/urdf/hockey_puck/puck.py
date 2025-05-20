import pybullet as p
import pybullet_data
import time
import math

# ------------------------------
# Regulation-size hockey puck
# ------------------------------
PUCK_RADIUS  = 0.0500      # [m] 3 in → 76.2 mm  ⇒ radius 38.1 mm
PUCK_HEIGHT  = 0.0254      # [m] 1 in  → 25.4 mm
PUCK_MASS    = 0.17        # [kg]  ≈ 6 oz
PUCK_COLOR   = [0, 0, 0, 1]  # opaque black

class HockeyPuck:
    def __init__(self, position=(0, 0, PUCK_HEIGHT/2)):
        """Spawn a rigid-body puck at the given world position (XYZ, metres)."""

        # ------------------------------------------------------------------
        # Collision & visual geometry
        #   • PyBullet’s cylinder axis is the local +Z direction.
        #   • For most PyBullet builds, 'height' (collision) and 'length'
        #     (visual) are the full height of the cylinder, NOT half-height.
        # ------------------------------------------------------------------
        collision = p.createCollisionShape(
            shapeType=p.GEOM_CYLINDER,
            radius=PUCK_RADIUS,
            height=PUCK_HEIGHT
        )

        visual = p.createVisualShape(
            shapeType=p.GEOM_CYLINDER,
            radius=PUCK_RADIUS,
            length=PUCK_HEIGHT,      # some PyBullet versions use 'length'
            rgbaColor=PUCK_COLOR
        )

        # ------------------------------------------------------------------
        # Inertia: PyBullet can compute it from the geometry + mass, but
        # we can also feed the exact numbers (optional).
        # ------------------------------------------------------------------
        # Ixx = (1/12)*PUCK_MASS*(3*PUCK_RADIUS**2 + PUCK_HEIGHT**2)
        # Izz = 0.5*PUCK_MASS*PUCK_RADIUS**2
        # inertia_diagonal = [Ixx, Ixx, Izz]

        self.body_id = p.createMultiBody(
            baseMass              = PUCK_MASS,
            baseCollisionShapeIndex = collision,
            baseVisualShapeIndex    = visual,
            basePosition            = position,
            baseInertialFramePosition=[0, 0, 0],
            baseInertialFrameOrientation=[0, 0, 0, 1],
            # baseInertialFrameInertiaDiag=inertia_diagonal
        )

        # ------------------------------------------------------------------
        # Puck–ice interaction parameters (tweak as needed):
        # ------------------------------------------------------------------
        p.changeDynamics(self.body_id, -1,
                         lateralFriction=0.4,   # ≈ ice + rubber
                         rollingFriction=1e-4,  # small but non-zero
                         spinningFriction=1e-4,
                         restitution=0.1)       # slight bounce

    def get_body_id(self):
        return self.body_id


# --------------------------------------------------------------------------
# Quick demo
# --------------------------------------------------------------------------
if __name__ == "__main__":
    physics_client = p.connect(p.GUI)       # use p.DIRECT for headless
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)

    # Load a flat plane as the ice rink surface
    plane_id = p.loadURDF("plane.urdf")

    # Spawn the puck
    puck = HockeyPuck(position=(0, 0, PUCK_HEIGHT/2 + 0.001))

    # Run a short sim so you can poke the puck with the mouse in GUI mode
    for _ in range(2_000):
        p.stepSimulation()
        time.sleep(1/240)

    p.disconnect()
