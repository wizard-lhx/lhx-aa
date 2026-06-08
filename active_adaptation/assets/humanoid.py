from pathlib import Path
from typing import Literal
from active_adaptation import ROBOT_MODEL_DIR
import active_adaptation.utils.symmetry as symmetry_utils

from active_adaptation.assets.asset_cfg import (
    AssetCfg,
    InitialStateCfg,
    ActuatorCfg,
    ContactSensorCfg,
)
from active_adaptation.registry import Registry

registry = Registry.instance()

FILE_DIR = Path(__file__).parent

ARMATURE_5020 = 0.003609725
ARMATURE_7520_14 = 0.010177520
ARMATURE_7520_22 = 0.025101925
ARMATURE_4010 = 0.00425

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz
DAMPING_RATIO = 2.0

STIFFNESS_5020 = ARMATURE_5020 * NATURAL_FREQ**2
STIFFNESS_7520_14 = ARMATURE_7520_14 * NATURAL_FREQ**2
STIFFNESS_7520_22 = ARMATURE_7520_22 * NATURAL_FREQ**2
STIFFNESS_4010 = ARMATURE_4010 * NATURAL_FREQ**2

DAMPING_5020 = 2.0 * DAMPING_RATIO * ARMATURE_5020 * NATURAL_FREQ
DAMPING_7520_14 = 2.0 * DAMPING_RATIO * ARMATURE_7520_14 * NATURAL_FREQ
DAMPING_7520_22 = 2.0 * DAMPING_RATIO * ARMATURE_7520_22 * NATURAL_FREQ
DAMPING_4010 = 2.0 * DAMPING_RATIO * ARMATURE_4010 * NATURAL_FREQ


INIT_POS = (0.0, 0.0, 0.85)
INIT_JOINT_POS = {
    ".*_hip_pitch_joint": -0.1,
    ".*_knee_joint": 0.3,
    ".*_ankle_pitch_joint": -0.2,
    ".*_elbow_joint": 0.6,
    "left_shoulder_roll_joint": 0.2,
    "left_shoulder_pitch_joint": 0.2,
    "right_shoulder_roll_joint": -0.2,
    "right_shoulder_pitch_joint": 0.2,
    "waist_yaw_joint": 0.0,
    "waist_roll_joint": 0.0,
    "waist_pitch_joint": 0.0,
}

ACTUATORS = {
    "base_legs": ActuatorCfg(
        joint_names_expr=".*",
        effort_limit={
            ".*_hip_yaw_joint.*": 88.0,
            ".*_hip_roll_joint.*": 139.0,
            ".*_hip_pitch_joint.*": 88.0,
            ".*_knee.*": 139.0,
            ".*_ankle.*": 50,
            ".*_shoulder.*": 25,
            ".*_elbow.*": 25,
            ".*_wrist_roll_joint": 25.0,
            ".*_wrist_pitch_joint": 5.0,
            ".*_wrist_yaw_joint": 5.0,
            "waist.*": 50,
        },
        velocity_limit={
            ".*_hip_yaw_joint": 32.0,
            ".*_hip_roll_joint": 20.0,
            ".*_hip_pitch_joint": 32.0,
            ".*_knee_joint": 20.0,
            ".*_ankle.*": 37.0,
            "waist.*": 37.0,
            ".*_shoulder_pitch_joint": 37.0,
            ".*_shoulder_roll_joint": 37.0,
            ".*_shoulder_yaw_joint": 37.0,
            ".*_elbow_joint": 37.0,
            ".*_wrist_roll_joint": 37.0,
            ".*_wrist_pitch_joint": 22.0,
            ".*_wrist_yaw_joint": 22.0,
        },
        stiffness={
            ".*_hip_yaw_joint": STIFFNESS_7520_14,
            ".*_hip_roll_joint": STIFFNESS_7520_22,
            ".*_hip_pitch_joint": STIFFNESS_7520_14,
            ".*_knee_joint": STIFFNESS_7520_22,
            ".*ankle.*": 2.0 * STIFFNESS_5020,
            "waist.*": 2.0 * STIFFNESS_5020,
            ".*_shoulder_pitch_joint": STIFFNESS_5020,
            ".*_shoulder_roll_joint": STIFFNESS_5020,
            ".*_shoulder_yaw_joint": STIFFNESS_5020,
            ".*_elbow_joint": STIFFNESS_5020,
            ".*_wrist_roll_joint": STIFFNESS_5020,
            ".*_wrist_pitch_joint": STIFFNESS_4010,
            ".*_wrist_yaw_joint": STIFFNESS_4010,
        },
        damping={
            ".*_hip_pitch_joint": DAMPING_7520_14,
            ".*_hip_roll_joint": DAMPING_7520_22,
            ".*_hip_yaw_joint": DAMPING_7520_14,
            ".*_knee_joint": DAMPING_7520_22,
            ".*ankle.*": 2.0 * DAMPING_5020,
            "waist.*": 2.0 * DAMPING_5020,
            ".*_shoulder_pitch_joint": DAMPING_5020,
            ".*_shoulder_roll_joint": DAMPING_5020,
            ".*_shoulder_yaw_joint": DAMPING_5020,
            ".*_elbow_joint": DAMPING_5020,
            ".*_wrist_roll_joint": DAMPING_5020,
            ".*_wrist_pitch_joint": DAMPING_4010,
            ".*_wrist_yaw_joint": DAMPING_4010,
        },
        armature={
            ".*_hip_pitch_joint": ARMATURE_7520_14,
            ".*_hip_roll_joint": ARMATURE_7520_22,
            ".*_hip_yaw_joint": ARMATURE_7520_14,
            ".*_knee_joint": ARMATURE_7520_22,
            ".*ankle.*": 2.0 * ARMATURE_5020,
            "waist.*": 2.0 * ARMATURE_5020,
            ".*_shoulder_pitch_joint": ARMATURE_5020,
            ".*_shoulder_roll_joint": ARMATURE_5020,
            ".*_shoulder_yaw_joint": ARMATURE_5020,
            ".*_elbow_joint": ARMATURE_5020,
            ".*_wrist_roll_joint": ARMATURE_5020,
            ".*_wrist_pitch_joint": ARMATURE_4010,
            ".*_wrist_yaw_joint": ARMATURE_4010,
        },
        friction=0.01,
    ),
}

MJLAB_ACTUATOR_GROUPS = (
    (".*_hip_yaw_joint", 88.0, STIFFNESS_7520_14, DAMPING_7520_14, ARMATURE_7520_14),
    (".*_hip_roll_joint", 139.0, STIFFNESS_7520_22, DAMPING_7520_22, ARMATURE_7520_22),
    (".*_hip_pitch_joint", 88.0, STIFFNESS_7520_14, DAMPING_7520_14, ARMATURE_7520_14),
    (".*_knee_joint", 139.0, STIFFNESS_7520_22, DAMPING_7520_22, ARMATURE_7520_22),
    (".*_ankle.*", 50.0, 2.0 * STIFFNESS_5020, 2.0 * DAMPING_5020, 2.0 * ARMATURE_5020),
    ("waist.*", 50.0, 2.0 * STIFFNESS_5020, 2.0 * DAMPING_5020, 2.0 * ARMATURE_5020),
    (".*_shoulder_.*_joint", 25.0, STIFFNESS_5020, DAMPING_5020, ARMATURE_5020),
    (".*_elbow_joint", 25.0, STIFFNESS_5020, DAMPING_5020, ARMATURE_5020),
    (".*_wrist_roll_joint", 25.0, STIFFNESS_5020, DAMPING_5020, ARMATURE_5020),
    (".*_wrist_pitch_joint", 5.0, STIFFNESS_4010, DAMPING_4010, ARMATURE_4010),
    (".*_wrist_yaw_joint", 5.0, STIFFNESS_4010, DAMPING_4010, ARMATURE_4010),
)

JOINT_SYMMETRY_MAPPING = symmetry_utils.mirrored({
    "left_hip_pitch_joint": (1, "right_hip_pitch_joint"),
    "left_hip_roll_joint": (-1, "right_hip_roll_joint"),
    "left_hip_yaw_joint": (-1, "right_hip_yaw_joint"),
    "left_knee_joint": (1, "right_knee_joint"),
    "left_ankle_pitch_joint": (1, "right_ankle_pitch_joint"),
    "left_ankle_roll_joint": (-1, "right_ankle_roll_joint"),
    "waist_yaw_joint": (-1, "waist_yaw_joint"),
    "waist_roll_joint": (-1, "waist_roll_joint"),
    "waist_pitch_joint": (1, "waist_pitch_joint"),
    "left_shoulder_pitch_joint": (1, "right_shoulder_pitch_joint"),
    "left_shoulder_roll_joint": (-1, "right_shoulder_roll_joint"),
    "left_shoulder_yaw_joint": (-1, "right_shoulder_yaw_joint"),
    "left_elbow_joint": (1, "right_elbow_joint"),
    "left_wrist_roll_joint": (-1, "right_wrist_roll_joint"),
    "left_wrist_pitch_joint": (1, "right_wrist_pitch_joint"),
    "left_wrist_yaw_joint": (-1, "right_wrist_yaw_joint"),
})

SPATIAL_SYMMETRY_MAPPING = symmetry_utils.mirrored({
    "left_hip_pitch_link": "right_hip_pitch_link",
    "left_hip_roll_link": "right_hip_roll_link",
    "left_hip_yaw_link": "right_hip_yaw_link",
    "left_knee_link": "right_knee_link",
    "left_ankle_pitch_link": "right_ankle_pitch_link",
    "left_ankle_roll_link": "right_ankle_roll_link",
    "pelvis": "pelvis",
    "torso_link": "torso_link",
    "waist_yaw_link": "waist_yaw_link",
    "waist_roll_link": "waist_roll_link",
    "left_shoulder_pitch_link": "right_shoulder_pitch_link",
    "left_shoulder_roll_link": "right_shoulder_roll_link",
    "left_shoulder_yaw_link": "right_shoulder_yaw_link",
    "left_elbow_link": "right_elbow_link",
    "left_wrist_roll_link": "right_wrist_roll_link",
    "left_wrist_yaw_link": "right_wrist_yaw_link",
    "left_wrist_pitch_link": "right_wrist_pitch_link",
})

JOINT_NAMES_SIMULATION = [
    "left_hip_pitch_joint", "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint", "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint", "right_knee_joint",
    "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint", "right_ankle_pitch_joint",
    "left_shoulder_roll_joint", "right_shoulder_roll_joint",
    "left_ankle_roll_joint", "right_ankle_roll_joint",
    "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
    "left_elbow_joint", "right_elbow_joint",
    "left_wrist_roll_joint", "right_wrist_roll_joint",
    "left_wrist_pitch_joint", "right_wrist_pitch_joint",
    "left_wrist_yaw_joint", "right_wrist_yaw_joint",
]

BODY_NAMES_SIMULATION = [
    "pelvis",
    "left_hip_pitch_link", "right_hip_pitch_link",
    "waist_yaw_link",
    "left_hip_roll_link", "right_hip_roll_link",
    "waist_roll_link",
    "left_hip_yaw_link", "right_hip_yaw_link",
    "torso_link",
    "left_knee_link", "right_knee_link",
    "left_shoulder_pitch_link", "right_shoulder_pitch_link",
    "left_ankle_pitch_link", "right_ankle_pitch_link",
    "left_shoulder_roll_link", "right_shoulder_roll_link",
    "left_ankle_roll_link", "right_ankle_roll_link",
    "left_shoulder_yaw_link", "right_shoulder_yaw_link",
    "left_elbow_link", "right_elbow_link",
    "left_wrist_roll_link", "right_wrist_roll_link",
    "left_wrist_pitch_link", "right_wrist_pitch_link",
    "left_wrist_yaw_link", "right_wrist_yaw_link",
]


def make_asset_cfg() -> AssetCfg:
    return AssetCfg(
        mjcf_path=FILE_DIR / "G1" / "mjcf" / "g1.xml",
        usd_path=FILE_DIR / "G1" / "waist_unlocked.usd",
        init_state=InitialStateCfg(
            pos=INIT_POS,
            joint_pos=INIT_JOINT_POS,
            joint_vel={".*": 0.0},
        ),
        self_collisions=False,
        actuators=ACTUATORS,
        joint_symmetry_mapping=JOINT_SYMMETRY_MAPPING,
        spatial_symmetry_mapping=SPATIAL_SYMMETRY_MAPPING,
        sensors_isaaclab=[
            ContactSensorCfg(
                name="contact_forces",
                primary=".*",
                secondary=[],
                track_air_time=True,
                history_length=3,
            ),
        ],
        sensors_mjlab=[
            ContactSensorCfg(
                name="contact_forces",
                primary=".*",
                secondary=[],
                track_air_time=True,
                history_length=3,
            ),
        ],
        joint_names_simulation=JOINT_NAMES_SIMULATION,
        body_names_simulation=BODY_NAMES_SIMULATION,
    )


def make_isaaclab_cfg():
    asset_cfg = make_asset_cfg()
    sensors = {sensor.name: sensor.isaaclab() for sensor in asset_cfg.sensors_isaaclab}
    return asset_cfg.isaaclab(), sensors


def make_mjlab_cfg():
    import mujoco
    from active_adaptation.assets.asset_cfg import EntityCfg
    from mjlab.actuator import BuiltinPositionActuatorCfg
    from mjlab.entity import EntityArticulationInfoCfg
    from mjlab.sensor import ContactMatch, ContactSensorCfg as MjlabContactSensorCfg
    from mjlab.utils.spec_config import CollisionCfg

    def spec_fn():
        return mujoco.MjSpec.from_file(
            str(ROBOT_MODEL_DIR / "g1_mjlab" / "g1_mjlab.xml")
        )

    cfg = EntityCfg(
        init_state=EntityCfg.InitialStateCfg(
            pos=INIT_POS,
            joint_pos=INIT_JOINT_POS,
            joint_vel={".*": 0.0},
        ),
        spec_fn=spec_fn,
        articulation=EntityArticulationInfoCfg(
            actuators=tuple(
                BuiltinPositionActuatorCfg(
                    target_names_expr=(pattern,),
                    effort_limit=effort_limit,
                    stiffness=stiffness,
                    damping=damping,
                    armature=armature,
                    frictionloss=0.01,
                )
                for pattern, effort_limit, stiffness, damping, armature in MJLAB_ACTUATOR_GROUPS
            ),
        ),
        collisions=(
            CollisionCfg(
                geom_names_expr=(".*_collision.*",),
                contype=0,
                conaffinity=1,
                condim=3,
            ),
        ),
        joint_symmetry_mapping=JOINT_SYMMETRY_MAPPING,
        spatial_symmetry_mapping=SPATIAL_SYMMETRY_MAPPING,
        joint_names_simulation=JOINT_NAMES_SIMULATION,
        body_names_simulation=BODY_NAMES_SIMULATION,
    )
    sensors = (
        MjlabContactSensorCfg(
            name="contact_forces",
            primary=ContactMatch(mode="body", pattern=".*", entity="robot"),
            secondary=ContactMatch(mode="body", pattern="terrain", entity=None),
            fields=("found", "force"),
            reduce="maxforce",
            num_slots=1,
            track_air_time=True,
            history_length=3,
        ),
    )
    return cfg, sensors


def make_mujoco_cfg():
    return make_asset_cfg().mujoco()


def make_cfg(backend: Literal["isaaclab", "mjlab", "mujoco"]):
    if backend == "isaaclab":
        return make_isaaclab_cfg()
    if backend == "mjlab":
        return make_mjlab_cfg()
    if backend == "mujoco":
        return make_mujoco_cfg()
    raise ValueError(f"Invalid backend: {backend}")


G1_WAIST_UNLOCKED_CFG = make_asset_cfg()
registry.register("asset", "g1_waist_unlocked", make_cfg)
