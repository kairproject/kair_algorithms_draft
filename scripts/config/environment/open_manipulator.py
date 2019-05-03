from math import pi

from geometry_msgs.msg import Quaternion


config = {
    "ENV_NAME": "OpenManipulatorReacher",
    "MAX_EPISODE_STEPS": 100,
    "TERM_COUNT": 5,
    "SUCCESS_COUNT": 5,
    "OVERHEAD_ORIENTATION": Quaternion(
        x=-0.00142460053167, y=0.999994209902, z=-0.00177030764765, w=0.00253311793936
    ),
    # box boundary
    "POLAR_RADIAN_BOUNDARY": (0.134, 0.32),
    "POLAR_THETA_BOUNDARY": (-pi * 0.7 / 4, pi * 0.7 / 4),
    "Z_BOUNDARY": (0.05, 0.28),
    "JOINT_LIMITS": {
        "HIGH": {
            "J1": pi * 0.9,
            "J2": pi * 0.5,
            "J3": pi * 0.44,
            "J4": pi * 0.65,
            "GRIP": 0.019,
        },
        "LOW": {
            "J1": -pi * 0.9,
            "J2": -pi * 0.57,
            "J3": -pi * 0.3,
            "J4": -pi * 0.57,
            "GRIP": -0.001,
        },
    },
    # Global variables
    "ACTION_DIM": 5,  # Cartesian
    "OBSERVATION_DIM": (25,),
    # terminal condition
    "INNER_RADIAN": 0.134,
    "OUTER_RADIAN": 0.3,
    "LOWER_RADIAN": 0.384,
    "INNER_Z": 0.321,
    "OUTER_Z": 0.250,
    "LOWER_Z": 0.116,
    "ENV_MODE": "sim",
    "TRAIN_MODE": True,
    "DISTANCE_THRESHOLD": 0.05,
    "REWARD_RESCALE_RATIO": 1.0,
    "REWARD_FUNC": "l2",
    "CONTROL_MODE": "position",
}


def get():
    return config
