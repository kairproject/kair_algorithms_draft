import numpy as np


env_name = "CAGCTRegistratorEnv"
stl_path = "./iSight_clamp.stl"
stl_resize_ratio = 0.3
max_action = 1
action_low = np.array([-1, -1, -1])
action_high = np.array([1, 1, 1])
max_episode_steps = 50
succeed_iou = 0.9
