import gym
from trimesh import voxel

from .utils import *


class CAGCTRegistratorEnv(gym.Env):

    def __init__(self, cfg):
        self.env_name = cfg.env_name
        self.cfg = cfg
        self.action_space = gym.spaces.Box(low=self.cfg.action_low,
            high=self.cfg.action_high, dtype=np.float32)
        self._max_episode_steps = self.cfg.max_episode_steps
        self.episode_steps = 0

    def compute_reward(self):
        state_arr = stl_to_arr(self.state_stl)
        projected_state_arr = project_3d_arr_to_2d_arr(state_arr)
        self.iou = get_iou(projected_state_arr, self.label_2d_arr)
        reward = self.iou - 1
        return reward

    def reset(self):
        self.state_stl = load_stl(self.cfg.stl_path)
        # resize stl
        self.state_stl.apply_scale(self.cfg.stl_resize_ratio)
        _, self.label_2d_arr, _ = generate_label(self.state_stl)
        return self.get_observation()

    def get_observation(self):
        state_3d_arr = stl_to_arr(self.state_stl).astype('float')
        return state_3d_arr.reshape(1, 1, *state_3d_arr.shape)

    def step(self, action):
        done = False
        succeed = None
        self.episode_steps += 1

        action = unnormalize_action(action)
        self.state_stl = rotate_stl(self.state_stl, action)
        obs = self.get_observation()
        reward = self.compute_reward()
        if self.iou > self.cfg.succeed_iou:
            done = True
            succeed = True
        if self.episode_steps == self._max_episode_steps:
            done = True
            self.episode_steps = 0
        return obs, reward, done, succeed

    def render(self):
        pass

