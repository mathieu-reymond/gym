from gym import RewardWrapper, make
from gym.spaces import Box
import numpy as np

try:
    import pybullet_envs
except ImportError as e:
    raise error.DependencyNotInstalled(f'{e} (HINT), you need to install pybullet from https://github.com/bulletphysics/bullet3')


def make_mo_pybullet_env(env_name):
    env = make(env_name)
    env = MOWrapper(env)
    return env


class MOWrapper(RewardWrapper):
    """Multi-objective extension of pybullet locomotion environments.
       There are 5 reward signals:
        - still alive
        - progress made
        - electricity cost
        - stuck joints cost
        - feet collision cost
    """
    def __init__(self, env):
        super(MOWrapper, self).__init__(env)
        low = -np.ones(5, dtype=np.float32)*np.inf
        high = np.ones(5, dtype=np.float32)*np.inf
        self.reward_space = Box(low=low, high=high, dtype=np.float32)

    def reward(self, reward):
        # we will ignore the reward (simply sum of different reward signals)
        # [alive, progress, electricity_cost, joints_at_limit_cost, feet_collision_cost]
        return np.array(self.rewards, dtype=np.float32)
