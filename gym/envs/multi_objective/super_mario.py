from gym import Wrapper, make
from gym.spaces import Box
import numpy as np

try:
    import gym_super_mario_bros
    from gym_super_mario_bros.actions import RIGHT_ONLY, SIMPLE_MOVEMENT, COMPLEX_MOVEMENT
    from nes_py.wrappers import JoypadSpace
except ImportError as e:
    raise error.DependencyNotInstalled(f'{e} (HINT), you need to install gym_super_mario_bros from https://github.com/Kautenja/gym-super-mario-bros')


def make_mo_mario_env(env_name, actions='RIGHT_ONLY'):
    assert 'SuperMarioBros' in env_name, 'This is meant for super mario envs'
    movements = {'RIGHT_ONLY': RIGHT_ONLY,
                 'SIMPLE_MOVEMENT': SIMPLE_MOVEMENT,
                 'COMPLEX_MOVEMENT': COMPLEX_MOVEMENT}
    env = make(env_name)
    env = MOWrapper(env)
    env = JoypadSpace(env, movements[actions])
    return env


class MOWrapper(Wrapper):
    """Multi objective Multimario, inspired from https://github.com/RunzheYang/MORL/
       There are 5 objectives:
        - how much it moved to the right
        - time penalty
        - death
        - collected coins
        - score increase (corresponds to hitting an enemy)
    """

    DEATH_PENALTY = -25

    def __init__(self, env):
        super(MOWrapper, self).__init__(env)
        low = np.array([-np.inf, -np.inf, MOWrapper.DEATH_PENALTY, 0, 0])
        high = np.array([np.inf, 0, 0, np.inf, np.inf])
        self.reward_space = Box(low=low, high=high, dtype=np.float32)

    def reset(self):
        self.lives = 2
        self.coins = 0
        self.x_pos = 0
        self.time = 0
        self.score = 0
        obs = super(MOWrapper, self).reset()
        return obs

    def step(self, action):
        # ignore single-objective reward
        n_obs, _, done, info = super(MOWrapper, self).step(action)
        # 1. position
        r_xpos = info['x_pos'] - self.x_pos
        self.x_pos = info['x_pos']
        # resolve an issue where after death the x position resets
        if r_xpos < -5:
            r_xpos = 0
        # 2.time penalty
        r_time = info['time'] - self.time
        self.time = info['time']
        if r_time > 0:
            r_time = 0
        # 3. death
        if self.lives > info['life']:
            r_death = MOWrapper.DEATH_PENALTY
        else:
            r_death = 0
        self.lives = info['life']
        # 4. coin
        r_coin = 100*(info['coins'] - self.coins)
        self.coins = info['coins']
        # 5. enemy
        r_enemy = info['score'] - self.score
        if r_coin or done:
            r_enemy = 0
        self.score = info['score']

        return n_obs, np.array([r_xpos, r_time, r_death, r_coin, r_enemy]), done, info
