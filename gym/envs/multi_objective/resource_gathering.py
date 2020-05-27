from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
import cv2


class ResourceGatheringEnv(Env):
    """ The classic 3-objective resource gathering env.
        A player collects gems, while avoiding being hit by an ambushing thief.
        The 4 actions are: going up, right, down, left.
        In this environment, there are 3 goals:
         - take first gem
         - take second gem
         - avoid getting hit
        The episode ends either with the player being hit, or reaching its start-position.
        When hit, the player loses all his treasures
    """
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self):

        self.shape = (5, 5)
        self.n_tiles = np.prod(self.shape)
        self.transitions = np.array([[-1, 0], [0, 1], [1, 0], [0, -1]])

        self.start_pos = np.array([4, 2])
        self.current_pos = None

        self.treasure_poss = np.array([[0, 2], [1, 4]])
        self.treasures = None
        self.treasure_possibilities = [2]*len(self.treasure_poss)

        self.enemy_poss = np.array([[0, 3], [1, 2]])
        self.enemy_probs = np.array([.1, .1])

        self.nS = np.prod(self.shape)*2**len(self.treasure_possibilities)
        self.nA = 4
        # position-index is shifted, depending on treasures in pocket
        self.observation_space = Discrete(self.nS*4)
        self.action_space = Discrete(self.nA)
        self.reward_space = Box(low=np.array([0, 0, -1]), high=np.array([1, 1, 0]), dtype=np.float32)

        super(ResourceGatheringEnv, self).__init__()

    def reset(self):
        self.current_pos = self.start_pos
        self.treasures = np.zeros(len(self.treasure_poss), dtype=np.bool)

        start_state = np.ravel_multi_index(self.start_pos, self.shape)
        return start_state

    def step(self, action):
        # move
        new_position = np.array(self.current_pos) + self.transitions[action]
        new_position = self._limit_coordinates(new_position).astype(int)

        # take treasure
        self.treasures = self.treasures | np.all(self.treasure_poss == new_position, 1)

        # cope with enemies
        on_enemy = np.where(np.all(self.enemy_poss == new_position, 1).any())[0]
        hit = False
        if len(on_enemy):
            hit = np.random.rand() < self.enemy_probs[on_enemy[0]]

        if hit:
            self.treasures = np.zeros(len(self.treasure_poss), dtype=np.bool)

        terminal = np.all(new_position == self.start_pos) or hit
        new_state = np.ravel_multi_index(new_position, self.shape)
        treasure_shift = np.ravel_multi_index(self.treasures, self.treasure_possibilities)
        new_state += treasure_shift*self.n_tiles
        if terminal:
            reward = np.append(self.treasures, -int(hit))
        else:
            reward = np.zeros(len(self.treasures)+1)

        self.current_pos = new_position
        return new_state, reward.astype(np.float32), terminal, {}

    def _limit_coordinates(self, coord):
        """
        Prevent the agent from falling out of the grid world
        :param coord:
        :return:
        """
        coord[0] = min(coord[0], self.shape[0] - 1)
        coord[0] = max(coord[0], 0)
        coord[1] = min(coord[1], self.shape[1] - 1)
        coord[1] = max(coord[1], 0)
        return coord

    def render(self, mode='rgb_array'):
        tile_size = 30
        img = np.full((self.shape[0]*tile_size, self.shape[1]*tile_size, 3), 255, np.uint8)

        y = np.tile(np.arange(tile_size, (self.shape[0]+1)*tile_size, tile_size), self.shape[1])
        x = np.repeat(np.arange(tile_size, (self.shape[1]+1)*tile_size, tile_size), self.shape[0])
        for x_i in x:
            for y_i in y:
                cv2.circle(img, (x_i, y_i), 0, (255, 0, 0))

        for i, c in enumerate(self.treasure_poss):
            if not self.treasures[i]:
                cv2.putText(img, str('t_{}'.format(i)), (tile_size*c[1]+tile_size//2, tile_size*c[0]+tile_size//2), cv2.FONT_HERSHEY_SIMPLEX, .2, 255)
        for i, c in enumerate(self.enemy_poss):
            cv2.putText(img, str('e_{}'.format(i)), (tile_size*c[1]+tile_size//2, tile_size*c[0]+tile_size//2), cv2.FONT_HERSHEY_SIMPLEX, .2, 255)
        cv2.putText(img, 'H', (tile_size * self.start_pos[1] + tile_size // 2, tile_size * self.start_pos[0] + tile_size // 2), cv2.FONT_HERSHEY_SIMPLEX, .2, 255)
        position = self.current_pos
        cv2.putText(img, 'P', (tile_size*position[1]+tile_size//2, tile_size*position[0]+tile_size//2), cv2.FONT_HERSHEY_SIMPLEX, .2, 255)

        return img
