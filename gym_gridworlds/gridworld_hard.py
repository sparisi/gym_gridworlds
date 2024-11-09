import numpy as np

from gym_gridworlds.gridworld import EMPTY, GOOD_SMALL, GOOD, BAD, BAD_SMALL, WALL, PIT, QCKSND
from gym_gridworlds.gridworld import LEFT, DOWN, RIGHT, UP, STAY, REWARDS, GRIDS
from gym_gridworlds.gridworld import Gridworld

# fmt: off
# add grids here
# fmt: on


class DirtCleaning(Gridworld):
    """
    Dirt appear in empty tiles and the agent has to go and clean it by doing
    STAY. Doing so yields a positive reward (+1) and dirt disappears.
    The agent also gets a small negative reward (-0.1) for every dirt on the grid
    at every timestep. 
    Episodes do not end when the agent collect dirt. Only other terminal
    conditions (e.g., falling into pits) can terminate an episode.
    The idea is to mimic a simple cleaning agent that keeps a room clean.

    Dirt is encoded as REWARDS tiles, thus will be rendered as green tiles.

    The default tabular observation is not sufficient to encode an optimal policy,
    because the agent must observe the whole grid (not just its current position).
    Use pixel-observations to solve this task.
    """

    def __init__(self, dirt_prob=0.05, **kwargs):
        Gridworld.__init__(self, **kwargs)
        self.dirt_prob = dirt_prob

    def step(self, action):
        obs, rwd, terminated, truncated, info = Gridworld.step(self, action)

        if self.grid[self.agent_pos] == GOOD and action == STAY:  # dirt was collected
            self.grid[self.agent_pos] = EMPTY
            terminated = False

        rwd -= REWARDS[BAD_SMALL] * (self.grid == GOOD).sum()  # add penalty for every dirt

        if self.np_random.random() < self.dirt_prob:  # spawn new dirt
            allowed_tiles = np.argwhere(self.grid == EMPTY)
            n_allowed = allowed_tiles.shape[0]
            if n_allowed != 0:
                new_dirt_tile = allowed_tiles[self.np_random.integers(n_allowed)]
                self.grid[new_dirt_tile[0], new_dirt_tile[1]] = GOOD

        return obs, rwd, terminated, truncated, info
