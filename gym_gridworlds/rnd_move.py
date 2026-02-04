import numpy as np

from gym_gridworlds.gridworld import EMPTY, GOOD_SMALL, GOOD, BAD, BAD_SMALL, WALL, PIT, QCKSND
from gym_gridworlds.gridworld import LEFT, DOWN, RIGHT, UP, STAY, REWARDS, RND_MOVE
from gym_gridworlds.gridworld import Gridworld


class RandomizedTiles(Gridworld):
    """
    Same as Gridworld, but with the addition of random-move tiles.
    These tiles work like directional tiles (the ones with an arrow denoting the
    only allowed action), but the allowed action changes randomly at every step.
    The randomization happens at the end of the transition, so the agent knows
    what action is allowed before acting.
    When rendered, the gray arrow denoting the allowed action changes as well.

    Tabular observations make it partially observable, as the agent can only
    know that any move has 25% chance of succeeding.
    Pixel observations make it fully observable.
    """

    def __init__(self, **kwargs):
        Gridworld.__init__(self, **kwargs)
        self.original_grid = self.grid.copy()

    def randomize_tiles(self):
        rnd_tiles = self.original_grid == RND_MOVE
        rnd_moves = self.np_random.integers(4, size=rnd_tiles.sum())
        self.grid[rnd_tiles] = rnd_moves

    def _reset(self, seed: int = None, **kwargs):
        Gridworld._reset(self, seed=seed, **kwargs)
        self.randomize_tiles()
        return {}

    def _step(self, action: int):
        obs, reward, terminated, truncated, info = Gridworld._step(self, action)
        self.randomize_tiles()
        return self.get_state(), reward, terminated, truncated, info
