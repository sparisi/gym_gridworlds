import numpy as np
import gymnasium as gym

from gym_gridworlds.gridworld import (
    LEFT, DOWN, RIGHT, UP, STAY, WALL, GOOD, Color, _move,
)
from gym_gridworlds.gridworld import Gridworld

GOAL = GOOD
GRASS = 99
ROAD = 98
SWAMP = 97

GRID_ENCODING = {
    ".": GRASS,
    "□": WALL,
    "+": ROAD,
    "-": SWAMP,
    "O": GOAL,
}


class TravelField(Gridworld):
    """
    The agent has to reach the goal while traveling through a field of different tiles:
    GRASS (green) gives -0.5 reward, SWAMP (brown) gives -1, ROAD (yellow) gives
    -0.1, and WALL (gray) cannot be walked on.
    The optimal policy goes through ROAD tiles only, but the path is hard to find.
    There are also dead-end roads that may mislead the agent.
    The agent can also move diagonally.
    If made `wall_is_terminal=True` episodes end if the agent hits a WALL.
    """

    def __init__(self, wall_is_terminal=False, **kwargs):
        Gridworld.__init__(self, encoding=GRID_ENCODING, **kwargs)
        self.action_space = gym.spaces.Discrete(9)  # cardinal + diagonal + stay
        self.wall_is_terminal = wall_is_terminal

        self.colormap[GRASS] = Color.GREEN
        self.colormap[ROAD] = Color.PALE_YELLOW
        self.colormap[SWAMP] = Color.BROWN
        self.colormap[GOAL] = Color.RED
        self.rewards[GRASS] = -0.5
        self.rewards[ROAD] = -0.1
        self.rewards[SWAMP] = -1.0
        self.rewards[GOAL] = 0

    def _step(self, action: int):
        obs, reward, terminated, truncated, info = Gridworld._step(self, action)
        if self.wall_is_terminal:
            tried_to_move_to = _move(
                self.last_pos,
                self.last_action,
                (self.n_rows, self.n_cols),
            )
            if self.grid[tried_to_move_to] == WALL:
                terminated = True
        return obs, reward, terminated, truncated, info
