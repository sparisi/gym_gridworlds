import numpy as np
import gymnasium as gym

from gym_gridworlds.gridworld import (
    LEFT, DOWN, RIGHT, UP, STAY, WALL, GOOD, REWARDS, COLORMAP, Color, _move,
)
from gym_gridworlds.gridworld import Gridworld

ROCK = WALL
GOAL = GOOD
GRASS = 99
ROAD = 98
SWAMP = 97

COLORMAP[ROCK] = Color.GRAY
COLORMAP[GRASS] = Color.GREEN
COLORMAP[ROAD] = Color.PALE_YELLOW
COLORMAP[SWAMP] = Color.BROWN
COLORMAP[GOAL] = Color.RED

REWARDS[ROCK] = -1.0
REWARDS[GRASS] = -0.5
REWARDS[ROAD] = -0.1
REWARDS[SWAMP] = -1.0
REWARDS[GOAL] = 0

GRID_ENCODING = {
    ".": GRASS,
    "□": ROCK,
    "+": ROAD,
    "-": SWAMP,
    "O": GOAL,
}


class TravelField(Gridworld):
    """
    The agent has to reach the goal while traveling through a field of different tiles:
    GRASS (green) gives -0.5 reward, SWAMP (brown) gives -1, ROAD (yellow) gives
    -0.1, and ROCK (gray) cannot be walked on.
    The optimal policy goes through ROAD tiles only, but the path is hard to find.
    There are also dead-end roads that may mislead the agent.
    The agent can also move diagonally.
    By default, the episode terminates if the agent hits a ROCK.
    """

    def __init__(self, rock_is_terminal=False, **kwargs):
        Gridworld.__init__(self, encoding=GRID_ENCODING, **kwargs)
        self.action_space = gym.spaces.Discrete(9)  # cardinal + diagonal + stay
        self.rock_is_terminal = rock_is_terminal

    def _step(self, action: int):
        obs, reward, terminated, truncated, info = Gridworld._step(self, action)
        if self.rock_is_terminal:
            tried_to_move_to = _move(
                self.last_pos,
                self.last_action,
                (self.n_rows, self.n_cols),
            )
            if self.grid[tried_to_move_to] == ROCK:
                terminated = True
        return obs, reward, terminated, truncated, info
