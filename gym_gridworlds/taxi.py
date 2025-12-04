import numpy as np
from itertools import product
import gymnasium as gym

from gym_gridworlds.gridworld import EMPTY, GOOD_SMALL, GOOD, BAD, BAD_SMALL, WALL, PIT, QCKSND
from gym_gridworlds.gridworld import LEFT, DOWN, RIGHT, UP, STAY, REWARDS, GRIDS
from gym_gridworlds.gridworld import Gridworld

PASS = GOOD_SMALL  # passenger, encoded as "GOOD_SMALL" just for rendering it with dark green tiles

# fmt: off
GRIDS["taxi_6x7"] = [
    [EMPTY, WALL,  PASS,  EMPTY, WALL,  EMPTY, GOOD ],
    [EMPTY, WALL,  EMPTY, EMPTY, WALL,  EMPTY, EMPTY],
    [EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY],
    [WALL,  WALL,  EMPTY, EMPTY, EMPTY, WALL,  WALL ],
    [EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, PASS ],
    [PASS,  EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, WALL ],
]
# fmt: on


class Taxi(Gridworld):
    """
    First presented in "An alternative softmax operator for reinforcement learning".
    The agent (taxi) must pick up passengers and drive them to a destination.
    By default, there are three passengers (at fixed locations) and one destination
    (also fixed, top right corner).
    The taxi always starts in the top left corner.
    If one, two, or all three passengers reach the goal, the agent is rewarded
    with 1, 3, or 15, respectively. Otherwise the reward is 0.
    The state consists of the taxi position and three booleans denoting if a
    passenger is in the taxi.
    Observations are enumerated to account all possible combinations of position
    and passengers picked.
    The agent can move LEFT, RIGHT, UP, or DOWN.
    An episode ends if the taxi goes to the destination, even without passengers.
    A passenger is picked by simply walking on its tile.
    A passenger cannot be dropped once picked.

    """

    def __init__(self, **kwargs):
        Gridworld.__init__(self, **kwargs)
        self.passengers = np.argwhere(self.grid == PASS)
        N = self.n_rows * self.n_cols
        P = len(self.passengers)
        self.passengers_picked = [False] * P
        self.original_grid = self.grid.copy()
        self.observation_space = gym.spaces.Discrete(N * (2**P))

    def reset(self, seed: int = None, **kwargs):
        self.grid = self.original_grid.copy()
        self.passengers_picked = [False] * len(self.passengers)
        return Gridworld.reset(self, seed=seed, **kwargs)

    def get_state(self):
        obs = Gridworld.get_state(self)
        picked = np.array(self.passengers_picked, dtype=int)
        picked_id = 0
        for bit in picked:
            picked_id = picked_id * 2 + bit
        state = obs * (2 ** len(self.passengers_picked)) + picked_id
        return state

    def set_state(self, state):
        N = self.n_rows * self.n_cols
        P = len(self.passengers)
        base = 2 ** P
        pos = state // base
        picked_id = state % base
        row = pos // self.n_cols
        col = pos % self.n_cols
        picked = []
        for i in reversed(range(P)):
            bit = (picked_id >> i) & 1
            picked.append(bool(bit))
        self.agent_pos = (row, col)
        self.passengers_picked = picked

    def step(self, action):
        rwd = 0.0
        terminated = False
        if self.grid[self.agent_pos] == GOOD:
            rewards_per_passengers = [0.0, 1.0, 3.0, 15.0]
            rwd = rewards_per_passengers[np.sum(self.passengers_picked)]
            terminated = True
        _, _, _, truncated, info = Gridworld.step(self, action)
        if self.grid[self.agent_pos] == PASS:
            self.grid[self.agent_pos] = EMPTY
            for i, passenger in enumerate(self.passengers):
                if np.all(passenger == self.agent_pos):
                    self.passengers_picked[i] = True
                    break
        return self.get_state(), rwd, terminated, truncated, info
