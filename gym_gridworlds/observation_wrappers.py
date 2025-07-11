import numpy as np
import gymnasium

from gym_gridworlds.gridworld import REWARDS, GOOD, GOOD_SMALL


class AddGoalWrapper(gymnasium.ObservationWrapper):
    """Also returns the position (ravel index) of the agent's goal.
    The grid must have only one goal.

    Example:

    >>> import gymnasium
    >>> import gym_gridworlds
    >>> from gym_gridworlds.observation_wrappers import AddGoalWrapper
    >>> env = gymnasium.make("Gym-Gridworlds/Penalty-3x3-v0", render_mode="human", random_goals=True)
    >>> obs, _ = env.reset(seed=42)
    >>> print(obs)
    0
    >>> env = AddGoalWrapper(env)
    >>> obs, _ = env.reset(seed=42)
    >>> print(obs)
    [0, 0]
    >>> obs, _ = env.reset(seed=24)
    >>> print(obs)
    [0, 3]
    """

    def __init__(self, env):
        super().__init__(env)
        assert (
            (env.unwrapped.grid == GOOD).sum() == 1 and
            (env.unwrapped.grid == GOOD_SMALL).sum() == 0
        ), "AddGoalWrapper supports only grids with one goal"  # fmt: skip
        self._n_rows = env.unwrapped.n_rows
        self._n_cols = env.unwrapped.n_cols
        size = self._n_rows * self._n_cols
        self.observation_space = gymnasium.spaces.MultiDiscrete([size, size])

    def observation(self, obs):
        goal = np.argwhere(self.env.unwrapped.grid == GOOD)[0]
        return [obs, np.ravel_multi_index(goal, (self._n_rows, self._n_cols))]


class CoordinateWrapper(gymnasium.ObservationWrapper):
    """Unravels matrix indices into coordinates.

    Example:

    >>> import gymnasium
    >>> import gym_gridworlds
    >>> from gym_gridworlds.observation_wrappers import CoordinateWrapper
    >>> env = gymnasium.make("Gym-Gridworlds/Penalty-3x3-v0", render_mode="human")
    >>> obs, _ = env.reset()
    >>> print(obs)
    0
    >>> env = CoordinateWrapper(env)
    >>> obs, _ = env.reset()
    >>> print(obs)
    (0, 0)
    """

    def __init__(self, env):
        super().__init__(env)
        self._n_rows = env.unwrapped.n_rows
        self._n_cols = env.unwrapped.n_cols
        self.observation_space = gymnasium.spaces.Box(
            low=np.array([0, 0]),
            high=np.array([self._n_rows - 1, self._n_cols - 1]),
            dtype=np.uint8,
        )

    def observation(self, obs):
        return np.unravel_index(obs, (self._n_rows, self._n_cols))


class MatrixWrapper(gymnasium.ObservationWrapper):
    """Binary map of the agent's position.

    Example:

    >>> import gymnasium
    >>> import gym_gridworlds
    >>> from gym_gridworlds.observation_wrappers import MatrixWrapper
    >>> env = gymnasium.make("Gym-Gridworlds/Penalty-3x3-v0", render_mode="human")
    >>> obs, _ = env.reset()
    >>> print(obs)
    0
    >>> env = MatrixWrapper(env)
    >>> obs, _ = env.reset()
    >>> print(obs)
    [[1 0 0]
     [0 0 0]
     [0 0 0]]
    """

    def __init__(self, env):
        super().__init__(env)
        self._n_rows = env.unwrapped.n_rows
        self._n_cols = env.unwrapped.n_cols
        self.observation_space = gymnasium.spaces.Box(
            shape=(self._n_rows, self._n_cols),
            low=0,
            high=1,
            dtype=np.uint8,
        )

    def observation(self, obs):
        position = np.unravel_index(obs, (self._n_rows, self._n_cols))
        map = np.zeros((self._n_rows, self._n_cols), dtype=np.uint8)
        map[position] = 1
        return map


class MatrixWithGoalWrapper(gymnasium.ObservationWrapper):
    """Map of the agent's position (first channel) and positive rewards'
    positions (second channel).

    Example:

    >>> import gymnasium
    >>> import gym_gridworlds
    >>> from gym_gridworlds.observation_wrappers import MatrixWithGoalWrapper
    >>> env = gymnasium.make("Gym-Gridworlds/Full-4x5-v0", render_mode="human", random_goals=True)
    >>> env = MatrixWithGoalWrapper(env)
    >>> obs, _ = env.reset(seed=42)
    >>> print(obs[..., 0])
    [[1. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0.]]
    >>> print(obs[..., 1])
    [[0.  0.1 0.  0.  0. ]
     [0.  0.  0.  0.  0. ]
     [0.  0.  0.  0.1 0. ]
     [0.  1.  0.  0.  0. ]]
    >>> obs, _ = env.reset(seed=24)
    >>> print(obs[..., 0])
    [[1. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0.]]
    >>> print(obs[..., 1])
    [[0.  0.  0.  0.  0. ]
     [1.  0.1 0.  0.  0. ]
     [0.  0.  0.  0.  0. ]
     [0.  0.1 0.  0.  0. ]]
    """

    def __init__(self, env):
        super().__init__(env)
        self._n_rows = env.unwrapped.n_rows
        self._n_cols = env.unwrapped.n_cols
        self.observation_space = gymnasium.spaces.Box(
            shape=(self._n_rows, self._n_cols, 2),
            low=0,
            high=1,  # Positive rewards are either 0.1 or 1
            dtype=np.float32,
        )

    def observation(self, obs):
        position = np.unravel_index(obs, (self._n_rows, self._n_cols))
        map = np.zeros((self._n_rows, self._n_cols))
        map[position] = 1
        map_rewards = np.zeros((self._n_rows, self._n_cols))
        map_rewards[self.env.unwrapped.grid == GOOD] = REWARDS[GOOD]
        map_rewards[self.env.unwrapped.grid == GOOD_SMALL] = REWARDS[GOOD_SMALL]
        return np.stack((map, map_rewards), axis=2)
