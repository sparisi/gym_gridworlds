import numpy as np
import gymnasium

from gym_gridworlds.gridworld import REWARDS, GOOD, GOOD_SMALL


class AddGoalWrapper(gymnasium.ObservationWrapper):
    """See README.
    It assumes there is only one goal!

    Example:

    >>> import gymnasium
    >>> import gym_gridworlds
    >>> from gym_gridworlds.observation_wrappers import AddGoalWrapper
    >>> env = gymnasium.make("Gym-Gridworlds/Penalty-3x3-v0", render_mode="human", random_goals=True)
    >>> obs, _ = env.reset()
    >>> print(obs)
    0
    >>> env = AddGoalWrapper(env)
    >>> obs, _ = env.reset()
    >>> print(obs)
    (0, 0)
    """

    def __init__(self, env):
        super().__init__(env)
        self._rows = env.unwrapped.n_rows
        self._cols = env.unwrapped.n_cols
        size = self._rows * self._cols
        self.observation_space = gymnasium.spaces.MultiDiscrete([size, size])
        self.goal_pos = None


    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        goal_location = np.argwhere(self.env.unwrapped.grid == GOOD)
        if goal_location.shape[0] > 1:
            raise NotImplementedError("Only one goal at a time is supported!")
        self.goal_pos = np.ravel_multi_index(goal_location[0], (self._rows,  self._cols))
        return [obs, self.goal_pos], info


    def observation(self, obs):
        return [obs, self.goal_pos]


class CoordinateWrapper(gymnasium.ObservationWrapper):
    """See README.

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
        self.observation_space = gymnasium.spaces.Box(
            low=np.array([0, 0]),
            high=np.array([env.unwrapped.n_rows - 1, env.unwrapped.n_cols - 1]),
            dtype=np.uint8,
        )

    def observation(self, obs):
        return np.unravel_index(
            obs,
            (self.env.unwrapped.n_rows, self.env.unwrapped.n_cols),
        )


class MatrixWrapper(gymnasium.ObservationWrapper):
    """See README.

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
        self.observation_space = gymnasium.spaces.Box(
            shape=env.unwrapped.grid.shape,
            low=0,
            high=1,
            dtype=np.uint8,
        )

    def observation(self, obs):
        position = np.unravel_index(
            obs,
            (self.env.unwrapped.n_rows, self.env.unwrapped.n_cols),
        )
        map = np.zeros(self.env.unwrapped.grid.shape, dtype=np.uint8)
        map[position] = 1
        return map


class MatrixWithGoalWrapper(gymnasium.ObservationWrapper):
    """See README.

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
        shp = env.unwrapped.grid.shape + (2,)  # Two channels
        self.observation_space = gymnasium.spaces.Box(
            shape=shp,
            low=0,
            high=1,  # Positive rewards are either 0.1 or 1
            dtype=np.float32,
        )

    def observation(self, obs):
        position = np.unravel_index(
            obs,
            (self.env.unwrapped.n_rows, self.env.unwrapped.n_cols),
        )
        map = np.zeros(self.env.unwrapped.grid.shape)
        map[position] = 1
        map_rewards = np.zeros(self.env.unwrapped.grid.shape)
        map_rewards[self.env.unwrapped.grid == GOOD] = REWARDS[GOOD]
        map_rewards[self.env.unwrapped.grid == GOOD_SMALL] = REWARDS[GOOD_SMALL]
        return np.stack((map, map_rewards), axis=2)
