import numpy as np
import gymnasium
from gym_gridworlds.gridworld import Gridworld


class MatrixCoordinateWrapper(gymnasium.ObservationWrapper):
    """See README.

    Example:

    >>> import gymnasium
    >>> import gym_gridworlds
    >>> from gym_gridworlds.observation_wrappers import MatrixCoordinateWrapper
    >>> env = gymnasium.make("Gym-Gridworlds/Penalty-3x3-v0", render_mode="human")
    >>> env = MatrixCoordinateWrapper(env)
    >>> env.reset()
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


class MatrixBinaryWrapper(gymnasium.ObservationWrapper):
    """See README.

    Example:

    >>> import gymnasium
    >>> import gym_gridworlds
    >>> from gym_gridworlds.observation_wrappers import MatrixBinaryWrapper
    >>> env = gymnasium.make("Gym-Gridworlds/Penalty-3x3-v0", render_mode="human")
    >>> env = MatrixBinaryWrapper(env)
    >>> env.reset()
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
