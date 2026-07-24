import numpy as np
import gymnasium

from gym_gridworlds.gridworld import REWARDS, GOOD, GOOD_SMALL, WALL, GRID_ENCODING

GRID_DECODING = {v: k for k, v in GRID_ENCODING.items()}


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

    def set_state(self, obs):
        unwrapped = self.env.unwrapped
        current_goal = tuple(np.argwhere(unwrapped.grid == GOOD)[0])
        unwrapped.grid[current_goal] = unwrapped.original_grid[current_goal]
        new_goal = np.unravel_index(obs[1], (self._n_rows, self._n_cols))
        unwrapped.grid[new_goal] = GOOD
        unwrapped.set_state(obs[0])
        unwrapped.last_action = None


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

    def set_state(self, obs):
        self.env.unwrapped.set_state(np.ravel_multi_index(obs, (self._n_rows, self._n_cols)))
        self.env.unwrapped.last_action = None


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
        assert (
            self._n_rows * self._n_cols == env.observation_space.n
        ), (
            "Cannot use MatrixWrapper, the observation space encodes more "
            "information than just the position. Try MatrixWithGoalWrapper."
        )
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

    def set_state(self, obs):
        pos = np.unravel_index(np.argmax(obs), (self._n_rows, self._n_cols))
        self.env.unwrapped.set_state(np.ravel_multi_index(pos, (self._n_rows, self._n_cols)))
        self.env.unwrapped.last_action = None


class BirdEyeWrapper(gymnasium.ObservationWrapper):
    """Bird's-eye view of the grid: the observation is a (2r+1, 2r+1) window
    centered on the agent. Each cell holds the character that encodes its
    tile (see `GRID_ENCODING` in `gridworld.py`) and out-of-bounds cells are
    filled with the `WALL` character.

    The view radius `r` is taken from the `view_radius` argument if given;
    otherwise it falls back to the wrapped env's `view_radius`. Either way,
    it must be strictly less than `max(n_rows, n_cols)`.

    Example (radius supplied to the wrapper):

    >>> import gymnasium
    >>> import gym_gridworlds
    >>> from gym_gridworlds.observation_wrappers import BirdEyeWrapper
    >>> env = gymnasium.make("Gym-Gridworlds/Penalty-3x3-v0", render_mode="human", start_pos=[("max", "max")])
    >>> env = BirdEyeWrapper(env, view_radius=1)
    >>> obs, _ = env.reset()
    >>> print(obs)  # . = EMPTY, □ = WALL, X = BAD
    [['X' '.' '□']
     ['.' '.' '□']
     ['□' '□' '□']]
    """

    def __init__(self, env, view_radius=None):
        super().__init__(env)
        self._n_rows = env.unwrapped.n_rows
        self._n_cols = env.unwrapped.n_cols
        if view_radius is None:
            view_radius = env.unwrapped.view_radius
        self._view_radius = view_radius
        assert self._view_radius < max(self._n_rows, self._n_cols), (
            "BirdEyeWrapper requires a `view_radius` < "
            f"max(n_rows, n_cols)={max(self._n_rows, self._n_cols)}, "
            f"but got view_radius={self._view_radius}. Pass one to the "
            "wrapper or set it on the env."
        )
        env.unwrapped.view_radius = self._view_radius  # keep rendering in sync
        size = 2 * self._view_radius + 1
        self.observation_space = gymnasium.spaces.Text(
            max_length=size * size,
            charset=frozenset(GRID_ENCODING.keys()),
        )

    def observation(self, obs):
        grid = self.env.unwrapped.grid
        r = self._view_radius
        size = 2 * r + 1
        row, col = self.env.unwrapped.agent_pos
        window = np.full((size, size), WALL, dtype=np.int32)
        row_start = max(row - r, 0)
        row_end = min(row + r + 1, self._n_rows)
        col_start = max(col - r, 0)
        col_end = min(col + r + 1, self._n_cols)
        w_row_start = row_start - (row - r)
        w_col_start = col_start - (col - r)
        w_row_end = w_row_start + (row_end - row_start)
        w_col_end = w_col_start + (col_end - col_start)
        window[w_row_start:w_row_end, w_col_start:w_col_end] = grid[
            row_start:row_end, col_start:col_end
        ]
        return np.vectorize(GRID_DECODING.get)(window).astype("<U1")


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
            low=0.0,
            high=1.0,  # Positive rewards are either 0.1 or 1.0
            dtype=np.float32,
        )

    def observation(self, obs):
        # If there is observation noise, apply it only to the agent's position
        pos = self.unwrapped.agent_pos
        if self.unwrapped.observation_noise > 0.0:
            if self.np_random.random() < self.observation_noise:
                pos = (
                    self.np_random.integers(0, self._n_rows),
                    self.np_random.integers(0, self._n_cols),
                )
        map = np.zeros((self._n_rows, self._n_cols))
        map[pos] = 1.0
        map_rewards = np.zeros((self._n_rows, self._n_cols))
        map_rewards[self.env.unwrapped.grid == GOOD] = REWARDS[GOOD]
        map_rewards[self.env.unwrapped.grid == GOOD_SMALL] = REWARDS[GOOD_SMALL]
        return np.stack((map, map_rewards), axis=2)

    def set_state(self, obs):
        pos = np.unravel_index(np.argmax(obs[..., 0]), (self._n_rows, self._n_cols))
        self.env.unwrapped.set_state(np.ravel_multi_index(pos, (self._n_rows, self._n_cols)))
        self.env.unwrapped.last_action = None

class ContinuousObservationWrapper(gymnasium.ObservationWrapper):
    """Observations are agent's coordinates normalized to [-1, 1].
    A continuous observation is "associated" with the closest tile within an
    absolute distance of 0.5.
    That is, given a discrete coordinate (x, y), any continuous observation within
    (x, y) ± 0.5 is treated as (x, y) for rewards and dynamics.
    To ensure coverage of the entire continuous state space, uniform noise in [-0.5, 0.5)
    is added to the initial position. The noise changes at every reset and is
    kept fixed for the whole episode.

    Example:

    >>> import gymnasium
    >>> import gym_gridworlds
    >>> from gym_gridworlds.observation_wrappers import ContinuousObservationWrapper
    >>> env = gymnasium.make("Gym-Gridworlds/Full-4x5-v0", render_mode="human", random_goals=True)
    >>> env = ContinuousObservationWrapper(env)
    >>> obs, _ = env.reset(seed=42)
    >>> print(obs)
    [-0.80675932, -0.63102737]
    >>> obs, *_ = env.step(1)
    >>> print(obs)
    [-0.80675932, -0.23102737]

    Note:

    Learning with such observations can be difficult, because tiles with similar
    observations — e.g., (0.2, 0.495) and (0.2, 0.505) — are very close in the
    observation space, but may be significantly different (e.g., one may be a pit
    and the other the goal). We advise using sparse function approximation when
    this wrapper is used, such as Fuzzy Tiling.
    """

    def __init__(self, env):
        super().__init__(env)
        self.grid_shape = np.array(env.unwrapped.grid.shape, dtype=np.float32)
        self.observation_space = gymnasium.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32,
        )
        self.agent_pos_offset = np.zeros(2, dtype=np.float32)

    def observation(self, obs):
        pos = np.array(self.env.unwrapped.agent_pos, dtype=np.float32)
        pos = (pos + self.agent_pos_offset + 0.5) / self.grid_shape  # in [0, 1]
        return pos * 2.0 - 1.0  # in [-1, 1]

    def set_state(self, obs):
        pos = (np.asarray(obs) + 1.0) / 2.0 * self.grid_shape - self.agent_pos_offset - 0.5
        pos = np.clip(np.round(pos).astype(int), 0, self.grid_shape.astype(int) - 1)
        self.env.unwrapped.set_state(np.ravel_multi_index(tuple(pos), tuple(self.grid_shape.astype(int))))
        self.env.unwrapped.last_action = None

    def reset(self, seed: int = None, **kwargs):
        self.agent_pos_offset = self.np_random.uniform(-0.5, 0.5, size=(2,))
        return super().reset(seed=seed, **kwargs)
