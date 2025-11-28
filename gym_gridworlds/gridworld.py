import numpy as np
import gymnasium as gym
from gymnasium.error import DependencyNotInstalled
from typing import Optional
from collections import defaultdict
from enum import Enum

# state IDs
EMPTY = -1
QCKSND = -2
GOOD_SMALL = 9
GOOD = 10
BAD = 11
BAD_SMALL = 12
WALL = -3
PIT = -4

# action IDs (and for one-directional states)
LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3
STAY = 4

# default rewards
REWARDS = defaultdict(lambda: 0)
REWARDS[GOOD] = 1
REWARDS[BAD] = -10
REWARDS[GOOD_SMALL] = 0.1
REWARDS[BAD_SMALL] = -0.1
REWARDS[PIT] = -100

# rendering colors
class Color(tuple, Enum):
    RED = (255, 0, 0)
    PALE_RED = (155, 0, 0)
    GREEN = (0, 255, 0)
    DARK_GREEN = (0, 155, 0)
    BLUE = (0, 0, 255)
    ORANGE = (255, 175, 0)
    PALE_YELLOW = (255, 255, 155)
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    GRAY = (100, 100, 100)
    PURPLE = (102, 51, 153)
    BROWN = (139, 69, 19)

COLORMAP = dict()
COLORMAP[EMPTY] = Color.BLACK
COLORMAP[QCKSND] = Color.PALE_YELLOW
COLORMAP[GOOD_SMALL] = Color.DARK_GREEN
COLORMAP[GOOD] = Color.GREEN
COLORMAP[BAD] = Color.RED
COLORMAP[BAD_SMALL] = Color.PALE_RED
COLORMAP[WALL] = Color.GRAY
COLORMAP[PIT] = Color.PURPLE
COLORMAP[LEFT] = Color.BLACK
COLORMAP[RIGHT] = Color.BLACK
COLORMAP[UP] = Color.BLACK
COLORMAP[DOWN] = Color.BLACK

# fmt: off
GRIDS = {
    "river_swim_6": [
        [GOOD_SMALL] + [EMPTY for _ in range(4)] + [GOOD],
    ],
    "20_straight": [
        [EMPTY for _ in range(19)] + [GOOD],
    ],
    "2x2_empty": [
        [EMPTY, EMPTY],
        [EMPTY, GOOD],
    ],
    "3x3_empty": [
        [EMPTY, EMPTY, EMPTY],
        [EMPTY, EMPTY, EMPTY],
        [EMPTY, EMPTY, GOOD],
    ],
    "3x3_empty_loop": [
        [EMPTY, LEFT, EMPTY],
        [EMPTY, RIGHT, UP],
        [EMPTY, EMPTY, GOOD],
    ],
    "3x3_penalty": [
        [EMPTY, BAD, GOOD],
        [EMPTY, BAD, EMPTY],
        [EMPTY, EMPTY, EMPTY],
    ],
    "10x10_empty":
        [[EMPTY for _ in range(10)] for _ in range(9)] +
        [[EMPTY for _ in range(9)] + [GOOD]]
    ,
    "6x6_distract":
        [[EMPTY for _ in range(6)] for _ in range(5)] +
        [[GOOD_SMALL] + [EMPTY for _ in range(4)] + [GOOD]]
    ,
    "4x4_quicksand": [
        [EMPTY, EMPTY, BAD, GOOD],
        [EMPTY, EMPTY, BAD, EMPTY],
        [EMPTY, QCKSND, EMPTY, EMPTY],
        [EMPTY, EMPTY, EMPTY, EMPTY],
    ],
    "4x4_quicksand_distract": [
        [EMPTY, GOOD_SMALL, BAD, GOOD],
        [EMPTY, BAD, EMPTY, EMPTY],
        [EMPTY, QCKSND, GOOD_SMALL, EMPTY],
        [EMPTY, EMPTY, EMPTY, EMPTY],
    ],
    "4x5_full": [
        [EMPTY, EMPTY, GOOD_SMALL, BAD, GOOD],
        [EMPTY, EMPTY, BAD, EMPTY, EMPTY],
        [RIGHT, EMPTY, QCKSND, GOOD_SMALL, EMPTY],
        [UP, EMPTY, EMPTY, EMPTY, EMPTY],
    ],
    "3x5_two_room_quicksand": [
        [EMPTY, EMPTY, LEFT, EMPTY, GOOD],
        [EMPTY, EMPTY, QCKSND, EMPTY, EMPTY],
        [EMPTY, EMPTY, EMPTY, EMPTY, EMPTY],
    ],
    "3x4_corridor": [
        [EMPTY, LEFT, LEFT, LEFT],
        [GOOD_SMALL, BAD_SMALL, BAD_SMALL, GOOD],
        [EMPTY, LEFT, LEFT, LEFT],
    ],
    "2x11_two_room_distract": [
        [GOOD_SMALL, EMPTY, EMPTY, EMPTY, RIGHT, DOWN, LEFT, EMPTY, EMPTY, EMPTY, GOOD],
        [EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY],
    ],
    "5x5_barrier": [
        [EMPTY, EMPTY, EMPTY, EMPTY, EMPTY],
        [EMPTY, LEFT, UP, RIGHT, EMPTY],
        [EMPTY, LEFT, GOOD, EMPTY, EMPTY],
        [EMPTY, LEFT, DOWN, RIGHT, EMPTY],
        [EMPTY, EMPTY, EMPTY, EMPTY, EMPTY],
    ],
    "4x12_cliffwalk": [
        [EMPTY, PIT, PIT, PIT, PIT, PIT, PIT, PIT, PIT, PIT, PIT, GOOD],
        [EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY],
        [EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY],
        [EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY],
    ],
    "6x6_danger_maze": [
        [EMPTY, PIT, PIT, PIT, EMPTY, EMPTY],
        [EMPTY, EMPTY, BAD, BAD, EMPTY, WALL],
        [EMPTY, PIT, EMPTY, WALL, EMPTY, WALL],
        [EMPTY, BAD, EMPTY, EMPTY, EMPTY, EMPTY],
        [EMPTY, EMPTY, EMPTY, PIT, PIT, GOOD],
    ],
    "11x11_four_rooms_symmetrical" : [
        [WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL],
        [WALL, EMPTY, EMPTY, EMPTY, EMPTY, WALL, EMPTY, EMPTY, EMPTY, EMPTY, WALL],
        [WALL, EMPTY, EMPTY, EMPTY, EMPTY, WALL, EMPTY, EMPTY, EMPTY, EMPTY, WALL],
        [WALL, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, WALL],
        [WALL, EMPTY, EMPTY, EMPTY, EMPTY, WALL, EMPTY, EMPTY, EMPTY, EMPTY, WALL],
        [WALL, WALL, WALL, EMPTY, WALL, WALL, WALL, EMPTY, WALL, WALL, WALL],
        [WALL, EMPTY, EMPTY, EMPTY, EMPTY, WALL, EMPTY, EMPTY, EMPTY, EMPTY, WALL],
        [WALL, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, WALL],
        [WALL, EMPTY, EMPTY, EMPTY, EMPTY, WALL, EMPTY, EMPTY, EMPTY, EMPTY, WALL],
        [WALL, EMPTY, EMPTY, EMPTY, EMPTY, WALL, EMPTY, EMPTY, EMPTY, GOOD, WALL],
        [WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL],
    ],
    "13x13_four_rooms_original": [
        [WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL],
        [WALL, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, WALL, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, WALL],
        [WALL, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, WALL, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, WALL],
        [WALL, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, WALL],
        [WALL, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, WALL, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, WALL],
        [WALL, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, WALL, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, WALL],
        [WALL, WALL, EMPTY, WALL, WALL, WALL, WALL, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, WALL],
        [WALL, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, WALL, WALL, WALL, EMPTY, WALL, WALL, WALL],
        [WALL, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, WALL, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, WALL],
        [WALL, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, WALL, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, WALL],
        [WALL, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, WALL],
        [WALL, EMPTY, EMPTY, EMPTY, EMPTY, EMPTY, WALL, EMPTY, EMPTY, EMPTY, EMPTY, GOOD, WALL],
        [WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL, WALL],
    ],
}
# fmt: on


def _move(row, col, a, nrow, ncol):
    if a == LEFT:
        col = max(col - 1, 0)
    elif a == DOWN:
        row = min(row + 1, nrow - 1)
    elif a == RIGHT:
        col = min(col + 1, ncol - 1)
    elif a == UP:
        row = max(row - 1, 0)
    elif a == STAY:
        pass
    else:
        raise ValueError("illegal action")
    return (row, col)


class Gridworld(gym.Env):
    """
    Gridworld where the agent has to reach a goal while avoid penalty tiles.
    Harder versions include:
    - Quicksand tiles, where the agent gets stuck with 90% probability,
    - Distracting rewards,
    - One-directional tiles, where the agent can only move in one direction
    (all other actions will fail).

    ## Grid
    The grid is defined by a 2D array of integers. It is possible to define
    custom grids.

    ## Action Space
    The action is discrete in the range `{0, 4}`.

    - 0: Move left
    - 1: Move down
    - 2: Move right
    - 3: Move up
    - 4: Stay (do not move)

    It is possible to remove the `STAY` action by making the environment with `no_stay=True`.

    If the agent is in a "quicksand" tile, any action will fail with 90% probability.

    ## Observation Space

    #### Default
    The observation is discrete in the range `{0, n_rows * n_cols - 1}`.
    Each integer denotes the current location of the agent.
    For example, in a 3x3 grid the observations are

     0 1 2
     3 4 5
     6 7 8

    #### Coordinate Wrapper
    `gym_gridworlds.observation_wrappers.MatrixCoordinateWrapper(env)` returns
    matrix coordinates `(row, col)`. In the above example, `obs = 3` becomes `obs = (1, 0)`.

    #### Binary Wrapper
    `gym_gridworlds.observation_wrappers.MatrixBinaryWrapper(env)` returns a map
    of the environment with one 1 in the agent's position. In the above example,
    `obs = 3` becomes

     0 0 0
     1 0 0
     0 0 0

    #### RGB
    To use classic RGB pixel observations, make the environment with
    `render_mode="rgb_array"` and then wrap it with `gymnasium.wrappers.AddRenderObservation`.

    #### Partial RGB
    Pixel observations can be made partial by passing `view_radius`. For example,
    if `view_radius=1` the rendering will show the content of only the tiles
    around the agent, while all other tiles will be filled with white noise.

    #### Noisy Observations
    All types of observations can be made noisy by making the environment with
    `observation_noise=0.2` (or any other float in `[0, 1)`).
    For non-pixels observations, the float represents the probability that the
    position observed by the agent is random.
    For pixels (RGB) observations: the float represents the probability that a
    pixel will be white noise.

    ## Starting State
    By default, the episode starts with the agent at the top-left tile `(0, 0)`.
    You can manually select the starting position by making the environment with
    the argument `start_pos`, e.g., `start_pos=(3, 4)`.
    You can use the key "max" to automatically select the end of the grid, e.g.,
    `start_pos=("max", 0)` will place the agent at the bottom-right corner.
    If you make the environment with `start_pos=None`, the starting position will be random.
    In both cases (fixed and random), the starting position cannot be a tile with
    a wall or a pit.

    ## Transition
    By default, the transition is deterministic except in quicksand tiles,
    where any action fails with 90% probability (the agent does not move).
    Transition can be made stochastic everywhere by passing `random_action_prob`.
    This is the probability that the action will be random.
    For example, if `random_action_prob=0.1` there is a 10% chance that the agent
    will do a random action instead of doing the one passed to `self.step(action)`.
    Another way to add stochasticity is with `slippery_prob`, which is the probability
    that the agent slips and moves twice (similar to "sticky actions" in other environments).

    ## Rewards
    - Doing STAY at the goal: +1
    - Doing STAY at a distracting goal: 0.1
    - Any action in penalty tiles: -10
    - Any action in small penalty tiles: -0.1
    - Walking on a pit tile: -100
    - Otherwise: 0

    If the environment is made with `no_stay=True`, then the agent receives positive
    rewards for any action done in a goal state. Note that the reward still depends
    on the current state and not on the next state.

    Positive rewards position can be randomized at every reset by making the
    environment with `random_goals=True`.

    #### Noisy Rewards
    White noise can be added to all rewards by passing `reward_noise_std`,
    or only to nonzero rewards with `nonzero_reward_noise_std`.

    #### Auxiliary Rewards
    An auxiliary negative reward based on the Manhattan distance to the closest
    goal can be added by passing `distance_reward=True`. The distance is scaled
    according to the size of the grid.

    ## Episode End
    By default, an episode ends if any of the following happens:
    - A positive reward is collected (termination),
    - Walking on a pit tile (termination),
    - The length of the episode is max_episode_steps (truncation).

    ## Rendering
    Human mode renders the environment as a grid with colored tiles.

    - Black: empty tiles
    - White: pits
    - Purple: walls
    - Black with gray arrow: empty one-directional tile
    - Green: goal
    - Pale green: distracting goal
    - Red: penalty tiles
    - Pale red: penalty tiles
    - Blue: agent
    - Pale yellow: quicksand
    - Orange arrow: last agent's action (LEFT, RIGHT, UP, DOWN)
    - Orange dot: last agent's action (STAY)

    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    def __init__(
        self,
        grid: str,
        start_pos: Optional[tuple] = (0, 0),
        random_goals: Optional[bool] = False,
        no_stay: Optional[bool] = False,
        distance_reward: Optional[bool] = False,
        render_mode: Optional[str] = None,
        random_action_prob: Optional[float] = 0.0,
        slippery_prob: Optional[float] = 0.0,
        reward_noise_std: Optional[float] = 0.0,
        nonzero_reward_noise_std: Optional[float] = 0.0,
        observation_noise: Optional[float] = 0.0,
        view_radius: Optional[int] = 99999,
        max_resolution: Optional[tuple] = (256, 256),
        **kwargs,
    ):
        self.random_goals = random_goals
        self.grid_key = grid
        self.grid = np.asarray(GRIDS[self.grid_key])
        self.start_pos = start_pos
        if self.start_pos is not None:
            self.start_pos = tuple(
                y - 1 if x == "max" else x for x, y in zip(start_pos, self.grid.shape)
            )
        self.n_rows, self.n_cols = self.grid.shape

        if start_pos is not None:
            assert (
                0 <= self.start_pos[0] < self.n_rows and
                0 <= self.start_pos[1] < self.n_cols
            ), f"received {self.start_pos} starting position, but bounds are {(self.n_rows, self.n_cols)})"   # fmt: skip
            assert (
                self.grid[self.start_pos] not in [WALL, PIT]
            ), "the agent cannot start in a pit or a wall tile"   # fmt: skip

        self.no_stay = no_stay
        self.random_action_prob = random_action_prob
        self.slippery_prob = slippery_prob
        self.reward_noise_std = reward_noise_std
        self.nonzero_reward_noise_std = nonzero_reward_noise_std
        assert 0.0 <= observation_noise < 1.0, "observation_noise must be in [0.0, 1.0)"
        self.observation_noise = observation_noise
        self.distance_reward = distance_reward
        self.observation_space = gym.spaces.Discrete(self.n_cols * self.n_rows)

        self.action_space = gym.spaces.Discrete(4 if no_stay else 5)
        self.agent_pos = None
        self.last_action = None

        self.view_radius = view_radius
        self.render_mode = render_mode
        self.window_surface = None
        self.clock = None

        # Rendering attributes
        max_width, max_height = max_resolution
        base_tile_size = 64

        # Base rendering size (without any borders)
        base_width = self.n_cols * base_tile_size
        base_height = self.n_rows * base_tile_size

        # Scale down if it exceeds max_resolution (preserve aspect ratio)
        scale = min(max_width / base_width, max_height / base_height, 1.0)
        tile_size = (base_tile_size * scale) // 1

        # White padding around each tile: 5% of tile size
        white_pad_size = (0.05 * tile_size) // 1

        # Total rendering with white border
        render_width = self.n_cols * (tile_size + white_pad_size) + white_pad_size
        render_height = self.n_rows * (tile_size + white_pad_size) + white_pad_size

        # Black padding around the whole grid: 5% of full rendering
        black_pad_size = (0.05 * max(render_width, render_height)) // 1

        # Total window size including black border
        window_width = int(render_width + 2 * black_pad_size)
        window_height = int(render_height + 2 * black_pad_size)

        self.window_size = (window_width, window_height)
        self.tile_size = int(tile_size)
        self.white_pad_size = int(white_pad_size)
        self.black_pad_size = int(black_pad_size)


    def set_state(self, state):
        self.agent_pos = np.unravel_index(state, (self.n_rows, self.n_cols))

    def get_state(self):
        pos = self.agent_pos
        if self.observation_noise > 0.0:
            if self.np_random.random() < self.observation_noise:
                pos = (
                    self.np_random.integers(0, self.n_rows),
                    self.np_random.integers(0, self.n_cols),
                )  # note that the random position can be also a wall or a pit
        return np.ravel_multi_index(pos, (self.n_rows, self.n_cols))

    def reset(self, seed: int = None, **kwargs):
        super().reset(seed=seed, **kwargs)
        self._reset(seed, **kwargs)
        if self.render_mode is not None and self.render_mode == "human":
            self.render()
        return self.get_state(), {}

    def step(self, action: int):
        obs, reward, terminated, truncated, info = self._step(action)
        if self.render_mode is not None and self.render_mode == "human":
            self.render()
        return obs, reward, terminated, truncated, info

    def _randomize_agent_pos(self):
        allowed_tiles = np.argwhere(
            np.logical_and(self.grid != WALL, self.grid != PIT),
        )
        n_allowed = allowed_tiles.shape[0]
        assert n_allowed != 0, "there is no tile where the agent can spawn"
        self.agent_pos = tuple(allowed_tiles[self.np_random.integers(n_allowed)])

    def _randomize_goals(self):
        original_grid = self.grid.copy()
        goals_bool = np.logical_or(self.grid == GOOD, self.grid == GOOD_SMALL)
        goals = np.argwhere(goals_bool)
        self.grid[goals_bool] = EMPTY
        for goal in goals:
            allowed_tiles = np.argwhere(self.grid == EMPTY)
            n_allowed = allowed_tiles.shape[0]
            assert n_allowed != 0, "there is no tile where the agent can spawn"
            new_goal = allowed_tiles[self.np_random.integers(n_allowed)]
            self.grid[tuple(new_goal)] = original_grid[tuple(goal)]

    def _reset(self, seed: int = None, **kwargs):
        self.grid = np.asarray(GRIDS[self.grid_key])
        if self.random_goals:
            self._randomize_goals()
        if self.start_pos is None:
            self._randomize_agent_pos()
        else:
            self.agent_pos = self.start_pos
        self.last_action = None
        self.last_pos = None

    def _step(self, action: int):
        self.last_pos = self.agent_pos
        if self.np_random.random() < self.random_action_prob:
            action = self.action_space.sample()  # random action if transition is noisy
        self.last_action = action

        terminated = False
        reward = REWARDS[self.grid[self.agent_pos]] * 1.0  # float
        if self.grid[self.agent_pos] in [GOOD, GOOD_SMALL]:
            if action == STAY or self.no_stay:  # positive rewards are collected only with STAY
                terminated = True
            else:
                reward = 0

        if self.reward_noise_std > 0.0:
            reward += self.np_random.normal() * self.reward_noise_std
        if reward != 0.0 and self.nonzero_reward_noise_std > 0.0:
            reward += self.np_random.normal() * self.nonzero_reward_noise_std

        if self.distance_reward:
            closest_goal = np.abs(
                np.argwhere(self.grid == GOOD) - self.agent_pos
            ).sum(1).min()
            reward -= closest_goal / (self.n_rows * self.n_cols)

        if self.grid[self.agent_pos] == QCKSND and self.np_random.random() > 0.1:
            pass  # fail to move in quicksand
        else:
            if (
                self.grid[self.agent_pos] == LEFT and action != LEFT or
                self.grid[self.agent_pos] == RIGHT and action != RIGHT or
                self.grid[self.agent_pos] == UP and action != UP or
                self.grid[self.agent_pos] == DOWN and action != DOWN
            ):  # fmt: skip
                pass  # fail to move in one-directional tile
            else:
                if self.no_stay and action == STAY:
                    raise ValueError("illegal action")
                self.agent_pos = _move(
                    self.agent_pos[0],
                    self.agent_pos[1],
                    action,
                    self.n_rows,
                    self.n_cols,
                )

            if self.grid[self.agent_pos] == PIT:
                terminated = True  # agent dies
                reward = REWARDS[PIT]
            elif self.grid[self.agent_pos] == WALL:
                self.agent_pos = self.last_pos  # can't walk on walls

            # Move again if slipped
            if self.slippery_prob > 0.0 and self.np_random.random() < self.slippery_prob:
                    self.agent_pos = _move(
                        self.agent_pos[0],
                        self.agent_pos[1],
                        action,
                        self.n_rows,
                        self.n_cols,
                    )
                    if self.grid[self.agent_pos] == PIT:
                        terminated = True
                        reward = REWARDS[PIT]
                    elif self.grid[self.agent_pos] == WALL:
                        self.agent_pos = self.last_pos

        if self.distance_difference_reward:
            closest_goal_new_dist = np.abs(
                np.argwhere(self.grid == GOOD) - self.agent_pos
            ).sum(1).min()
            reward -= (closest_goal_new_dist - closest_goal_dist) * (100 / self.n_rows / self.n_cols)

        return self.get_state(), reward, terminated, False, {}

    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return
        else:  # self.render_mode in {"human", "rgb_array"}:
            return self._render_gui(self.render_mode)

    def _render_gui(self, mode):
        try:
            import pygame
        except ImportError as e:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gymnasium[toy-text]`"
            ) from e

        if self.window_surface is None:
            pygame.init()
            if mode == "human":
                pygame.display.init()
                pygame.display.set_caption(self.unwrapped.spec.id)
                self.window_surface = pygame.display.set_mode(self.window_size)
            elif mode == "rgb_array":
                self.window_surface = pygame.Surface(self.window_size)

        assert (
            self.window_surface is not None
        ), "Something went wrong with pygame. This should never happen."

        if self.clock is None:
            self.clock = pygame.time.Clock()

        def arrow_head(pos, size, dir):
            if dir == LEFT:
                return (
                    (end_pos[0], end_pos[1] - size[1]),
                    (end_pos[0], end_pos[1] + size[1]),
                    (end_pos[0] - size[0], end_pos[1]),
                )
            elif dir == DOWN:
                return (
                    (end_pos[0] - size[0], end_pos[1]),
                    (end_pos[0] + size[0], end_pos[1]),
                    (end_pos[0], end_pos[1] + size[1]),
                )
            elif dir == RIGHT:
                return (
                    (end_pos[0], end_pos[1] - size[1]),
                    (end_pos[0], end_pos[1] + size[1]),
                    (end_pos[0] + size[0], end_pos[1]),
                )
            elif dir == UP:
                return (
                    (end_pos[0] - size[0], end_pos[1]),
                    (end_pos[0] + size[0], end_pos[1]),
                    (end_pos[0], end_pos[1] - size[1]),
                )

        tile_with_pad_size = self.tile_size + self.white_pad_size
        bw_pad = self.black_pad_size + self.white_pad_size

        # draw black background for the padding
        self.window_surface.fill(Color.BLACK)

        # draw white background for the grid (shift according to black padding)
        background_init = (self.black_pad_size, self.black_pad_size)
        background_dims = (
            self.n_cols * tile_with_pad_size + self.white_pad_size,
            self.n_rows * tile_with_pad_size + self.white_pad_size,
        )
        background = pygame.Rect(background_init, background_dims)
        pygame.draw.rect(self.window_surface, Color.WHITE, background)

        # draw tiles
        for y in range(self.n_rows):
            for x in range(self.n_cols):
                pos = (
                    x * tile_with_pad_size + bw_pad,
                    y * tile_with_pad_size + bw_pad,
                )
                rect = pygame.Rect(pos[0], pos[1], self.tile_size, self.tile_size)

                # mask unobservable tiles with white noise
                if not (
                    y >= self.agent_pos[0] - self.view_radius and
                    y <= self.agent_pos[0] + self.view_radius and
                    x >= self.agent_pos[1] - self.view_radius and
                    x <= self.agent_pos[1] + self.view_radius
                ):
                    grain = 5
                    for i in range(grain):
                        for j in range(grain):
                            rect = pygame.Rect(
                                pos[0] + i / grain * self.tile_size,
                                pos[1] + j / grain * self.tile_size,
                                tile_with_pad_size / grain,
                                tile_with_pad_size / grain,
                            )
                            rnd_color = self.np_random.random(3) * 255
                            rnd_color = [(rnd_color * (0.2989, 0.5870, 0.1140)).sum()] * 3  # grayscale
                            pygame.draw.rect(self.window_surface, rnd_color, rect)
                    continue

                # draw environment elements
                pygame.draw.rect(self.window_surface, COLORMAP[self.grid[y][x]], rect)

                # draw agent
                if (y, x) == self.agent_pos:
                    pygame.draw.ellipse(self.window_surface, Color.BLUE, rect)

                # draw arrow for one-directional tiles
                if self.grid[y][x] in [LEFT, RIGHT, UP, DOWN]:
                    arrow_width = self.tile_size // 3
                    if self.grid[y][x] == LEFT:
                        start_pos = (
                            pos[0] + self.tile_size,
                            pos[1] + self.tile_size // 2,
                        )
                        end_pos = (
                            pos[0] + self.tile_size // 2,
                            pos[1] + self.tile_size // 2,
                        )
                    elif self.grid[y][x] == DOWN:
                        start_pos = (
                            pos[0] + self.tile_size // 2,
                            pos[1],
                        )
                        end_pos = (
                            pos[0] + self.tile_size // 2,
                            pos[1] + self.tile_size // 2,
                        )
                    elif self.grid[y][x] == RIGHT:
                        start_pos = (
                            pos[0],
                            pos[1] + self.tile_size // 2
                        )
                        end_pos = (
                            pos[0] + self.tile_size // 2,
                            pos[1] + self.tile_size // 2,
                        )
                    elif self.grid[y][x] == UP:
                        start_pos = (
                            pos[0] + self.tile_size // 2,
                            pos[1] + self.tile_size,
                        )
                        end_pos = (
                            pos[0] + self.tile_size // 2,
                            pos[1] + self.tile_size // 2,
                        )
                    else:
                        pass
                    pygame.draw.polygon(
                        self.window_surface,
                        Color.GRAY,
                        (start_pos, end_pos),
                        arrow_width,
                    )
                    arr_pos = arrow_head(
                        end_pos,
                        (self.tile_size // 2,) * 2,
                        self.grid[y][x],
                    )
                    pygame.draw.polygon(self.window_surface, Color.GRAY, arr_pos, 0)

                # some pixels are white noise
                if self.observation_noise > 0.0:
                    grain = 5
                    for i in range(grain):
                        for j in range(grain):
                            if self.np_random.random() < self.observation_noise:
                                rect = pygame.Rect(
                                    pos[0] + i / grain * self.tile_size,
                                    pos[1] + j / grain * self.tile_size,
                                    tile_with_pad_size / grain,
                                    tile_with_pad_size / grain,
                                )
                                rnd_color = self.np_random.random(3) * 255
                                rnd_color = [(rnd_color * (0.2989, 0.5870, 0.1140)).sum()] * 3  # grayscale
                                pygame.draw.rect(self.window_surface, rnd_color, rect)

        # draw last action
        if self.last_pos is not None:
            x = self.last_pos[1]
            y = self.last_pos[0]

            if self.last_action == STAY:  # draw circle
                pos = (
                    x * tile_with_pad_size + bw_pad,
                    y * tile_with_pad_size + bw_pad,
                )
                rect = pygame.Rect(
                    (pos[0], pos[1]),
                    (self.tile_size, self.tile_size),
                )
                rect.centerx = pos[0] + self.tile_size / 2
                rect.centery = pos[1] + self.tile_size / 2
                pygame.draw.ellipse(self.window_surface, Color.ORANGE, rect.scale_by(0.5))
            else:  # draw arrow
                pos = (
                    x * tile_with_pad_size + bw_pad + self.tile_size // 2,
                    y * tile_with_pad_size + bw_pad + self.tile_size // 2,
                )
                arrow_width = self.tile_size // 6
                if self.last_action == LEFT:
                    end_pos = (pos[0] - self.tile_size // 4, pos[1])
                elif self.last_action == DOWN:
                    end_pos = (pos[0], pos[1] + self.tile_size // 4)
                elif self.last_action == RIGHT:
                    end_pos = (pos[0] + self.tile_size // 4, pos[1])
                elif self.last_action == UP:
                    end_pos = (pos[0], pos[1] - self.tile_size // 4)
                else:
                    raise ValueError("illegal action")

                pygame.draw.polygon(
                    self.window_surface,
                    Color.ORANGE,
                    (pos, end_pos),
                    arrow_width,
                )
                arr_pos = arrow_head(
                    end_pos,
                    (self.tile_size // 5,) * 2,
                    self.last_action,
                )
                pygame.draw.polygon(self.window_surface, Color.ORANGE, arr_pos, 0)

        if mode == "human":
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        elif mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.window_surface)), axes=(1, 0, 2)
            )
        else:
            raise NotImplementedError

    def close(self):
        if self.window_surface is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()


class RiverSwim(Gridworld):
    """
    First presented in "An analysis of model-based Interval Estimation for Markov Decision Processes".
    Implementation according to https://rlgammazero.github.io/docs/2020_AAAI_tut_part1.pdf

    One-dimensional grid with positive rewards at its ends, 0.01 in the leftmost
    tile and 1 in the rightmost.
    The agent starts either in the 2nd or 3rd leftmost tile, the only actions are
    LEFT and RIGHT, and the transition is stochastic.

    This is an infinite horizon MDP, there is no terminal state.
    Episodes are truncated after max_episode_steps (default 200).
    """

    def __init__(self, **kwargs):
        Gridworld.__init__(self, **kwargs)
        self.grid[0] = 0.01  # we use self.grid for rendering
        self.grid[-1] = 1.0
        self.action_space = gym.spaces.Discrete(2)  # only LEFT and RIGHT

    def _reset(self, seed: int = None, **kwargs):
        Gridworld._reset(self, seed=seed, **kwargs)
        self.agent_pos = (0, self.np_random.integers(1, 3))  # 2nd or 3rd tile
        return self.get_state(), {}

    def _step(self, action: int):
        state = np.ravel_multi_index(self.agent_pos, (self.n_rows, self.n_cols))
        first = 0
        last = self.n_rows * self.n_cols - 1

        # Map action to match Gridworld notation
        if action == 0:
            action = LEFT
        elif action == 1:
            action = RIGHT
        else:
            raise NotImplementedError("illegal action")

        # Stochastic transition
        original_action = action
        r = self.np_random.random()
        if action == RIGHT:
            if state == first:
                if r < 0.4:
                    action = LEFT  # or stay, it's equivalent
            elif state == last:
                if r < 0.4:
                    action = LEFT
            else:
                if r < 0.05:
                    action = LEFT
                elif r < 0.65:
                    action = STAY

        obs, _, _, truncated, info = Gridworld._step(self, action)
        terminated = False  # infinite horizon
        self.last_action = original_action

        reward = 0.0
        if state == last and action == RIGHT and original_action == RIGHT:
            reward = 1.0
        elif state == first and action == LEFT and original_action == LEFT:
            reward = 0.01

        return obs, reward, terminated, truncated, info
