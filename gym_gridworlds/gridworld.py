import os
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
RND_MOVE = 13  # see rnd_move.py

# action IDs (also used for one-directional states)
LEFT = 0
RIGHT = 1
DOWN = 2
UP = 3
STAY = 4

# diagonal actions are not used in the default grids, but can be used in harder grids (e.g., travel_field.py)
UP_LEFT = 5
DOWN_LEFT = 6
DOWN_RIGHT = 7
UP_RIGHT = 8

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

# tile movement mapping
ACTION_TO_VEC = {
    STAY: (0, 0),
    LEFT: (0, -1),
    DOWN: (+1, 0),
    RIGHT: (0, +1),
    UP: (-1, 0),
    UP_LEFT: (-1, -1),
    UP_RIGHT: (-1, +1),
    DOWN_LEFT: (+1, -1),
    DOWN_RIGHT: (+1, +1),
}

def _move(pos, action, shape):
    movement = ACTION_TO_VEC.get(action, None)
    if movement is None:
        raise ValueError("illegal action")
    new_row = pos[0] + movement[0]
    new_col = pos[1] + movement[1]
    if not (0 <= new_col < shape[1] and 0 <= new_row < shape[0]):
        return pos
    return (new_row, new_col)

# tile colors
COLORMAP = dict()
COLORMAP[EMPTY] = Color.BLACK
COLORMAP[QCKSND] = Color.PALE_YELLOW
COLORMAP[GOOD_SMALL] = Color.DARK_GREEN
COLORMAP[GOOD] = Color.GREEN
COLORMAP[BAD] = Color.RED
COLORMAP[BAD_SMALL] = Color.PALE_RED
COLORMAP[WALL] = Color.GRAY
COLORMAP[PIT] = Color.WHITE
for action in ACTION_TO_VEC:
    COLORMAP[action] = Color.BLACK

# to parse grids from txt file
GRID_ENCODING = {
    ".": EMPTY,
    "□": WALL,
    "_": QCKSND,
    " ": PIT,
    "O": GOOD,
    "o": GOOD_SMALL,
    "X": BAD,
    "x": BAD_SMALL,
    "←": LEFT,
    "→": RIGHT,
    "↑": UP,
    "↓": DOWN,
    "↖": UP_LEFT,
    "↗": UP_RIGHT,
    "↙": DOWN_LEFT,
    "↘": DOWN_RIGHT,
    "*": RND_MOVE,
}

def load_grid(file_path, encoding):
    cwd = os.path.dirname(__file__)
    file_path = os.path.join(cwd, "grids", file_path + ".txt")
    with open(file_path, "r", encoding="utf-8") as f:
        return np.asarray([
            [encoding[c] for c in line.strip()]
            for line in f
            if line.strip()  # remove empty lines
        ])


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
    - 1: Move right
    - 2: Move down
    - 3: Move up
    - 4: Stay (do not move)

    It is possible to remove the `STAY` action by making the environment with `no_stay=True`.
    Diagonal actions `{5, 8}` for `{UP_LEFT, DOWN_LEFT, DOWN_RIGHT, UP_RIGHT}`
    are also supported but not used in the default MDP.

    If the agent is in a "quicksand" tile, any action will fail with 90% probability.

    ## Observation Space

    #### Default (True State)
    The observation is discrete in the range `{0, n_rows * n_cols - 1}`.
    Each integer denotes the current location of the agent.
    For example, in a 3x3 grid the observations are

     0 1 2
     3 4 5
     6 7 8

     > The true state is always passed with the `info` dictionary, to retrieve
     it even when wrappers are used. This makes debugging easier (e.g., it is
     possible to count state visits even when RGB wrappers are used).

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
    Auxiliary rewards based on the Manhattan distance to the closest goal can be
    added by passing `distance_reward=True` or `distance_difference_reward=True`.
    The former is `distance_at_current_state / max_distance`, i.e., the distance
    from the current state scaled according to the size of the grid to be in the range [-1, 0].
    The latter is `distance_at_current_state - distance_at_next_state`, thus it
    can be +1 (if the agent moves closer to the goal), 0 (if it does STAY),
    or -1 (if it moves further from the goal).

    ## Episode End
    By default, an episode ends if any of the following happens:
    - A positive reward is collected (termination),
    - Walking on a pit tile (termination),
    - The length of the episode is max_episode_steps (truncation).

    It is possible to remove termination altogether by making the environment
    with `infinite_horizon=True`.

    ## Rendering
    Human mode renders the environment as a grid with colored tiles.

    - Black: empty tiles
    - White: pits
    - Gray: walls
    - Black with purple arrow: empty one-directional tile
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
        encoding: Optional[dict] = GRID_ENCODING,
        start_pos: Optional[tuple] = (0, 0),
        infinite_horizon: Optional[bool] = False,
        random_goals: Optional[bool] = False,
        no_stay: Optional[bool] = False,
        distance_reward: Optional[bool] = False,
        distance_difference_reward: Optional[bool] = False,
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
        self.original_grid = load_grid(grid, encoding)
        self.grid = self.original_grid.copy()
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
        self.infinite_horizon = infinite_horizon
        self.random_action_prob = random_action_prob
        self.slippery_prob = slippery_prob
        self.reward_noise_std = reward_noise_std
        self.nonzero_reward_noise_std = nonzero_reward_noise_std
        assert 0.0 <= observation_noise < 1.0, "observation_noise must be in [0.0, 1.0)"
        self.observation_noise = observation_noise
        self.distance_reward = distance_reward
        self.distance_difference_reward = distance_difference_reward
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
        info = self._reset(seed, **kwargs)
        if self.render_mode is not None and self.render_mode == "human":
            self.render()
        obs = self.get_state()
        info["state"] = obs
        return obs, info

    def step(self, action: int):
        obs, reward, terminated, truncated, info = self._step(action)
        if self.render_mode is not None and self.render_mode == "human":
            self.render()
        info["state"] = obs
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
        self.grid = self.original_grid.copy()
        if self.random_goals:
            self._randomize_goals()
        if self.start_pos is None:
            self._randomize_agent_pos()
        else:
            self.agent_pos = self.start_pos
        self.last_action = None
        self.last_pos = None
        return {}

    def _step(self, action: int):
        self.last_pos = self.agent_pos
        if self.np_random.random() < self.random_action_prob:
            action = self.action_space.sample()  # random action if transition is noisy
        self.last_action = action

        terminated = False
        reward = REWARDS[self.grid[self.agent_pos]] * 1.0  # float
        if self.grid[self.agent_pos] in [GOOD, GOOD_SMALL]:
            if action == STAY or self.no_stay:  # positive rewards are collected only with STAY
                terminated = True  # positive rewards terminate the episode (unless self.infinite_horizon=True)
            else:
                reward = 0

        if self.reward_noise_std > 0.0:
            reward += self.np_random.normal() * self.reward_noise_std
        if reward != 0.0 and self.nonzero_reward_noise_std > 0.0:
            reward += self.np_random.normal() * self.nonzero_reward_noise_std

        if self.grid[self.agent_pos] == QCKSND and self.np_random.random() > 0.1:
            pass  # fail to move in quicksand
        else:
            if (
                self.grid[self.agent_pos] in ACTION_TO_VEC and
                self.grid[self.agent_pos] != action
            ):  # fmt: skip
                pass  # fail to move in one-directional tile
            else:
                self.agent_pos = _move(
                    self.agent_pos,
                    action,
                    (self.n_rows, self.n_cols),
                )
            if self.grid[self.agent_pos] == PIT:
                terminated = True  # agent dies
                reward = REWARDS[PIT]
            elif self.grid[self.agent_pos] == WALL:
                self.agent_pos = self.last_pos  # can't walk on walls

            # Move again if slipped
            if self.slippery_prob > 0.0 and self.np_random.random() < self.slippery_prob:
                    self.agent_pos = _move(
                        self.agent_pos,
                        action,
                        (self.n_rows, self.n_cols),
                    )
                    if self.grid[self.agent_pos] == PIT:
                        terminated = True
                        reward = REWARDS[PIT]
                    elif self.grid[self.agent_pos] == WALL:
                        self.agent_pos = self.last_pos

        # Auxiliary reward based on distance to the closest goal
        def distance_from_closest_tile_type(tile_type, pos):
            dist = np.linalg.norm(
                np.argwhere(self.grid == tile_type) - pos,
                ord=1,
                axis=1,
            )
            if len(dist) == 0:
                return 0.0
            return dist.min()

        if self.distance_reward or self.distance_difference_reward:
            goal_dist = distance_from_closest_tile_type(GOOD, self.agent_pos)
            if self.distance_difference_reward:
                old_goal_dist = distance_from_closest_tile_type(GOOD, self.last_pos)
                reward -= goal_dist - old_goal_dist
            else:
                max_dist = np.linalg.norm(
                    [self.n_rows - 1, self.n_cols - 1],
                    ord=1,
                )
                reward -= goal_dist / max_dist

        if self.infinite_horizon:
            terminated = False
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
            movement = ACTION_TO_VEC.get(dir, None)
            if movement is None:
                raise ValueError("illegal action")
            return (
                (pos[0] - movement[0] * size[0], pos[1] + movement[1] * size[1]),
                (pos[0] + movement[0] * size[0], pos[1] - movement[1] * size[1]),
                (pos[0] + movement[1] * size[0], pos[1] + movement[0] * size[1]),
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
                movement = ACTION_TO_VEC.get(self.grid[y][x], None)
                if movement is not None:
                    arrow_width = max(self.tile_size // 3, 1)
                    center = (
                        pos[0] + self.tile_size // 2,
                        pos[1] + self.tile_size // 2,
                    )
                    start_pos = (
                        center[0] - movement[1] * (self.tile_size // 2 - self.white_pad_size),
                        center[1] - movement[0] * (self.tile_size // 2 - self.white_pad_size),
                    )
                    pygame.draw.polygon(
                        self.window_surface,
                        Color.PURPLE,
                        (start_pos, center),
                        arrow_width,
                    )
                    arr_pos = arrow_head(
                        center,
                        ((self.tile_size // 2 - self.white_pad_size),) * 2,
                        self.grid[y][x],
                    )
                    pygame.draw.polygon(self.window_surface, Color.PURPLE, arr_pos, 0)

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
        if self.last_pos is not None and self.last_action is not None:
            x = self.last_pos[1]
            y = self.last_pos[0]

            if self.last_action == STAY:  # draw circle
                pos = (
                    x * tile_with_pad_size + bw_pad,
                    y * tile_with_pad_size + bw_pad,
                )
                rect = pygame.Rect(pos, (self.tile_size,) * 2)
                rect.centerx = pos[0] + self.tile_size / 2
                rect.centery = pos[1] + self.tile_size / 2
                pygame.draw.ellipse(self.window_surface, Color.ORANGE, rect.scale_by(0.5))
            else:  # draw arrow
                movement = ACTION_TO_VEC.get(self.last_action, None)
                if movement is None:
                    raise ValueError("illegal action")
                arrow_width = max(self.tile_size // 6, 1)
                pos = (
                    x * tile_with_pad_size + bw_pad + self.tile_size // 2,
                    y * tile_with_pad_size + bw_pad + self.tile_size // 2,
                )
                end_pos = (
                    pos[0] + movement[1] * self.tile_size // 4,
                    pos[1] + movement[0] * self.tile_size // 4,
                )
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

    One-dimensional grid with positive rewards at its ends: 0.01 in the leftmost
    tile and 1 in the rightmost.
    The agent starts either in the 2nd or 3rd leftmost tile, the only actions are
    LEFT and RIGHT, and the transition is stochastic.

    This is an infinite horizon MDP, there is no terminal state.
    Episodes are truncated after `max_episode_steps` (default 200).
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
