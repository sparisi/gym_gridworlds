## Overview

Minimalistic implementation of gridworlds based on
[Gymnasium](https://github.com/Farama-Foundation/Gymnasium), useful for quickly
testing and prototyping reinforcement learning algorithms (both tabular and with
function approximation).  
The default class `Gridworld` implements a "go-to-goal" task where the agent has
five actions (left, right, up, down, stay) and default transition function
(e.g., doing "stay" in goal states ends the episode).  
You can change actions and transition function by implementing more classes.
For example, in `RiverSwim` there are only two actions and no terminal state.  


## Install and Examples

To install the environments run
```
pip install -e .
```

Run `python` and then

```python
import gymnasium
import gym_gridworlds
env = gymnasium.make("Gym-Gridworlds/TravelField-32x32-v0", render_mode="human")
env.reset()
env.step(1) # DOWN
env.step(4) # STAY
```
to render the `Penalty-3x3-v0` gridworld (left figure),

```python
import gymnasium
import gym_gridworlds
env = gymnasium.make("Gym-Gridworlds/Full-4x5-v0", render_mode="human")
env.reset()
env.step(1) # DOWN
```
to render the `Full-4x5-v0` gridworld (middle figure), and

```python
import gymnasium
import gym_gridworlds
env = gymnasium.make("Gym-Gridworlds/DangerMaze-6x6-v0", render_mode="human")
env.reset()
env.step(1) # DOWN
```
to render the `DangerMaze-6x6-v0` gridworld (right figure).

<p align="center">
  <img src="figures/gridworld_penalty_3x3.png" width="170" alt="Gridworld Penalty"> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <img src="figures/gridworld_full_4x5.png" width="220" alt="Gridworld Full"> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <img src="figures/gridworld_danger_maze_6x6.png" width="215" alt="Gridworld Full">
</p>

- Black tiles are empty,
- White tiles are pits (walking on them yields a large negative reward and the episode ends),
- Purple tiles are walls (the agent cannot step on them),
- Black tiles with gray arrows are tiles where the agent can move only in one direction (other actions will fail),
- Red tiles give negative rewards,
- Green tiles give positive rewards (the brighter, the higher),
- Yellow tiles are quicksands, where all actions will fail with 90% probability,
- The agent is the blue circle,
- The orange arrow denotes the agent's last action,
- The orange dot denotes that the agent did not try to move with its last action.

<table>
  <tr>
    <td align="center">
      <img src="figures/gridworld_empty_2x2.png" width="120">
    </td>
    <td>
      <p>
        The smallest pre-built environment is <code>Gym-Gridworlds/Empty-RandomStart-2x2-v0</code> (on the left):
        there are only 4 states, 5 actions, and the initial position is random.
        It is the simplest environment you can use to debug your algorithm.
      </p>
    </td>
  </tr>
</table>


## Optional Features

&#10148; <strong>Noisy Transition and Reward Functions</strong>  
```python
import gymnasium
import gym_gridworlds
env = gymnasium.make("Gym-Gridworlds/Full-4x5-v0", random_action_prob=0.1, reward_noise_std=0.05)
```
This makes the environment take a random action (instead of the action passed by
the agent) with 10% probability, and Gaussian noise with 0.05 standard deviation
is added to the reward.

&#10148; <strong>POMDP</strong>  
To turn the MDP into a POMDP and learn from partially-observable pixels, make
the environment with `view_radius=1` (or any integer). This way, only
the tiles close to the agent (within the view radius) will be visible, while
far away tiles will be masked by white noise. For example,
this is the partially-observable version of the `Full-4x5-v0` gridworld above.

```python
import gymnasium
import gym_gridworlds
env = gymnasium.make("Gym-Gridworlds/Full-4x5-v0", render_mode="human", view_radius=1)
env.reset()
env.step(1) # DOWN
```

<p align="center">
  <img src="figures/gridworld_full_4x5_partial.png" width="200" alt="Gridworld Full Partial">
</p>

&#10148; <strong>Noisy Observations</strong>  
Make the environment with `observation_noise=0.2` (or any float between 0 and 1).
With default observations, the float represents the probability that the position
observed by the agent is random. With RGB observations, it represents the
probability that a pixel is white noise, as shown below.

```python
import gymnasium
import gym_gridworlds
env = gymnasium.make("Gym-Gridworlds/Full-4x5-v0", render_mode="human", observation_noise=0.2)
env.reset()
env.step(1) # DOWN
```

<p align="center">
  <img src="figures/gridworld_full_4x5_noisy.png" width="200" alt="Gridworld Full Noisy">
</p>

&#10148; <strong>Random Goals</strong>  
Make the environment with `random_goals=True` to randomize the position of positive
rewards (positive only!) at every reset. To learn in this setting, you need to add
the rewards position to the observation (`MatrixWithGoalWrapper`), or to learn from pixels.


## Make Your Own Gridworld

1. Define your grid in `gym_gridworlds/gridworld.py`, for example
```python
GRIDS["5x5_wall"] = [
    [EMPTY, EMPTY, EMPTY, EMPTY, EMPTY],
    [EMPTY, WALL, WALL, WALL, EMPTY],
    [EMPTY, WALL, GOOD, EMPTY, EMPTY],
    [EMPTY, WALL, WALL, WALL, EMPTY],
    [EMPTY, EMPTY, EMPTY, EMPTY, EMPTY],
]
```

2. Register the environment in `gym_gridworlds/__init__.py`, for example
```python
register(
    id="Gym-Gridworlds/Wall-RandomStart-5x5-v0",
    entry_point="gym_gridworlds.gridworld:Gridworld",
    max_episode_steps=50,
    kwargs={
        "grid": "5x5_wall",
        "start_pos": None,  # random
    },
)
```

3. Try it
```python
import gymnasium
import gym_gridworlds
env = gymnasium.make("Gym-Gridworlds/Wall-RandomStart-5x5-v0", grid="5x5_wall", render_mode="human")
env.reset(seed=42)
```

<p align="center">
  <img src="figures/gridworld_wall_5x5.png" width="200" alt="Gridworld Full">
</p>



## Default MDP (`Gridworld` Class)

### <ins>Action Space</ins>
The action is discrete in the range `{0, 4}` for `{LEFT, DOWN, RIGHT, UP, STAY}`.
It is possible to remove the `STAY` action by making the environment with `no_stay=True`.

### <ins>Observation Space</ins>
&#10148; <strong>Default</strong>  
The observation is discrete in the range `{0, n_rows * n_cols - 1}`.
Each integer denotes the current location of the agent.
For example, in a 3x3 grid the observations are
```
 0 1 2
 3 4 5
 6 7 8
```

The observation can be transformed to better fit function approximation.
- `gym_gridworlds.observation_wrappers.CoordinateWrapper` returns matrix coordinates
`(row, col)`. In the above example, `obs = 3` becomes `obs = (1, 0)`.
- `gym_gridworlds.observation_wrappers.MatrixWrapper` returns a map of the environment
with one 1 at the agent's position. In the above example, `obs = 3` becomes
```
 0 0 0
 1 0 0
 0 0 0
 ```
 - See `gym_gridworlds.observation_wrappers` for more wrappers and examples.

&#10148; <strong>RGB</strong>  
To use classic RGB pixel observations, make the environment with
`render_mode="rgb_array"` and then wrap it with `gymnasium.wrappers.AddRenderObservation`.

&#10148; <strong>Partial RGB</strong>  
Pixel observations can be made partial by making the environment with `view_radius`.
For example, if `view_radius=1` the rendering will show the content of only the tiles
around the agent, while all other tiles will be filled with white noise.

&#10148; <strong>Noisy Observations</strong>  
Make the environment with `observation_noise=0.2` (or any float between 0 and 1).
With default observations, the float represents the probability that the position
observed by the agent is random. With RGB observations, it represents the
probability that a pixel is white noise.

### <ins>Starting State</ins>
By default, the episode starts with the agent at the top-left tile `(0, 0)`.
You can manually select the starting position by making the environment with
the argument `start_pos`, e.g., `start_pos=(3, 4)`.
You can use the key "max" to automatically select the end of the grid, e.g.,
`start_pos=("max", 0)` will place the agent at the bottom-right corner.
If you make the environment with `start_pos=None`, the starting position will be random.
In both cases (fixed and random), the starting position cannot be a tile with
a wall or a pit.

### <ins>Transition</ins>
By default, the transition is deterministic except in quicksand tiles,
where any action fails with 90% probability (the agent does not move).  
Transition can be made stochastic everywhere by passing `random_action_prob`.
This is the probability that the action will be random.
For example, if `random_action_prob=0.1` there is a 10% chance that the agent
will do a random action instead of doing the one passed to `self.step(action)`.  

### <ins>Rewards</ins>
- Doing `STAY` at the goal: +1
- Doing `STAY` at a distracting goal: 0.1
- Any action in penalty tiles: -10
- Any action in small penalty tiles: -0.1
- Walking on a pit tile: -100
- Otherwise: 0

If the environment is made with `no_stay=True`, then the agent receives positive
rewards for any action done in a goal state. Note that the reward still depends
on the current state and not on the next state.

Positive rewards position can be randomized at every reset by making the
environment with `random_goals=True`.

&#10148; <strong>Noisy Rewards</strong>  
White noise can be added to all rewards by passing `reward_noise_std`,
or only to nonzero rewards with `nonzero_reward_noise_std`.

&#10148; <strong>Auxiliary Rewards</strong>  
An auxiliary negative reward based on the Manhattan distance to the closest
goal can be added by passing `distance_reward=True`. The distance is scaled
according to the size of the grid.

### <ins>Episode End</ins>
By default, an episode ends if any of the following happens:
- A positive reward is collected (termination),
- Walking on a pit tile (termination),
- The length of the episode is `max_episode_steps` (truncation).
