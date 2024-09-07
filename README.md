## Description

Generic implementation of a gridworld environment for reinforcement learning based on [gymnasium](https://github.com/Farama-Foundation/Gymnasium).  
The default class `Gridworld` implements a "go-to goal" task where the agent has five actions (left, right, up, down, stay) and default transition function (e.g., doing "stay" in goal states ends the episode).  
You can change actions and transition function by implementing more classes. For example, in `RiverSwim` there are only two actions and no terminal state.  
You can create your own gridworld with different grids and rewards by defining its map.  
See `gym_gridworlds/gridworld.py` and `gym_gridworlds/gym.py` for more details.  


## Install and Examples

To install the environments run
```
pip install -e .
```

Run `python` and then
```python
import gymnasium
env = gymnasium.make("Gym-Gridworlds/Penalty-3x3-v0", render_mode="human")
env.reset()
env.step(1) # DOWN
env.step(4) # STAY
env.render()
```

to render the `Penalty-3x3-v0` gridworld (left figure), and
```python
import gymnasium
env = gymnasium.make("Gym-Gridworlds/Full-5x5-v0", render_mode="human")
env.reset()
env.step(1) # DOWN
env.render()
```

to render the `Full-5x5-v0` gridworld (right figure).

<p align="center">
  <img src="figures/gridworld_penalty_3x3.png" height=200 alt="Gridworld Penalty"> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <img src="figures/gridworld_full_5x5.png" height=200 alt="Gridworld Full">
</p>

- Black tiles are empty,
- White tiles are walls,
- Black tiles with gray arrows are tiles where the agent can move only in one direction (other actions will fail),
- Red tiles give negative rewards,
- Green tiles give positive rewards (the brighter, the higher),
- Yellow tiles are quicksands, where all actions will fail with 90% probability,
- The agent is the blue circle,
- The orange arrow denotes the agent's last action,
- The orange dot denotes that the agent did not try to move with its last action.

It is also possible to add noise to the transition and the reward functions.
For example, in the following environment
```python
import gymnasium
env = gymnasium.make("Gym-Gridworlds/Full-5x5-v0", random_action_prob=0.1, reward_noise_std=0.05)
```
the agent's action will fail with 10% probability and a random one will be performed instead,
and Gaussian noise with 0.05 standard deviation is added to the reward.

You can also turn the MDP into a POMDP and learn from partially-observable pixels,
by passing the `view_radius` argument. This way, only the tiles close to the agent
will be visible while far away tiles will be masked by white noise. For example,
this is the partially-observable version of the `Full-5x5-v0` gridworld above.

```python
import gymnasium
env = gymnasium.make("Gym-Gridworlds/Full-5x5-v0", render_mode="human", view_radius=1)
env.reset()
env.step(1) # DOWN
env.render()
```

<p align="center">
  <img src="figures/gridworld_full_5x5_partial.png" height=200 alt="Gridworld Full">
</p>
