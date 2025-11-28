"""
Move:   ← ↑ → ↓
Stay:   ENTER
Reset:  R
Quit:   Q

python playground.py ENVIRONMENT --record --env-arg= ...
--record to save a gif of the game
--env-arg to pass optional environment arguments

Example:
python playground.py Gym-Gridworlds/TravelField-28x28-v0 --env-arg distance_reward=True --env-arg no_stay=True --record
"""

import imageio
import numpy as np
import gymnasium
import gym_gridworlds
from gym_gridworlds.gridworld import LEFT, DOWN, RIGHT, UP, STAY
import argparse
from pynput import keyboard
import time
from pathlib import Path

# Mutable so we can update it in on_press`
program_running = [True]
sum_rewards = [0]

def parse_env_args(arg_list):
    env_kwargs = {}
    for item in arg_list:
        if "=" not in item:
            raise ValueError(f"Invalid format for --env-arg '{item}', expected key=value")
        key, value = item.split("=", 1)

        # Try to auto-convert type: int, float, bool
        if value.lower() == "true":
            value = True
        elif value.lower() == "false":
            value = False
        else:
            try:
                value = int(value)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    pass  # leave as string

        env_kwargs[key] = value

    return env_kwargs

parser = argparse.ArgumentParser()
parser.add_argument("env")
parser.add_argument(
    "--env-arg",
    action="append",
    default=[],
    help="Extra environment arguments in key=value format",
)
parser.add_argument("--record", action="store_true")
args = parser.parse_args()

env_kwargs = parse_env_args(args.env_arg)
env = gymnasium.make(args.env, render_mode="human", **env_kwargs)

if args.record:
    # Gymnasium human rendering does not return RGB array, so we must make a copy
    env_record = gymnasium.make(args.env, render_mode="rgb_array", **env_kwargs)
    frames = []


def step(action):
    if action not in env.action_space:
        print("Illegal action, skipped")
        return
    state = env.unwrapped.agent_pos
    next_obs, rwd, term, trunc, info = env.step(action)
    sum_rewards[0] = sum_rewards[0] + rwd
    next_state = env.unwrapped.agent_pos

    print(f"{state} | {action} | {rwd} | {next_state}")

    if term:
        print("Terminal state, sum of rewards:", sum_rewards[0])
        sum_rewards[0] = 0
    elif trunc:
        print("Time step limit, sum of rewards:", sum_rewards[0])
        sum_rewards[0] = 0

    if args.record:
        env_record.step(action)
        frame = env_record.render()
        if frame is not None:
            frames.append(frame)


def reset():
    seed = np.random.randint(999)
    env.reset(seed=seed)
    env.render()
    sum_rewards[0] = 0

    if args.record:
        env_record.reset(seed=seed)
        frame = env_record.render()
        if frame is not None:
            frames.append(frame)


def on_press(key):
    try:
        if key == keyboard.Key.up:
            step(UP)
        elif key == keyboard.Key.down:
            step(DOWN)
        elif key == keyboard.Key.left:
            step(LEFT)
        elif key == keyboard.Key.right:
            step(RIGHT)
        elif key == keyboard.Key.enter:
            step(STAY)
        elif key.char.isalpha() and key.char == "q":
            # Can't call env.close() or pygame will freeze everything
            program_running[0] = False
            return False
        elif key.char.isalpha() and key.char == "r":
            reset()
        else:
            pass
    except AttributeError:
        pass


print(
    "\n"
    "Move: \t← ↑ → ↓\n"
    "Stay: \tENTER\n"
    "Reset: \tR\n"
    "Quit: \tQ\n"
    "\n"
    "Prints are: (S | A | R | S')"
)

reset()

# Start listener in a non-blocking way
listener = keyboard.Listener(on_press=on_press)
listener.start()

try:
    while program_running[0]:
        time.sleep(0.05)
finally:
    # Cleanup in main thread
    if args.record:
        gif_name = "".join(Path(args.env).parts).replace("Gym-Gridworlds", "")
        imageio.mimsave(gif_name + ".gif", frames, fps=5, loop=0)
    listener.stop()
    env.close()
    if args.record:
        env_record.close()
