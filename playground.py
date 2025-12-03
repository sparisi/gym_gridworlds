"""
Move:   ↖ ↑ ↗      Q W E
        ←   →  or  A   D
        ↙ ↓ ↘      Z X C
Stay:   ENTER or S
Reset:  Backspace
Quit:   Esc

python playground.py ENVIRONMENT --record --env-arg= ...
--record to save a gif of the game
--discount to also compute discounted rewards
--env-arg to pass optional environment arguments

Example:
python playground.py Gym-Gridworlds/TravelField-28x28-v0 --env-arg distance_reward=True --env-arg no_stay=True --record --discount=0.99
"""

import imageio
import numpy as np
import gymnasium
import gym_gridworlds
from gym_gridworlds.gridworld import (
    LEFT, DOWN, RIGHT, UP, STAY, DOWN_LEFT, DOWN_RIGHT, UP_LEFT, UP_RIGHT
)
import argparse
from pynput import keyboard
import time
from pathlib import Path
import json

# Mutable so we can update it in on_press`
program_running = [True]
sum_rewards = [0.0, 0.0]
t = [0]

def parse_env_args(arg_list):
    env_kwargs = {}
    for item in arg_list:
        if "=" not in item:
            raise ValueError(f"Invalid format for --env-arg '{item}', expected key=value")
        key, value = item.split("=", 1)
        try:
            value = json.loads(value)  # Works for lists, dicts, numbers, booleans, null
        except json.JSONDecodeError:
            pass  # Fallback: keep as string
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
parser.add_argument("--discount", default=1.0)
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
    sum_rewards[1] = sum_rewards[1] + rwd * args.discount**t[0]
    next_state = env.unwrapped.agent_pos
    t[0] += 1

    print(f"{state} | {action} | {rwd} | {next_state}")

    if term or trunc:
        if term:
            print(f"Terminal state ...")
        elif trunc:
            print(f"Time step limit ...")
        print(f"  ... sum of rewards: {sum_rewards[0]} (undiscounted), {sum_rewards[1]} (discounted)")
        reset()
        sum_rewards[0] = 0.0
        sum_rewards[1] = 0.0

    if args.record:
        env_record.step(action)
        frame = env_record.render()
        if frame is not None:
            frames.append(frame)


def reset():
    print("\nEnvironment reset")
    seed = np.random.randint(999)
    env.reset(seed=seed)
    env.render()
    sum_rewards[0] = 0.0
    sum_rewards[1] = 0.0
    t[0] = 0

    if args.record:
        env_record.reset(seed=seed)
        frame = env_record.render()
        if frame is not None:
            frames.append(frame)


def on_press(key):
    vk = getattr(key, "vk", None)
    ch = getattr(key, "char", None)
    try:
        if key == keyboard.Key.up or vk == 104 or ch == "w":
            step(UP)
        elif key == keyboard.Key.down or vk == 98 or ch == "x":
            step(DOWN)
        elif key == keyboard.Key.left or vk == 100 or ch == "a":
            step(LEFT)
        elif key == keyboard.Key.right or vk == 102 or ch == "d":
            step(RIGHT)
        elif key == keyboard.Key.enter or vk == 101 or ch == "s":
            step(STAY)
        elif key == keyboard.Key.enter or vk == 105 or ch == "e":
            step(UP_RIGHT)
        elif key == keyboard.Key.enter or vk == 99 or ch == "c":
            step(DOWN_RIGHT)
        elif key == keyboard.Key.enter or vk == 103 or ch == "q":
            step(UP_LEFT)
        elif key == keyboard.Key.enter or vk == 97 or ch == "z":
            step(DOWN_LEFT)
        elif key == keyboard.Key.esc:
            # Can't call env.close() or pygame will freeze everything
            program_running[0] = False
            return False
        elif key == keyboard.Key.backspace:
            reset()
        else:
            pass
    except AttributeError:
        pass


print(
    "\n"
    "Move: \t↖ ↑ ↗      Q W E\n"
    "\t←   →  or  A   D\n"
    "\t↙ ↓ ↘      Z X C\n"
    "Stay: \tENTER or S\n"
    "Reset: \tBackspace\n"
    "Quit: \tEsc\n"
    "\n"
    "Prints are: (State | Action | Reward | Next State)"
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
