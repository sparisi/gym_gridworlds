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
    nargs="+",
    default=[],
    help="Environment arguments, for example: --env-arg no_stay=True infinite_horizon=True",
)
parser.add_argument("--record", action="store_true")
parser.add_argument("--discount", default=0.99)
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
    obs = env.unwrapped.get_state()
    next_obs, rwd, term, trunc, info = env.step(action)
    sum_rewards[0] = sum_rewards[0] + rwd
    sum_rewards[1] = sum_rewards[1] + rwd * args.discount**t[0]
    print(f"{t[0]}: {obs} | {action} | {rwd} | {next_obs}")
    t[0] += 1

    if term or trunc:
        if term:
            print(f"Terminal state ...")
        elif trunc:
            print(f"Time step limit ...")
        print(f"  ... sum of rewards: {sum_rewards[0]} (γ = 1), {sum_rewards[1]} (γ = {args.discount})")
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


KEY_TO_ACTION = {
    keyboard.Key.up: UP,
    keyboard.Key.down: DOWN,
    keyboard.Key.left: LEFT,
    keyboard.Key.right: RIGHT,
    keyboard.Key.enter: STAY,
}
CHAR_TO_ACTION = {
    "w": UP, "x": DOWN, "a": LEFT, "d": RIGHT, "s": STAY,
    "q": UP_LEFT, "e": UP_RIGHT, "z": DOWN_LEFT, "c": DOWN_RIGHT,
}
VK_TO_ACTION = {  # numpad
    104: UP, 98: DOWN, 100: LEFT, 102: RIGHT, 101: STAY,
    103: UP_LEFT, 105: UP_RIGHT, 97: DOWN_LEFT, 99: DOWN_RIGHT,
}


def on_press(key):
    try:
        if key == keyboard.Key.esc:
            # Can't call env.close() or pygame will freeze everything
            program_running[0] = False
            return False
        if key == keyboard.Key.backspace:
            reset()
            return
        # Careful: LEFT == 0 is falsy, so use explicit None checks not `or`.
        for mapping, lookup in (
            (KEY_TO_ACTION, key),
            (CHAR_TO_ACTION, getattr(key, "char", None)),
            (VK_TO_ACTION, getattr(key, "vk", None)),
        ):
            action = mapping.get(lookup)
            if action is not None:
                step(action)
                return
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
    "Prints are 'Timestep: (Obs | Action | Reward | Next Obs)'"
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
