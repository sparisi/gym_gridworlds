"""
Move:           ← ↑ → ↓
Stay:           ENTER
Reset Board:    R
Quit:           Q

python playground.py ENVIRONMENT --record
--record to save a gif of the game
"""

import imageio
import numpy as np
import gymnasium
import gym_gridworlds
from gym_gridworlds.gridworld import LEFT, DOWN, RIGHT, UP, STAY
import argparse
from pynput import keyboard
import time

# Mutable object (list) to signal when the program should exit
program_running = [True]
sum_rewards = [0]  # Mutable so we can update it in `on_press`

parser = argparse.ArgumentParser()
parser.add_argument("env")
parser.add_argument("--record", action="store_true")
args = parser.parse_args()

env = gymnasium.make(args.env, render_mode="human")

if args.record:
    # Gymnasium human rendering does not return RGB array, so we must make a copy
    env_record = gymnasium.make(args.env, render_mode="rgb_array")
    frames = []


def step(action):
    next_obs, rwd, term, trunc, info = env.step(action)
    sum_rewards[0] = sum_rewards[0] + rwd
    if term or trunc:
        print("Episode ended, sum of rewards:", sum_rewards[0])
    if args.record:
        env_record.step(action)
        frame = env_record.render()
        if frame is not None:
            frames.append(frame)


def reset():
    seed = np.random.randint(999)
    env.reset(seed=seed)
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
    "Move: \t\t← ↑ → ↓\n"
    "Stay: \t\tENTER\n"
    "Reset Board: \tR\n"
    "Quit: \t\tQ\n"
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
        imageio.mimsave(args.env + ".gif", frames, fps=5, loop=0)
    listener.stop()
    env.close()
    if args.record:
        env_record.close()
