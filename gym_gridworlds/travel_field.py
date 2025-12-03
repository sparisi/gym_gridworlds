import numpy as np

from gym_gridworlds.gridworld import (
    LEFT, DOWN, RIGHT, UP, STAY, WALL, GOOD, REWARDS, GRIDS, COLORMAP, Color
)
from gym_gridworlds.gridworld import Gridworld

ROCK = WALL
GOAL = GOOD
GRASS = 99
ROAD = 98
SWAMP = 97

COLORMAP[ROCK] = Color.GRAY
COLORMAP[GRASS] = Color.GREEN
COLORMAP[ROAD] = Color.PALE_YELLOW
COLORMAP[SWAMP] = Color.BROWN
COLORMAP[GOAL] = Color.RED

REWARDS[ROCK] = -1.0
REWARDS[GRASS] = -1.0
REWARDS[ROAD] = -0.25
REWARDS[SWAMP] = -2.0
REWARDS[GOAL] = 0

char_dict = {
    ".": GRASS,
    "□": ROCK,
    "+": ROAD,
    "-": SWAMP,
    "O": GOAL,
}

small_field = (
    "..........\n"
    ".□.□..--O+\n"
    ".□.□----□+\n"
    "--.□---.□+\n"
    "--□.----□+\n"
    "+.--□++--+\n"
    "+----□+++.\n"
    "..----□□+.\n"
    ".+□□□.□++.\n"
    "....□+++..\n"
)

large_field = (
    "............................\n"
    "....□........□□□□□□□□□....++\n"
    "..□..□....□□□□...---.□□.O.□+\n"
    ".□.....□□□□□....------...□□+\n"
    "..............□□□-----....□+\n"
    "..............□-------....□+\n"
    "......□----□□.□-------.---□+\n"
    "....--------□□□--□.-----.-□+\n"
    "...□---------□□--□..------++\n"
    ".□□□--□□□□□□--□--□....-..++.\n"
    ".-----□++++---□--□□......+..\n"
    ".-----□++-----□---□□.....++.\n"
    ".□--..□□□-□□□□□----□□□....+.\n"
    ".□□□□□..□-□------□--.□....+.\n"
    "..□□+□□□□□-------□--.□□...+.\n"
    "...+++----□------□--□□....+.\n"
    "-...+----□-------□--.□....+.\n"
    "-...+.--------------□□....++\n"
    "----+..-----.------+□□□.+++.\n"
    "---.+..---.--------+□□□..++.\n"
    "---.+..-----------.+□.....+.\n"
    "-.......---------..+□□....++\n"
    "..□□□..-□□□□□---..++.□□...++\n"
    ".□□□..□-------...++...□....+\n"
    "□□..--□----..-.+++++..□□...+\n"
    "....□-□---...++++..++..□..++\n"
    ".+..□-□--..+++......+..++++.\n"
    "....□---.+++................\n"
)

GRIDS["small_field"] = [[char_dict[c] for c in list(s)] for s in small_field.splitlines()]
GRIDS["large_field"] = [[char_dict[c] for c in list(s)] for s in large_field.splitlines()]



class TravelField(Gridworld):
    """
    The agent has to reach the goal while traveling through a field of different tiles:
    GRASS (green) gives -1 reward, SWAMP (brown) gives -2, ROAD (yellow) gives
    -0.25, and ROCK (gray) cannot be walked on.
    The optimal policy goes through ROAD tiles only, but the path is hard to find.
    There are also dead-end roads that may mislead the agent.
    """
