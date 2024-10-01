from gymnasium.envs.registration import register


def register_envs():
    register(
        id="Straight-20-v0",
        entry_point="gym_gridworlds.gridworld:Gridworld",
        max_episode_steps=200,
        kwargs={
            "grid": "20_straight",
        },
    )

    register(
        id="Empty-2x2-v0",
        entry_point="gym_gridworlds.gridworld:Gridworld",
        max_episode_steps=10,
        kwargs={
            "grid": "2x2_empty",
        },
    )

    register(
        id="Empty-3x3-v0",
        entry_point="gym_gridworlds.gridworld:Gridworld",
        max_episode_steps=50,
        kwargs={
            "grid": "3x3_empty",
        },
    )

    register(
        id="Empty-Loop-3x3-v0",
        entry_point="gym_gridworlds.gridworld:Gridworld",
        max_episode_steps=50,
        kwargs={
            "grid": "3x3_empty_loop",
        },
    )

    register(
        id="Empty-10x10-v0",
        entry_point="gym_gridworlds.gridworld:Gridworld",
        max_episode_steps=100,
        kwargs={
            "grid": "10x10_empty",
        },
    )

    register(
        id="Empty-Distract-6x6-v0",
        entry_point="gym_gridworlds.gridworld:Gridworld",
        max_episode_steps=50,
        kwargs={
            "grid": "6x6_distract",
        },
    )

    register(
        id="Penalty-3x3-v0",
        entry_point="gym_gridworlds.gridworld:Gridworld",
        max_episode_steps=50,
        kwargs={
            "grid": "3x3_penalty",
        },
    )

    register(
        id="Quicksand-4x4-v0",
        entry_point="gym_gridworlds.gridworld:Gridworld",
        max_episode_steps=50,
        kwargs={
            "grid": "4x4_quicksand",
        },
    )

    register(
        id="Quicksand-Distract-4x4-v0",
        entry_point="gym_gridworlds.gridworld:Gridworld",
        max_episode_steps=50,
        kwargs={
            "grid": "4x4_quicksand_distract",
        },
    )

    register(
        id="TwoRoom-Quicksand-3x5-v0",
        entry_point="gym_gridworlds.gridworld:Gridworld",
        max_episode_steps=50,
        kwargs={
            "grid": "3x5_two_room_quicksand",
        },
    )

    register(
        id="Corridor-3x4-v0",
        entry_point="gym_gridworlds.gridworld:Gridworld",
        max_episode_steps=50,
        kwargs={
            "grid": "3x4_corridor",
        },
    )
    register(
        id="Full-4x5-v0",
        entry_point="gym_gridworlds.gridworld:Gridworld",
        max_episode_steps=50,
        kwargs={
            "grid": "4x5_full",
        },
    )

    register(
        id="TwoRoom-Distract-Middle-2x11-v0",
        entry_point="gym_gridworlds.gridworld:GridworldMiddleStart",
        max_episode_steps=200,
        kwargs={
            "grid": "2x11_two_room_distract",
        },
    )

    register(
        id="Barrier-5x5-v0",
        entry_point="gym_gridworlds.gridworld:Gridworld",
        max_episode_steps=50,
        kwargs={
            "grid": "5x5_barrier",
        },
    )

    register(
        id="RiverSwim-6-v0",
        entry_point="gym_gridworlds.gridworld:RiverSwim",
        max_episode_steps=200,
        kwargs={
            "grid": "river_swim_6",
        },
    )

    register(
        id="CliffWalk-4x12-v0",
        entry_point="gym_gridworlds.gridworld:Gridworld",
        max_episode_steps=200,
        kwargs={
            "grid": "4x12_cliffwalk",
        },
    )

    register(
        id="DangerMaze-6x6-v0",
        entry_point="gym_gridworlds.gridworld:Gridworld",
        max_episode_steps=200,
        kwargs={
            "grid": "6x6_danger_maze",
        },
    )
