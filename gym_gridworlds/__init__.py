from gymnasium.envs.registration import register

# ------------------------------------------------------------------------------
# Early grids
# ------------------------------------------------------------------------------

register(
    id="Gym-Gridworlds/Straight-20-v0",
    entry_point="gym_gridworlds.gridworld:Gridworld",
    max_episode_steps=200,
    kwargs={
        "grid": "20_straight",
    },
)

register(
    id="Gym-Gridworlds/Empty-RandomStart-2x2-v0",
    entry_point="gym_gridworlds.gridworld:Gridworld",
    max_episode_steps=10,
    kwargs={
        "grid": "2x2_empty",
        "start_pos": None,  # random
    },
)

register(
    id="Gym-Gridworlds/Empty-RandomStart-3x3-v0",
    entry_point="gym_gridworlds.gridworld:Gridworld",
    max_episode_steps=50,
    kwargs={
        "grid": "3x3_empty",
        "start_pos": None,  # random
    },
)

register(
    id="Gym-Gridworlds/Empty-RandomGoal-3x3-v0",
    entry_point="gym_gridworlds.gridworld:Gridworld",
    max_episode_steps=50,
    kwargs={
        "grid": "3x3_empty",
        "random_goals": True,
    },
)

register(
    id="Gym-Gridworlds/Empty-Loop-3x3-v0",
    entry_point="gym_gridworlds.gridworld:Gridworld",
    max_episode_steps=50,
    kwargs={
        "grid": "3x3_empty_loop",
    },
)

register(
    id="Gym-Gridworlds/Empty-10x10-v0",
    entry_point="gym_gridworlds.gridworld:Gridworld",
    max_episode_steps=100,
    kwargs={
        "grid": "10x10_empty",
    },
)

register(
    id="Gym-Gridworlds/Empty-Distract-6x6-v0",
    entry_point="gym_gridworlds.gridworld:Gridworld",
    max_episode_steps=50,
    kwargs={
        "grid": "6x6_distract",
    },
)

register(
    id="Gym-Gridworlds/Penalty-3x3-v0",
    entry_point="gym_gridworlds.gridworld:Gridworld",
    max_episode_steps=50,
    kwargs={
        "grid": "3x3_penalty",
    },
)

register(
    id="Gym-Gridworlds/Quicksand-4x4-v0",
    entry_point="gym_gridworlds.gridworld:Gridworld",
    max_episode_steps=50,
    kwargs={
        "grid": "4x4_quicksand",
    },
)

register(
    id="Gym-Gridworlds/Quicksand-Distract-4x4-v0",
    entry_point="gym_gridworlds.gridworld:Gridworld",
    max_episode_steps=50,
    kwargs={
        "grid": "4x4_quicksand_distract",
    },
)

register(
    id="Gym-Gridworlds/TwoRoom-Quicksand-3x5-v0",
    entry_point="gym_gridworlds.gridworld:Gridworld",
    max_episode_steps=50,
    kwargs={
        "grid": "3x5_two_room_quicksand",
    },
)

register(
    id="Gym-Gridworlds/Corridor-3x4-v0",
    entry_point="gym_gridworlds.gridworld:Gridworld",
    max_episode_steps=50,
    kwargs={
        "grid": "3x4_corridor",
    },
)
register(
    id="Gym-Gridworlds/Full-4x5-v0",
    entry_point="gym_gridworlds.gridworld:Gridworld",
    max_episode_steps=50,
    kwargs={
        "grid": "4x5_full",
    },
)

register(
    id="Gym-Gridworlds/Full-RandomGoalAndStart-4x5-v0",
    entry_point="gym_gridworlds.gridworld:Gridworld",
    max_episode_steps=50,
    kwargs={
        "grid": "4x5_full",
        "random_goals": True,
        "start_pos": None,  # random
    },
)

register(
    id="Gym-Gridworlds/TwoRoom-Distract-Middle-2x11-v0",
    entry_point="gym_gridworlds.gridworld:Gridworld",
    max_episode_steps=200,
    kwargs={
        "grid": "2x11_two_room_distract",
        "start_pos": (1, 5),
    },
)

register(
    id="Gym-Gridworlds/Barrier-5x5-v0",
    entry_point="gym_gridworlds.gridworld:Gridworld",
    max_episode_steps=50,
    kwargs={
        "grid": "5x5_barrier",
    },
)

register(
    id="Gym-Gridworlds/RiverSwim-6-v0",
    entry_point="gym_gridworlds.gridworld:RiverSwim",
    max_episode_steps=200,
    kwargs={
        "grid": "river_swim_6",
        "no_stay": True,
        "infinite_horizon": True,
    },
)

register(
    id="Gym-Gridworlds/CliffWalk-4x12-v0",
    entry_point="gym_gridworlds.gridworld:Gridworld",
    max_episode_steps=200,
    kwargs={
        "grid": "4x12_cliffwalk",
    },
)

register(
    id="Gym-Gridworlds/DangerMaze-5x6-v0",
    entry_point="gym_gridworlds.gridworld:Gridworld",
    max_episode_steps=200,
    kwargs={
        "grid": "5x6_danger_maze",
    },
)

# ------------------------------------------------------------------------------
# Four rooms and variations
# ------------------------------------------------------------------------------

register(
    id="Gym-Gridworlds/FourRooms-Symmetrical-11x11-v0",
    entry_point="gym_gridworlds.gridworld:Gridworld",
    max_episode_steps=200,
    kwargs={
        "grid": "11x11_four_rooms_symmetrical",
        "no_stay": True,
        "start_pos": None,  # random
        "random_goals": True,
    },
)

register(
    id="Gym-Gridworlds/FourRooms-Original-13x13-v0",
    entry_point="gym_gridworlds.gridworld:Gridworld",
    max_episode_steps=200,
    kwargs={
        "grid": "13x13_four_rooms_original",
        "no_stay": True,
        "start_pos": None,  # random
        "random_goals": True,
    },
)

register(
    id="Gym-Gridworlds/FourRooms-Original-13x13-Stuck-v0",
    entry_point="gym_gridworlds.gridworld:Gridworld",
    max_episode_steps=200,
    kwargs={
        "grid": "13x13_four_rooms_original_stuck",
        "no_stay": True,
        "start_pos": None,  # random
        "random_goals": True,
    },
)

register(
    id="Gym-Gridworlds/FourRooms-Original-13x13-Loop-v0",
    entry_point="gym_gridworlds.gridworld:Gridworld",
    max_episode_steps=200,
    kwargs={
        "grid": "13x13_four_rooms_original_loop",
        "no_stay": True,
        "start_pos": None,  # random
        "random_goals": True,
    },
)

register(
    id="Gym-Gridworlds/FourRooms-Crossed-14x16-v0",
    entry_point="gym_gridworlds.gridworld:Gridworld",
    max_episode_steps=200,
    kwargs={
        "grid": "14x16_four_rooms_crossed",
        "no_stay": True,
        "start_pos": None,  # random
        "random_goals": True,
    },
)

# ------------------------------------------------------------------------------
# Harder environments
# ------------------------------------------------------------------------------

register(
    id="Gym-Gridworlds/CleanDirt-10x10-v0",
    entry_point="gym_gridworlds.clean_dirt:CleanDirt",
    max_episode_steps=500,
    kwargs={
        "grid": "10x10_empty",
    },
)

register(
    id="Gym-Gridworlds/TravelField-28x28-v0",
    entry_point="gym_gridworlds.travel_field:TravelField",
    max_episode_steps=1000,
    kwargs={
        "grid": "large_field",
        "start_pos": ("max", 0),
        "rock_is_terminal": True,
    },
)

register(
    id="Gym-Gridworlds/TravelField-10x10-v0",
    entry_point="gym_gridworlds.travel_field:TravelField",
    max_episode_steps=200,
    kwargs={
        "grid": "small_field",
        "start_pos": ("max", 0),
        "rock_is_terminal": True,
    },
)

register(
    id="Gym-Gridworlds/Penalty-Randomized-4x4-v0",
    entry_point="gym_gridworlds.rnd_move:RandomizedTiles",
    max_episode_steps=500,
    kwargs={
        "grid": "4x4_penalty_rnd_move",
    },
)

register(
    id="Gym-Gridworlds/Taxi-6x7-v0",
    entry_point="gym_gridworlds.taxi:Taxi",
    max_episode_steps=100,
    kwargs={
        "grid": "6x7_taxi",
        "start_pos": (0, 0),
        "no_stay": True,
    },
)

register(
    id="Gym-Gridworlds/Wall-50x50-v0",
    entry_point="gym_gridworlds.gridworld:Gridworld",
    max_episode_steps=500,
    kwargs={
        "grid": "50x50_wall",
        "start_pos": ("max", 0),
    },
)
