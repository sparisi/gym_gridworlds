from gymnasium.envs.registration import register

# ------------------------------------------------------------------------------
# Early grids
# ------------------------------------------------------------------------------

register(
    id="Gym-Gridworlds/Straight-1x20-v0",
    entry_point="gym_gridworlds.gridworld:Gridworld",
    max_episode_steps=200,
    kwargs={
        "grid": "1x20_straight",
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
        "start_pos": [(1, 5)],
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
    id="Gym-Gridworlds/RiverSwim-1x6-v0",
    entry_point="gym_gridworlds.gridworld:RiverSwim",
    max_episode_steps=200,
    kwargs={
        "grid": "1x6_river_swim",
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
    id="Gym-Gridworlds/FourRooms-13x13-v0",
    entry_point="gym_gridworlds.gridworld:Gridworld",
    max_episode_steps=200,
    kwargs={
        "grid": "13x13_four_rooms",
        "no_stay": True,
        "start_pos": None,  # random
        "random_goals": True,
    },
)

register(
    id="Gym-Gridworlds/FourRooms-Stuck-13x13-v0",
    entry_point="gym_gridworlds.gridworld:Gridworld",
    max_episode_steps=200,
    kwargs={
        "grid": "13x13_four_rooms_stuck",
        "no_stay": True,
        "start_pos": None,  # random
        "random_goals": True,
    },
)

register(
    id="Gym-Gridworlds/FourRooms-Loop-13x13-v0",
    entry_point="gym_gridworlds.gridworld:Gridworld",
    max_episode_steps=200,
    kwargs={
        "grid": "13x13_four_rooms_loop",
        "no_stay": True,
        "start_pos": None,  # random
        "random_goals": True,
    },
)

register(
    id="Gym-Gridworlds/FourRooms-8x7-v0",
    entry_point="gym_gridworlds.gridworld:Gridworld",
    max_episode_steps=100,
    kwargs={
        "grid": "8x7_four_rooms_mini",
        "no_stay": True,
        "start_pos": None,  # random
        "random_goals": True,
    },
)

register(
    id="Gym-Gridworlds/FourRooms-Loop-8x7-v0",
    entry_point="gym_gridworlds.gridworld:Gridworld",
    max_episode_steps=200,
    kwargs={
        "grid": "8x7_four_rooms_loop_mini",
        "no_stay": True,
        "start_pos": None,  # random
        "random_goals": True,
    },
)

register(
    id="Gym-Gridworlds/FourRooms-Stuck-8x7-v0",
    entry_point="gym_gridworlds.gridworld:Gridworld",
    max_episode_steps=200,
    kwargs={
        "grid": "8x7_four_rooms_stuck_mini",
        "no_stay": True,
        "start_pos": None,  # random
        "random_goals": True,
    },
)

register(
    id="Gym-Gridworlds/FourRooms-Wall-7x7-v0",
    entry_point="gym_gridworlds.gridworld:Gridworld",
    max_episode_steps=100,
    kwargs={
        "grid": "7x7_four_rooms_wall_mini",
        "no_stay": True,
        "start_pos": [
            (2, 2),   # first room
            (2, 4),   # second room
            (4, 2),   # third room
            (4, 4),   # fourth room
        ],
    },
)

register(
    id="Gym-Gridworlds/FourRooms-Cross-14x16-v0",
    entry_point="gym_gridworlds.gridworld:Gridworld",
    max_episode_steps=50,
    kwargs={
        "grid": "14x16_four_rooms_cross",
        "no_stay": True,
        "start_pos": None,  # random
        "random_goals": True,
    },
)

register(
    id="Gym-Gridworlds/ThreeRooms-Wall-14x16-v0",
    entry_point="gym_gridworlds.gridworld:Gridworld",
    max_episode_steps=500,
    kwargs={
        "grid": "14x16_three_rooms_wall",
        "no_stay": True,
        "start_pos": [
            (2, 0),   # first room
            (7, 0),   # second room
            (12, 0),  # third room
        ],
    },
)

register(
    id="Gym-Gridworlds/ThreeRooms-Quicksand-14x16-v0",
    entry_point="gym_gridworlds.gridworld:Gridworld",
    max_episode_steps=500,
    kwargs={
        "grid": "14x16_three_rooms_quicksand",
        "no_stay": True,
        "start_pos": [
            (2, 0),   # first room
            (7, 0),   # second room
            (12, 0),  # third room
        ],
    },
)

register(
    id="Gym-Gridworlds/ThreeRooms-Wall-11x8-v0",
    entry_point="gym_gridworlds.gridworld:Gridworld",
    max_episode_steps=100,
    kwargs={
        "grid": "11x8_three_rooms_wall_mini",
        "no_stay": True,
        "start_pos": [
            (2, 0),   # first room
            (4, 0),   # second room
            (10, 0),  # third room
        ],
    },
)

register(
    id="Gym-Gridworlds/ThreeRooms-Quicksand-11x8-v0",
    entry_point="gym_gridworlds.gridworld:Gridworld",
    max_episode_steps=100,
    kwargs={
        "grid": "11x8_three_rooms_quicksand_mini",
        "no_stay": True,
        "start_pos": [
            (2, 0),   # first room
            (4, 0),   # second room
            (10, 0),  # third room
        ],
    },
)

# https://arxiv.org/pdf/2505.01336
register(
    id="Gym-Gridworlds/Maze-12x12-v0",
    entry_point="gym_gridworlds.gridworld:Gridworld",
    max_episode_steps=500,
    kwargs={
        "grid": "12x12_maze",
        "no_stay": True,
        "start_pos": [(6, 7)],  # center of the maze
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
        "start_pos": [(5, 5)],  # center of the grid
    },
)

register(
    id="Gym-Gridworlds/TravelField-28x28-v0",
    entry_point="gym_gridworlds.travel_field:TravelField",
    max_episode_steps=1000,
    kwargs={
        "grid": "travel_field_28x28_v0",
        "start_pos": [("max", 0)],
        # "rock_is_terminal": True,
    },
)

register(
    id="Gym-Gridworlds/TravelField-12x12-v0",
    entry_point="gym_gridworlds.travel_field:TravelField",
    max_episode_steps=200,
    kwargs={
        "grid": "travel_field_12x12_v0",
        "start_pos": [("max", 0)],
        # "rock_is_terminal": True,
    },
)

register(
    id="Gym-Gridworlds/TravelField-28x28-v1",
    entry_point="gym_gridworlds.travel_field:TravelField",
    max_episode_steps=1000,
    kwargs={
        "grid": "travel_field_28x28_v1",
        "start_pos": [("max", 0)],
        # "rock_is_terminal": True,
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
        "start_pos": [(0, 0)],
        "no_stay": True,
    },
)

register(
    id="Gym-Gridworlds/Wall-50x50-v0",
    entry_point="gym_gridworlds.gridworld:Gridworld",
    max_episode_steps=500,
    kwargs={
        "grid": "50x50_wall",
        "start_pos": [("max", 0)],
    },
)
