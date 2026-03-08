Unless stated otherwise, the agent must do `STAY` to get positive rewards in green tiles.  
Starting position is always the top-leftmost tile, unless the name of the environment has `RandomStart` or stated otherwise.  
Check [__init__.py](https://github.com/sparisi/gym_gridworlds/blob/main/gym_gridworlds/__init__.py) for more details.


`Straight-1x20-v0`  
<img src="Straight-1x20-v0.png" width="220">

`Empty-RandomStart-2x2-v0`  
<img src="Empty-RandomStart-2x2-v0.png" width="220">

`Empty-RandomStart-3x3-v0`  
<img src="Empty-RandomStart-3x3-v0.png" width="220">

`Empty-RandomGoal-3x3-v0`Goal is randomized at every reset.  
<img src="Empty-RandomGoal-3x3-v0.png" width="220">

`Empty-Loop-3x3-v0`: One-directional tiles make a "loop-like" path in the top-right corner.  
<img src="Empty-Loop-3x3-v0.png" width="220">

`Empty-10x10-v0`  
<img src="Empty-10x10-v0.png" width="220">

`Empty-Distract-6x6-v0`: Bottom-right goal is a "distractor".  
<img src="Empty-Distract-6x6-v0.png" width="220">

`Penalty-3x3-v0`: Red tiles with negative reward force the optimal agent to take a detour.  
<img src="Penalty-3x3-v0.png" width="220">

`Quicksand-4x4-v0`: Yellow tile is quicksand (the agent has only 10% chance of leaving it).  
<img src="Quicksand-4x4-v0.png" width="220">

`Quicksand-Distract-4x4-v0`: Quicksand and two distractors.  
<img src="Quicksand-Distract-4x4-v0.png" width="220">

`TwoRoom-Quicksand-3x5-v0`: The one-directional tile and the quicksand split the grid into two rooms.  
<img src="TwoRoom-Quicksand-3x5-v0.png" width="220">

`Corridor-3x4-v0`: To get the highest reward, the agent must walk over small negative rewards.  
<img src="Corridor-3x4-v0.png" width="220">

`Full-4x5-v0`: This grid has all the basic components: distractors, penalties, quicksand, and one-directional tiles.  
<img src="Full-4x5-v0.png" width="220">

`Full-RandomGoalAndStart-4x5-v0`: Like above, but starting position and green tiles positions are randomized at every reset.  
<img src="Full-RandomGoalAndStart-4x5-v0.png" width="220">

`TwoRoom-Distract-Middle-2x11-v0`: One-directional tiles split the grid into two rooms. The agent starts in the middle.  
<img src="TwoRoom-Distract-Middle-2x11-v0.png" width="220">

`Barrier-5x5-v0`: One-directional tiles make a barrier around the goal.  
<img src="Barrier-5x5-v0.png" width="220">

`RiverSwim-1x6-v0`: Classic RL benchmark. The agent starts either in the 2nd or 3rd leftmost tile, the only actions are `LEFT` and `RIGHT`, and the transition is stochastic. No need to `STAY` in green tiles. There are no terminal states.  
<img src="RiverSwim-1x6-v0.png" width="220">

`CliffWalk-4x12-v0`: Classic RL benchmark. The agent starts in the top-leftmost tile, and the goal is located on the other side of a long pit.  
<img src="CliffWalk-4x12-v0.png" width="220">

`DangerMaze-5x6-v0`: Maze-like grid with pits, walls, and negative rewards.  
<img src="DangerMaze-5x6-v0.png" width="220">

`FourRooms-Symmetrical-11x11-v0`: Classic RL benchmark. Start position and goal are randomized at every reset. No need to `STAY` in green tiles.  
<img src="FourRooms-Symmetrical-11x11-v0.png" width="220">

`FourRooms-Original-13x13-v0`: Classic RL benchmark. Start position and goal are randomized at every reset. No need to `STAY` in green tiles.  
<img src="FourRooms-Original-13x13-v0.png" width="220">


`FourRooms-Original-13x13-Stuck-v0`: The agent cannot exit the bottom-left room. Start position and goal are randomized at every reset. No need to `STAY` in green tiles.  
<img src="FourRooms-Original-13x13-Stuck-v0.png" width="220">

`FourRooms-Original-13x13-Loop-v0`: One-directional arrows enforce a "loop-like" path to visit all rooms. Start position and goal are randomized at every reset. No need to `STAY` in green tiles.  
<img src="FourRooms-Original-13x13-Loop-v0.png" width="220">

`FourRooms-Mini-8x7-v0`: Smaller version of the above environment.  
<img src="FourRooms-Mini-8x7-v0.png" width="220">

`FourRooms-Mini-8x7-Loop-v0`: Smaller version of the above environment.  
<img src="FourRooms-Mini-8x7-Loop-v0.png" width="220">

`FourRooms-Mini-8x7-Stuck-v0`: Smaller version of the above environment.  
<img src="FourRooms-Mini-8x7-Stuck-v0.png" width="220">

`FourRooms-Cross-14x16-v0`: Rooms are spread across the grid in a "cross-like" shape. Start position and goal are randomized at every reset.  
<img src="FourRooms-Cross-14x16-v0.png" width="220">

`Maze-12x12-v0`: [The agent starts in the middle. No need to `STAY` in green tiles.](https://arxiv.org/pdf/2505.01336).  
<img src="Maze-12x12-v0.png" width="220">

`CleanDirt-10x10-v0`: The agent must clean dirt (green tiles) that randomly appear over time. The starting position is in the middle of the grid. No need to `STAY` to clean dirt.  
<img src="CleanDirt-10x10-v0.png" width="220">

`TravelField-28x28-v0`: The agent start in the bottom-leftmost tile and has to reach the red tile. The shortest path is through a swamp (brown tiles), but it yields large negative rewards. A longer but easy-to-find path is through grass (left side of the grid), but grass also yields negative rewards. The best path is through a road (yellow tiles, smallest negative reward), but it's harder to find. Beside the default actions (including `STAY`) the agent can also move diagonally. Designed to be solved without discount factor.  
<img src="TravelField-28x28-v0.png" width="220">

`TravelField-10x10-v0`: Smaller version of the above grid.   
<img src="TravelField-10x10-v0.png" width="220">

`TravelField-28x28-v1`: Easier version without distracting roads.  
<img src="TravelField-28x28-v1.png" width="220">

`TravelField-10x10-v1`: Smaller version of the above grid.  
<img src="TravelField-10x10-v1.png" width="220">

`Penalty-Randomized-4x4-v0`: The one-directional tile randomly changes at every step.  
<img src="Penalty-Randomized-4x4-v0.png" width="220">

`Taxi-6x7-v0`: Classic RL benchmark. The agent starts in the top-rightmost tile and has to collect passengers (dark green tiles) before reaching the goal. No need to `STAY` in green tiles.  
<img src="Taxi-6x7-v0.png" width="220">

`Wall-50x50-v0`: Large grid with a wall dividing it into two areas. One distracting reward is in the top-rightmost tile, while the goal is at the bottom-rightmost tile. The agent always starts in the bottom-leftmost tile.  
<img src="Wall-50x50-v0.png" width="220">
