Unless stated otherwise, the agent must do <code>STAY</code> to get positive rewards in green tiles.  
Starting position is always the top-leftmost tile, unless the name of the environment has <code>RandomStart</code>.

<table>
  <tr>
    <td align="center">
      <img src="Straight-1x20-v0.png" width="120">
    </td>
    <td>
      <p>
        <code>Straight-1x20-v0</code>
      </p>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="Empty-RandomStart-2x2-v0.png" width="120">
    </td>
    <td>
      <p>
        <code>Empty-RandomStart-2x2-v0</code
      </p>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="Empty-RandomStart-3x3-v0.png" width="120">
    </td>
    <td>
      <p>
        <code>Empty-RandomStart-3x3-v0</code>
      </p>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="Empty-RandomGoal-3x3-v0.png" width="120">
    </td>
    <td>
      <p>
        <code>Empty-RandomGoal-3x3-v0</code>Goal is randomized at every reset.
      </p>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="Empty-Loop-3x3-v0.png" width="120">
    </td>
    <td>
      <p>
        <code>Empty-Loop-3x3-v0</code>: One-directional tiles make a "loop-like" path in the top-right corner.
      </p>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="Empty-10x10-v0.png" width="120">
    </td>
    <td>
      <p>
        <code>Empty-10x10-v0</code>
      </p>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="Empty-Distract-6x6-v0.png" width="120">
    </td>
    <td>
      <p>
        <code>Empty-Distract-6x6-v0</code>: Bottom-right goal is a "distractor".
      </p>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="Penalty-3x3-v0.png" width="120">
    </td>
    <td>
      <p>
        <code>Penalty-3x3-v0</code>: Red tiles with negative reward force the optimal agent to take a detour.
      </p>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="Quicksand-4x4-v0.png" width="120">
    </td>
    <td>
      <p>
        <code>Quicksand-4x4-v0</code>: Yellow tile is quicksand (the agent has only 10% chance of leaving it).
      </p>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="Quicksand-Distract-4x4-v0.png" width="120">
    </td>
    <td>
      <p>
        <code>Quicksand-Distract-4x4-v0</code>: Quicksand and two distractors.
      </p>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="TwoRoom-Quicksand-3x5-v0.png" width="120">
    </td>
    <td>
      <p>
        <code>TwoRoom-Quicksand-3x5-v0</code>: The one-directional tile and the quicksand split the grid into two rooms.
      </p>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="Corridor-3x4-v0.png" width="120">
    </td>
    <td>
      <p>
        <code>Corridor-3x4-v0</code>: To get the highest reward, the agent must walk over small negative rewards.
      </p>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="Full-4x5-v0.png" width="120">
    </td>
    <td>
      <p>
        <code>Full-4x5-v0</code>: This grid has all the basic components: distractors, penalties, quicksand, and one-directional tiles.
      </p>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="Full-RandomGoalAndStart-4x5-v0.png" width="120">
    </td>
    <td>
      <p>
        <code>Full-RandomGoalAndStart-4x5-v0</code>: Like above, but starting position and green tiles positions are randomized at every reset.
      </p>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="TwoRoom-Distract-Middle-2x11-v0.png" width="120">
    </td>
    <td>
      <p>
        <code>TwoRoom-Distract-Middle-2x11-v0</code>: One-directional tiles split the grid into two rooms. The agent starts in the middle.
      </p>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="Barrier-5x5-v0.png" width="120">
    </td>
    <td>
      <p>
        <code>Barrier-5x5-v0</code>: One-directional tiles make a barrier around the goal.
      </p>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="RiverSwim-1x6-v0.png" width="120">
    </td>
    <td>
      <p>
        <code>RiverSwim-1x6-v0</code>: Classic RL benchmark. No need to <code>STAY</code> in green tiles.
      </p>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="CliffWalk-4x12-v0.png" width="120">
    </td>
    <td>
      <p>
        <code>CliffWalk-4x12-v0</code>: Classic RL benchmark.
      </p>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="DangerMaze-5x6-v0.png" width="120">
    </td>
    <td>
      <p>
        <code>DangerMaze-5x6-v0</code>: Maze-like grid with pits, walls, and negative rewards.
      </p>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="FourRooms-Symmetrical-11x11-v0.png" width="120">
    </td>
    <td>
      <p>
        <code>FourRooms-Symmetrical-11x11-v0</code>: Classic RL benchmark. Start position and goal are randomized at every reset. No need to <code>STAY</code> in green tiles.
      </p>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="FourRooms-Original-13x13-v0.png" width="120">
    </td>
    <td>
      <p>
        <code>FourRooms-Original-13x13-v0</code>: Classic RL benchmark. Start position and goal are randomized at every reset. No need to <code>STAY</code> in green tiles.
      </p>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="FourRooms-Original-13x13-Stuck-v0.png" width="120">
    </td>
    <td>
      <p>
        <code>FourRooms-Original-13x13-Stuck-v0</code>: The agent cannot exit the bottom-left room. Start position and goal are randomized at every reset. No need to <code>STAY</code> in green tiles.
      </p>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="FourRooms-Original-13x13-Loop-v0.png" width="120">
    </td>
    <td>
      <p>
        <code>FourRooms-Original-13x13-Loop-v0</code>: One-directional arrows enforce a "loop-like" path to visit all rooms. Start position and goal are randomized at every reset. No need to <code>STAY</code> in green tiles.
      </p>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="FourRooms-Mini-8x7-v0.png" width="120">
    </td>
    <td>
      <p>
        <code>FourRooms-Mini-8x7-v0</code>: Smaller version of the above environment.
      </p>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="FourRooms-Mini-8x7-Loop-v0.png" width="120">
    </td>
    <td>
      <p>
        <code>FourRooms-Mini-8x7-Loop-v0</code>: Smaller version of the above environment.
      </p>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="FourRooms-Mini-8x7-Stuck-v0.png" width="120">
    </td>
    <td>
      <p>
        <code>FourRooms-Mini-8x7-Stuck-v0</code>: Smaller version of the above environment.
      </p>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="FourRooms-Cross-14x16-v0.png" width="120">
    </td>
    <td>
      <p>
        <code>FourRooms-Cross-14x16-v0</code>: Rooms are spread across the grid in a "cross-like" shape. Start position and goal are randomized at every reset.
      </p>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="Maze-12x12-v0.png" width="120">
    </td>
    <td>
      <p>
        <code>Maze-12x12-v0</code>: [The agent starts in the middle. No need to <code>STAY</code> in green tiles.](https://arxiv.org/pdf/2505.01336).
      </p>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="CleanDirt-10x10-v0.png" width="120">
    </td>
    <td>
      <p>
        <code>CleanDirt-10x10-v0</code>: [Check full description here](gym_gridworlds/gym_gridworlds/clean_dirt.py)
      </p>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="TravelField-28x28-v0.png" width="120">
    </td>
    <td>
      <p>
        <code>TravelField-28x28-v0</code>: [Check full description here](gym_gridworlds/gym_gridworlds/travel_field.py). It has diagonal actions.
      </p>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="TravelField-10x10-v0.png" width="120">
    </td>
    <td>
      <p>
        <code>TravelField-10x10-v0</code>: [Check full description here](gym_gridworlds/gym_gridworlds/travel_field.py). It has diagonal actions.
      </p>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="TravelField-28x28-v1.png" width="120">
    </td>
    <td>
      <p>
        <code>TravelField-28x28-v1</code>: [Check full description here](gym_gridworlds/gym_gridworlds/travel_field.py). It has diagonal actions.
      </p>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="TravelField-10x10-v1.png" width="120">
    </td>
    <td>
      <p>
        <code>TravelField-10x10-v1</code>: [Check full description here](gym_gridworlds/gym_gridworlds/travel_field.py). It has diagonal actions.
      </p>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="Penalty-Randomized-4x4-v0.png" width="120">
    </td>
    <td>
      <p>
        <code>Penalty-Randomized-4x4-v0</code>: The one-directional tile randomly changes at every step.
      </p>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="Taxi-6x7-v0.png" width="120">
    </td>
    <td>
      <p>
        <code>Taxi-6x7-v0</code>: Classic RL benchmark. The agent starts in the top-rightmost tile. No need to <code>STAY</code> in green tiles.
      </p>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="Wall-50x50-v0.png" width="120">
    </td>
    <td>
      <p>
        <code>Wall-50x50-v0</code>: [Very large grid with a wall dividing it into two areas. One distracting reward is in the top-rightmost tile, while the goal is at the bottom-rightmost tile. The agent always starts in the bottom-leftmost tile.](https://arxiv.org/abs/2001.00119).
      </p>
    </td>
  </tr>
</table>
