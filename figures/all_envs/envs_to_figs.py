import gymnasium
import gym_gridworlds
import matplotlib.pyplot as plt
import os

for env_id in gymnasium.envs.registry:
    if env_id.startswith("Gym-Gridworlds"):
        print(env_id)
        # env = gymnasium.make(env_id, render_mode="human")  # to render
        env = gymnasium.make(env_id, render_mode="rgb_array")  # to save as png
        obs, info = env.reset()
        frame = env.render()
        root_dir = os.path.dirname(os.path.dirname(gym_gridworlds.__file__))
        save_path = os.path.join(root_dir, "figures", "all_envs")
        os.makedirs(save_path, exist_ok=True)
        fig_name = env_id.replace("Gym-Gridworlds/", "")
        plt.imsave(os.path.join(save_path, fig_name + ".png"), frame)
