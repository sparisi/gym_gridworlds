# fmt: off
from setuptools import setup

packages = ["gym_gridworlds"]
install_requires = [
    "gymnasium",
    "pygame"
]
entry_points = {
    "gymnasium.envs": ["Gym-Gridworlds=gym_gridworlds.gym:register_envs"]
}

setup(
    name="Gym-Gridworlds",
    version="1.0",
    license="CC-BY-4.0",
    author="Simone Parisi",
    packages=packages,
    entry_points=entry_points,
    install_requires=install_requires,
)
