import os
import wandb
from ray.rllib.algorithms import PPOConfig
from ray.tune.registry import register_env


def env_creator(env_config):
    import gymnasium as gym
    import miniworld
    env = gym.make("MiniWorld-Maze-v0")
    resized_env = gym.wrappers.ResizeObservation(env, (84, 84))
    return resized_env


register_env("MiniWorld-Maze-v0", env_creator)

config = (
    PPOConfig()
    .environment("MiniWorld-Maze-v0")
    .rollouts(num_rollout_workers=4)
    .framework("torch")
    .training(model={
        "dim": 84,
        "fcnet_hiddens": [64, 64],
    })
    .evaluation(evaluation_num_workers=1)
)

local_rank = int(os.getenv("LOCAL_RANK", 0))
if local_rank == 0:
    wandb.init(project="minioff", name=f'pruebitas', sync_tensorboard=True,
               config=config.to_dict())

algo = config.build()

for _ in range(3):
    print(algo.train())

algo.evaluate()


