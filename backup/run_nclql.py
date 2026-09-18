import importlib
import random

import gymnasium as gym
import numpy as np
import torch
from gymnasium.wrappers import RecordEpisodeStatistics
from tensordict import TensorDict
from torchrl.data import LazyTensorStorage, ListStorage, ReplayBuffer
from tqdm import tqdm, trange

from nclql.models import AnnealedLangevinDynamics, NoiseConditionedCritic
from nclql.trainer import NCLQLTrainer


def seed_all(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)


config_path = "./configs/nclql_config.py"
spec = importlib.util.spec_from_file_location("config", config_path)

config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)

cfg = config.TotalConfig()

seed_all(cfg.train.random_seed)

# env = gym.make("HalfCheetah-v5", render_mode="human")
# env = gym.make("Reacher-v4", render_mode="human")
# env = gym.make("HumanoidStandup-v5", impact_cost_weight=0.5e-6, render_mode="human")
# env = gym.make("Humanoid-v5", ctrl_cost_weight=0.1, render_mode="human")
env = gym.make("InvertedPendulum-v5", reset_noise_scale=0.1, render_mode="human")
env = gym.wrappers.RecordEpisodeStatistics(env)
test = env.action_space
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]

buffer = ReplayBuffer(storage=LazyTensorStorage(cfg.train.buffer_capacity))

critic = NoiseConditionedCritic(
    obs_dim + act_dim + cfg.model.time_dim, 1, h_dims=cfg.model.h_dims
)

critic_optimizer = torch.optim.Adam(critic.parameters(), lr=cfg.train.lr)

ald = AnnealedLangevinDynamics(
    model=critic,
    L=cfg.model.L,
    T=cfg.model.T,
    w=cfg.model.w,
    act_dim=act_dim,
    act_max=env.action_space.high,
    act_min=env.action_space.low,
    # act_max=1,
    # act_min=-1,
    q_grad_norm=cfg.model.q_grad_norm,
)

ald = torch.compile(ald)

trainer = NCLQLTrainer(
    ald=ald,
    critic=critic,
    replay_buffer=buffer,
    critic_optimizer=critic_optimizer,
    batch_size=cfg.train.batch_size,
)

observation, info = env.reset()

with tqdm(
    range(cfg.train.total_it),
    desc=cfg.train.training_alg + " Training",
    dynamic_ncols=True,
) as pbar:
    for i, ch in enumerate(pbar):
        if len(buffer) > cfg.train.batch_size:
            # if len(buffer) > start_steps:
            action = ald.sample(
                torch.tensor(observation, dtype=torch.float32).reshape(1, -1),
                shape=(1, act_dim),
            )
            action = action.data.numpy()[0]
        else:
            action = env.action_space.sample()

        next_observation, reward, terminated, truncated, info = env.step(action)

        sample = TensorDict(
            {
                "obs": torch.as_tensor(observation, dtype=torch.float32),
                "next_obs": torch.as_tensor(next_observation, dtype=torch.float32),
                "action": torch.as_tensor(action, dtype=torch.float32),
                "reward": torch.as_tensor([reward], dtype=torch.float32),
                "done": torch.as_tensor([int(terminated)], dtype=torch.float32),
            }
        )

        buffer.add(sample)
        observation = next_observation
        if len(buffer) > cfg.train.batch_size:
            # if len(buffer) > start_steps:
            trainer.update()
            trainer.update_target()

        if terminated or truncated:
            # pbar.set_postfix(EpisodicReturn=f"{info['episode']['r']:.2f}")
            print("Episodic Return: {}, Time Step {}".format(info["episode"]["r"], i))
            observation, info = env.reset()
