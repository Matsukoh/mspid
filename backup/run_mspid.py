import importlib
import random

import gymnasium as gym
import numpy as np
import torch
from gymnasium.wrappers import RecordEpisodeStatistics
from tensordict import TensorDict
from torchrl.data import LazyTensorStorage, ListStorage, ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from tqdm import tqdm, trange

from flow.models import ConditionalMeanVelocityNet, MeanFlowPolicy
from nclql.models import AnnealedLangevinDynamics, NoiseConditionedCritic
from utils.trainer import MSPIDTrainer


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

env = gym.make("HalfCheetah-v5", render_mode="human")
# env = gym.make("Reacher-v4", render_mode="human")
# env = gym.make("HumanoidStandup-v5", impact_cost_weight=0.5e-6, render_mode="human")
# env = gym.make("Humanoid-v5", ctrl_cost_weight=0.1, render_mode="human")
# env = gym.make("InvertedPendulum-v5", reset_noise_scale=0.1, render_mode="human")
env = gym.wrappers.RecordEpisodeStatistics(env)
test = env.action_space
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]

buffer = ReplayBuffer(storage=LazyTensorStorage(cfg.train.buffer_capacity))

ep_buffer = ReplayBuffer(
    storage=LazyTensorStorage(cfg.train.buffer_capacity),
    sampler=SamplerWithoutReplacement(drop_last=False),
    batch_size=64,
)

critic = NoiseConditionedCritic(
    obs_dim + act_dim + cfg.model.time_dim, 1, h_dims=cfg.model.h_dims
)

ald = AnnealedLangevinDynamics(
    model=critic,
    L=cfg.model.L,
    T=cfg.model.T,
    w=cfg.model.w,
    act_dim=act_dim,
    act_max=env.action_space.high,
    act_min=env.action_space.low,
    q_grad_norm=cfg.model.q_grad_norm,
)

ald = torch.compile(ald)

vnet = ConditionalMeanVelocityNet(
    obs_dim=obs_dim, act_dim=act_dim, h_dim=256, time_dim=cfg.model.time_dim
)

actor = MeanFlowPolicy(
    vnet=vnet,
    act_dim=act_dim,
    act_max=env.action_space.high,
    act_min=env.action_space.low,
)

# actor = torch.compile(actor)

actor_optimizer = torch.optim.Adam(actor.parameters(), lr=cfg.train.lr)
critic_optimizer = torch.optim.Adam(critic.parameters(), lr=cfg.train.lr)

trainer = MSPIDTrainer(
    ald=ald,
    actor=actor,
    critic=critic,
    replay_buffer=buffer,
    episodic_buffer=ep_buffer,
    actor_optimizer=actor_optimizer,
    critic_optimizer=critic_optimizer,
    batch_size=cfg.train.batch_size,
)

observation, info = env.reset()
max_return = -np.inf
with tqdm(
    range(cfg.train.total_it),
    desc=cfg.train.training_alg + " Training",
    dynamic_ncols=True,
) as pbar:
    for i, ch in enumerate(pbar):
        if len(buffer) > cfg.train.batch_size:
            # if len(buffer) > start_steps:
            action = actor.sample(
                torch.tensor(observation, dtype=torch.float32).reshape(1, -1),
                shape=(1, act_dim),
            )
            # action = ald.sample(
            #     torch.tensor(observation, dtype=torch.float32).reshape(1, -1),
            #     shape=(1, act_dim),
            # )
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
        ep_buffer.add(sample)
        observation = next_observation
        if len(buffer) > cfg.train.batch_size:
            # if len(buffer) > start_steps:
            trainer.update()
            trainer.update_target()

        test = len(ep_buffer)

        if terminated or truncated:
            if info["episode"]["r"] > max_return:
                if not (max_return == -np.inf):
                    trainer.update_imitation(epoch_num=10)
                max_return = info["episode"]["r"]
            # pbar.set_postfix(EpisodicReturn=f"{info['episode']['r']:.2f}")
            print("Episodic Return: {}, Time Step {}".format(info["episode"]["r"], i))
            observation, info = env.reset()
            ep_buffer.empty()
