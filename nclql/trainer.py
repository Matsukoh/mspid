import copy

import torch
import torch.nn.functional as F
from tqdm import tqdm


class NCLQLTrainer:
    def __init__(
        self,
        ald,
        critic,
        replay_buffer,
        critic_optimizer,
        batch_size,
        polyak=0.995,
        gamma=0.99,
        device="cpu",
    ):
        self.alg_name = "NCLQL"
        self.ald = ald
        self.critic = critic
        self.target_critic = copy.deepcopy(critic)
        self.replay_buffer = replay_buffer
        self.critic_optimizer = critic_optimizer
        self.batch_size = batch_size
        self.polyak = polyak
        self.gamma = torch.as_tensor([gamma])

        self.device = device

    def update(self, data_for_logging=None):
        sample = self.replay_buffer.sample(self.batch_size)
        obs, next_obs, act, rwd, done = list(sample.values())
        # rwd *= 0.2
        with torch.no_grad():
            next_act_target = self.ald.sample(
                next_obs.to(self.device), shape=(self.batch_size, self.ald.act_dim)
            )
            L_minus_1 = torch.full((self.batch_size,), self.ald.L - 1)
            Q_target_1, Q_target_2 = self.target_critic(
                next_obs.to(self.device), next_act_target, L_minus_1.to(self.device)
            )
            Q_target_min = torch.min(torch.cat((Q_target_1, Q_target_2), 1), dim=1)[
                0
            ].unsqueeze(-1)

            Q_target = rwd.to(self.device) + (self.gamma * Q_target_min) * (
                1 - done
            ).to(self.device)

        Q1, Q2 = self.critic(
            obs.to(self.device),
            act.to(self.device),
            L_minus_1.to(self.device),
        )

        loss_critic_td = F.mse_loss(Q_target, Q1) + F.mse_loss(Q_target, Q2)
        # self.critic_optimizer.zero_grad()
        # loss_critic_td.backward()
        # self.critic_optimizer.step()
        lc_td = loss_critic_td.data.item()

        Q_cat = torch.stack([Q1, Q2], axis=0)
        Q_mean = torch.mean(Q_cat, axis=0).detach()

        l = torch.randint(
            low=0,
            high=self.ald.L - 1,
            size=(self.batch_size,),
            device=self.device,
        )

        noise = torch.randn(
            size=(self.batch_size, self.ald.act_dim),
            device=self.device,
        )

        sigmas = self.ald.sigma_schedule()
        sigmas_l = sigmas[l]
        a_l = (
            act
            + sigmas_l.reshape(
                (self.batch_size, 1),
            )
            * noise
        )

        Q1_t, Q2_t = self.critic(
            obs.to(self.device),
            a_l.to(self.device),
            l.to(self.device),
        )

        loss_critic_t = F.mse_loss(Q_mean, Q1_t) + F.mse_loss(Q_mean, Q2_t)
        # self.critic_optimizer.zero_grad()
        # loss_critic_t.backward()
        # self.critic_optimizer.step()
        lc_t = loss_critic_t.data.item()
        self.critic_optimizer.zero_grad()
        (loss_critic_td + loss_critic_t).backward()
        self.critic_optimizer.step()

        if data_for_logging is not None:
            data_for_logging[0].log(
                {
                    "loss/critic_td": lc_td,
                    "loss/critic_t": lc_t,
                },
                step=data_for_logging[1],
            )

    def update_target(self):
        for param, target_param in zip(
            self.critic.parameters(), self.target_critic.parameters()
        ):
            target_param.data.mul_(self.polyak)
            target_param.data.add_((1 - self.polyak) * param.data)
