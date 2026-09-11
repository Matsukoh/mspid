import copy
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchrl.data.replay_buffers
from torch.func import jvp
from tqdm import tqdm


class SocialMSPIDTrainer:
    def __init__(
        self,
        ald: nn.Module,
        actor: nn.Module,
        critic: nn.Module,
        target_critic: nn.Module,
        replay_buffer: torchrl.data.replay_buffers,
        imitation_buffer: torchrl.data.replay_buffers,
        actor_optimizer: torch.optim.Optimizer,
        critic_optimizer: torch.optim.Optimizer,
        batch_size: int,
        time_sampler: Literal["uniform", "logit_normal"] = "uniform",
        unequal_time_ratio: float = 0.75,
        td_sample_size: int = 10,
        distil_sample_size: int = 10,
        polyak: float = 0.995,
        gamma: float = 0.99,
        device: str = "cpu",
    ):
        self.alg_name = "MSPID"
        self.ald = ald
        self.actor = actor
        self.critic = critic
        # self.target_critic = copy.deepcopy(critic)
        self.target_critic = target_critic
        self.replay_buffer = replay_buffer
        self.imitation_buffer = imitation_buffer
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.batch_size = batch_size
        self.time_sampler = time_sampler
        self.unequal_time_ratio = unequal_time_ratio
        self.td_sample_size = td_sample_size
        self.distil_sample_size = distil_sample_size
        self.polyak = polyak
        self.gamma = torch.as_tensor([gamma])

        self.device = device

    def _sample_times(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.time_sampler == "uniform":
            times = torch.rand(batch_size, 2, device=device, dtype=dtype)
        elif self.time_sampler == "logit_normal":
            normal = torch.randn(batch_size, 2, device=device, dtype=dtype)
            times = torch.sigmoid(
                self.logit_normal_mu + self.logit_normal_sigma * normal
            )
        else:
            raise ValueError(f"Unknown time_sampler: {self.time_sampler}")

        r, t = torch.sort(times, dim=-1).values.unbind(dim=-1)

        # Include r=t samples so that the boundary identity u(x_t,t,t)=v_t
        # is explicitly trained.
        equal_mask = torch.rand(batch_size, device=device) > self.unequal_time_ratio
        r = torch.where(equal_mask, t, r)
        return r, t

    def _conditional_mean_flow_loss(
        self,
        actions: torch.Tensor,
        obs: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Compute the reverse-time conditional MeanFlow matching loss."""
        actions = actions.to(self.device)
        obs = tuple(value.to(self.device) for value in obs)
        batch_size = actions.shape[0]
        r, t = self._sample_times(
            batch_size=batch_size,
            device=actions.device,
            dtype=actions.dtype,
        )

        # This project uses data at time 0 and Gaussian noise at time 1.
        noise = torch.randn_like(actions)
        t_column = t[:, None]
        x_t = (1.0 - t_column) * actions + t_column * noise
        v_t = noise - actions
        interval = (t - r)[:, None]

        u_prediction = self.actor.vnet(x_t, obs, r, t)

        def model_along_path(
            current_x: torch.Tensor,
            current_r: torch.Tensor,
            current_t: torch.Tensor,
        ) -> torch.Tensor:
            return self.actor.vnet(current_x, obs, current_r, current_t)

        _, du_dt = jvp(
            model_along_path,
            primals=(x_t, r, t),
            tangents=(v_t, torch.zeros_like(r), torch.ones_like(t)),
        )

        # Reverse-time form of equation (4): u = v - (t-r) D_t u.
        u_target = (v_t - interval * du_dt).detach()
        return (u_prediction - u_target).square().mean()

    def _actor_loss(
        self,
        actions: torch.Tensor,
        obs: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        return self._conditional_mean_flow_loss(actions, obs)

    def update_nclql(self):
        sample = self.replay_buffer.sample(self.batch_size)
        r_obs, next_r_obs, h_obs, next_h_obs, act, rwd, done = list(sample.values())
        # rwd *= 0.2
        with torch.no_grad():
            next_act_target = self.actor.sample(
                (
                    next_r_obs.to(self.device),
                    next_h_obs.to(self.device),
                ),
                shape=(self.batch_size, self.ald.act_dim),
            )
            # next_act_target = self.ald.sample(
            #     (
            #         next_r_obs.to(self.device),
            #         next_h_obs.to(self.device),
            #     ),
            #     shape=(self.batch_size, self.ald.act_dim),
            # )
            L_minus_1 = torch.full((self.batch_size,), self.ald.L - 1)
            Q_target_1, Q_target_2 = self.target_critic(
                (
                    next_r_obs.to(self.device),
                    next_h_obs.to(self.device),
                ),
                next_act_target,
                L_minus_1.to(self.device),
            )
            Q_target_min = torch.min(torch.cat((Q_target_1, Q_target_2), 1), dim=1)[
                0
            ].unsqueeze(-1)

            Q_target = rwd.to(self.device) + (self.gamma * Q_target_min) * (
                1 - done
            ).to(self.device)

        Q1, Q2 = self.critic(
            (
                r_obs.to(self.device),
                h_obs.to(self.device),
            ),
            act.to(self.device),
            L_minus_1.to(self.device),
        )

        loss_critic_td = F.mse_loss(Q_target, Q1) + F.mse_loss(Q_target, Q2)
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
            (
                r_obs.to(self.device),
                h_obs.to(self.device),
            ),
            a_l.to(self.device),
            l.to(self.device),
        )

        loss_critic_t = F.mse_loss(Q_mean, Q1_t) + F.mse_loss(Q_mean, Q2_t)
        lc_t = loss_critic_t.data.item()
        self.critic_optimizer.zero_grad()
        (loss_critic_td + loss_critic_t).backward()
        self.critic_optimizer.step()

        # if data_for_logging is not None:
        #     data_for_logging[0].log(
        #         {
        #             "loss/critic_td": lc_td,
        #             "loss/critic_t": lc_t,
        #         },
        #         step=data_for_logging[1],
        #     )

        return lc_td, lc_t

    def update(self):
        sample = self.replay_buffer.sample(self.batch_size)
        r_obs, next_r_obs, h_obs, next_h_obs, act, rwd, done = list(sample.values())
        # rwd *= 0.2
        with torch.no_grad():
            next_r_obs_repeated = next_r_obs.repeat_interleave(
                self.td_sample_size, dim=0
            ).to(self.device)

            next_h_obs_repeated = next_h_obs.repeat_interleave(
                self.td_sample_size, dim=0
            ).to(self.device)

            # next_act_target = self.ald.sample(
            #     (
            #         next_r_obs_repeated,
            #         next_h_obs_repeated,
            #     ),
            #     shape=(self.batch_size * self.td_sample_size, self.ald.act_dim),
            # )
            next_act_target = self.actor.sample(
                (
                    next_r_obs_repeated,
                    next_h_obs_repeated,
                ),
                shape=(self.batch_size * self.td_sample_size, self.ald.act_dim),
            )
            L_minus_1 = torch.full((self.batch_size,), self.ald.L - 1)
            Q_target_1, Q_target_2 = self.target_critic(
                (
                    next_r_obs_repeated,
                    next_h_obs_repeated,
                ),
                next_act_target,
                L_minus_1.repeat_interleave(self.td_sample_size, dim=0).to(self.device),
            )
            Q_target_min = torch.min(
                torch.cat(
                    (
                        Q_target_1.view(self.batch_size, self.td_sample_size, 1).mean(
                            dim=1
                        ),
                        Q_target_2.view(self.batch_size, self.td_sample_size, 1).mean(
                            dim=1
                        ),
                    ),
                    1,
                ),
                dim=1,
            )[0].unsqueeze(-1)

            Q_target = rwd.to(self.device) + (
                self.gamma.to(self.device) * Q_target_min
            ) * (1 - done).to(self.device)

        Q1, Q2 = self.critic(
            (
                r_obs.to(self.device),
                h_obs.to(self.device),
            ),
            act.to(self.device),
            L_minus_1.to(self.device),
        )

        loss_critic_td = F.mse_loss(Q_target, Q1) + F.mse_loss(Q_target, Q2)
        # lc_td = loss_critic_td.data.item()

        Q_cat = torch.stack([Q1, Q2], axis=0)
        Q_mean = torch.mean(Q_cat, axis=0).detach()

        l = torch.randint(
            low=0,
            high=self.ald.L - 1,
            size=(self.batch_size,),
        )

        noise = torch.randn(
            size=(self.batch_size, self.ald.act_dim),
            device=self.device,
        )

        # sigmas = self.ald.sigma_schedule()
        sigmas_l = self.ald.sigmas[l]
        a_l = (
            act.to(self.device)
            + sigmas_l.reshape(
                (self.batch_size, 1),
            )
            * noise
        )

        Q1_t, Q2_t = self.critic(
            (
                r_obs.to(self.device),
                h_obs.to(self.device),
            ),
            a_l.to(self.device),
            l.to(self.device),
        )

        loss_critic_t = F.mse_loss(Q_mean, Q1_t) + F.mse_loss(Q_mean, Q2_t)
        # lc_t = loss_critic_t.data.item()
        self.critic_optimizer.zero_grad()
        (loss_critic_td + loss_critic_t).backward()
        self.critic_optimizer.step()

        r_obs_repeated = r_obs.repeat_interleave(self.distil_sample_size, dim=0).to(
            self.device
        )
        h_obs_repeated = h_obs.repeat_interleave(self.distil_sample_size, dim=0).to(
            self.device
        )

        act_sample = self.ald.sample(
            # (
            #     r_obs.to(self.device),
            #     h_obs.to(self.device),
            # ),
            # self.target_critic,
            (
                r_obs_repeated,
                h_obs_repeated,
            ),
            # shape=(self.batch_size, self.ald.act_dim),
            shape=(self.batch_size * self.distil_sample_size, self.ald.act_dim),
        )

        loss_actor = self._actor_loss(
            act_sample,
            (r_obs_repeated, h_obs_repeated),
        )
        # la = loss_actor.data.item()
        self.actor_optimizer.zero_grad()
        loss_actor.backward()
        self.actor_optimizer.step()

        # if data_for_logging is not None:
        #     data_for_logging[0].log(
        #         {
        #             "loss/critic_td": lc_td,
        #             "loss/critic_t": lc_t,
        #             "loss/actor": la,
        #         },
        #         step=data_for_logging[1],
        #     )

        return loss_critic_td.detach(), loss_critic_t.detach(), loss_actor.detach()

    def update_imitation(self, epoch_num=100, data_for_logging=None):
        for e in tqdm(range(epoch_num)):
            # sample = self.episodic_buffer.sample(self.batch_size)
            # obs, next_obs, act, rwd, done = list(sample.values())
            # batch_size = self.episodic_buffer.batch_size
            for batch in self.imitation_buffer:
                r_obs, h_obs, act = (
                    batch["robot_obs"],
                    batch["humans_obs"],
                    batch["action"],
                )
                # rwd *= 0.2

                loss_actor = self._actor_loss(act, (r_obs, h_obs))
                la = loss_actor.data.item()
                self.actor_optimizer.zero_grad()
                loss_actor.backward()
                self.actor_optimizer.step()

        if data_for_logging is not None:
            data_for_logging[0].log(
                {
                    "loss/actor_imitation": la,
                },
                step=data_for_logging[1],
            )

    def update_target(self):
        for param, target_param in zip(
            self.critic.parameters(), self.target_critic.parameters()
        ):
            target_param.data.mul_(self.polyak)
            target_param.data.add_((1 - self.polyak) * param.data)


class SocialRMFlowTrainer(SocialMSPIDTrainer):
    """MSPID trainer whose actor objective is the RMFlow objective."""

    def __init__(self, *args, nll_weight: float = 0.1, **kwargs):
        super().__init__(*args, **kwargs)
        if nll_weight < 0:
            raise ValueError("nll_weight must be non-negative.")
        for attribute in ("sigma", "sigma_min"):
            if not hasattr(self.actor, attribute):
                raise TypeError(
                    "SocialRMFlowTrainer requires an RMFlowPolicy actor with "
                    f"an {attribute} attribute."
                )

        self.alg_name = "MSPID-RMFlow"
        self.nll_weight = nll_weight
        self.last_actor_losses: dict[str, torch.Tensor] = {}

    def _actor_loss(
        self,
        actions: torch.Tensor,
        obs: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        actions = actions.to(self.device)
        obs = tuple(value.to(self.device) for value in obs)

        # Equation (6), adapted to this repository's reverse-time convention:
        # the MeanFlow endpoint is a slightly noisy action.
        intermediate_target = actions + self.actor.sigma.to(
            device=actions.device, dtype=actions.dtype
        ) * torch.randn_like(actions)
        mean_flow_loss = self._conditional_mean_flow_loss(intermediate_target, obs)

        # Equation (10): match a noisy observed action to the 1-NFE transport
        # mean. The prior used here is independent of the interpolation sample.
        batch_size = actions.shape[0]
        prior = torch.randn_like(actions)
        r = torch.zeros(batch_size, device=actions.device, dtype=actions.dtype)
        t = torch.ones(batch_size, device=actions.device, dtype=actions.dtype)
        generated_mean = prior - self.actor.vnet(prior, obs, r, t)
        noisy_target = actions + self.actor.sigma_min.to(
            device=actions.device, dtype=actions.dtype
        ) * torch.randn_like(actions)
        nll_loss = (noisy_target - generated_mean).square().mean()

        total_loss = mean_flow_loss + self.nll_weight * nll_loss
        self.last_actor_losses = {
            "mean_flow": mean_flow_loss.detach(),
            "nll": nll_loss.detach(),
            "total": total_loss.detach(),
        }
        return total_loss


class SocialNCLQLTrainer:
    def __init__(
        self,
        ald: nn.Module,
        critic: nn.Module,
        replay_buffer: torchrl.data.replay_buffers,
        critic_optimizer: torch.optim.Optimizer,
        batch_size: int,
        time_sampler: Literal["uniform", "logit_normal"] = "uniform",
        unequal_time_ratio: float = 0.75,
        td_sample_size: int = 10,
        polyak: float = 0.995,
        gamma: float = 0.99,
        device: str = "cpu",
    ):
        self.alg_name = "NCLQL"
        self.ald = ald
        self.critic = critic
        self.target_critic = copy.deepcopy(critic)
        self.replay_buffer = replay_buffer
        self.critic_optimizer = critic_optimizer
        self.batch_size = batch_size
        self.time_sampler = time_sampler
        self.unequal_time_ratio = unequal_time_ratio
        self.td_sample_size = td_sample_size
        self.polyak = polyak
        self.gamma = torch.as_tensor([gamma])

        self.device = device

    def _sample_times(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.time_sampler == "uniform":
            times = torch.rand(batch_size, 2, device=device, dtype=dtype)
        elif self.time_sampler == "logit_normal":
            normal = torch.randn(batch_size, 2, device=device, dtype=dtype)
            times = torch.sigmoid(
                self.logit_normal_mu + self.logit_normal_sigma * normal
            )
        else:
            raise ValueError(f"Unknown time_sampler: {self.time_sampler}")

        r, t = torch.sort(times, dim=-1).values.unbind(dim=-1)

        # Include r=t samples so that the boundary identity u(x_t,t,t)=v_t
        # is explicitly trained.
        equal_mask = torch.rand(batch_size, device=device) > self.unequal_time_ratio
        r = torch.where(equal_mask, t, r)
        return r, t

    def update(self):
        sample = self.replay_buffer.sample(self.batch_size)
        r_obs, next_r_obs, h_obs, next_h_obs, act, rwd, done = list(sample.values())
        # rwd *= 0.2
        with torch.no_grad():
            # next_act_target = self.actor.sample(
            #     next_obs.to(self.device), shape=(self.batch_size, self.ald.act_dim)
            # )

            next_r_obs_repeated = next_r_obs.repeat_interleave(
                self.td_sample_size, dim=0
            ).to(self.device)

            next_h_obs_repeated = next_h_obs.repeat_interleave(
                self.td_sample_size, dim=0
            ).to(self.device)

            next_act_target = self.ald.sample(
                (
                    next_r_obs_repeated,
                    next_h_obs_repeated,
                ),
                shape=(self.batch_size * self.td_sample_size, self.ald.act_dim),
            )
            L_minus_1 = torch.full((self.batch_size,), self.ald.L - 1)
            Q_target_1, Q_target_2 = self.target_critic(
                (
                    next_r_obs_repeated,
                    next_h_obs_repeated,
                ),
                next_act_target,
                L_minus_1.repeat_interleave(self.td_sample_size, dim=0).to(self.device),
            )
            Q_target_min = torch.min(
                torch.cat(
                    (
                        Q_target_1.view(self.batch_size, self.td_sample_size, 1).mean(
                            dim=1
                        ),
                        Q_target_2.view(self.batch_size, self.td_sample_size, 1).mean(
                            dim=1
                        ),
                    ),
                    1,
                ),
                dim=1,
            )[0].unsqueeze(-1)

            Q_target = rwd.to(self.device) + (
                self.gamma.to(self.device) * Q_target_min
            ) * (1 - done).to(self.device)

        Q1, Q2 = self.critic(
            (
                r_obs.to(self.device),
                h_obs.to(self.device),
            ),
            act.to(self.device),
            L_minus_1.to(self.device),
        )

        loss_critic_td = F.mse_loss(Q_target, Q1) + F.mse_loss(Q_target, Q2)
        # lc_td = loss_critic_td.data.item()

        Q_cat = torch.stack([Q1, Q2], axis=0)
        Q_mean = torch.mean(Q_cat, axis=0).detach()

        l = torch.randint(
            low=0,
            high=self.ald.L - 1,
            size=(self.batch_size,),
        )

        noise = torch.randn(
            size=(self.batch_size, self.ald.act_dim),
            device=self.device,
        )

        # sigmas = self.ald.sigma_schedule()
        sigmas_l = self.ald.sigmas[l]
        a_l = (
            act.to(self.device)
            + sigmas_l.reshape(
                (self.batch_size, 1),
            )
            * noise
        )

        Q1_t, Q2_t = self.critic(
            (
                r_obs.to(self.device),
                h_obs.to(self.device),
            ),
            a_l.to(self.device),
            l.to(self.device),
        )

        loss_critic_t = F.mse_loss(Q_mean, Q1_t) + F.mse_loss(Q_mean, Q2_t)
        # lc_t = loss_critic_t.data.item()
        self.critic_optimizer.zero_grad()
        (loss_critic_td + loss_critic_t).backward()
        self.critic_optimizer.step()

        # if data_for_logging is not None:
        #     data_for_logging[0].log(
        #         {
        #             "loss/critic_td": lc_td,
        #             "loss/critic_t": lc_t,
        #         },
        #         step=data_for_logging[1],
        #     )

        return loss_critic_td.detach(), loss_critic_t.detach()

    def update_target(self):
        for param, target_param in zip(
            self.critic.parameters(), self.target_critic.parameters()
        ):
            target_param.data.mul_(self.polyak)
            target_param.data.add_((1 - self.polyak) * param.data)
