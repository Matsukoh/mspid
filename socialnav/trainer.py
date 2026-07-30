import copy
from typing import Literal, Optional

import torch
import torch.nn.functional as F
from torch.func import jvp
from tqdm import tqdm


class SocialMSPIDTrainer:
    def __init__(
        self,
        ald,
        actor,
        critic,
        replay_buffer,
        imitation_buffer,
        actor_optimizer,
        critic_optimizer,
        batch_size,
        time_sampler: Literal["uniform", "logit_normal"] = "uniform",
        unequal_time_ratio: float = 0.75,
        polyak=0.995,
        gamma=0.99,
        device="cpu",
    ):
        self.alg_name = "MSPID"
        self.ald = ald
        self.actor = actor
        self.critic = critic
        self.target_critic = copy.deepcopy(critic)
        self.replay_buffer = replay_buffer
        self.imitation_buffer = imitation_buffer
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.batch_size = batch_size
        self.time_sampler = time_sampler
        self.unequal_time_ratio = unequal_time_ratio
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

    def update_nclql(self, data_for_logging=None):
        sample = self.replay_buffer.sample(self.batch_size)
        r_obs, next_r_obs, h_obs, next_h_obs, act, rwd, done = list(sample.values())
        # rwd *= 0.2
        with torch.no_grad():
            # next_act_target = self.actor.sample(
            #     (
            #         next_r_obs.to(self.device),
            #         next_h_obs.to(self.device),
            #     ),
            #     shape=(self.batch_size, self.ald.act_dim),
            # )
            next_act_target = self.ald.sample(
                (
                    next_r_obs.to(self.device),
                    next_h_obs.to(self.device),
                ),
                shape=(self.batch_size, self.ald.act_dim),
            )
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

        if data_for_logging is not None:
            data_for_logging[0].log(
                {
                    "loss/critic_td": lc_td,
                    "loss/critic_t": lc_t,
                },
                step=data_for_logging[1],
            )

    def update(self, data_for_logging=None):
        sample = self.replay_buffer.sample(self.batch_size)
        r_obs, next_r_obs, h_obs, next_h_obs, act, rwd, done = list(sample.values())
        # rwd *= 0.2
        with torch.no_grad():
            # next_act_target = self.actor.sample(
            #     (
            #         next_r_obs.to(self.device),
            #         next_h_obs.to(self.device),
            #     ),
            #     shape=(self.batch_size, self.ald.act_dim),
            # )
            sample_size = 10

            next_r_obs_repeated = next_r_obs.repeat_interleave(sample_size, dim=0).to(
                self.device
            )

            next_h_obs_repeated = next_h_obs.repeat_interleave(sample_size, dim=0).to(
                self.device
            )

            next_act_target = self.ald.sample(
                (
                    next_r_obs_repeated,
                    next_h_obs_repeated,
                ),
                shape=(self.batch_size * sample_size, self.ald.act_dim),
            )
            L_minus_1 = torch.full((self.batch_size,), self.ald.L - 1)
            Q_target_1, Q_target_2 = self.target_critic(
                (
                    next_r_obs_repeated,
                    next_h_obs_repeated,
                ),
                next_act_target,
                L_minus_1.repeat_interleave(sample_size, dim=0).to(self.device),
            )
            Q_target_min = torch.min(
                torch.cat(
                    (
                        Q_target_1.view(self.batch_size, sample_size, 1).mean(dim=1),
                        Q_target_2.view(self.batch_size, sample_size, 1).mean(dim=1),
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
        lc_td = loss_critic_td.data.item()

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

        sigmas = self.ald.sigma_schedule()
        sigmas_l = sigmas[l].to(self.device)
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
        lc_t = loss_critic_t.data.item()
        self.critic_optimizer.zero_grad()
        (loss_critic_td + loss_critic_t).backward()
        self.critic_optimizer.step()

        r_obs_repeated = r_obs.repeat_interleave(sample_size, dim=0).to(self.device)
        h_obs_repeated = h_obs.repeat_interleave(sample_size, dim=0).to(self.device)

        act_sample = self.ald.sample(
            # (
            #     r_obs.to(self.device),
            #     h_obs.to(self.device),
            # ),
            (
                r_obs_repeated,
                h_obs_repeated,
            ),
            # shape=(self.batch_size, self.ald.act_dim),
            shape=(self.batch_size * sample_size, self.ald.act_dim),
        )

        r, t = self._sample_times(
            # batch_size=self.batch_size,
            batch_size=self.batch_size * sample_size,
            device=self.device,
            dtype=act_sample.dtype,
        )

        # x_0=data_action, x_1=noise, x_t=(1-t)x_0+t*x_1
        noise = torch.randn_like(act_sample)
        t_column = t[:, None]
        x_t = (1.0 - t_column) * act_sample + t_column * noise
        v_t = noise - act_sample
        interval = (t - r)[:, None]

        # Current prediction at the primal point.
        u_prediction = self.actor.vnet(
            x_t,
            # (
            #     r_obs.to(self.device),
            #     h_obs.to(self.device),
            # ),
            (
                r_obs_repeated,
                h_obs_repeated,
            ),
            r,
            t,
        )

        # Total derivative along the interpolation:
        # d/dt u(x_t, obs, r, t) =
        #     J_x u * v_t + partial_t u.
        #
        # obs and r are held fixed. jvp remains differentiable w.r.t. parameters,
        # while the bootstrap target is detached below.
        def model_along_path(
            current_x: torch.Tensor,
            current_r: torch.Tensor,
            current_t: torch.Tensor,
        ) -> torch.Tensor:
            return self.actor.vnet(
                current_x,
                # (
                #     r_obs.to(self.device),
                #     h_obs.to(self.device),
                # ),
                (
                    r_obs_repeated,
                    h_obs_repeated,
                ),
                current_r,
                current_t,
            )

        _, du_dt = jvp(
            model_along_path,
            primals=(x_t, r, t),
            tangents=(v_t, torch.zeros_like(r), torch.ones_like(t)),
        )

        # MeanFlow bootstrap target:
        # u = v - (t-r) * d_t u
        u_target = (v_t - interval * du_dt).detach()

        per_element_error = (u_prediction - u_target).square()
        per_sample_mse = per_element_error.mean(dim=-1)
        loss_actor = per_sample_mse.mean()
        la = loss_actor.data.item()
        self.actor_optimizer.zero_grad()
        loss_actor.backward()
        self.actor_optimizer.step()

        if data_for_logging is not None:
            data_for_logging[0].log(
                {
                    "loss/critic_td": lc_td,
                    "loss/critic_t": lc_t,
                    "loss/actor": la,
                },
                step=data_for_logging[1],
            )

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
                batch_size = r_obs.shape[0]
                # rwd *= 0.2

                r, t = self._sample_times(
                    batch_size=batch_size,
                    device=self.device,
                    dtype=act.dtype,
                )

                # x_0=data_action, x_1=noise, x_t=(1-t)x_0+t*x_1
                noise = torch.randn_like(act)
                t_column = t[:, None]
                x_t = (1.0 - t_column) * act + t_column * noise
                v_t = noise - act
                interval = (t - r)[:, None]

                # Current prediction at the primal point.
                u_prediction = self.actor.vnet(x_t, (r_obs, h_obs), r, t)

                # Total derivative along the interpolation:
                # d/dt u(x_t, obs, r, t) =
                #     J_x u * v_t + partial_t u.
                #
                # obs and r are held fixed. jvp remains differentiable w.r.t. parameters,
                # while the bootstrap target is detached below.
                def model_along_path(
                    current_x: torch.Tensor,
                    current_r: torch.Tensor,
                    current_t: torch.Tensor,
                ) -> torch.Tensor:
                    return self.actor.vnet(
                        current_x, (r_obs, h_obs), current_r, current_t
                    )

                _, du_dt = jvp(
                    model_along_path,
                    primals=(x_t, r, t),
                    tangents=(v_t, torch.zeros_like(r), torch.ones_like(t)),
                )

                # MeanFlow bootstrap target:
                # u = v - (t-r) * d_t u
                u_target = (v_t - interval * du_dt).detach()

                per_element_error = (u_prediction - u_target).square()
                per_sample_mse = per_element_error.mean(dim=-1)
                loss_actor = per_sample_mse.mean()
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


class SocialNCLQLTrainer:
    def __init__(
        self,
        ald,
        critic,
        replay_buffer,
        critic_optimizer,
        batch_size,
        time_sampler: Literal["uniform", "logit_normal"] = "uniform",
        unequal_time_ratio: float = 0.75,
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
        self.time_sampler = time_sampler
        self.unequal_time_ratio = unequal_time_ratio
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

    def update(self, data_for_logging=None):
        sample = self.replay_buffer.sample(self.batch_size)
        r_obs, next_r_obs, h_obs, next_h_obs, act, rwd, done = list(sample.values())
        # rwd *= 0.2
        with torch.no_grad():
            # next_act_target = self.actor.sample(
            #     next_obs.to(self.device), shape=(self.batch_size, self.ald.act_dim)
            # )
            sample_size = 10

            next_r_obs_repeated = next_r_obs.repeat_interleave(sample_size, dim=0).to(
                self.device
            )

            next_h_obs_repeated = next_h_obs.repeat_interleave(sample_size, dim=0).to(
                self.device
            )

            next_act_target = self.ald.sample(
                (
                    next_r_obs_repeated,
                    next_h_obs_repeated,
                ),
                shape=(self.batch_size * sample_size, self.ald.act_dim),
            )
            L_minus_1 = torch.full((self.batch_size,), self.ald.L - 1)
            Q_target_1, Q_target_2 = self.target_critic(
                (
                    next_r_obs_repeated,
                    next_h_obs_repeated,
                ),
                next_act_target,
                L_minus_1.repeat_interleave(sample_size, dim=0).to(self.device),
            )
            Q_target_min = torch.min(
                torch.cat(
                    (
                        Q_target_1.view(self.batch_size, sample_size, 1).mean(dim=1),
                        Q_target_2.view(self.batch_size, sample_size, 1).mean(dim=1),
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
        lc_td = loss_critic_td.data.item()

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

        sigmas = self.ald.sigma_schedule()
        sigmas_l = sigmas[l].to(self.device)
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
