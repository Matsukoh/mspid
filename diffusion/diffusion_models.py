"""DDPM policy adapted from legacy/diffusion for current SocialNav runners."""

import torch
import torch.nn as nn
from diffusion.helpers import (
    extract,
    linear_beta_schedule,
    cosine_beta_schedule,
    vp_beta_schedule,
)


class DiffusionActor(nn.Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        model,
        act_min=(-1.0, -1.0),
        act_max=(1.0, 1.0),
        beta_schedule="linear",
        n_timesteps=100,
        clip_denoised=True,
        predict_epsilon=True,
        random_sample=True,
        sampling_noise_scale=0.2,
    ):
        super(DiffusionActor, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        if n_timesteps < 1:
            raise ValueError("n_timesteps must be positive")
        if sampling_noise_scale < 0:
            raise ValueError("sampling_noise_scale must be non-negative")
        self.act_dim = action_dim
        self.sampling_noise_scale = sampling_noise_scale
        self.register_buffer("act_min", torch.as_tensor(act_min, dtype=torch.float32))
        self.register_buffer("act_max", torch.as_tensor(act_max, dtype=torch.float32))
        if self.act_min.shape != (action_dim,) or self.act_max.shape != (action_dim,):
            raise ValueError("Action bounds must have shape (action_dim,)")
        if not torch.all(self.act_min < self.act_max):
            raise ValueError("Each lower action bound must be below its upper bound")
        self.model = model
        self.random_sample = random_sample
        if beta_schedule == "linear":
            betas = linear_beta_schedule(n_timesteps)
        elif beta_schedule == "cosine":
            betas = cosine_beta_schedule(n_timesteps)
        elif beta_schedule == "vp":
            betas = vp_beta_schedule(n_timesteps)
        else:
            raise ValueError(f"Unknown beta_schedule: {beta_schedule}")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.n_timesteps = int(n_timesteps)
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon

        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )
        self.register_buffer(
            "log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod)
        )
        self.register_buffer(
            "sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod)
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1)
        )

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.register_buffer("posterior_variance", posterior_variance)

        ## log calculation clipped because the posterior variance
        ## is 0 at the beginning of the diffusion chain
        self.register_buffer(
            "posterior_log_variance_clipped",
            torch.log(torch.clamp(posterior_variance, min=1e-20)),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

    # ------------------------------------------ sampling ------------------------------------------#

    def predict_start_from_noise(self, x_t, t, noise):
        """
        if self.predict_epsilon, model output is (scaled) noise;
        otherwise, model predicts x0 directly
        """
        if self.predict_epsilon:
            return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
                - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, s):
        x_recon = self.predict_start_from_noise(x, t=t, noise=self.model(x, t, s))

        if self.clip_denoised:
            x_recon.clamp_(self.act_min, self.act_max)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t
        )
        return model_mean, posterior_variance, posterior_log_variance

    def p_sample(self, x, t, s):
        b = x.shape[0]
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, s=s)
        noise = torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return (
            model_mean
            + nonzero_mask
            * (0.5 * model_log_variance).exp()
            * noise
            * self.sampling_noise_scale
            * self.random_sample
        )

    @torch.no_grad()
    def sample(self, obs, shape=None):
        """Return bounded [batch, action_dim] actions, like the flow policies."""
        batch_size = obs[0].shape[0]
        if shape is None:
            shape = (batch_size, self.act_dim)
        if tuple(shape) != (batch_size, self.act_dim):
            raise ValueError(
                "shape must match the observation batch and action dimension"
            )
        x = (
            torch.randn(shape, device=self.betas.device)
            if self.random_sample
            else torch.zeros(shape, device=self.betas.device)
        )
        for i in reversed(range(self.n_timesteps)):
            t = torch.full((batch_size,), i, device=x.device, dtype=torch.long)
            x = self.p_sample(x, t, obs)
        return x.clamp(self.act_min, self.act_max)

    def forward(self, obs, shape=None):
        return self.sample(obs, shape=shape)

    # ------------------------------------------ training ------------------------------------------#

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sample = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        return sample
