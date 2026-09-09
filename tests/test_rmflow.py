import unittest

import torch
import torch.nn as nn

from flow.models import RMFlowPolicy
from socialnav.trainer import SocialRMFlowTrainer


class ZeroVelocityNet(nn.Module):
    def forward(self, x, obs, r, t):
        return torch.zeros_like(x)


class TinyVelocityNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 2)

    def forward(self, x, obs, r, t):
        robot_obs, human_obs = obs
        inputs = torch.cat((x, robot_obs[:, :1], human_obs[:, :1]), dim=-1)
        return self.linear(inputs)


class RMFlowPolicyTest(unittest.TestCase):
    def test_noise_injection_matches_rmflow_equation(self):
        policy = RMFlowPolicy(
            ZeroVelocityNet(),
            act_dim=2,
            act_max=[10.0, 10.0],
            act_min=[-10.0, -10.0],
            sigma=0.3,
            sigma_min=0.5,
        )
        prior = torch.tensor([[1.0, -1.0]])
        refinement_noise = torch.tensor([[2.0, -2.0]])

        sample = policy.sample(
            torch.zeros(1, 1),
            shape=(1, 2),
            noise=prior,
            refinement_noise=refinement_noise,
        )

        expected = prior + 0.4 * refinement_noise
        torch.testing.assert_close(sample, expected)

    def test_only_one_nfe_is_supported(self):
        policy = RMFlowPolicy(
            ZeroVelocityNet(),
            act_dim=2,
            act_max=[1.0, 1.0],
            act_min=[-1.0, -1.0],
        )
        with self.assertRaisesRegex(ValueError, "only 1-NFE"):
            policy.sample(torch.zeros(1, 1), shape=(1, 2), num_steps=2)


class SocialRMFlowTrainerTest(unittest.TestCase):
    def test_joint_actor_loss_backpropagates(self):
        actor = RMFlowPolicy(
            TinyVelocityNet(),
            act_dim=2,
            act_max=[1.0, 1.0],
            act_min=[-1.0, -1.0],
            sigma=1e-4,
            sigma_min=1e-3,
        )
        optimizer = torch.optim.Adam(actor.parameters(), lr=1e-3)
        trainer = SocialRMFlowTrainer(
            ald=None,
            actor=actor,
            critic=None,
            target_critic=None,
            replay_buffer=None,
            imitation_buffer=None,
            actor_optimizer=optimizer,
            critic_optimizer=None,
            batch_size=4,
        )
        observations = (torch.randn(4, 2), torch.randn(4, 2))

        loss = trainer._actor_loss(torch.randn(4, 2), observations)
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(actor.vnet.linear.weight.grad)
        self.assertEqual(set(trainer.last_actor_losses), {"mean_flow", "nll", "total"})


if __name__ == "__main__":
    unittest.main()
