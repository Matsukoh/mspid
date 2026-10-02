import copy
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock

import torch
from tensordict import TensorDict

from configs.sac_socialnav_config import TotalConfig
from sac.models import SocialGaussianPolicy, SocialSACCritic
from socialnav.aggregators import GATAggregator
from socialnav.checkpoints import build_evaluation_policy
from socialnav.trainer import SocialSACTrainer

ROOT = Path(__file__).resolve().parents[1]


def networks(device="cpu"):
    def aggregator():
        return GATAggregator(4, 5, projection_dim=8, enc_hdims=[8])

    actor = SocialGaussianPolicy(
        8, 2, aggregator(), h_dims=[8], act_min=[-0.7, -0.3], act_max=[0.2, 0.8]
    ).to(device)
    critic = SocialSACCritic(8, 2, aggregator(), h_dims=[8]).to(device)
    return actor, critic


def batch(size):
    return TensorDict(
        {
            "done": (torch.arange(size) % 2 == 0).reshape(size, 1),
            "reward": torch.randn(size, 1),
            "action": torch.randn(size, 2),
            "humans_obs": torch.randn(size, 3, 4),
            "robot_obs": torch.randn(size, 5),
            "next_humans_obs": torch.randn(size, 3, 4),
            "next_robot_obs": torch.randn(size, 5),
        },
        batch_size=[size],
    )


class SACTest(unittest.TestCase):
    def test_transformed_density_and_saturated_actions(self):
        actor, _ = networks()
        obs = (torch.randn(1, 5), torch.randn(1, 3, 4))
        torch.manual_seed(1)
        dist = actor.distribution(obs)
        raw = dist.rsample()
        expected = (
            dist.log_prob(raw)
            - torch.log(1 - raw.tanh().square())
            - actor.action_scale.log()
        ).sum(-1, keepdim=True)
        torch.manual_seed(1)
        actions, log_prob = actor.sample_with_log_prob(obs)
        torch.testing.assert_close(log_prob, expected)
        self.assertEqual(actions.shape, (1, 2))
        self.assertEqual(log_prob.shape, (1, 1))
        self.assertTrue(
            (actions >= actor.act_min).all() and (actions <= actor.act_max).all()
        )
        (actions.sum() + log_prob.sum()).backward()
        self.assertIsNotNone(actor.mean.weight.grad)
        with torch.no_grad():
            actor.mean.weight.zero_()
            actor.mean.bias.fill_(100)
        _, log_prob = actor.sample_with_log_prob(obs)
        self.assertTrue(torch.isfinite(log_prob).all())
        actor.eval()
        torch.testing.assert_close(actor.sample(obs), actor.sample(obs))
        self.assertFalse(actor.sample(obs).requires_grad)
        with self.assertRaises(ValueError):
            actor.sample(obs, shape=(2, 2))

    def test_sac_objectives_terminal_mask_and_temperature(self):
        actor, critic = networks()
        data = batch(2)
        trainer = SocialSACTrainer(
            actor,
            critic,
            Mock(sample=Mock(return_value=data)),
            torch.optim.SGD(actor.parameters(), lr=0),
            torch.optim.SGD(critic.parameters(), lr=0),
            2,
            gamma=0.5,
            init_alpha=0.3,
            target_entropy=100,
        )
        obs = (data["robot_obs"], data["humans_obs"])
        next_obs = (data["next_robot_obs"], data["next_humans_obs"])
        with torch.no_grad():
            torch.manual_seed(42)
            next_actions, next_lp = actor.sample_with_log_prob(next_obs)
            tq1, tq2 = trainer.target_critic(next_obs, next_actions)
            targets = data["reward"] + 0.5 * (~data["done"]).float() * (
                torch.minimum(tq1, tq2) - 0.3 * next_lp
            )
            q1, q2 = critic(obs, data["action"])
            expected_critic = (q1 - targets).square().mean() + (
                q2 - targets
            ).square().mean()
            policy_actions, lp = actor.sample_with_log_prob(obs)
            q1, q2 = critic(obs, policy_actions)
            expected_actor = (0.3 * lp - torch.minimum(q1, q2)).mean()
            expected_alpha = -(math.log(0.3) * (lp + 100)).mean()
        torch.manual_seed(42)
        losses = trainer.update()
        for actual, expected in zip(
            losses, (expected_critic, expected_actor, expected_alpha)
        ):
            torch.testing.assert_close(actual, expected)
        self.assertGreater(trainer.alpha.item(), 0.3)
        # With a very low entropy target, alpha must move down.
        trainer.target_entropy = -100
        trainer.alpha_optimizer = torch.optim.Adam([trainer.log_alpha], lr=3e-4)
        previous = trainer.alpha.item()
        trainer.update()
        self.assertLess(trainer.alpha.item(), previous)
        self.assertTrue(all(p.grad is None for p in critic.parameters()))

    def test_parameter_updates_target_and_state_roundtrip(self):
        devices = ["cpu"]
        if torch.cuda.is_available():
            devices.append("cuda")
        if torch.backends.mps.is_available():
            devices.append("mps")
        for device in devices:
            for size in (1, 4):
                with self.subTest(device=device, size=size):
                    actor, critic = networks(device)
                    trainer = SocialSACTrainer(
                        actor,
                        critic,
                        Mock(sample=Mock(return_value=batch(size))),
                        torch.optim.Adam(actor.parameters()),
                        torch.optim.Adam(critic.parameters()),
                        size,
                        device=device,
                        polyak=0.6,
                    )
                    before_actor = [p.clone() for p in actor.parameters()]
                    before_critic = [p.clone() for p in critic.parameters()]
                    losses = trainer.update()
                    self.assertTrue(all(torch.isfinite(loss) for loss in losses))
                    self.assertTrue(
                        any(
                            not torch.equal(a, b)
                            for a, b in zip(before_actor, actor.parameters())
                        )
                    )
                    self.assertTrue(
                        any(
                            not torch.equal(a, b)
                            for a, b in zip(before_critic, critic.parameters())
                        )
                    )
                    trainer.update_target()
                    for old, current, target in zip(
                        before_critic,
                        critic.parameters(),
                        trainer.target_critic.parameters(),
                    ):
                        torch.testing.assert_close(target, 0.6 * old + 0.4 * current)
                        self.assertFalse(target.requires_grad)
                    state = copy.deepcopy(trainer.state_dict())
                    trainer.log_alpha.data.fill_(5)
                    trainer.load_state_dict(state)
                    torch.testing.assert_close(trainer.log_alpha, state["log_alpha"])

    def test_cli_training_save_and_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "run_sac_socialnav.py"),
                    "--train.preliminary-exp-n",
                    "1",
                    "--train.total-it",
                    "1",
                    "--train.batch-size",
                    "1",
                    "--train.buffer-capacity",
                    "32",
                    "--train.updates-per-episode",
                    "2",
                    "--model.h-dims",
                    "[8]",
                    "--model.projection-dim",
                    "8",
                    "--model.aggregator-enc-hdims",
                    "[8]",
                    "--env.time-limit",
                    "1",
                    "--env.val-size",
                    "1",
                    "--env.test-size",
                    "1",
                    "--sim.human-num",
                    "1",
                    "--log.save-model",
                ],
                cwd=tmp,
                capture_output=True,
                text=True,
                timeout=90,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            checkpoint = next(Path(tmp).rglob("model_best.pth"))
            saved = torch.load(checkpoint, weights_only=True)
            self.assertEqual(saved["algorithm"], "SAC")
            self.assertIn("log_alpha", saved["trainer_state_dict"])
            cfg = TotalConfig(_cli_parse_args=False, _env_file=None, **saved["config"])
            policy = build_evaluation_policy(cfg, saved, torch.device("cpu"))
            self.assertFalse(policy.training)
            obs = (torch.randn(1, 5), torch.randn(1, 3, 4))
            torch.testing.assert_close(policy.sample(obs), policy.sample(obs))
            output = Path(tmp) / "evaluation"
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "run_sac_socialnav.py"),
                    "--eval.run-path",
                    str(checkpoint),
                    "--eval.device",
                    "cpu",
                    "--eval.episodes",
                    "1",
                    "--eval.output-dir",
                    str(output),
                ],
                cwd=tmp,
                capture_output=True,
                text=True,
                timeout=90,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            report = json.loads((output / "results.json").read_text())
            self.assertEqual(report["algorithm"], "SAC")
            self.assertEqual(len(report["metrics"]), 7)
            self.assertNotIn("SAC Training", result.stderr)


if __name__ == "__main__":
    unittest.main()
