import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock

import torch
from tensordict import TensorDict
from torchrl.data import LazyTensorStorage, ReplayBuffer

from configs.qsm_socialnav_config import TotalConfig
from diffusion.diffusion_models import DiffusionActor
from diffusion.models import DMLP, SocialCritic
from socialnav.aggregators import GATAggregator
from socialnav.checkpoints import build_evaluation_policy
from socialnav.trainer import SocialQSMTrainer

ROOT = Path(__file__).resolve().parents[1]


def networks(device="cpu", schedule="linear", random_sample=True):
    def aggregator():
        return GATAggregator(4, 5, projection_dim=8, enc_hdims=[8])

    actor = DiffusionActor(
        8,
        2,
        DMLP(8, 2, h_dims=[8], aggregator=aggregator(), t_dim=4),
        n_timesteps=3,
        beta_schedule=schedule,
        random_sample=random_sample,
        act_min=[-0.7, -0.3],
        act_max=[0.2, 0.8],
    ).to(device)
    critic = SocialCritic(10, 1, h_dims=[8], aggregator=aggregator()).to(device)
    return actor, critic


class QSMTest(unittest.TestCase):
    def test_diffusion_sampling_and_checkpoint_roundtrip(self):
        for schedule in ("linear", "cosine", "vp"):
            with self.subTest(schedule=schedule):
                actor, _ = networks(schedule=schedule, random_sample=False)
                obs = (torch.randn(1, 5), torch.randn(1, 3, 4))
                sampled = actor.sample(obs, shape=(1, 2))
                self.assertEqual(sampled.shape, (1, 2))
                self.assertFalse(sampled.requires_grad)
                self.assertTrue(torch.isfinite(sampled).all())
                self.assertTrue((sampled >= actor.act_min).all())
                self.assertTrue((sampled <= actor.act_max).all())
                cfg = TotalConfig(
                    _cli_parse_args=False,
                    _env_file=None,
                    env={
                        "action_space_low": [-0.7, -0.3],
                        "action_space_high": [0.2, 0.8],
                    },
                    model={
                        "h_dims": [8],
                        "projection_dim": 8,
                        "aggregator_enc_hdims": [8],
                        "time_dim": 4,
                        "n_timesteps": 3,
                        "beta_schedule": schedule,
                        "random_sample": False,
                    },
                )
                restored = build_evaluation_policy(
                    cfg, {"actor_state_dict": actor.state_dict()}, torch.device("cpu")
                )
                torch.testing.assert_close(restored.sample(obs), sampled)
                with self.assertRaises(ValueError):
                    actor.sample(obs, shape=(2, 2))

    def test_update_shared_encoder_singleton_batch_and_targets(self):
        devices = ["cpu"]
        if torch.cuda.is_available():
            devices.append("cuda")
        if torch.backends.mps.is_available():
            devices.append("mps")
        for device in devices:
            for batch_size in (1, 4):
                with self.subTest(device=device, batch_size=batch_size):
                    actor, critic = networks(device)
                    # Deliberately use a different insertion order from the runners.
                    buffer = ReplayBuffer(storage=LazyTensorStorage(20))
                    buffer.extend(
                        TensorDict(
                            {
                                "done": torch.tensor(
                                    [[True], [False], [True], [False]]
                                ),
                                "action": torch.randn(4, 2),
                                "reward": torch.randn(4, 1),
                                "humans_obs": torch.randn(4, 3, 4),
                                "next_humans_obs": torch.randn(4, 3, 4),
                                "robot_obs": torch.randn(4, 5),
                                "next_robot_obs": torch.randn(4, 5),
                            },
                            batch_size=[4],
                        )
                    )
                    trainer = SocialQSMTrainer(
                        actor,
                        critic,
                        buffer,
                        torch.optim.Adam(actor.parameters()),
                        torch.optim.Adam(critic.parameters()),
                        batch_size,
                        polyak=0.6,
                        device=device,
                    )
                    actor_before = copy.deepcopy(actor.state_dict())
                    critic_before = copy.deepcopy(critic.state_dict())
                    target_before = [
                        p.clone() for p in trainer.target_actor.parameters()
                    ]
                    losses = trainer.update()
                    self.assertTrue(all(torch.isfinite(loss) for loss in losses))
                    self.assertTrue(
                        any(
                            not torch.equal(p, actor_before[n])
                            for n, p in actor.named_parameters()
                        )
                    )
                    self.assertTrue(
                        any(
                            not torch.equal(p, critic_before[n])
                            for n, p in critic.named_parameters()
                        )
                    )
                    self.assertTrue(all(p.grad is None for p in critic.parameters()))
                    trainer.update_target()
                    for old, current, target in zip(
                        target_before,
                        actor.parameters(),
                        trainer.target_actor.parameters(),
                    ):
                        torch.testing.assert_close(target, 0.6 * old + 0.4 * current)
                        self.assertFalse(target.requires_grad)

    def test_legacy_gradient_sign_scale_and_terminal_mask(self):
        actor, _ = networks(random_sample=False)

        class LinearCritic(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(1.0))

            def forward(self, obs, act):
                q = act.sum(-1, keepdim=True) * self.weight
                return q, 3 * q

        critic = LinearCritic()
        actions = torch.tensor([[0.1, 0.2], [-0.2, 0.1]])
        obs = (torch.randn(2, 5), torch.randn(2, 3, 4))
        sample = TensorDict(
            {
                "robot_obs": obs[0],
                "humans_obs": obs[1],
                "next_robot_obs": obs[0],
                "next_humans_obs": obs[1],
                "action": actions,
                "reward": torch.tensor([[2.0], [1.0]]),
                "done": torch.tensor([[True], [False]]),
            },
            batch_size=[2],
        )
        trainer = SocialQSMTrainer(
            actor,
            critic,
            Mock(sample=Mock(return_value=sample)),
            torch.optim.SGD(actor.parameters(), lr=0),
            torch.optim.SGD(critic.parameters(), lr=0),
            2,
            M=5,
            gamma=0.5,
            use_eta=False,
        )
        trainer.target_actor.sample = Mock(
            return_value=torch.tensor([[0.4, 0.6], [0.4, 0.6]])
        )
        torch.testing.assert_close(
            trainer.q_actions_grad(obs, actions), torch.full_like(actions, 2)
        )
        torch.manual_seed(42)
        t = torch.randint(3, (2,))
        noisy = actor.q_sample(actions, t)
        predicted = actor.model(noisy, t, obs)
        expected_actor_loss = (predicted + 10).square().mean()
        torch.manual_seed(42)
        critic_loss, actor_loss = trainer.update()
        target = torch.tensor([[2.0], [1.5]])
        q = actions.sum(-1, keepdim=True)
        torch.testing.assert_close(
            critic_loss, (q - target).square().mean() + (3 * q - target).square().mean()
        )
        torch.testing.assert_close(actor_loss, expected_actor_loss)

    def test_cli_training_save_and_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "run_qsm_socialnav.py"),
                    "--train.preliminary-exp-n",
                    "1",
                    "--train.total-it",
                    "1",
                    "--train.batch-size",
                    "1",
                    "--train.buffer-capacity",
                    "32",
                    "--model.n-timesteps",
                    "2",
                    "--model.h-dims",
                    "[8]",
                    "--model.projection-dim",
                    "8",
                    "--model.aggregator-enc-hdims",
                    "[8]",
                    "--model.time-dim",
                    "4",
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
            self.assertEqual(saved["algorithm"], "QSM")
            self.assertEqual(saved["step"], 1)
            output = Path(tmp) / "evaluation"
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "run_qsm_socialnav.py"),
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
            self.assertEqual(report["algorithm"], "QSM")
            self.assertEqual(len(report["metrics"]), 7)
            self.assertNotIn("QSM Training", result.stderr)


if __name__ == "__main__":
    unittest.main()
