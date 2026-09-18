import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import torch
import yaml

from socialnav.checkpoints import (
    build_evaluation_policy,
    load_evaluation_config,
    load_weights,
    resolve_checkpoint,
    save_best_to_wandb,
)

ROOT = Path(__file__).resolve().parents[1]


class SavedEvaluationTest(unittest.TestCase):
    def config(self, name="mspid", **kwargs):
        cls = importlib.import_module(f"configs.{name}_socialnav_config").TotalConfig
        return cls(_cli_parse_args=False, _env_file=None, **kwargs)

    def test_resolves_run_files_directory_and_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            model_dir = root / "files" / "trained_models"
            model_dir.mkdir(parents=True)
            checkpoint = model_dir / "model_best.pth"
            checkpoint.touch()
            for source in (root, root / "files", model_dir, checkpoint):
                self.assertEqual(resolve_checkpoint(source), checkpoint.resolve())
            with self.assertRaises(FileNotFoundError):
                resolve_checkpoint(root, "model_123.pth")

    def test_only_best_is_registered_and_local_checkpoints_remain_loadable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            run_dir = root / "files"
            run_dir.mkdir()
            local_dir = root / "local_checkpoints" / "trained_models"
            local_dir.mkdir(parents=True)
            intermediate = local_dir / "model_100.pth"
            torch.save({"step": 100}, intermediate)
            run = Mock(dir=str(run_dir))
            checkpoint = {
                "critic_state_dict": {"weight": torch.tensor([1.0])},
                "config": self.config().model_dump(),
                "algorithm": "MSPID",
                "step": 50,
                "best_cdr": 0.75,
            }
            best = save_best_to_wandb(run, checkpoint)
            run.save.assert_called_once_with(
                str(best),
                base_path=str(run_dir),
                policy="end",
                glob=False,
            )
            self.assertEqual(list(run_dir.rglob("*.pth")), [best])
            restored = torch.load(best, weights_only=True)
            self.assertEqual(restored["step"], 50)
            self.assertEqual(restored["best_cdr"], 0.75)
            torch.testing.assert_close(
                restored["critic_state_dict"]["weight"],
                torch.tensor([1.0]),
            )
            for source in (root, run_dir):
                self.assertEqual(resolve_checkpoint(source), best)
                self.assertEqual(
                    resolve_checkpoint(source, "model_100.pth"), intermediate
                )

    def test_restores_wandb_config_without_environment_overrides(self):
        cfg = self.config(model={"projection_dim": 64}, train={"random_seed": 41})
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            (root / "trained_models").mkdir()
            data = {key: {"value": value} for key, value in cfg.model_dump().items()}
            data["wandb_version"] = 1
            data["_wandb"] = {"value": {}}
            (root / "config.yaml").write_text(yaml.safe_dump(data))
            cli = self.config(eval={"episodes": 3})
            with patch.dict(os.environ, {"EXPERIMENT_MODEL__PROJECTION_DIM": "128"}):
                loaded = load_evaluation_config(
                    cli, type(cfg), {}, root / "trained_models/model_best.pth"
                )
            self.assertEqual(loaded.model.projection_dim, 64)
            self.assertEqual(loaded.train.random_seed, 41)
            self.assertEqual(loaded.eval.episodes, 3)
            self.assertFalse(loaded.log.wandb)

    def test_missing_config_and_wrong_algorithm_fail(self):
        cli = self.config()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trained_models/model_best.pth"
            with self.assertRaises(FileNotFoundError):
                load_evaluation_config(cli, type(cli), {}, path)
            with self.assertRaises(ValueError):
                load_evaluation_config(
                    cli,
                    type(cli),
                    {
                        "config": cli.model_dump(),
                        "algorithm": "NC-LQL",
                    },
                    path,
                )

    def test_legacy_compiled_weights_and_strict_shape_check(self):
        model = torch.nn.Linear(3, 2)
        legacy = {
            "_orig_mod." + key: value.clone()
            for key, value in model.state_dict().items()
        }
        restored = torch.nn.Linear(3, 2)
        load_weights(restored, legacy)
        torch.testing.assert_close(restored.weight, model.weight)
        with self.assertRaises(RuntimeError):
            load_weights(torch.nn.Linear(4, 2), legacy)

    def test_all_three_cli_evaluate_without_training(self):
        for name in ("mspid", "mspid_rmflow", "nclql"):
            with self.subTest(algorithm=name), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp).resolve()
                cfg = self.config(
                    name,
                    env={"time_limit": 2, "test_size": 1},
                    sim={"human_num": 1},
                    model={"L": 2, "T": 1},
                    # Training would fail if accidentally entered.
                    train={"preliminary_exp_n": 0, "total_it": 0},
                    log={"wandb": True},
                )
                with patch("socialnav.checkpoints.load_weights"):
                    policy = build_evaluation_policy(
                        cfg,
                        {
                            "actor_state_dict": {},
                            "critic_state_dict": {},
                        },
                        torch.device("cpu"),
                    )
                if name == "nclql":
                    checkpoint = {"critic_state_dict": policy.model.state_dict()}
                else:
                    checkpoint = {
                        "actor_state_dict": {
                            "_orig_mod." + key: value
                            for key, value in policy.state_dict().items()
                        }
                    }
                checkpoint.update(
                    config=cfg.model_dump(), algorithm=cfg.train.training_alg, step=100
                )
                model_dir = root / "files/trained_models"
                model_dir.mkdir(parents=True)
                torch.save(checkpoint, model_dir / "model_best.pth")
                output = root / "results"
                result = subprocess.run(
                    [
                        sys.executable,
                        str(ROOT / f"run_{name}_socialnav.py"),
                        "--eval.run-path",
                        str(root),
                        "--eval.episodes",
                        "1",
                        "--eval.device",
                        "cpu",
                        "--eval.output-dir",
                        str(output),
                    ],
                    cwd=ROOT,
                    capture_output=True,
                    text=True,
                    timeout=90,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                report = json.loads((output / "results.json").read_text())
                self.assertEqual(report["algorithm"], cfg.train.training_alg)
                self.assertEqual(report["episodes"], 1)
                self.assertEqual(report["step"], 100)
                self.assertEqual(len(report["metrics"]), 7)
                self.assertTrue((output / "results.csv").is_file())
                self.assertNotIn("Training:", result.stderr)


if __name__ == "__main__":
    unittest.main()
