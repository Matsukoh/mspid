"""Save and restore SocialNav checkpoints for training and evaluation."""

import datetime
import json
from pathlib import Path

import torch
import yaml

METRICS = (
    "episode_reward",
    "cdr",
    "mean_step_return",
    "success_rate",
    "collision_rate",
    "timeout_rate",
    "avg_nav_time",
)


def resolve_checkpoint(source, filename="model_best.pth"):
    source = Path(source).expanduser().resolve()
    if source.is_file():
        return source
    local_root = source.parent if source.name == "files" else source
    for candidate in (
        source / "files" / "trained_models" / filename,
        source / "trained_models" / filename,
        source / filename,
        local_root / "local_checkpoints" / "trained_models" / filename,
    ):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Checkpoint not found: {source} ({filename})")


def read_config(path):
    path = Path(path).expanduser()
    if path.suffix == ".json":
        data = json.loads(path.read_text())
    elif path.suffix in (".yaml", ".yml"):
        data = yaml.safe_load(path.read_text())
        if not isinstance(data, dict):
            raise ValueError(f"Expected a config mapping in {path}")
        # W&B config.yaml wraps each user config field in {value: ...}.
        data = {
            key: value.get("value", value) if isinstance(value, dict) else value
            for key, value in data.items()
            if key != "wandb_version" and not key.startswith("_")
        }
    else:
        raise ValueError(
            "Config must be JSON or YAML (Python config files are not executed)."
        )
    if not isinstance(data, dict):
        raise ValueError(f"Expected a config mapping in {path}")
    return data


def load_evaluation_config(cli_cfg, config_type, checkpoint, checkpoint_path):
    if cli_cfg.eval.config_path:
        data = read_config(cli_cfg.eval.config_path)
    elif "config" in checkpoint:
        data = checkpoint["config"]
    else:
        data = None
        for directory in (checkpoint_path.parent, checkpoint_path.parent.parent):
            for filename in ("config.json", "config.yaml", "config.yml"):
                path = directory / filename
                if path.is_file():
                    data = read_config(path)
                    break
            if data is not None:
                break
        if data is None:
            raise FileNotFoundError(
                "Training config not found in checkpoint or adjacent config.json/config.yaml. "
                "Pass --eval.config-path with the original training config (JSON/YAML)."
            )
    # Validate fields directly, without rereading CLI, .env, or environment settings.
    # This also prevents current shell settings from altering the saved experiment.
    if not isinstance(data, dict):
        raise ValueError("Saved config must be a mapping")
    unknown = set(data) - set(config_type.model_fields)
    if unknown:
        raise ValueError(f"Unknown saved config fields: {sorted(unknown)}")
    values = {}
    for name, field in config_type.model_fields.items():
        if name not in data:
            raise ValueError(
                f"Saved config is missing {name!r}; supply a complete training config"
            )
        annotation = field.annotation
        if hasattr(annotation, "model_validate"):
            values[name] = annotation.model_validate(data[name])
        else:
            from pydantic import TypeAdapter

            values[name] = TypeAdapter(annotation).validate_python(data[name])
    cfg = config_type.model_construct(**values)
    expected = cli_cfg.train.training_alg
    saved = checkpoint.get("algorithm", cfg.train.training_alg)
    if saved != expected or cfg.train.training_alg != expected:
        raise ValueError(
            f"Algorithm mismatch: expected {expected}, checkpoint/config uses {saved}/{cfg.train.training_alg}"
        )
    # Evaluation controls come from this invocation; model/environment/seed from training.
    cfg.eval = cli_cfg.eval.model_copy(deep=True)
    cfg.log.wandb = False
    cfg.log.save_model = False
    return cfg


def load_weights(model, state_dict):
    # Legacy actor checkpoints were saved after torch.compile().
    state_dict = {
        key.removeprefix("_orig_mod."): value for key, value in state_dict.items()
    }
    model.load_state_dict(state_dict, strict=True)


def build_evaluation_policy(cfg, checkpoint, device):
    from flow.models import MeanFlowPolicy, RMFlowPolicy
    from nclql.models import AnnealedLangevinDynamics
    from socialnav.aggregators import GATAggregator
    from socialnav.models import (
        SocialConditionalMeanVelocityNet,
        SocialNoiseConditionedCritic,
    )

    aggregator = GATAggregator(
        cfg.env.obs_dim,
        cfg.env.r_obs_dim,
        projection_dim=cfg.model.projection_dim,
        enc_hdims=cfg.model.aggregator_enc_hdims,
    )
    if cfg.train.training_alg == "NC-LQL":
        critic = SocialNoiseConditionedCritic(
            cfg.model.projection_dim + cfg.env.act_dim + cfg.model.time_dim,
            1,
            time_dim=cfg.model.time_dim,
            h_dims=cfg.model.h_dims,
            aggregator=aggregator,
        ).to(device)
        load_weights(critic, checkpoint["critic_state_dict"])
        policy = AnnealedLangevinDynamics(
            model=critic,
            L=cfg.model.L,
            T=cfg.model.T,
            w=cfg.model.w,
            act_dim=cfg.env.act_dim,
            act_max=cfg.env.action_space_high,
            act_min=cfg.env.action_space_low,
            q_grad_norm=cfg.model.q_grad_norm,
        ).to(device)
    else:
        policy_class = {"MSPID": MeanFlowPolicy, "MSPID-RMFlow": RMFlowPolicy}[
            cfg.train.training_alg
        ]
        vnet = SocialConditionalMeanVelocityNet(
            obs_dim=cfg.model.projection_dim,
            act_dim=cfg.env.act_dim,
            h_dim=256,
            time_dim=cfg.model.time_dim,
            aggregator=aggregator,
        )
        policy = policy_class(
            vnet=vnet,
            act_dim=cfg.env.act_dim,
            act_max=cfg.env.action_space_high,
            act_min=cfg.env.action_space_low,
        ).to(device)
        load_weights(policy, checkpoint["actor_state_dict"])
    policy.eval()
    return policy


def evaluate_saved_run(cli_cfg, config_type, define_env, seed_all, default_device):
    from rewacs.envs.utils.action import ActionXY
    from rewacs.envs.utils.transformations import GetRobotFrameObs
    from socialnav.evaluation import eval_policy

    path = resolve_checkpoint(cli_cfg.eval.run_path, cli_cfg.eval.checkpoint)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    cfg = load_evaluation_config(cli_cfg, config_type, checkpoint, path)
    device = (
        default_device if cfg.eval.device == "auto" else torch.device(cfg.eval.device)
    )
    seed_all(cfg.train.random_seed)
    policy = build_evaluation_policy(cfg, checkpoint, device)
    env, _ = define_env(config=cfg)
    transfunc = GetRobotFrameObs(
        with_peds_vel=cfg.transfunc.with_peds_vel,
        peds_vel_as_relative=cfg.transfunc.peds_vel_as_relative,
        use_omega=cfg.transfunc.use_omega,
    )
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
    output = Path(
        cfg.eval.output_dir or f"evaluations/{path.stem}_{stamp}"
    ).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    render_path = output / "renders"
    if cfg.eval.render:
        render_path.mkdir(exist_ok=True)
    episodes = cfg.eval.episodes or env.case_size["test"]
    print(f"Evaluating {path} on {device} ({episodes} test episodes)")
    # ALD needs action gradients, so do not wrap this in inference_mode/no_grad.
    logs = eval_policy(
        eval_env=env,
        model=policy,
        transfunc=transfunc,
        convert_action=lambda action: ActionXY(action[0], action[1]),
        eval_episodes=episodes,
        scenario="test",
        render=cfg.eval.render,
        render_type=cfg.eval.render_type,
        path=str(render_path),
        print_results=True,
        output_name=str(output / "results"),
    )
    result = {
        "checkpoint": str(path),
        "algorithm": cfg.train.training_alg,
        "step": checkpoint.get("step"),
        "seed": cfg.train.random_seed,
        "episodes": episodes,
        "discount": 0.99,
        "metrics": {name: float(value) for name, value in zip(METRICS, logs)},
    }
    (output / "results.json").write_text(json.dumps(result, indent=2))
    (output / "config.json").write_text(cfg.model_dump_json(indent=2))
    print(f"Evaluation results: {output.resolve()}")
    return result


def save_best_to_wandb(run, checkpoint):
    """Register only the final best checkpoint for upload when the run finishes."""
    run_dir = Path(run.dir).resolve()
    path = run_dir / "trained_models" / "model_best.pth"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)
    run.save(str(path), base_path=str(run_dir), policy="end", glob=False)
    return path
