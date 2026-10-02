import copy
import datetime
import importlib
import os
import random
import shutil
from pathlib import Path

import numpy as np
import torch
from tensordict import TensorDict
from torchrl.data import LazyTensorStorage, ReplayBuffer
from tqdm import tqdm

from rewacs.envs import CrowdSim
from rewacs.envs.policy.policy_factory import policy_factory
from rewacs.envs.utils.action import ActionXY
from rewacs.envs.utils.robot import Robot
from rewacs.envs.utils.transformations import GetRobotFrameObs
from sac.models import SocialGaussianPolicy, SocialSACCritic
from socialnav.aggregators import GATAggregator
from socialnav.checkpoints import evaluate_saved_run, save_best_to_wandb
from socialnav.evaluation import eval_policy
from socialnav.trainer import SocialSACTrainer

try:
    import wandb
except ModuleNotFoundError:
    pass


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def define_env(
    config,
    debug=False,
):
    cfg = config
    env = CrowdSim()
    env.configure(cfg)
    robot = Robot(cfg, "robot")
    robot.time_step = env.time_step
    env.set_robot(robot)

    if robot.visible:
        safety_space = 0
    else:
        safety_space = 0.15

    policy = policy_factory[cfg.robot.policy]()
    policy.safety_space = safety_space

    robot.set_policy(policy)

    if debug:
        print(cfg)

    return env, robot


def main():
    start_time_log = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    config_path = str(
        Path(__file__).resolve().parent / "configs/sac_socialnav_config.py"
    )
    spec = importlib.util.spec_from_file_location("config", config_path)

    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)

    cfg = config.TotalConfig()

    if cfg.eval.run_path is not None:
        evaluate_saved_run(cfg, config.TotalConfig, define_env, seed_all, device)
        return

    if cfg.log.wandb:
        run = wandb.init(
            project=cfg.log.wandb_project, save_code=True, mode=cfg.log.wandb_mode
        )
        run.config.update(cfg.model_dump())

        results_log_columns = [
            "reward",
            "cdr",
            "return",
            "success_rate",
            "collision_rate",
            "timeout_rate",
            "avg_nav_time",
        ]
        val_log_columns = ["step_num"] + results_log_columns
        val_table = wandb.Table(columns=val_log_columns)

        shutil.copy(config_path, os.path.join(run.dir, "config.py"))

        code_artifact = wandb.Artifact(name="config_code_artifact", type="code")
        code_artifact.add_file(os.path.join(run.dir, "config.py"))
        wandb.log_artifact(code_artifact)

    if cfg.log.save_model:
        output_dir = (
            Path(run.dir).parent / "local_checkpoints"
            if cfg.log.wandb
            else Path("models") / (start_time_log + "_" + cfg.train.training_alg)
        )
        trained_models_dir = str(output_dir / "trained_models")
        os.makedirs(trained_models_dir, exist_ok=True)
        (output_dir / "config.json").write_text(cfg.model_dump_json(indent=2))

    seed_all(cfg.train.random_seed)

    env, robot = define_env(debug=True, config=cfg)

    transfunc = GetRobotFrameObs(
        with_peds_vel=cfg.transfunc.with_peds_vel,
        peds_vel_as_relative=cfg.transfunc.peds_vel_as_relative,
        use_omega=cfg.transfunc.use_omega,
    )

    def convert_action(action):
        action = ActionXY(action[0], action[1])

        return action

    buffer = ReplayBuffer(storage=LazyTensorStorage(cfg.train.buffer_capacity))

    critic_aggregator = GATAggregator(
        cfg.env.obs_dim,
        cfg.env.r_obs_dim,
        projection_dim=cfg.model.projection_dim,
        enc_hdims=cfg.model.aggregator_enc_hdims,
    )

    critic = SocialSACCritic(
        cfg.model.projection_dim,
        cfg.env.act_dim,
        h_dims=cfg.model.h_dims,
        aggregator=critic_aggregator,
    ).to(device)

    actor_aggregator = GATAggregator(
        cfg.env.obs_dim,
        cfg.env.r_obs_dim,
        projection_dim=cfg.model.projection_dim,
        enc_hdims=cfg.model.aggregator_enc_hdims,
    )

    actor = SocialGaussianPolicy(
        obs_dim=cfg.model.projection_dim,
        act_dim=cfg.env.act_dim,
        aggregator=actor_aggregator,
        h_dims=cfg.model.h_dims,
        act_min=cfg.env.action_space_low,
        act_max=cfg.env.action_space_high,
        log_std_min=cfg.model.log_std_min,
        log_std_max=cfg.model.log_std_max,
        deterministic_eval=cfg.eval.deterministic,
    ).to(device)

    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=cfg.train.lr)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=cfg.train.lr)

    trainer = SocialSACTrainer(
        actor=actor,
        critic=critic,
        replay_buffer=buffer,
        actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer,
        batch_size=cfg.train.batch_size,
        polyak=cfg.train.polyak,
        gamma=cfg.train.gamma,
        init_alpha=cfg.train.init_alpha,
        alpha_lr=cfg.train.alpha_lr,
        target_entropy=cfg.train.target_entropy,
        device=device,
    )

    for i in tqdm(range(cfg.train.preliminary_exp_n)):
        robot_state, human_state = env.reset("train")
        done = False
        robot_obs, humans_obs = transfunc(robot_state, human_state)
        while not done:
            action = env.robot.act(human_state)
            action = convert_action(action)

            robot_state, human_state, reward, done, info = env.step(action)
            next_robot_obs, next_humans_obs = transfunc(robot_state, human_state)

            sample = TensorDict(
                {
                    "robot_obs": robot_obs,
                    "next_robot_obs": next_robot_obs,
                    "humans_obs": humans_obs,
                    "next_humans_obs": next_humans_obs,
                    "action": torch.as_tensor(action, dtype=torch.float32),
                    "reward": [reward],
                    "done": [int(done)],
                }
            )

            buffer.add(sample)

            robot_obs = next_robot_obs
            humans_obs = next_humans_obs

    max_cdr = -np.inf
    with tqdm(
        range(cfg.train.total_it),
        desc=cfg.train.training_alg + " Training",
        dynamic_ncols=True,
    ) as pbar:
        for i in pbar:
            robot_state, human_state = env.reset("train")
            done = False
            robot_obs, humans_obs = transfunc(robot_state, human_state)

            while not done:
                action = actor.sample(
                    (
                        robot_obs.unsqueeze(0).to(device),
                        humans_obs.unsqueeze(0).to(device),
                    ),
                    shape=(1, cfg.env.act_dim),
                )
                action = action.cpu().numpy()[0]
                action = convert_action(action)

                robot_state, human_state, reward, done, info = env.step(action)
                next_robot_obs, next_humans_obs = transfunc(robot_state, human_state)

                sample = TensorDict(
                    {
                        "robot_obs": robot_obs,
                        "next_robot_obs": next_robot_obs,
                        "humans_obs": humans_obs,
                        "next_humans_obs": next_humans_obs,
                        "action": torch.as_tensor(action, dtype=torch.float32),
                        "reward": [reward],
                        "done": [int(done)],
                    }
                )

                buffer.add(sample)

                robot_obs = next_robot_obs
                humans_obs = next_humans_obs
                if len(buffer) > cfg.train.batch_size:
                    lc, la, l_alpha = trainer.update()
                    trainer.update_target()

            if done:
                # for _ in range(cfg.train.updates_per_episode):
                #     lc, la, l_alpha = trainer.update()
                #     trainer.update_target()

                if cfg.log.wandb:
                    wandb.log(
                        {
                            "loss/critic": lc.item(),
                            "loss/actor": la.item(),
                            "loss/alpha": l_alpha.item(),
                            "train/alpha": trainer.alpha.item(),
                            "train/entropy": trainer.last_entropy.item(),
                        },
                        step=i + 1,
                    )
                pbar.set_postfix(Reward=f"{reward:.2f}")
                if (i + 1) % cfg.eval.eval_interval == 0 or i + 1 == cfg.train.total_it:
                    actor.eval()
                    val_logs = eval_policy(
                        eval_env=env,
                        model=actor,
                        transfunc=transfunc,
                        convert_action=convert_action,
                        eval_episodes=env.case_size["val"],
                        scenario="val",
                        render=cfg.eval.val_render,
                        print_results=True,
                    )
                    actor.train()
                    if cfg.log.wandb:
                        wandb.log(
                            {
                                "val/reward": val_logs[0],
                                "val/cdr": val_logs[1],
                                "val/mean_step_return": val_logs[2],
                                "val/success_rate": val_logs[3],
                                "val/collision_rate": val_logs[4],
                                "val/timeout_rate": val_logs[5],
                                "val/avg_nav_time": val_logs[6],
                            },
                            step=i + 1,
                        )

                        val_log_data = [i + 1] + list(val_logs)
                        val_table.add_data(*val_log_data)
                        if i + 1 == cfg.train.total_it:
                            run.log({"Validation Table": val_table})

                    update_best = val_logs[1] > max_cdr
                    if update_best:
                        best_actor_model = copy.deepcopy(actor.state_dict())
                        best_critic_model = copy.deepcopy(critic.state_dict())
                        best_trainer_state = copy.deepcopy(trainer.state_dict())
                        best_step_num = i + 1
                        max_cdr = val_logs[1]

                    if cfg.log.save_model:
                        torch.save(
                            {
                                "actor_state_dict": actor.state_dict(),
                                "critic_state_dict": critic.state_dict(),
                                "trainer_state_dict": trainer.state_dict(),
                                "config": cfg.model_dump(),
                                "algorithm": cfg.train.training_alg,
                                "step": i + 1,
                            },
                            trained_models_dir + "/model_{}.pth".format(i + 1),
                        )
                        if update_best:
                            torch.save(
                                {
                                    "actor_state_dict": actor.state_dict(),
                                    "critic_state_dict": critic.state_dict(),
                                    "trainer_state_dict": trainer.state_dict(),
                                    "config": cfg.model_dump(),
                                    "algorithm": cfg.train.training_alg,
                                    "step": i + 1,
                                },
                                trained_models_dir + "/model_best.pth",
                            )

    if cfg.log.wandb:
        save_best_to_wandb(
            run,
            {
                "actor_state_dict": best_actor_model,
                "critic_state_dict": best_critic_model,
                "trainer_state_dict": best_trainer_state,
                "config": cfg.model_dump(),
                "algorithm": cfg.train.training_alg,
                "step": best_step_num,
                "best_cdr": float(max_cdr),
            },
        )

    render = cfg.eval.render
    render_type = cfg.eval.render_type
    if render:
        if cfg.log.wandb:
            path_v = os.path.join(run.dir, "videos/training_results")
            os.makedirs(path_v, exist_ok=True)
        else:
            path_v = "videos/{}_{}_{}".format(
                start_time_log, trainer.alg_name, "CrowdSim"
            )
            os.makedirs(path_v, exist_ok=True)
    else:
        path_v = None

    actor.load_state_dict(best_actor_model)
    actor.eval()
    critic.load_state_dict(best_critic_model)
    print(f"The best model number is {best_step_num}")

    test_logs = eval_policy(
        eval_env=env,
        model=actor,
        transfunc=transfunc,
        convert_action=convert_action,
        eval_episodes=env.case_size["test"],
        scenario="test",
        render=render,
        render_type=render_type,
        path=path_v,
        print_results=True,
    )

    if cfg.log.wandb:
        test_log_columns = ["best_step_num"] + results_log_columns
        test_log_data = [best_step_num] + list(test_logs)
        test_table = wandb.Table(columns=test_log_columns)
        test_table.add_data(*test_log_data)

        run.log({"Test Table": test_table})
        wandb.finish()


if __name__ == "__main__":
    main()
