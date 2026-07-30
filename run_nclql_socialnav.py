import copy
import datetime
import importlib
import os
import random
import shutil

import gymnasium as gym
import numpy as np
import torch
from gymnasium.wrappers import RecordEpisodeStatistics
from tensordict import TensorDict
from torchrl.data import LazyTensorStorage, ListStorage, ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from tqdm import tqdm, trange

from flow.models import MeanFlowPolicy
from nclql.models import AnnealedLangevinDynamics
from rewacs.envs import CrowdSim
from rewacs.envs.policy.policy_factory import policy_factory
from rewacs.envs.utils.action import ActionRot, ActionXY, ActionXYW
from rewacs.envs.utils.robot import Robot
from rewacs.envs.utils.transformations import GetRobotFrameObs
from socialnav.aggregators import GATAggregator
from socialnav.evaluation import eval_policy
from socialnav.explorer import ExploerCrowdSim
from socialnav.models import (
    SocialConditionalMeanVelocityNet,
    SocialNoiseConditionedCritic,
)
from socialnav.trainer import SocialNCLQLTrainer

try:
    import wandb
except ModuleNotFoundError:
    pass


def seed_all(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)


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

config_path = "./configs/nclql_socialnav_config.py"
spec = importlib.util.spec_from_file_location("config", config_path)

config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)

cfg = config.TotalConfig()

if cfg.log.wandb:
    # wandb.tensorboard.patch(root_logdir=f"logs/{start_time_log}")

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
        trained_models_dir = os.path.join(run.dir, "trained_models")
        os.makedirs(trained_models_dir, exist_ok=True)


seed_all(cfg.train.random_seed)


##################################################################################
# load env
env, robot = define_env(debug=True, config=cfg)
##################################################################################

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

critic = SocialNoiseConditionedCritic(
    cfg.model.projection_dim + cfg.env.act_dim + cfg.model.time_dim,
    1,
    time_dim=cfg.model.time_dim,
    h_dims=cfg.model.h_dims,
    aggregator=critic_aggregator,
)

ald = AnnealedLangevinDynamics(
    model=critic,
    L=cfg.model.L,
    T=cfg.model.T,
    w=cfg.model.w,
    act_dim=cfg.env.act_dim,
    act_max=cfg.env.action_space_high,
    act_min=cfg.env.action_space_low,
    q_grad_norm=cfg.model.q_grad_norm,
)

ald = torch.compile(ald)

critic_optimizer = torch.optim.Adam(critic.parameters(), lr=cfg.train.lr)

critic.to(device)

expl = ExploerCrowdSim(
    env=env,
    # num_samples=5000,
    obs_dim=cfg.env.obs_dim,
    act_dim=cfg.env.act_dim,
    r_obs_dim=cfg.env.r_obs_dim,
    transfunc=transfunc,
    convert_action=convert_action,
    render=False,
)

trainer = SocialNCLQLTrainer(
    ald=ald,
    critic=critic,
    replay_buffer=buffer,
    critic_optimizer=critic_optimizer,
    batch_size=cfg.train.batch_size,
    device=device,
)


for i in tqdm(range(cfg.train.preliminary_exp_n)):
    # action = env.action_space.sample()
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
        # ep_buffer.add(sample)

        robot_obs = next_robot_obs
        humans_obs = next_humans_obs


# val_logs = eval_policy(
#     eval_env=env,
#     model=actor,
#     transfunc=transfunc,
#     convert_action=convert_action,
#     eval_episodes=env.case_size["test"],
#     scenario="test",
#     render=cfg.eval.val_render,
#     print_results=True,
# )

max_return = -np.inf
with tqdm(
    range(cfg.train.total_it),
    desc=cfg.train.training_alg + " Training",
    dynamic_ncols=True,
) as pbar:
    for i, ch in enumerate(pbar):
        # action = env.action_space.sample()
        robot_state, human_state = env.reset("train")
        done = False
        robot_obs, humans_obs = transfunc(robot_state, human_state)

        while not done:
            # action = actor.sample(
            #     (robot_obs.unsqueeze(0), humans_obs.unsqueeze(0)),
            #     shape=(1, cfg.env.act_dim),
            # )
            action = ald.sample(
                (robot_obs.unsqueeze(0).to(device), humans_obs.unsqueeze(0).to(device)),
                shape=(1, cfg.env.act_dim),
            )
            # action = ald.sample(
            #     torch.tensor(observation, dtype=torch.float32).reshape(1, -1),
            #     shape=(1, act_dim),
            # )
            action = action.cpu().data.numpy()[0]
            action = convert_action(action)

            robot_state, human_state, reward, done, info = env.step(action)
            next_robot_obs, next_humans_obs = transfunc(robot_state, human_state)

            sample = TensorDict(
                {
                    "robot_obs": robot_obs,
                    "next_robot_obs": next_robot_obs,
                    "humans_obs": humans_obs,
                    "next_humans_obs": next_humans_obs,
                    "action": action,
                    "reward": [reward],
                    "done": [int(done)],
                }
            )

            buffer.add(sample)
            # ep_buffer.add(sample)

            robot_obs = next_robot_obs
            humans_obs = next_humans_obs

            # if len(buffer) > cfg.train.batch_size:
            #     # if len(buffer) > start_steps:
            #     trainer.update()
            #     trainer.update_target()

        if done:
            lc_td, lc_t = trainer.update()
            trainer.update_target()

            if cfg.log.wandb:
                wandb.log(
                    {
                        "loss/critic_td": lc_td,
                        "loss/critic_t": lc_t,
                    },
                    step=i + 1,
                )
            # trainer.update_imitation(epoch_num=1)
            # if info["episode"]["r"] > max_return:
            #     if not (max_return == -np.inf):
            #         trainer.update_imitation(epoch_num=10)
            #     max_return = info["episode"]["r"]
            pbar.set_postfix(Reward=f"{reward:.2f}")
            # print("Episodic Return: {}, Time Step {}".format(info["episode"]["r"], i))
            if (i + 1) % cfg.eval.eval_interval == 0:
                val_logs = eval_policy(
                    eval_env=env,
                    model=ald,
                    transfunc=transfunc,
                    convert_action=convert_action,
                    eval_episodes=env.case_size["val"],
                    scenario="val",
                    render=cfg.eval.val_render,
                    print_results=True,
                )
                if cfg.log.wandb:
                    wandb.log(
                        {
                            "val/reward": val_logs[0],
                            "val/cdr": val_logs[1],
                            "val/return": val_logs[2],
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

                update_best = val_logs[1] > max_return
                if update_best:
                    best_critic_model = copy.deepcopy(critic.state_dict())
                    best_step_num = i + 1
                    max_cdr = val_logs[1]

                if cfg.log.save_model:
                    torch.save(
                        {
                            "critic_state_dict": critic.state_dict(),
                        },
                        trained_models_dir + "/model_{}.pth".format(i + 1),
                    )
                    if update_best:
                        torch.save(
                            {
                                "critic_state_dict": critic.state_dict(),
                            },
                            trained_models_dir + "/model_best.pth",
                        )

                    # model load
                    # checkpoint = torch.load("models.pt", weights_only=True)
                    # critic.load_state_dict(checkpoint["critic_state_dict"])


render = cfg.eval.render
render_type = cfg.eval.render_type
if render and (render_type == "video"):
    if cfg.log.wandb:
        path_v = os.path.join(run.dir, "videos/training_results")
        os.makedirs(path_v, exist_ok=True)
    else:
        path_v = "videos/{}_{}_{}".format(start_time_log, trainer.alg_name, "CrowdSim")
        os.mkdir(path_v)
else:
    path_v = None

critic.load_state_dict(best_critic_model)
print(f"The best model number is {best_step_num}")

test_logs = eval_policy(
    eval_env=env,
    model=ald,
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
    # run.log({"Validation Table": val_table})

    test_log_columns = ["bset_step_num"] + results_log_columns
    test_log_data = [best_step_num] + list(test_logs)
    test_table = wandb.Table(columns=test_log_columns)
    test_table.add_data(*test_log_data)

    run.log({"Test Table": test_table})
    wandb.finish()
