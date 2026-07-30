import csv
import datetime

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
from numpy import average
from tqdm import tqdm

from rewacs.envs.utils.info import Collision, Discomfort, ReachGoal, Timeout
from rewacs.envs.utils.state import JointState


def MSE(Y, YH):
    return np.square(Y - YH).mean()


def RMSE(Y, YH):
    return np.sqrt(np.square(Y - YH).mean())


def eval_policy(
    eval_env,
    model,
    transfunc,
    convert_action,
    discount=0.9,
    render=False,
    render_type="",
    path=None,
    eval_episodes=10,
    scenario="test",
    random_p_num=False,
    p_range=(1, 11),
    ax=None,
    output_name=None,
    print_results=False,
):
    rewards = []
    sum_rewards = 0.0
    sum_cdrs = 0.0
    sum_returns = 0.0

    eval_env.robot.print_info()

    try:
        parameter = next(model.parameters())
    except (AttributeError, StopIteration):
        parameter = None

    device = parameter.device if parameter is not None else torch.device("cpu")

    ###############################
    success_times = []
    collision_times = []
    timeout_times = []
    success = 0
    collision = 0
    timeout = 0
    too_close = 0
    min_dist = []
    collision_cases = []
    timeout_cases = []
    ###############################

    for i in tqdm(range(eval_episodes), dynamic_ncols=True):
        if random_p_num:
            p_num = np.random.randint(*p_range)
            eval_env.set_human_num(p_num)
        robot_state, human_state = eval_env.reset(scenario)
        done = False
        robot_obs, humans_obs = transfunc(
            robot_state,
            human_state,
        )

        # robot_obs, human_obs = robot_obs/4., human_obs/4.

        while not done:
            action = model.sample(
                (
                    robot_obs.unsqueeze(0).to(device),
                    humans_obs.unsqueeze(0).to(device),
                ),
                shape=(1, model.act_dim),
            )
            action = action.cpu().data.numpy()[0]
            action = convert_action(action)
            # action = eval_env.robot.act(obs_r)
            # obs_pred = model.predict_obs(
            #     human_obs.reshape(1, -1).to(model.device), robot_obs.to(model.device)
            # )

            robot_state, human_state, reward, done, info = eval_env.step(action)
            robot_obs, humans_obs = transfunc(
                robot_state,
                human_state,
            )

            # robot_obs, human_obs = robot_obs/4., human_obs/4.
            # if ax != None:
            #     pos = obs_pred[0, :, :2]
            #     ax.clear()
            #     for human in pos:
            #         ax.set_xlim(-4, 4)
            #         ax.set_ylim(-4, 4)
            #         human_circle = plt.Circle(human, 0.3, fill=False, color='b')
            #         ax.add_artist(human_circle)
            #     plt.pause(0.5)

            sum_rewards += reward
            rewards.append(reward)
            if render and done:
                if render_type == "video":
                    eval_env.render("video", path + "/" + str(i) + ".mp4")
                elif render_type == "traj":
                    if path:
                        eval_env.render("traj", path + "/" + str(i) + ".png")
                    else:
                        eval_env.render("traj")
                else:
                    eval_env.render()
            # if isinstance(info, Danger):
            if isinstance(info, Discomfort):
                too_close += 1
                min_dist.append(info.min_dist)
        if isinstance(info, ReachGoal):
            success += 1
            success_times.append(eval_env.global_time)
        elif isinstance(info, Collision):
            collision += 1
            collision_cases.append(i)
            collision_times.append(eval_env.global_time)
        elif isinstance(info, Timeout):
            timeout += 1
            timeout_cases.append(i)
            timeout_times.append(eval_env.time_limit)
        cdr = sum(
            [
                pow(discount, t * eval_env.robot.time_step * eval_env.robot.v_pref) * r
                for t, r in enumerate(rewards)
            ]
        )
        step_returns = []
        for step in range(len(rewards)):
            step_return = sum(
                [
                    pow(discount, t * eval_env.robot.time_step * eval_env.robot.v_pref)
                    * reward
                    for t, reward in enumerate(rewards[step:])
                ]
            )
            step_returns.append(step_return)
        sum_returns += average(step_returns)
        del rewards[:]
        sum_cdrs += cdr

    avg_reward = sum_rewards / eval_episodes
    avg_cdr = sum_cdrs / eval_episodes
    avg_return = sum_returns / eval_episodes
    success_rate = success / eval_episodes
    collision_rate = collision / eval_episodes
    timeout_rate = timeout / eval_episodes
    assert success + collision + timeout == eval_episodes
    avg_nav_time = (
        sum(success_times) / len(success_times)
        if success_times
        else eval_env.time_limit
    )

    if print_results:
        print("\n")
        print("----------------------------")
        print("Scenario : " + str(eval_env.test_scenario) + "-" + str(scenario))
        print("----------------------------")
        print(
            f"Evaluation over {eval_episodes} Average Reward: {avg_reward:.3f} Average Cumulative Discounted Reward: {avg_cdr:.3f}, Average Return: {avg_return:.3f}"
        )
        print(
            f"Success Rate {success_rate} Collision Rate: {collision_rate:.3f} Timeout Rate: {timeout_rate:.3f} Success Time: {avg_nav_time:.3f}"
        )
        print("----------------------------")

        if output_name is not None:
            with open(output_name + ".csv", mode="w") as f:
                writer = csv.writer(f)
                header = [
                    "Average Reward",
                    "Average Cumulative Discounted Reward",
                    "Average Return",
                    "Success Rate",
                    "Collision Rate",
                    "Timeout Rate",
                    "Success Time",
                ]
                results = [
                    avg_reward,
                    avg_cdr,
                    avg_return,
                    success_rate,
                    collision_rate,
                    timeout_rate,
                    avg_nav_time,
                ]
                writer.writerow(header)
                writer.writerow(results)

            with open(output_name + ".txt", mode="w") as f:
                f.write("----------------------------" + "\n")
                f.write(
                    "Scenario : "
                    + str(eval_env.test_scenario)
                    + "-"
                    + str(scenario)
                    + "\n"
                )
                f.write("----------------------------" + "\n")
                f.write(
                    f"Evaluation over {eval_episodes} Average Reward: {avg_reward:.3f} Average Cumulative Discounted Reward: {avg_cdr:.3f}, Average Return: {avg_return:.3f}\n"
                )
                f.write(
                    f"Success Rate {success_rate} Collision Rate: {collision_rate:.3f} Timeout Rate: {timeout_rate:.3f} Success Time: {avg_nav_time:.3f}\n"
                )
                f.write("----------------------------\n")
    return (
        avg_reward,
        avg_cdr,
        avg_return,
        success_rate,
        collision_rate,
        timeout_rate,
        avg_nav_time,
    )


def eval_policy_ma(
    eval_env,
    model,
    transfunc,
    convert_action,
    discount=0.9,
    render=False,
    render_type="",
    path=None,
    eval_episodes=10,
    scenario="test",
    random_p_num=False,
    p_range=(1, 11),
    ax=None,
    output_name=None,
    print_results=False,
):
    rewards = []
    sum_rewards = 0.0
    sum_cdrs = 0.0
    sum_returns = 0.0

    for robot in eval_env.robots:
        robot.print_info()

    ###############################
    success_times = []
    collision_times = []
    timeout_times = []
    success = 0
    collision = 0
    timeout = 0
    too_close = 0
    min_dist = []
    collision_cases = []
    timeout_cases = []
    ###############################

    for i in tqdm(range(eval_episodes), dynamic_ncols=True):
        if random_p_num:
            p_num = np.random.randint(*p_range)
            eval_env.set_human_num(p_num)
        ob_r, ob_h = eval_env.reset(scenario)
        done = False
        robots_obs, other_robots_obs, humans_obs = transfunc(ob_r, ob_h)

        # robot_obs, human_obs = robot_obs/4., human_obs/4.
        # state = psr.state0.expand(5, 20).to(psr.device)
        while not done:
            _, _, actions = model.generate_action(
                (
                    robots_obs.unsqueeze(0).to(model.device),
                    other_robots_obs.unsqueeze(0).to(model.device),
                    humans_obs.unsqueeze(0).to(model.device),
                )
            )
            actions = (
                actions.clamp(model.actor.act_min, model.actor.act_max)
                .cpu()
                .data.numpy()
                .squeeze()
            )
            actions = convert_action(actions)
            # action = eval_env.robot.act(obs_r)
            # obs_pred = model.predict_obs(
            #     human_obs.reshape(1, -1).to(model.device), robot_obs.to(model.device)
            # )

            ob_r, ob_h, reward, done, info = eval_env.step(actions)

            robots_obs, other_robots_obs, humans_obs = transfunc(ob_r, ob_h)

            # robot_obs, human_obs = robot_obs/4., human_obs/4.
            # if ax != None:
            #     pos = obs_pred[0, :, :2]
            #     ax.clear()
            #     for human in pos:
            #         ax.set_xlim(-4, 4)
            #         ax.set_ylim(-4, 4)
            #         human_circle = plt.Circle(human, 0.3, fill=False, color='b')
            #         ax.add_artist(human_circle)
            #     plt.pause(0.5)

            sum_rewards += reward
            rewards.append(reward)
            if render and done:
                if render_type == "video":
                    eval_env.render("video", path + "/" + str(i) + ".mp4")
                elif render_type == "traj":
                    if path:
                        eval_env.render("traj", path + "/" + str(i) + ".png")
                    else:
                        eval_env.render("traj")
                else:
                    eval_env.render()
            # if isinstance(info, Danger):
            if isinstance(info, Discomfort):
                too_close += 1
                min_dist.append(info.min_dist)
        if isinstance(info, ReachGoal):
            success += 1
            success_times.append(eval_env.global_time)
        elif isinstance(info, Collision):
            collision += 1
            collision_cases.append(i)
            collision_times.append(eval_env.global_time)
        elif isinstance(info, Timeout):
            timeout += 1
            timeout_cases.append(i)
            timeout_times.append(eval_env.time_limit)
        cdr = sum(
            [
                pow(discount, t * eval_env.time_step * eval_env.config.robot.v_pref) * r
                for t, r in enumerate(rewards)
            ]
        )
        step_returns = []
        for step in range(len(rewards)):
            step_return = sum(
                [
                    pow(
                        discount,
                        t * eval_env.time_step * eval_env.config.robot.v_pref,
                    )
                    * reward
                    for t, reward in enumerate(rewards[step:])
                ]
            )
            step_returns.append(step_return)
        sum_returns += average(step_returns)
        del rewards[:]
        sum_cdrs += cdr

    avg_reward = sum_rewards / eval_episodes
    avg_cdr = sum_cdrs / eval_episodes
    avg_return = sum_returns / eval_episodes
    success_rate = success / eval_episodes
    collision_rate = collision / eval_episodes
    timeout_rate = timeout / eval_episodes
    assert success + collision + timeout == eval_episodes
    avg_nav_time = (
        sum(success_times) / len(success_times)
        if success_times
        else eval_env.time_limit
    )

    if print_results:
        print("\n")
        print("----------------------------")
        print("Scenario : " + str(eval_env.test_scenario) + "-" + str(scenario))
        print("----------------------------")
        print(
            f"Evaluation over {eval_episodes} Average Reward: {avg_reward:.3f} Average Cumulative Discounted Reward: {avg_cdr:.3f}, Average Return: {avg_return:.3f}"
        )
        print(
            f"Success Rate {success_rate} Collision Rate: {collision_rate:.3f} Timeout Rate: {timeout_rate:.3f} Success Time: {avg_nav_time:.3f}"
        )
        print("----------------------------")

        if output_name is not None:
            with open(output_name + ".csv", mode="w") as f:
                writer = csv.writer(f)
                header = [
                    "Average Reward",
                    "Average Cumulative Discounted Reward",
                    "Average Return",
                    "Success Rate",
                    "Collision Rate",
                    "Timeout Rate",
                    "Success Time",
                ]
                results = [
                    avg_reward,
                    avg_cdr,
                    avg_return,
                    success_rate,
                    collision_rate,
                    timeout_rate,
                    avg_nav_time,
                ]
                writer.writerow(header)
                writer.writerow(results)

            with open(output_name + ".txt", mode="w") as f:
                f.write("----------------------------" + "\n")
                f.write(
                    "Scenario : "
                    + str(eval_env.test_scenario)
                    + "-"
                    + str(scenario)
                    + "\n"
                )
                f.write("----------------------------" + "\n")
                f.write(
                    f"Evaluation over {eval_episodes} Average Reward: {avg_reward:.3f} Average Cumulative Discounted Reward: {avg_cdr:.3f}, Average Return: {avg_return:.3f}\n"
                )
                f.write(
                    f"Success Rate {success_rate} Collision Rate: {collision_rate:.3f} Timeout Rate: {timeout_rate:.3f} Success Time: {avg_nav_time:.3f}\n"
                )
                f.write("----------------------------\n")
    return (
        avg_reward,
        avg_cdr,
        avg_return,
        success_rate,
        collision_rate,
        timeout_rate,
        avg_nav_time,
    )
