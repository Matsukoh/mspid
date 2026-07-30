# from utils.buffers import OffPolicyTuple
from time import time

import numpy as np
import torch
from numpy import average
from tensordict import TensorDict
from tqdm import tqdm

from rewacs.envs.utils.info import Collision, Discomfort, ReachGoal, Timeout


class ExploerCrowdSim:
    def __init__(
        self,
        env,
        obs_dim=0,
        act_dim=0,
        state_dim=0,
        r_obs_dim=0,
        discount=0.9,
        policy=None,
        render=False,
        transfunc=None,
        convert_action=None,
    ):
        super().__init__()

        self.env = env
        self.policy = policy
        # self.num_trajs = num_trajs
        # self.num_samples = num_samples
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.state_dim = state_dim
        self.r_obs_dim = r_obs_dim
        self.discount = discount
        self.render = render
        self.transfunc = transfunc
        self.convert_action = convert_action

    def get_r_state(self):
        r_state = self.env.robot.get_full_state()
        r_pos = np.array(r_state.position)
        r_theta = np.array(r_state.theta)
        goal = np.array(r_state.goal_position)
        diff = goal - r_pos
        dist = np.linalg.norm(diff)
        rot = np.arctan2(diff[1], diff[0])
        if self.env.robot.kinematics == "unicycle":
            theta = r_theta - rot
        else:
            # set theta to be zero since it's not used
            theta = 0

        return np.array([dist, theta]), np.array([r_pos[0], r_pos[1], r_theta])

    def calc_cdr(self, rewards):
        sum_cdrs = sum(
            [
                pow(
                    self.discount,
                    t * self.env.robot.time_step * self.env.robot.v_pref,
                )
                * reward
                for t, reward in enumerate(rewards)
            ]
        )

        return sum_cdrs

    def calc_returns(self, rewards):
        step_returns = []
        for step in range(len(rewards)):
            step_return = sum(
                [
                    pow(
                        self.discount,
                        t * self.env.robot.time_step * self.env.robot.v_pref,
                    )
                    * reward
                    for t, reward in enumerate(rewards[step:])
                ]
            )
            step_returns.append(step_return.cpu().data.numpy())
        sum_return = average(step_returns)

        return sum_return

    def exploration_k_ep_orca(
        self, buffer, k=1, render=False, random_p_num=False, p_range=(1, 5)
    ):
        # d_a = self.act_dim
        sum_rewards = 0
        sum_cdrs = 0
        sum_returns = 0
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
        for _ in tqdm(range(k), desc="Preliminary Exploration", dynamic_ncols=True):
            if random_p_num:
                p_num = np.random.randint(*p_range)
                self.env.set_human_num(p_num)
                # d_o = self.obs_dim * p_num
            else:
                d_o = self.obs_dim * self.env.human_num
            d_ro = self.r_obs_dim
            d_s = self.state_dim * self.env.human_num

            # Temporary buffer list
            next_obs = []
            obs = []
            act = []
            next_r_obs = []
            r_obs = []
            rwd = []
            is_done = []

            robot_state, ob = self.env.reset("train")
            done = False
            robot_obs, humans_obs = self.transfunc(robot_state, ob)

            while not done:
                a = self.env.robot.act(ob)

                obs.append(torch.as_tensor(humans_obs.reshape(-1), dtype=torch.float32))

                r_obs.append(torch.as_tensor(robot_obs, dtype=torch.float32))

                robot_state, ob, r, done, info = self.env.step(a)

                robot_obs, humans_obs = self.transfunc(
                    robot_state,
                    ob,
                )

                act.append(torch.as_tensor(a, dtype=torch.float32))

                next_obs.append(
                    torch.as_tensor(humans_obs.reshape(-1), dtype=torch.float32)
                )

                next_r_obs.append(torch.as_tensor(robot_obs, dtype=torch.float32))

                rwd.append(torch.as_tensor([r], dtype=torch.float32).reshape(1))

                is_done.append(
                    torch.as_tensor([int(done)], dtype=torch.float32).reshape(1)
                )

                if isinstance(info, Discomfort):
                    too_close += 1
                    min_dist.append(info.min_dist)

                if done:
                    if render:
                        self.env.render()
                    sum_rewards += r

            if isinstance(info, ReachGoal):
                success += 1
                success_times.append(self.env.global_time)
            elif isinstance(info, Collision):
                collision += 1
                collision_cases.append(k)
                collision_times.append(self.env.global_time)
            elif isinstance(info, Timeout):
                timeout += 1
                timeout_cases.append(k)
                timeout_times.append(self.env.time_limit)

            # j += 1
            # rewards = rwd[:j, :]
            sum_cdrs = self.calc_cdr(rwd)

            sum_returns += self.calc_returns(rwd)

            # if isinstance(info, ReachGoal):
            if isinstance(info, ReachGoal) or isinstance(info, Collision):
                for i in range(len(obs)):
                    # test = v[i]
                    # yy = r_obs[i]
                    samples = TensorDict(
                        {
                            "humans_obs": obs[i].reshape(-1, self.obs_dim),
                            "next_humans_obs": next_obs[i].reshape(-1, self.obs_dim),
                            "robot_obs": r_obs[i],
                            "next_robot_obs": next_r_obs[i],
                            "action": act[i].reshape(-1, self.act_dim),
                            "reward": rwd[i],
                            "done": is_done[i],
                        }
                    )
                    buffer.add(samples)

        avg_reward = sum_rewards / k
        avg_cdr = sum_cdrs / k
        avg_return = sum_returns / k
        success_rate = success / k
        collision_rate = collision / k
        timeout_rate = timeout / k
        assert success + collision + timeout == k
        avg_nav_time = (
            sum(success_times) / len(success_times)
            if success_times
            else self.env.time_limit
        )

        return (
            avg_reward,
            avg_cdr,
            avg_return,
            success_rate,
            collision_rate,
            timeout_rate,
            avg_nav_time,
        )

    def exploration_k_ep(
        self,
        buffer,
        model=None,
        k=1,
        epsilon=0.5,
        render=False,
        pbar=None,
        random_p_num=False,
        p_range=(1, 5),
    ):
        trajs = []
        d_a = self.act_dim
        sum_rewards = 0
        sum_cdrs = 0
        sum_returns = 0
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
        for _ in range(k):
            if random_p_num:
                p_num = np.random.randint(*p_range)
                self.env.set_human_num(p_num)
                d_o = self.obs_dim * p_num
            else:
                d_o = self.obs_dim * self.env.human_num
            d_ro = self.r_obs_dim
            d_s = self.state_dim * self.env.human_num

            # Temporary buffer list
            next_obs = []
            obs = []
            act = []
            next_r_obs = []
            r_obs = []
            rwd = []
            is_done = []

            robot_state, human_state = self.env.reset("train")
            robot_obs, humans_obs = self.transfunc(
                robot_state,
                human_state,
            )
            o0 = torch.clone(humans_obs.reshape(-1))
            r_o0 = torch.clone(robot_obs)
            # if model != None:
            #     s = psr.estimate_state(human_obs.reshape(-1).to(psr.device))
            #     # integrated_states = []
            done = False
            while not done:
                a, _, _ = model.generate_action(
                    (
                        humans_obs.unsqueeze(0).to(model.device),
                        robot_obs.reshape(1, 1, -1).to(model.device),
                    )
                )
                a = (
                    a.clamp(model.actor.act_min, model.actor.act_max)
                    .cpu()
                    .data.numpy()
                    .squeeze()
                )

                a = self.convert_action(a)

                obs.append(torch.as_tensor(humans_obs.reshape(-1), dtype=torch.float32))

                r_obs.append(torch.as_tensor(robot_obs, dtype=torch.float32))

                robot_state, human_state, r, done, info = self.env.step(a)

                robot_obs, humans_obs = self.transfunc(
                    robot_state,
                    human_state,
                )

                act.append(torch.as_tensor(a, dtype=torch.float32))

                next_obs.append(
                    torch.as_tensor(humans_obs.reshape(-1), dtype=torch.float32)
                )

                next_r_obs.append(torch.as_tensor(robot_obs, dtype=torch.float32))

                rwd.append(torch.as_tensor([r], dtype=torch.float32))

                is_done.append(torch.as_tensor([int(done)], dtype=torch.float32))

                if isinstance(info, Discomfort):
                    too_close += 1
                    min_dist.append(info.min_dist)

                if done:
                    if render:
                        self.env.render()
                    sum_rewards += r

            if isinstance(info, ReachGoal):
                success += 1
                success_times.append(self.env.global_time)
            elif isinstance(info, Collision):
                collision += 1
                collision_cases.append(k)
                collision_times.append(self.env.global_time)
            elif isinstance(info, Timeout):
                timeout += 1
                timeout_cases.append(k)
                timeout_times.append(self.env.time_limit)

            # j += 1
            # rewards = rwd[:j, :]
            sum_cdrs = self.calc_cdr(rwd)

            sum_returns += self.calc_returns(rwd)

            # if isinstance(info, ReachGoal):
            # if isinstance(info, ReachGoal) or isinstance(info, Collision):
            for i in range(len(obs)):
                # test = v[i]
                # yy = r_obs[i]
                samples = TensorDict(
                    {
                        "humans_obs": obs[i].reshape(-1, self.obs_dim),
                        "next_humans_obs": next_obs[i].reshape(-1, self.obs_dim),
                        "robot_obs": r_obs[i],
                        "next_robot_obs": next_r_obs[i],
                        "action": act[i].reshape(-1, self.act_dim),
                        "reward": rwd[i],
                        "done": is_done[i],
                    }
                )
                buffer.add(samples)
        if pbar:
            pbar.set_postfix(Reward=str(sum_rewards / k))

        avg_reward = sum_rewards / k
        avg_cdr = (sum_cdrs / k).item()
        avg_return = (sum_returns / k).item()
        success_rate = success / k
        collision_rate = collision / k
        timeout_rate = timeout / k
        assert success + collision + timeout == k
        avg_nav_time = (
            sum(success_times) / len(success_times)
            if success_times
            else self.env.time_limit
        )

        return (
            avg_reward,
            avg_cdr,
            avg_return,
            success_rate,
            collision_rate,
            timeout_rate,
            avg_nav_time,
        )
