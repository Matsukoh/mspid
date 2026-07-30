import importlib

import numpy as np

from rewacs.envs import CrowdSim
from rewacs.envs.policy.policy_factory import policy_factory
from rewacs.envs.utils.robot import Robot
from rewacs.envs.utils.transformations import GetRobotFrameObs


def define_env(
    debug=False,
    path="./configs/test_env_config.py",
):
    # configure policy
    spec = importlib.util.spec_from_file_location("config", path)

    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)

    env_config = config.TestConfig()
    env = CrowdSim()
    env.configure(env_config)
    robot = Robot(env_config, "robot")
    robot.time_step = env.time_step
    env.set_robot(robot)

    if robot.visible:
        safety_space = 0
    else:
        safety_space = 0.15

    policy = policy_factory[env_config.robot.policy]()
    policy.safety_space = safety_space

    robot.set_policy(policy)

    if debug:
        print(env_config)

    return env, robot


env, robot = define_env(debug=True)
rng = np.random.RandomState(0)


robot_state, ob = env.reset("train")

scenario_count = 0

transfunc = GetRobotFrameObs(
    with_peds_vel=True, peds_vel_as_relative=True, use_omega=True
)

while True:
    action = env.robot.act(ob)
    robot_obs, humans_obs = transfunc(
        robot_state,
        ob,
    )
    robot_state, ob, reward, done, info = env.step(action)
    if done:
        env.render()
        robot_state, ob = env.reset("train")
        scenario_count += 1
