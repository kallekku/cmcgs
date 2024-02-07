import copy
import os
#if os.name != 'nt':  # Don't run this line on Windows.
#    os.environ["MUJOCO_GL"] = "egl"
from pathlib import Path
import random

import hydra
import numpy as np
from dm_control import suite
import time
import wandb
#import cv2

import utils
from wrappers import DMC2GYMWrapper, SimulatorWrapper, ActionRepeatWrapper
from agents import CMCGS
from toy_envs.two_d_navigation_env import TwoDimNavigationEnv
from toy_envs import two_d_reacher_env
from toy_envs.two_d_reacher_env import TwoDimReacherEnv


# action repeats for each domain.
ACTION_REPEATS = {'cartpole': 8, 'reacher': 4, 'finger': 2, 'cheetah': 4, 'ball_in_cup': 6, 'walker': 2, 'humanoid': 2
                  , 'hopper': 2, 'quadruped': 2, 'acrobot': 2, 'dog': 2, 'fish': 2, 'lqr': 2, 'manipulator': 2
                  , 'pendulum': 2, 'point_mass': 2, 'stacker': 2, 'swimmer': 2}


def make_env_and_model(env_name, seed=0):
    if "2d-navigation" in env_name:
        env = TwoDimNavigationEnv(obstacle_mode=env_name[14:])  # remove "2d-navigation" from the env_name to get the obstacle mode
    elif "2d-reacher" in env_name:
        env = TwoDimReacherEnv(env_name=env_name, seed=seed)
    else:
        # DMCS env
        domain_name, task_name = env_name.split("-")

        # get action_repeat, reference: https://github.com/Kaixhin/PlaNet/blob/master/env.py
        num_repeats = ACTION_REPEATS[domain_name] if domain_name in ACTION_REPEATS else 1
        env = suite.load(domain_name, task_name, {"random": seed})
        env = DMC2GYMWrapper(env)

        env = ActionRepeatWrapper(env, action_repeat=num_repeats)

    # create a dynamic model used in planning.    
    model = SimulatorWrapper(env)

    return env, model


# used for customed envs.
def save_video(rendering_buffer, file_name='video.mp4', fps=50):
    path = os.path.join(os.getcwd(), 'eval_video')
    os.makedirs(os.path.join(os.getcwd(), 'eval_video'), exist_ok=True)

    path = os.path.join(path, file_name)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    h, w = rendering_buffer[0].shape[0], rendering_buffer[0].shape[1]
    writer = cv2.VideoWriter(path, fourcc, fps, (w, h))

    for img in rendering_buffer:
        writer.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    # cv2.destroyAllWindows()
    writer.release()

    print('Saved video to: %s' % path)


def env_supports_video_recorder(env_name):
    if "2d-navigation" in env_name or "2d-reacher" in env_name:
        return False
    return True


def get_budget_string(agent_config):
    num_simulations = agent_config.params.simulation_budget
    if num_simulations % 1000 == 0:
        return '%dK' % (num_simulations // 1000)
    return ('%.1fK' % (num_simulations / 1000)).replace(".", ",")


@hydra.main(config_path="configs", config_name="mpc_config")
def main(config):
    os.chdir(hydra.utils.get_original_cwd())  # set to original working path, see https://github.com/facebookresearch/hydra/issues/922
    
    # seeding
    seed = config.seed
    if seed == 0:
        seed = random.randint(0, 10000)
    random.seed(seed)
    np.random.seed(seed)

    file_name = f"{config.agent.name}_{config.env_name}_{config.run_suffix}_{seed}"

    print('env_name: %s\nsave_video: %d\nuse_wandb: %d\nagent: %s\nrun_suffix: %s'
          % (config.env_name, config.save_video, config.use_wandb, config.agent, config.run_suffix))

    # set logging
    if config.use_wandb:
        wandb.init(
            project="cmcgs",
            group=config.env_name,
            config=vars(config),
            name=file_name,
            monitor_gym=True,)

    # setup env and model
    env, model = make_env_and_model(config.env_name, seed)

    # initialize the cmcgs agent
    agent = CMCGS(config=config.agent.params, model=model)
    if "2d-navigation" in config.env_name:
        agent.set_save_exploration(config.save_exploration)
        agent_copy = copy.deepcopy(agent)
    
    # set video_recoder
    video_recorder = None
    rendering_buffer = None
    if config.save_video:
        if "2d-navigation" in config.env_name or "2d-reacher" in config.env_name:
            rendering_buffer = []
        else: 
            video_recorder = utils.VideoRecorder(Path.cwd(), camera_id=0, use_wandb=config.use_wandb)
            video_recorder.init(env, enabled=True)

    # begin to run the agent
    obs = env.reset()
    total_rewards, done, step, log_rewards = 0, False, 0, []

    while not done:
        action = agent.act(obs)

        if "2d-navigation" in config.env_name:
            if config.save_exploration:
                env.plot_visited_states(config.agent.name, agent.exploration_trajs, step=step, rollout_indices=agent.rollout_indices)

                agent.reset_exploration_trajs()
            obs, reward, done, info = env.step(action, render=config.save_video)
            if config.save_video:
                rendering_buffer += info['render']
        elif "2d-reacher" in config.env_name:
            obs, reward, done, info = env.step(action)
            if config.save_video:
                rendering_buffer.append(env.render('rgb_array'))
        else:
            # DMCS
            obs, reward, done, _ = env.step(action)
            if config.save_video:
                video_recorder.record(env)

        total_rewards += reward
        step += 1

        if config.use_wandb:
            wandb.log({"CMCGS/": {'reward': reward, 'step': step}})

        print(' i: %d, cumulative_reward: %.2f, perstep_reward: %.2f' % (step, total_rewards, reward))

        log_rewards.append(reward)

    if config.save_video:
        file_name = f'{config.env_name}_{config.agent.name}_{seed}.mp4'
        if env_supports_video_recorder(config.env_name):
            video_recorder.save(file_name)
        else:
            if 'video.frames_per_second' in env.metadata.keys():
                fps = env.metadata['video.frames_per_second']
            else:
                fps = env.env.metadata['video.frames_per_second']
            save_video(rendering_buffer, file_name=file_name, fps=fps)
    
    if config.use_wandb:
        wandb.log({"MPC/": {"total_reward": total_rewards}})
    print(f'total_rewards: %.2f' % total_rewards)

    if not config.use_wandb:
        log_dir = os.path.join(os.getcwd(), 'results-csv')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        agent_name = config.agent.name
        budget = get_budget_string(config.agent)
        log_dir = os.path.join(log_dir, "%s_%s_%s_%s.csv" % (agent_name, budget, config.env_name, config.run_suffix))
        np.savetxt(log_dir, np.array(log_rewards), delimiter=",", header="rewards", comments="", fmt='%.2f')
        print('Logged results into ', log_dir)


if __name__ == "__main__":
    main()
