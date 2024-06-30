import td3_fork_pendulum as td3_fork
import td3
import sac_new as sac

import gymnasium as gym
from gymnasium.envs.registration import register
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from mujoco_mod.envs.domain_randomization import DomainRandomization, simp_angle
import torch as T
import os

# Mujoco Modded by Charles
# ----------------------------------------
# modded inverted_pendulum.xml
# changed inverted_pendulum.xml file path in inverted_pendulum_mod.py
# changed to fromto="0 0 0 0.001 0 -0.6" in "cpole" (pole resets downwards)
# changed to range="-180 180" in "pole"
# rewards to cos(theta)
# does not terminate until max timesteps reached
# changed reset_model to have a min and high of -0.1 and 0.1 respectively
# changed reward function to -(theta^2 + 10*cart^2 + 0.1*theta_dt^2 + 0.1*cart_dt^2 + 0.001*torque^2)

register(
    id="InvertedPendulumModded",
    entry_point="mujoco_mod.envs.inverted_pendulum_mod:InvertedPendulumEnv",
    max_episode_steps=1000,
    reward_threshold=0.0,
)

if __name__ == "__main__":
    algo = "td3-fork"
    isDomainRandomization = False
    isMonteCarlo = False  # True = train every episode
    game_id = "InvertedPendulumModded"
    filename = "testing"
    chkpt_dir = "./tmp/td3 fork"
    log_dir = f"runs/inverted_pendulum_sim/{filename}"
    # chkpt_dir = "./drive/MyDrive/pendulum/tmp/td3 fork"
    # log_dir = f"./drive/MyDrive/pendulum/runs/inverted_pendulum_sim/{filename}"
    env = gym.make(game_id)
    domain_randomizer = DomainRandomization()

    seed = None
    if seed is not None:
        np.random.seed(seed)
        T.manual_seed(seed)

    if algo == "td3-fork":
        # a warmup of 20,000 steps irl
        lr = 0.001  # now is 0.005
        agent = td3_fork.Agent(
            alpha=lr,
            beta=lr,
            ln=True,
            warmup=10000,  # can be changed now
            input_dims=env.observation_space.shape,
            tau=0.005,
            env=env,
            gamma=0.98,
            noise=0.1,
            policy_noise=0.2,
            layer1_size=400,  # constant don't change orig now it is 16x16
            layer2_size=300,
            sys1_size=400,
            sys2_size=300,
            r1_size=256,
            r2_size=256,
            sys_weight=0.6,
            update_actor_interval=1,
            max_size=1_000_000,
            n_actions=env.action_space.shape[0],
            game_id=game_id,
            chkpt_dir=chkpt_dir,
        )
        # remove this after transfer learning
        # agent.chkpt_file_pth = (
        #     f"./drive/MyDrive/pendulum/tmp/td3_fork_learned/{game_id} td3_fork.chkpt"
        # )
        # agent.load_models(load_all_weights=True, load_optimizers=False)
        # agent.freeze_layer(first_layer=False, second_layer=False)
        # agent.chkpt_file_pth = os.path.join(chkpt_dir, f"{game_id} td3 fork.chkpt")
    elif algo == "sac":
        agent = sac.Agent(
            a_lr=0.00073,
            q_lr=0.00073,
            batch_size=256,
            max_size=1_000_000,
            gamma=0.98,
            layer1_size=400,
            layer2_size=300,
            tau=0.02,
            warmup=10_000,
            input_dims=env.observation_space.shape,
            env=env,
            n_actions=env.action_space.shape[0],
            game_id=game_id,
            chkpt_dir=chkpt_dir,
        )
    elif algo == "td3":
        buffer_size = 200_000
        lr = 0.001
        fc1 = 400
        fc2 = 300
        agent = td3.Agent(
            alpha=lr,
            beta=lr,
            input_dims=env.observation_space.shape,
            tau=0.005,
            env=env,
            batch_size=100,
            layer1_size=fc1,
            layer2_size=fc2,
            noise=0.1,
            max_size=buffer_size,
            gamma=0.98,
            update_actor_interval=1,
            warmup=10_000,
            n_actions=env.action_space.shape[0],
            game_id=game_id,
        )

    # agent.partial_load_models()
    writer = SummaryWriter(log_dir=log_dir)
    n_timesteps = 1_000_000
    episode = 1_000  # 1 episode = 1k timesteps

    best_score = env.reward_range[0]
    best_avg_score = best_score
    score_history = []
    critic_loss_count = 0
    actor_loss_count = 0
    critic_loss = 0
    actor_loss = 0
    if algo == "td3-fork":
        system_loss_count = 0
        reward_loss_count = 0
        system_loss = 0
        reward_loss = 0
    score = 0
    steps = 0
    done = True
    domain_randomizer.reset_environment()

    for step in range(n_timesteps):
        if done:
            if isMonteCarlo:
                for _ in range(steps):
                    if algo == "td3-fork":
                        c_loss, a_loss, s_loss, r_loss = agent.learn()
                    else:
                        c_loss, a_loss = agent.learn()

                    if c_loss is not None:
                        critic_loss_count += 1
                        critic_loss += c_loss
                    if a_loss is not None:
                        actor_loss_count += 1
                        actor_loss += a_loss
                    if algo == "td3-fork":
                        if s_loss is not None:
                            system_loss_count += 1
                            system_loss += s_loss
                        if r_loss is not None:
                            reward_loss_count += 1
                            reward_loss += r_loss
            steps = 0
            if isDomainRandomization:
                ep = int(step / episode)
                if domain_randomizer.check_level_up(np.mean(score_history[-10:]), ep):
                    best_avg_score -= 800
                    best_score -= 800
                domain_randomizer.environment()
            env = gym.make(game_id)
            observation, info = env.reset(seed=seed)
            if isDomainRandomization:
                observation = domain_randomizer.observation(observation)
            observation[1] = simp_angle(observation[1])

        action = agent.choose_action(observation)
        action = domain_randomizer.action(action)
        observation_, reward, terminated, truncated, info = env.step(action)
        if isDomainRandomization:
            observation_ = domain_randomizer.observation(observation_)
        observation_[1] = simp_angle(observation_[1])
        done = terminated or truncated
        agent.remember(observation, action, reward, observation_, done)
        steps += 1
        observation = observation_
        writer.add_scalar("train/return", reward, step)
        score += reward

        if not isMonteCarlo:
            if algo == "td3-fork":
                c_loss, a_loss, s_loss, r_loss = agent.learn()
            else:
                c_loss, a_loss = agent.learn()

            if c_loss is not None:
                critic_loss_count += 1
                critic_loss += c_loss
            if a_loss is not None:
                actor_loss_count += 1
                actor_loss += a_loss
            if algo == "td3-fork":
                if s_loss is not None:
                    system_loss_count += 1
                    system_loss += s_loss
                if r_loss is not None:
                    reward_loss_count += 1
                    reward_loss += r_loss

        if (step + 1) % episode == 0:
            i = int(step / episode)
            score_history.append(score)
            avg_score = np.mean(score_history[-100:])
            if critic_loss_count > 0:
                critic_loss /= critic_loss_count
            if actor_loss_count > 0:
                actor_loss /= actor_loss_count
            if algo == "td3-fork":
                if system_loss_count > 0:
                    system_loss /= system_loss_count
                if reward_loss_count > 0:
                    reward_loss /= reward_loss_count
            if avg_score > best_score:
                best_score = avg_score

            writer.add_scalar("train/reward", score, i)
            writer.add_scalar("train/critic_loss", critic_loss, i)
            writer.add_scalar("train/actor_loss", actor_loss, i)
            if algo == "td3-fork":
                writer.add_scalar("train/system_loss", system_loss, i)
                writer.add_scalar("train/reward_loss", reward_loss, i)
                print(
                    "episode",
                    i,
                    "score %.1f" % score,
                    "avg score %.1f" % avg_score,
                    "critic loss %.5f" % critic_loss,
                    "actor loss %.5f" % actor_loss,
                    "system loss %.5f" % system_loss,
                    "reward loss %.5f" % reward_loss,
                )
            else:
                print(
                    "episode",
                    i,
                    "score %.1f" % score,
                    "avg score %.1f" % avg_score,
                    "critic loss %.5f" % critic_loss,
                    "actor loss %.5f" % actor_loss,
                )

            if score >= 990:
                perfect_score_count += 1
                if perfect_score_count >= 10:
                    print("...environment solved...")
                    agent.save_models()
                    break
            else:
                perfect_score_count = 0

            if avg_score >= best_avg_score:
                best_avg_score = avg_score
                agent.save_models()
            critic_loss_count = 0
            actor_loss_count = 0
            critic_loss = 0
            actor_loss = 0
            if algo == "td3-fork":
                system_loss_count = 0
                reward_loss_count = 0
                system_loss = 0
                reward_loss = 0
            score = 0
            writer.flush()

    writer.close()
