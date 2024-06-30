import numpy as np
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import os
import csv
from mujoco_mod.envs.domain_randomization import simp_angle


def run_env(game_id="InvertedPendulumModded", eps=1):
    from gymnasium.envs.registration import register

    register(
        id="InvertedPendulumModded",
        entry_point="mujoco_mod.envs.inverted_pendulum_mod:InvertedPendulumEnv",
        max_episode_steps=500,
        reward_threshold=1000.0,
    )

    env = gym.make(game_id, render_mode="human")

    for _ in range(eps):
        steps = 0
        obs, info = env.reset()
        done = False
        max_action = [1.6]
        max_speed = float("inf")
        while not done:
            # action = env.action_space.sample()
            if steps < 200:
                action = max_action
                if obs[0] < -0.8:
                    action = [max_action[0] / 2 * 0.6]
                elif obs[0] > 0.8:
                    action = [-max_action[0] / 2 * 0.6]
            elif steps > 200:
                action = [-max_action[0]]
                if obs[0] < -0.8:
                    action = [max_action[0] / 2 * 0.6]
                elif obs[0] > 0.8:
                    action = [-max_action[0] / 2 * 0.6]

            observation_, reward, terminated, truncated, info = env.step(action)
            obs = observation_
            print(
                f"pos:{obs[0]:.2f} angle:{obs[1]:.4f} lin_vel:{obs[2]:.2f} ang_vel:{obs[3]:.2f} reward:{reward:.2f}"
            )
            if abs(obs[1]) < max_speed:
                max_speed = abs(obs[1])
            done = terminated or truncated
            steps += 1
    print(max_speed)
    env.close()


def agent_play(game_id, agent, eps=3, save=True, find_best=False, save_data=False):
    if save:
        render_mode = "rgb_array"
    else:
        render_mode = "human"
    if game_id == "Pendulum-v1":
        env = gym.make(game_id, render_mode=render_mode, g=9.80665)
    else:
        env = gym.make(game_id, render_mode=render_mode)
    if save:
        env = RecordVideo(
            env,
            "recordings",
            name_prefix="InvertedPendulum",
            episode_trigger=lambda _: True,
        )
    if save_data:
        eps = 1
        data = []
    if not find_best:
        rewards = 0
        for ep in range(eps):
            observation, info = env.reset()
            obs = observation
            obs[1] = simp_angle(obs[1])
            print(
                f"pos:{obs[0]:.2f} angle:{obs[1]:.2f} lin_vel:{obs[2]:.2f} ang_vel:{obs[3]:.2f}"
            )
            if save:
                env.start_video_recorder()
            done = False
            max_speed = float("-inf")
            time_steps = 0
            while not done:
                action = agent.choose_action(observation, evaluate=True)
                observation_, reward, terminated, truncated, info = env.step(action)
                observation_[1] = simp_angle(observation_[1])
                if not save:
                    obs = observation_
                    print(
                        f"pos:{obs[0]:.2f} angle:{obs[1]:.2f} lin_vel:{obs[2]:.2f} ang_vel:{obs[3]:.2f} reward:{reward:.2f} action:{action.item():.2f}"
                    )
                if save_data:
                    data.append(
                        [
                            observation[0],
                            observation[1],
                            observation[2],
                            observation[3],
                            reward,
                            action.item(),
                        ]
                    )
                rewards += reward
                observation = observation_
                done = terminated or truncated
                time_steps += 1
            print(time_steps)
        env.close()

        if save_data:
            with open("recordings/sample_data_16x16.csv", "w") as file:
                writer = csv.writer(file)
                for d in data:
                    writer.writerow(d)

        return rewards / eps
    else:
        trial = 0
        while True:
            trial += 1
            rewards = 0
            observation, info = env.reset()
            if save:
                env.start_video_recorder()
            done = False
            while not done:
                action = agent.choose_action(observation, evaluate=True)
                observation_, reward, terminated, truncated, info = env.step(action)
                rewards += reward
                observation = observation_
                done = terminated or truncated
            print(f"trial #{trial} rewards {rewards}")
            if rewards > 888:
                print("Found the best episode")
                break
            else:
                [
                    os.remove(os.path.join(".\\recordings", file))
                    for file in os.listdir(".\\recordings")
                ]
        env.close()


def testing(env):
    for _ in range(2):
        env.reset()
        done = False
        env.unwrapped.state = np.array([2 * np.pi, 1.0])
        while not done:
            _, _, terminated, truncated, _ = env.step([env.max_torque])
            done = terminated or truncated
    env.close()
