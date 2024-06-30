import td3_fork
import numpy as np
from pybullet_sim import Environment

if __name__ == "__main__":
    filename = "testing"
    chkpt_dir = "C:/Users/Charles/Documents/Python Scripts/Personal/Artificial Intelligence/FluxxIO/models"
    log_dir = f"C:/Users/Charles/Documents/Python Scripts/Personal/Artificial Intelligence/FluxxIO/plots/{filename}"
    robot_path = "C:/Users/Charles/Documents/Python Scripts/Personal/Artificial Intelligence/FluxxIO/src/main/simulation_robot/robot.urdf"
    env = Environment(robot_path, record=False, render="human", print_reward=1)  # if debug true it means print rewards
    env.curriculum_level = 0

    lr = 0.001
    agent = td3_fork.Agent(
        alpha=lr,
        beta=lr,
        env=env,
        chkpt_dir=chkpt_dir,
    )

    agent.load_models()
    observation, _ = env.reset()
    score = 0
    done = False
    while True:
        action = agent.choose_action(observation)
        # action = np.random.uniform(-1, 1, 4)
        observation_, reward, terminated, truncated, _ = env.step(action)
        score += reward
        done = terminated or truncated
        observation = observation_
        if env.time_step % int(1/env.dt) == 0:
            seconds = int(env.time_step * env.dt)
            print(f"{seconds} seconds\t {score=}")
        if done:
            break
    print(f"Total Reward: {score}")
