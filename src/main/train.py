import td3_fork
import numpy as np
from pybullet_sim import Environment
from torch.utils.tensorboard import SummaryWriter

if __name__ == "__main__":
    filename = "testing"
    chkpt_dir = "C:/Users/Charles/Documents/Python Scripts/Personal/Artificial Intelligence/FluxxIO/models"
    log_dir = f"C:/Users/Charles/Documents/Python Scripts/Personal/Artificial Intelligence/FluxxIO/plots/{filename}"
    robot_path = "C:/Users/Charles/Documents/Python Scripts/Personal/Artificial Intelligence/FluxxIO/src/main/simulation_robot/robot.urdf"
    # chkpt_dir = "./drive/MyDrive/FluxxIO/models"
    # log_dir = f"./drive/MyDrive/FluxxIO/runs/{filename}"
    # robot_path = "./drive/MyDrive/FluxxIO/src/main/simulation_robot/robot.urdf"
    env = Environment(robot_path)

    lr = 0.001
    agent = td3_fork.Agent(
        alpha=lr,
        beta=lr,
        env=env,
        chkpt_dir=chkpt_dir,
        warmup=env.time_step_max*3,
    )

    # agent.load_models()
    # agent.time_step = 0
    # agent.memory.load(agent.buffer_file_pth)

    writer = SummaryWriter(log_dir=log_dir)
    n_timesteps = 2_000_000
    episode = env.time_step_max // 4

    best_score = env.reward_range[0]
    best_avg_score = best_score
    score_history = []
    critic_loss_count = 0
    actor_loss_count = 0
    critic_loss = 0
    actor_loss = 0
    system_loss_count = 0
    reward_loss_count = 0
    system_loss = 0
    reward_loss = 0
    score = 0
    steps = 0
    eps = 0
    eps_jump_load = (agent.memory.mem_cntr // env.time_step_max) - 1
    done = True

    for step in range(n_timesteps):
        if done:
            eps += 1
            observation, _ = env.reset()
        action = agent.choose_action(observation)
        observation_, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        agent.remember(observation, action, reward, observation_, done)
        observation = observation_
        writer.add_scalar("train/return", reward, step)
        score += reward

        c_loss, a_loss, s_loss, r_loss = agent.learn()
        if c_loss is not None:
            critic_loss_count += 1
            critic_loss += c_loss
        if a_loss is not None:
            actor_loss_count += 1
            actor_loss += a_loss
        if s_loss is not None:
            system_loss_count += 1
            system_loss += s_loss
        if r_loss is not None:
            reward_loss_count += 1
            reward_loss += r_loss

        if ((step + 1) % episode) == 0:
            i = int(step / episode) + eps_jump_load
            score_history.append(score)
            avg_score = np.mean(score_history[-100:])
            if critic_loss_count > 0:
                critic_loss /= critic_loss_count
            if actor_loss_count > 0:
                actor_loss /= actor_loss_count
            if system_loss_count > 0:
                system_loss /= system_loss_count
            if reward_loss_count > 0:
                reward_loss /= reward_loss_count
            if avg_score > best_score:
                best_score = avg_score

            writer.add_scalar("train/reward", score, i)
            writer.add_scalar("train/critic_loss", critic_loss, i)
            writer.add_scalar("train/actor_loss", actor_loss, i)
            writer.add_scalar("train/system_loss", system_loss, i)
            writer.add_scalar("train/reward_loss", reward_loss, i)
            writer.flush()

            print(
                f"Episode {i}: "
                f"Score: {score:.1f}, "
                f"Average Score: {avg_score:.1f}, "
                f"Critic Loss: {critic_loss:.5f}, "
                f"Actor Loss: {actor_loss:.5f}, "
                f"System Loss: {system_loss:.5f}, "
                f"Reward Loss: {reward_loss:.5f}"
            )

            if avg_score >= best_avg_score:
                best_avg_score = avg_score
                agent.save_models()

            critic_loss_count = 0
            actor_loss_count = 0
            critic_loss = 0
            actor_loss = 0
            system_loss_count = 0
            reward_loss_count = 0
            system_loss = 0
            reward_loss = 0
            score = 0
    writer.close()
