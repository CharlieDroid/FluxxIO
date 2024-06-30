import numpy as np
import copy
import torch as T
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import os
import pickle
import zipfile


T.set_default_dtype(T.float32)


class ReplayBuffer:
    def __init__(self, max_size, input_privilege_shape, input_observe_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.privilege_state_memory = np.zeros((self.mem_size, *input_privilege_shape), dtype=np.float32)
        self.privilege_new_state_memory = np.zeros((self.mem_size, *input_privilege_shape), dtype=np.float32)
        self.observe_state_memory = np.zeros((self.mem_size, *input_observe_shape), dtype=np.float32)
        self.observe_new_state_memory = np.zeros((self.mem_size, *input_observe_shape), dtype=np.float32)
        self.action_memory = np.zeros((self.mem_size, n_actions), dtype=np.float32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)

    def store_transition(self, privilege_state, observe_state, action, reward, privilege_state_, observe_state_, done):
        index = self.mem_cntr % self.mem_size
        self.privilege_state_memory[index] = privilege_state
        self.observe_state_memory[index] = observe_state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.privilege_new_state_memory[index] = privilege_state_
        self.observe_new_state_memory[index] = observe_state_
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, batch_size)

        privilege_states = self.privilege_state_memory[batch]
        observe_states = self.observe_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        privilege_states_ = self.privilege_new_state_memory[batch]
        observe_states_ = self.observe_new_state_memory[batch]
        dones = self.terminal_memory[batch]

        return privilege_states, observe_states, actions, rewards, privilege_states_, observe_states_, dones

    def save(self, file_pth):
        print("...saving memory...")
        memory = (
            self.privilege_state_memory,
            self.observe_state_memory,
            self.privilege_new_state_memory,
            self.observe_new_state_memory,
            self.action_memory,
            self.terminal_memory,
            self.reward_memory,
            self.mem_cntr,
        )
        with open(file_pth, "wb") as outfile:
            pickle.dump(memory, outfile, pickle.HIGHEST_PROTOCOL)

    def load(self, file_pth):
        print("...loading memory...")
        with open(file_pth, "rb") as infile:
            result = pickle.load(infile)
        (
            self.privilege_state_memory,
            self.observe_state_memory,
            self.privilege_new_state_memory,
            self.observe_new_state_memory,
            self.action_memory,
            self.terminal_memory,
            self.reward_memory,
            self.mem_cntr,
        ) = result


class RewardNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims, n_actions, device):
        super(RewardNetwork, self).__init__()

        self.fc1 = nn.Linear(2 * input_dims[0] + n_actions, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.fc3 = nn.Linear(fc2_dims, 1)

        self.ln1 = nn.LayerNorm(fc1_dims)
        self.ln2 = nn.LayerNorm(fc2_dims)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)

        self.to(device)

    def forward(self, state, state_, action):
        sa = T.cat([state, state_, action], dim=1)

        q1 = F.relu(self.fc1(sa))
        q1 = self.ln1(q1)

        q1 = F.relu(self.fc2(q1))
        q1 = self.ln2(q1)

        q1 = self.fc3(q1)
        return q1


class SystemNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims, n_actions, device):
        super(SystemNetwork, self).__init__()

        self.fc1 = nn.Linear(input_dims[0] + n_actions, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.fc3 = nn.Linear(fc2_dims, input_dims[0])

        self.ln1 = nn.LayerNorm(fc1_dims)
        self.ln2 = nn.LayerNorm(fc2_dims)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)

        self.to(device)

    def forward(self, state, action):
        """Build a system model to predict the next state at a given state."""
        xa = T.cat([state, action], dim=1)

        x1 = F.relu(self.fc1(xa))
        x1 = self.ln1(x1)

        x1 = F.relu(self.fc2(x1))
        x1 = self.ln2(x1)

        x1 = self.fc3(x1)
        return x1


class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims, n_actions, device):
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

        self.fc1 = nn.Linear(self.input_dims[0] + n_actions, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.q1 = nn.Linear(self.fc2_dims, 1)

        self.ln1 = nn.LayerNorm(self.fc1_dims)
        self.ln2 = nn.LayerNorm(self.fc2_dims)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)

        self.to(device)

    def forward(self, state, action):
        q1_action_value = self.fc1(T.cat([state, action], dim=1))
        q1_action_value = F.relu(q1_action_value)
        q1_action_value = self.ln1(q1_action_value)

        q1_action_value = self.fc2(q1_action_value)
        q1_action_value = F.relu(q1_action_value)
        q1_action_value = self.ln2(q1_action_value)

        q1 = self.q1(q1_action_value)

        return q1


class ActorNetwork(nn.Module):
    def __init__(
        self, alpha, input_dims, fc1_dims, fc2_dims, n_actions, device
    ):
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.mu = nn.Linear(self.fc2_dims, self.n_actions)

        self.ln1 = nn.LayerNorm(self.fc1_dims)
        self.ln2 = nn.LayerNorm(self.fc2_dims)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)

        self.to(device)

    def forward(self, state):
        a = F.relu(self.fc1(state))
        a = self.ln1(a)

        a = F.relu(self.fc2(a))
        a = self.ln2(a)

        # activation is tanh because it bounds it between +- 1
        # just multiply this according to the maximum action of the environment
        mu = T.tanh(self.mu(a))
        return mu


class Agent:
    def __init__(
            self,
            alpha,
            beta,
            env,
            chkpt_dir,
            buffer_size=1_000_000,
            tau=0.005,
            gamma=0.98,
            update_actor_interval=1,
            warmup=10_000,
            action1_size=256,
            action2_size=256,
            critic1_size=256,
            critic2_size=256,
            sys1_size=400,
            sys2_size=300,
            r1_size=256,
            r2_size=256,
            batch_size=100,
            noise=0.1,
            policy_noise=0.2,
            sys_weight=0.6,
            sys_threshold=0.010,
            max_grad_norm=5.0
    ):
        # n_actions and input_dims to action_dims and obs_dims from env
        # game_id also from env
        # make sure all actions will be within -1 to 1
        # in env the actions will be scaled
        # environment can use list, this handles list
        # TODO: use float 32

        # hyperparams
        self.gamma = gamma
        self.tau = tau
        self.warmup = warmup
        self.update_actor_interval = update_actor_interval
        self.batch_size = batch_size
        self.noise = noise
        self.policy_noise = policy_noise
        self.max_grad_norm = max_grad_norm
        self.num_actions = env.action_space.shape[0]
        privilege_obs_dim = env.observation_space.privilege_shape
        observe_obs_dim = env.observation_space.observe_shape
        combined_obs_dim = env.observation_space.combined_shape

        # system and rewards (FORK)
        self.sys_weight = sys_weight
        self.sys_threshold = sys_threshold
        self.system_loss = 0
        self.reward_loss = 0

        # buffer
        self.memory = ReplayBuffer(buffer_size, privilege_obs_dim, observe_obs_dim, self.num_actions)

        # timestep counters
        self.learn_step_cntr = 0
        self.time_step = 0

        # filepaths for checkpoints and buffers
        self.chkpt_dir = chkpt_dir
        self.actor_file_zip = "td3_fork_actor.zip"
        self.chkpt_file_pth = os.path.join(chkpt_dir, f"{env.name} td3 fork.chkpt")
        self.buffer_file_pth = os.path.join(chkpt_dir, f"buffer td3 fork.pkl")
        self.actor_zipfile_pth = os.path.join(chkpt_dir, self.actor_file_zip).replace("\\", "/")

        # action bounds (note: array)
        self.action_upper_bound = env.action_space.bounds[1]
        self.action_lower_bound = env.action_space.bounds[0]

        # networks
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.actor = ActorNetwork(
            alpha,
            observe_obs_dim,
            action1_size,
            action2_size,
            self.num_actions,
            self.device,
        )
        self.actor.apply(self.init_weights)
        self.critic_1 = CriticNetwork(
            beta,
            combined_obs_dim,
            critic1_size,
            critic2_size,
            self.num_actions,
            self.device,
        )
        self.critic_1.apply(self.init_weights)
        self.critic_2 = CriticNetwork(
            beta,
            combined_obs_dim,
            critic1_size,
            critic2_size,
            self.num_actions,
            self.device,
        )
        self.critic_2.apply(self.init_weights)

        self.target_actor = copy.deepcopy(self.actor)
        self.target_critic_1 = copy.deepcopy(self.critic_1)
        self.target_critic_2 = copy.deepcopy(self.critic_2)
        self.target_actor.apply(self.init_weights)
        self.target_critic_1.apply(self.init_weights)
        self.target_critic_2.apply(self.init_weights)

        self.system = SystemNetwork(
            beta,
            combined_obs_dim,
            sys1_size,
            sys2_size,
            self.num_actions,
            self.device,
        )
        self.system.apply(self.init_weights)
        self.reward = RewardNetwork(
            beta, 
            combined_obs_dim,
            r1_size, 
            r2_size, 
            self.num_actions, 
            self.device,
        )
        self.reward.apply(self.init_weights)

        # Obs bounds
        self.obs_upper_bound = T.tensor(env.observation_space.bounds[1]).to(self.device)
        self.obs_lower_bound = T.tensor(env.observation_space.bounds[0]).to(self.device)
    
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            T.nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
            if m.bias is not None:
                T.nn.init.constant_(m.bias, 0)
    
    def choose_action(self, observation):
        if self.time_step < self.warmup:
            mu = T.tensor(np.random.normal(scale=self.noise, size=(self.num_actions,)))
        else:
            state = T.tensor(observation, dtype=T.float32).to(self.device)
            mu = self.actor.forward(state).to(self.device)
        # make sure the actions are bounded (can be removed if you are sure) since it is -1 to 1 anyway
        mu = T.clamp(mu, self.action_lower_bound, self.action_upper_bound)
        self.time_step += 1
        return mu.cpu().detach().tolist()

    def remember(self, privilege_state, observe_state, action, reward, privilege_new_state, observe_new_state, done):
        self.memory.store_transition(privilege_state, observe_state, action,
                                     reward, privilege_new_state, observe_new_state, done)
    
    def learn(self):
        if (self.memory.mem_cntr < self.batch_size) or (self.time_step < self.warmup):
            return None, None, None, None

        self.learn_step_cntr += 1
        privilege_state, observe_state, action, reward, privilege_new_state, observe_new_state, done = self.memory.sample_buffer(
            self.batch_size
        )
        reward = T.tensor(reward, dtype=T.float32).to(self.device)
        done = T.tensor(done).to(self.device)
        privilege_state_ = T.tensor(privilege_new_state, dtype=T.float32).to(self.device)
        observe_state_ = T.tensor(observe_new_state, dtype=T.float32).to(self.device)
        privilege_state = T.tensor(privilege_state, dtype=T.float32).to(self.device)
        observe_state = T.tensor(observe_state, dtype=T.float32).to(self.device)
        action = T.tensor(action, dtype=T.float32).to(self.device)

        combined_state = T.cat((privilege_state, observe_state), dim=1)
        combined_state_ = T.cat((privilege_state_, observe_state_), dim=1)

        with T.no_grad():
            noise = T.clamp(T.randn_like(action) * self.policy_noise, -0.5, 0.5)
            target_actions = T.clamp(
                self.target_actor(observe_state_) + noise,
                self.action_lower_bound,
                self.action_upper_bound,
            )

            q1_ = self.target_critic_1.forward(combined_state_, target_actions)
            q2_ = self.target_critic_2.forward(combined_state_, target_actions)

            q1_[done] = 0.0
            q2_[done] = 0.0

            q1_ = q1_.view(-1)
            q2_ = q2_.view(-1)

            critic_value_ = T.min(q1_, q2_)
            target = reward + self.gamma * critic_value_
            target = target.view(self.batch_size, 1)

        q1 = self.critic_1.forward(combined_state, action)
        q2 = self.critic_2.forward(combined_state, action)

        q1_loss = F.mse_loss(q1, target)
        q2_loss = F.mse_loss(q2, target)
        critic_loss = q1_loss + q2_loss

        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()
        critic_loss.backward()
        T.nn.utils.clip_grad_norm_(self.critic_1.parameters(), self.max_grad_norm)
        T.nn.utils.clip_grad_norm_(self.critic_2.parameters(), self.max_grad_norm)
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        predict_next_state = self.system.forward(combined_state, action)
        predict_next_state = predict_next_state.clamp(
            self.obs_lower_bound, self.obs_upper_bound
        )
        system_loss = F.smooth_l1_loss(predict_next_state, combined_state_.detach())

        self.system.optimizer.zero_grad()
        system_loss.backward()
        T.nn.utils.clip_grad_norm_(self.system.parameters(), self.max_grad_norm)
        self.system.optimizer.step()
        self.system_loss = system_loss.item()

        predict_reward = self.reward(combined_state, combined_state_, action)
        reward_loss = F.mse_loss(predict_reward.view(-1), reward.detach())
        self.reward.optimizer.zero_grad()
        reward_loss.backward()
        T.nn.utils.clip_grad_norm_(self.reward.parameters(), self.max_grad_norm)
        self.reward.optimizer.step()
        self.reward_loss = reward_loss.item()

        s_flag = 1 if system_loss.item() < self.sys_threshold else 0

        if self.learn_step_cntr % self.update_actor_interval != 0:
            return critic_loss, None, system_loss, reward_loss

        actor_q1_loss = self.critic_1.forward(combined_state, self.actor.forward(combined_state))
        actor_loss = -T.mean(actor_q1_loss)

        if s_flag:
            predict_next_state = self.system.forward(combined_state, self.actor.forward(combined_state))
            predict_next_state = T.clamp(
                predict_next_state, self.obs_lower_bound, self.obs_upper_bound
            )
            actions2 = self.actor.forward(predict_next_state.detach())

            # skipping to "TD3_FORK"
            predict_next_reward = self.reward.forward(
                combined_state, predict_next_state.detach(), self.actor.forward(combined_state)
            )
            predict_next_state2 = self.system.forward(predict_next_state, actions2)
            predict_next_state2 = T.clamp(
                predict_next_state2, self.obs_lower_bound, self.obs_upper_bound
            )
            predict_next_reward2 = self.reward(
                predict_next_state.detach(), predict_next_state2.detach(), actions2
            )
            actions3 = self.actor.forward(predict_next_state2.detach())

            actor_loss2 = self.critic_1.forward(predict_next_state2.detach(), actions3)
            actor_loss3 = (
                    predict_next_reward
                    + self.gamma * predict_next_reward2
                    + self.gamma ** 2 * actor_loss2
            )
            actor_loss3 = -T.mean(actor_loss3)

            actor_loss = actor_loss + self.sys_weight * actor_loss3

        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()
        self.system.optimizer.zero_grad()
        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        T.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.actor.optimizer.step()

        for param, target_param in zip(
                self.critic_1.parameters(), self.target_critic_1.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )
        for param, target_param in zip(
                self.critic_2.parameters(), self.target_critic_2.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )
        for param, target_param in zip(
                self.actor.parameters(), self.target_actor.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )
        return critic_loss, actor_loss, system_loss, reward_loss

    def save_models(self):
        print("...saving checkpoint...")
        T.save(
            {
                "actor": self.actor.state_dict(),
                "target_actor": self.target_actor.state_dict(),
                "critic_1": self.critic_1.state_dict(),
                "critic_2": self.critic_2.state_dict(),
                "target_critic_1": self.target_critic_1.state_dict(),
                "target_critic_2": self.target_critic_2.state_dict(),
                "actor_optimizer": self.actor.optimizer.state_dict(),
                "target_actor_optimizer": self.target_actor.optimizer.state_dict(),
                "critic_1_optimizer": self.critic_1.optimizer.state_dict(),
                "critic_2_optimizer": self.critic_2.optimizer.state_dict(),
                "target_critic_1_optimizer": self.target_critic_1.optimizer.state_dict(),
                "target_critic_2_optimizer": self.target_critic_2.optimizer.state_dict(),
                "system": self.system.state_dict(),
                "system_optimizer": self.system.optimizer.state_dict(),
                "reward": self.reward.state_dict(),
                "reward_optimizer": self.reward.optimizer.state_dict(),
                "timestep": self.time_step,
            },
            self.chkpt_file_pth,
        )
        self.memory.save(self.buffer_file_pth)

    def load_models(self):
        print("...loading checkpoint...")
        checkpoint = T.load(self.chkpt_file_pth, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.target_actor.load_state_dict(checkpoint["target_actor"])
        self.critic_1.load_state_dict(checkpoint["critic_1"])
        self.critic_2.load_state_dict(checkpoint["critic_2"])
        self.target_critic_1.load_state_dict(checkpoint["target_critic_1"])
        self.target_critic_2.load_state_dict(checkpoint["target_critic_2"])
        self.system.load_state_dict(checkpoint["system"])
        self.reward.load_state_dict(checkpoint["reward"])
        self.actor.optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.target_actor.optimizer.load_state_dict(
            checkpoint["target_actor_optimizer"]
        )
        self.critic_1.optimizer.load_state_dict(checkpoint["critic_1_optimizer"])
        self.critic_2.optimizer.load_state_dict(checkpoint["critic_2_optimizer"])
        self.target_critic_1.optimizer.load_state_dict(
            checkpoint["target_critic_1_optimizer"]
        )
        self.target_critic_2.optimizer.load_state_dict(
            checkpoint["target_critic_2_optimizer"]
        )
        self.system.optimizer.load_state_dict(checkpoint["system_optimizer"])
        self.reward.optimizer.load_state_dict(checkpoint["reward_optimizer"])
        self.time_step = checkpoint["timestep"]

    def save_actor_model_txt(self):
        print("...saving actor into csv...")
        fc1_w = self.actor.fc1.weight.detach().numpy()
        fc1_b = self.actor.fc1.bias.detach().numpy()
        fc2_w = self.actor.fc2.weight.detach().numpy()
        fc2_b = self.actor.fc2.bias.detach().numpy()
        mu_w = self.actor.mu.weight.detach().numpy()
        mu_b = self.actor.mu.bias.detach().numpy()
        ln1_w = self.actor.ln1.weight.detach().numpy()
        ln1_b = self.actor.ln1.bias.detach().numpy()
        ln2_w = self.actor.ln2.weight.detach().numpy()
        ln2_b = self.actor.ln2.bias.detach().numpy()
        files = [fc1_w, fc1_b, fc2_w, fc2_b, mu_w, mu_b, ln1_w, ln1_b, ln2_w, ln2_b]
        filenames = [
            "models/fc1_weights.csv",
            "models/fc1_biases.csv",
            "models/fc2_weights.csv",
            "models/fc2_biases.csv",
            "models/mu_weights.csv",
            "models/mu_biases.csv",
            "models/ln1_weights.csv",
            "models/ln1_biases.csv",
            "models/ln2_weights.csv",
            "models/ln2_biases.csv",
        ]
        for name, file in zip(filenames, files):
            np.savetxt(name, file, delimiter=",")
        if self.actor_file_zip in os.listdir(self.chkpt_dir):
            os.remove(self.actor_zipfile_pth)
        with zipfile.ZipFile(self.actor_zipfile_pth, "w") as zipf:
            os.chdir(self.chkpt_dir)
            for file in filenames:
                zipf.write(file[7:])  # write without "models/"
        zipf.close()
        os.chdir("..")
        for file in filenames:
            os.remove(file)
