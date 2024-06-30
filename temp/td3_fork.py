import numpy as np
import torch as T
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import copy
import pickle
import os
import zipfile


class ReplayBuffer:
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape))
        self.new_state_memory = np.zeros((self.mem_size, *input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = state_
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, states_, dones

    def save(self, file_pth):
        print("...saving memory...")
        memory = (
            self.state_memory,
            self.new_state_memory,
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
            self.state_memory,
            self.new_state_memory,
            self.action_memory,
            self.terminal_memory,
            self.reward_memory,
            self.mem_cntr,
        ) = result


class RewardNetwork(nn.Module):
    def __init__(
        self, beta, input_dims, fc1_dims, fc2_dims, n_actions, device, ln=False
    ):
        super(RewardNetwork, self).__init__()

        self.fc1 = nn.Linear(2 * input_dims[0] + n_actions, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.fc3 = nn.Linear(fc2_dims, 1)

        self.ln1 = None
        self.ln2 = None
        if ln:
            self.ln1 = nn.LayerNorm(fc1_dims)
            self.ln2 = nn.LayerNorm(fc2_dims)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.to(device)

    def forward(self, state, state_, action):
        sa = T.cat([state, state_, action], dim=1)
        q1 = F.relu(self.fc1(sa))
        if self.ln1:
            q1 = self.ln1(q1)
        q1 = F.relu(self.fc2(q1))
        if self.ln2:
            q1 = self.ln2(q1)
        q1 = self.fc3(q1)
        return q1


class SystemNetwork(nn.Module):
    def __init__(
        self, beta, input_dims, fc1_dims, fc2_dims, n_actions, device, ln=False
    ):
        super(SystemNetwork, self).__init__()

        self.fc1 = nn.Linear(input_dims[0] + n_actions, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.fc3 = nn.Linear(fc2_dims, input_dims[0])

        self.ln1 = None
        self.ln2 = None
        if ln:
            self.ln1 = nn.LayerNorm(fc1_dims)
            self.ln2 = nn.LayerNorm(fc2_dims)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.to(device)

    def forward(self, state, action):
        """Build a system model to predict the next state at a given state."""
        xa = T.cat([state, action], dim=1)

        x1 = F.relu(self.fc1(xa))
        if self.ln1:
            x1 = self.ln1(x1)
        x1 = F.relu(self.fc2(x1))
        if self.ln2:
            x1 = self.ln2(x1)
        x1 = self.fc3(x1)
        return x1


class CriticNetwork(nn.Module):
    def __init__(
        self,
        beta,
        input_dims,
        fc1_dims,
        fc2_dims,
        n_actions,
        device,
        ln=False,
    ):
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

        self.fc1 = nn.Linear(self.input_dims[0] + n_actions, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.q1 = nn.Linear(self.fc2_dims, 1)

        self.ln1 = None
        self.ln2 = None
        if ln:
            self.ln1 = nn.LayerNorm(self.fc1_dims)
            self.ln2 = nn.LayerNorm(self.fc2_dims)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.to(device)

    def forward(self, state, action):
        q1_action_value = self.fc1(T.cat([state, action], dim=1))
        q1_action_value = F.relu(q1_action_value)
        if self.ln1:
            q1_action_value = self.ln1(q1_action_value)

        q1_action_value = self.fc2(q1_action_value)
        q1_action_value = F.relu(q1_action_value)
        if self.ln2:
            q1_action_value = self.ln2(q1_action_value)

        q1 = self.q1(q1_action_value)
        return q1


class ActorNetwork(nn.Module):
    def __init__(
        self,
        alpha,
        input_dims,
        fc1_dims,
        fc2_dims,
        n_actions,
        device,
        optimizer=True,
        ln=False,
    ):
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.mu = nn.Linear(self.fc2_dims, self.n_actions)

        self.ln1 = None
        self.ln2 = None
        if ln:
            self.ln1 = nn.LayerNorm(self.fc1_dims)
            self.ln2 = nn.LayerNorm(self.fc2_dims)

        if optimizer:
            self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.to(device)

    def forward(self, state):
        a = F.relu(self.fc1(state))
        if self.ln1:
            a = self.ln1(a)

        a = F.relu(self.fc2(a))
        if self.ln2:
            a = self.ln2(a)

        mu = T.tanh(self.mu(a))
        return mu


class AgentActor:
    def __init__(
        self,
        env,
        layer1_size=16,
        layer2_size=16,
        chkpt_dir="./models",
        memory_dir="./memory",
    ):
        n_actions = env.action_space.shape[0]
        input_dims = env.observation_space.shape
        self.max_action = env.action_space.high
        self.chkpt_dir = chkpt_dir
        self.memory_dir = memory_dir
        self.chkpt_file_name = "td3_fork.chkpt"
        self.actor_file_name = "td3_fork_actor.chkpt"
        self.actor_file_zip = "td3_fork_actor.zip"
        self.memory_file_name = "td3_fork_memory.pkl"
        self.memory_file_zip = "td3_fork_memory.zip"
        self.buffer_file_name = "td3_fork_buffer.pkl"
        self.actor_file_pth = os.path.join(chkpt_dir, self.actor_file_name).replace(
            "\\", "/"
        )
        self.actor_zipfile_pth = os.path.join(chkpt_dir, self.actor_file_zip).replace(
            "\\", "/"
        )
        self.chkpt_file_pth = os.path.join(chkpt_dir, self.chkpt_file_name).replace(
            "\\", "/"
        )
        self.memory_file_pth = os.path.join(memory_dir, self.memory_file_name).replace(
            "\\", "/"
        )
        self.memory_zipfile_pth = os.path.join(
            memory_dir, self.memory_file_zip
        ).replace("\\", "/")
        self.buffer_file_pth = os.path.join(memory_dir, self.buffer_file_name).replace(
            "\\", "/"
        )
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.actor = ActorNetwork(
            0.001,
            input_dims,
            layer1_size,
            layer2_size,
            n_actions=n_actions,
            device=self.device,
            optimizer=False,
        )

    def choose_action(self, observation):
        state = T.tensor(observation, dtype=T.float).to(self.device)
        mu = self.actor.forward(state).to(self.device)
        return mu.cpu().detach().numpy() * self.max_action[0]
        # print(mu.cpu().detach().numpy(), mu.cpu().detach().numpy() * 500. + 500.)
        # return mu.cpu().detach().numpy() * 500. + 500.

    def save_model(self):
        print("...saving actor...")
        data = {"actor": self.actor.state_dict()}
        T.save(data, self.actor_file_pth)

    def load_model(self):
        print("...loading actor...")
        checkpoint = T.load(self.actor_file_pth, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])


class Agent(AgentActor):
    def __init__(
        self,
        env,
        lr=0.001,
        tau=0.005,
        gamma=0.98,
        update_actor_interval=1,
        max_size=1_000_000,
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
        sys_threshold=0.02,
        ln=False,
        save_buffer=False,
        load_buffer=False,
        chkpt_dir="./models",
    ):
        super().__init__(env, chkpt_dir=chkpt_dir)
        layer1_size, layer2_size = self.actor.fc1_dims, self.actor.fc2_dims
        n_actions = env.action_space.shape[0]
        input_dims = env.observation_space.shape
        self.ln = ln
        self.gamma = gamma
        self.tau = tau
        self.min_action = env.action_space.low
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.learn_step_cntr = 0
        self.n_actions = n_actions
        self.update_actor_iter = update_actor_interval
        self.noise = noise
        self.policy_noise = policy_noise
        self.system_loss = 0
        self.reward_loss = 0
        self.sys_weight = sys_weight
        self.sys_threshold = sys_threshold
        self.save_buffer = save_buffer
        self.load_buffer = load_buffer
        self.obs_upper_bound = T.tensor(env.observation_space.high).to(self.device)
        self.obs_lower_bound = T.tensor(env.observation_space.low).to(self.device)

        self.actor = ActorNetwork(
            lr,
            input_dims,
            layer1_size,
            layer2_size,
            n_actions=n_actions,
            device=self.device,
            ln=ln,
        )
        self.critic_1 = CriticNetwork(
            lr,
            input_dims,
            critic1_size,
            critic2_size,
            n_actions=n_actions,
            device=self.device,
            ln=ln,
        )
        self.critic_2 = CriticNetwork(
            lr,
            input_dims,
            critic1_size,
            critic2_size,
            n_actions=n_actions,
            device=self.device,
            ln=ln,
        )
        self.target_actor = copy.deepcopy(self.actor)
        self.target_critic_1 = copy.deepcopy(self.critic_1)
        self.target_critic_2 = copy.deepcopy(self.critic_2)
        self.system = SystemNetwork(
            lr,
            input_dims,
            sys1_size,
            sys2_size,
            n_actions=n_actions,
            device=self.device,
            ln=ln,
        )
        self.system.apply(self.init_weights)

        self.reward = RewardNetwork(
            lr,
            input_dims,
            r1_size,
            r2_size,
            n_actions=n_actions,
            device=self.device,
            ln=ln,
        )

    def init_weights(self, m):
        if type(m) == nn.Linear:
            T.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.001)

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return None, None, None, None

        self.learn_step_cntr += 1

        state, action, reward, new_state, done = self.memory.sample_buffer(
            self.batch_size
        )
        reward = T.tensor(reward, dtype=T.float).to(self.device)
        done = T.tensor(done).to(self.device)
        state_ = T.tensor(new_state, dtype=T.float).to(self.device)
        state = T.tensor(state, dtype=T.float).to(self.device)
        action = T.tensor(action, dtype=T.float).to(self.device)

        with T.no_grad():
            noise = T.clamp(T.randn_like(action) * self.policy_noise, -0.5, 0.5)
            target_actions = T.clamp(
                self.target_actor(state_) + noise,
                self.min_action[0],
                self.max_action[0],
            )

            q1_ = self.target_critic_1.forward(state_, target_actions)
            q2_ = self.target_critic_2.forward(state_, target_actions)

            q1_[done] = 0.0
            q2_[done] = 0.0

            q1_ = q1_.view(-1)
            q2_ = q2_.view(-1)

            critic_value_ = T.min(q1_, q2_)
            target = reward + self.gamma * critic_value_
            target = target.view(self.batch_size, 1)

        q1 = self.critic_1.forward(state, action)
        q2 = self.critic_2.forward(state, action)

        q1_loss = F.mse_loss(q1, target)
        q2_loss = F.mse_loss(q2, target)
        critic_loss = q1_loss + q2_loss

        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()
        critic_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        predict_next_state = self.system.forward(state, action)
        predict_next_state = predict_next_state.clamp(
            self.obs_lower_bound, self.obs_upper_bound
        )
        system_loss = F.smooth_l1_loss(predict_next_state, state_.detach())

        self.system.optimizer.zero_grad()
        system_loss.backward()
        self.system.optimizer.step()
        self.system_loss = system_loss.item()

        predict_reward = self.reward(state, state_, action)
        reward_loss = F.mse_loss(predict_reward.view(-1), reward.detach())
        self.reward.optimizer.zero_grad()
        reward_loss.backward()
        self.reward.optimizer.step()
        self.reward_loss = reward_loss.item()

        s_flag = 1 if system_loss.item() < self.sys_threshold else 0

        if self.learn_step_cntr % self.update_actor_iter != 0:
            return critic_loss, None, system_loss, reward_loss

        actor_q1_loss = self.critic_1.forward(state, self.actor.forward(state))
        actor_loss = -T.mean(actor_q1_loss)

        if s_flag:
            predict_next_state = self.system.forward(state, self.actor.forward(state))
            predict_next_state = T.clamp(
                predict_next_state, self.obs_lower_bound, self.obs_upper_bound
            )
            actions2 = self.actor.forward(predict_next_state.detach())

            # skipping to "TD3_FORK"
            predict_next_reward = self.reward.forward(
                state, predict_next_state.detach(), self.actor.forward(state)
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
                + self.gamma**2 * actor_loss2
            )
            actor_loss3 = -T.mean(actor_loss3)

            actor_loss = actor_loss + self.sys_weight * actor_loss3

        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()
        self.system.optimizer.zero_grad()
        self.actor.optimizer.zero_grad()
        actor_loss.backward()
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

    def freeze_layers(self):
        self.actor.fc1.requires_grad_(False)
        self.target_actor.fc1.requires_grad_(False)
        self.actor.fc2.requires_grad_(False)
        self.target_actor.fc2.requires_grad_(False)
        self.actor.mu.requires_grad_(False)
        self.target_actor.mu.requires_grad_(False)

        # self.critic_1.fc1.requires_grad_(False)
        # self.critic_2.fc1.requires_grad_(False)
        # self.target_critic_1.fc1.requires_grad_(False)
        # self.target_critic_2.fc1.requires_grad_(False)
        # self.system.fc1.requires_grad_(False)
        # self.reward.fc1.requires_grad_(False)
        #
        # self.critic_1.fc2.requires_grad_(False)
        # self.critic_2.fc2.requires_grad_(False)
        # self.target_critic_1.fc2.requires_grad_(False)
        # self.target_critic_2.fc2.requires_grad_(False)
        # self.system.fc2.requires_grad_(False)
        # self.reward.fc2.requires_grad_(False)
        #
        # self.critic_1.q1.requires_grad_(False)
        # self.critic_2.q1.requires_grad_(False)
        # self.target_critic_1.q1.requires_grad_(False)
        # self.target_critic_2.q1.requires_grad_(False)
        # self.system.fc3.requires_grad_(False)
        # self.reward.fc3.requires_grad_(False)

    def save_models(self):
        print("...saving checkpoint...")
        data = {
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
        }
        if self.save_buffer:
            self.memory.save(self.buffer_file_pth)
        T.save(data, self.chkpt_file_pth)

    def save_model_txt(self):
        print("...saving actor...")
        fc1_w = self.actor.fc1.weight.detach().numpy()
        fc1_b = self.actor.fc1.bias.detach().numpy()
        fc2_w = self.actor.fc2.weight.detach().numpy()
        fc2_b = self.actor.fc2.bias.detach().numpy()
        mu_w = self.actor.mu.weight.detach().numpy()
        mu_b = self.actor.mu.bias.detach().numpy()
        files = [fc1_w, fc1_b, fc2_w, fc2_b, mu_w, mu_b]
        filenames = [
            "models/fc1_weights.csv",
            "models/fc1_biases.csv",
            "models/fc2_weights.csv",
            "models/fc2_biases.csv",
            "models/mu_weights.csv",
            "models/mu_biases.csv",
        ]
        if self.ln:
            ln1_w = self.actor.ln1.weight.detach().numpy()
            ln1_b = self.actor.ln1.bias.detach().numpy()
            ln2_w = self.actor.ln2.weight.detach().numpy()
            ln2_b = self.actor.ln2.bias.detach().numpy()
            # default layer norm epsilon value is 1e-5
            files.append(ln1_w)
            files.append(ln1_b)
            files.append(ln2_w)
            files.append(ln2_b)
            filenames.append("models/ln1_weights.csv")
            filenames.append("models/ln1_biases.csv")
            filenames.append("models/ln2_weights.csv")
            filenames.append("models/ln2_biases.csv")
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

    def load_models(self, reset=False, freeze=False):
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
        if not reset:
            print("...loading optimizer weights...")
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
        if freeze:
            print("...freezing layers...")
            self.freeze_layers()
        if self.load_buffer:
            print("...loading last saved buffer...")
            self.memory.load(self.buffer_file_pth)

    def preload_buffer(self):
        print("...preloading buffer...")
        self.memory.load(
            os.path.join(self.memory_dir, "buffer_warmup.pkl").replace("\\", "/")
        )
