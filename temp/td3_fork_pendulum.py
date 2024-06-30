import numpy as np
import copy
import torch as T
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import os
import pickle


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
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims, n_actions, ln=False):
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
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")

        self.to(self.device)

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
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims, n_actions, ln=False):
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
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")

        self.to(self.device)

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
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims, n_actions, name, ln=False):
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name

        self.fc1 = nn.Linear(self.input_dims[0] + n_actions, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.q1 = nn.Linear(self.fc2_dims, 1)

        self.ln1 = None
        self.ln2 = None
        if ln:
            self.ln1 = nn.LayerNorm(self.fc1_dims)
            self.ln2 = nn.LayerNorm(self.fc2_dims)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")

        self.to(self.device)

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
        self, alpha, input_dims, fc1_dims, fc2_dims, n_actions, name, ln=False
    ):
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.mu = nn.Linear(self.fc2_dims, self.n_actions)

        self.ln1 = None
        self.ln2 = None
        if ln:
            self.ln1 = nn.LayerNorm(self.fc1_dims)
            self.ln2 = nn.LayerNorm(self.fc2_dims)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")

        self.to(self.device)

    def forward(self, state):
        a = F.relu(self.fc1(state))
        if self.ln1:
            a = self.ln1(a)

        a = F.relu(self.fc2(a))
        if self.ln2:
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
        input_dims,
        tau,
        env,
        gamma=0.99,
        update_actor_interval=2,
        warmup=10_000,
        n_actions=2,
        max_size=1_000_000,
        layer1_size=256,
        layer2_size=256,
        critic1_size=256,
        critic2_size=256,
        sys1_size=400,
        sys2_size=300,
        r1_size=256,
        r2_size=256,
        batch_size=100,
        noise=0.2,
        policy_noise=0.2,
        sys_weight=0.5,
        sys_weight2=0.4,
        sys_threshold=0.020,
        ln=False,
        chkpt_dir="./tmp/td3 fork",
        game_id="Pendulum-v2",
    ):
        # IN NEXT IMPLEMENTATION: Add gradient clipping
        self.gamma = gamma
        self.tau = tau
        self.max_action = [
            1.0
        ]  # moved to the environment, everything here is action = [-1, 1]
        self.min_action = [-1.0]
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.learn_step_cntr = 0
        self.time_step = 0
        self.warmup = warmup
        self.n_actions = n_actions
        self.update_actor_iter = update_actor_interval

        self.system_loss = 0
        self.reward_loss = 0

        self.sys_weight = sys_weight
        self.sys_weight2 = sys_weight2
        self.sys_threshold = sys_threshold
        self.chkpt_file_pth = os.path.join(chkpt_dir, f"{game_id} td3 fork.chkpt")
        self.buffer_file_pth = os.path.join(chkpt_dir, f"buffer td3 fork.pkl")

        self.actor = ActorNetwork(
            alpha,
            input_dims,
            layer1_size,
            layer2_size,
            n_actions=n_actions,
            name="actor",
            ln=ln,
        )
        self.critic_1 = CriticNetwork(
            beta,
            input_dims,
            critic1_size,
            critic2_size,
            n_actions=n_actions,
            name="critic_1",
            ln=ln,
        )
        self.critic_2 = CriticNetwork(
            beta,
            input_dims,
            critic1_size,
            critic2_size,
            n_actions=n_actions,
            name="critic_2",
            ln=ln,
        )
        self.target_actor = copy.deepcopy(self.actor)
        self.target_critic_1 = copy.deepcopy(self.critic_1)
        self.target_critic_2 = copy.deepcopy(self.critic_2)

        self.system = SystemNetwork(
            beta, input_dims, sys1_size, sys2_size, n_actions=n_actions, ln=ln
        )
        self.system.apply(self.init_weights)

        self.reward = RewardNetwork(
            beta, input_dims, r1_size, r2_size, n_actions=n_actions, ln=ln
        )

        self.obs_upper_bound = T.tensor([1.1, np.pi, 20.0, 100.0]).to(self.actor.device)
        self.obs_lower_bound = T.tensor([-1.1, -np.pi, -20.0, -100.0]).to(
            self.actor.device
        )
        self.noise = noise
        self.policy_noise = policy_noise

    def init_weights(self, m):
        if type(m) == nn.Linear:
            T.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.001)

    def choose_action(self, observation, evaluate=False):
        if (self.time_step < self.warmup) and not evaluate:
            # mu = T.tensor(np.random.normal(scale=self.noise, size=(self.n_actions,)))
            state = T.tensor(observation, dtype=T.float).to(self.actor.device)
            mu = self.actor.forward(state).to(self.actor.device)
        else:
            state = T.tensor(observation, dtype=T.float).to(self.actor.device)
            mu = self.actor.forward(state).to(self.actor.device)

        mu_prime = mu
        # decided to remove this since my input is bounded in [-1, 1] I'll
        # multiply max action inside environment
        self.time_step += 1
        # should be list if moving to real world now or like basta optional ra ang numpy
        return mu_prime.cpu().detach().numpy()

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def learn(self):
        if (self.memory.mem_cntr < self.batch_size) or (self.time_step < self.warmup):
            return None, None, None, None

        self.learn_step_cntr += 1

        state, action, reward, new_state, done = self.memory.sample_buffer(
            self.batch_size
        )
        reward = T.tensor(reward, dtype=T.float).to(self.critic_1.device)
        done = T.tensor(done).to(self.critic_1.device)
        state_ = T.tensor(new_state, dtype=T.float).to(self.critic_1.device)
        state = T.tensor(state, dtype=T.float).to(self.critic_1.device)
        action = T.tensor(action, dtype=T.float).to(self.critic_1.device)

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

    def freeze_layer(self, first_layer=True, second_layer=False):
        if first_layer:
            print("...freezing first layer...")
            self.actor.fc1.requires_grad_(False)
            self.critic_1.fc1.requires_grad_(False)
            self.critic_2.fc1.requires_grad_(False)
            self.target_actor.fc1.requires_grad_(False)
            self.target_critic_1.fc1.requires_grad_(False)
            self.target_critic_2.fc1.requires_grad_(False)
            self.system.fc1.requires_grad_(False)
            self.reward.fc1.requires_grad_(False)
        if second_layer:
            print("...freezing second layer...")
            self.actor.fc2.requires_grad_(False)
            self.critic_1.fc2.requires_grad_(False)
            self.critic_2.fc2.requires_grad_(False)
            self.target_actor.fc2.requires_grad_(False)
            self.target_critic_1.fc2.requires_grad_(False)
            self.target_critic_2.fc2.requires_grad_(False)
            self.system.fc2.requires_grad_(False)
            self.reward.fc2.requires_grad_(False)

    def load_models(self, load_all_weights=True, load_optimizers=False):
        print("...loading checkpoint...")
        device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        checkpoint = T.load(self.chkpt_file_pth, map_location=device)
        self.actor.load_state_dict(checkpoint["actor"])
        if load_all_weights:
            print("...loading all weights...")
            self.target_actor.load_state_dict(checkpoint["target_actor"])
            self.critic_1.load_state_dict(checkpoint["critic_1"])
            self.critic_2.load_state_dict(checkpoint["critic_2"])
            self.target_critic_1.load_state_dict(checkpoint["target_critic_1"])
            self.target_critic_2.load_state_dict(checkpoint["target_critic_2"])
            self.system.load_state_dict(checkpoint["system"])
            self.reward.load_state_dict(checkpoint["reward"])
        if load_optimizers:
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
        # self.time_step = checkpoint["timestep"]  # orig
        self.time_step = 0

    def partial_load_models(self):
        print("...partial loading checkpoint...")
        device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        checkpoint = T.load(self.chkpt_file_pth, map_location=device)
        self.system.load_state_dict(checkpoint["system"])
        self.system.optimizer.load_state_dict(checkpoint["system_optimizer"])
        self.reward.load_state_dict(checkpoint["reward"])
        self.reward.optimizer.load_state_dict(checkpoint["reward_optimizer"])
