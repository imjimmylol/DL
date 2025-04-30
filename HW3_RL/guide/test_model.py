import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import os
import types  # 用來建立假的 args
import wandb

# 設定初始參數
args = types.SimpleNamespace(
    memory_size=10000,
    lr=1e-3,
    batch_size=64,
    discount_factor=0.99,
    epsilon_start=0,
    epsilon_decay=0.99,
    epsilon_min=0.01,
    max_episode_steps=500,
    replay_start_size=1000,
    target_update_frequency=100,
    train_per_step=1,
    save_dir="./results"
)

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

class DQN(nn.Module):
    def __init__(self, state_dim, num_actions):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
           nn.Linear(state_dim, 64),
           nn.ReLU(),
           nn.Linear(64, 64),
           nn.ReLU(),
           nn.Linear(64, num_actions)
        )       

    def forward(self, x):
        return self.network(x)

class DQNAgent:
    def __init__(self, env_name="CartPole-v1", args=None):
        self.env = gym.make(env_name, render_mode="rgb_array")
        self.test_env = gym.make(env_name, render_mode="rgb_array", max_episode_steps=1500)
        self.num_actions = self.env.action_space.n
        self.state_dim = self.env.observation_space.shape[0]
        self.replay_buffer = deque(maxlen=args.memory_size)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)

        self.q_net = DQN(state_dim=self.state_dim, num_actions=self.num_actions).to(self.device)
        self.q_net.apply(init_weights)
        self.target_net = DQN(state_dim=self.state_dim, num_actions=self.num_actions).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=args.lr)

        self.batch_size = args.batch_size
        self.gamma = args.discount_factor
        self.epsilon = args.epsilon_start
        self.epsilon_decay = args.epsilon_decay
        self.epsilon_min = args.epsilon_min

        self.env_count = 0
        self.train_count = 0
        self.best_reward = 0
        self.max_episode_steps = args.max_episode_steps
        self.replay_start_size = args.replay_start_size
        self.target_update_frequency = args.target_update_frequency
        self.train_per_step = args.train_per_step
        self.save_dir = args.save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        state_tensor = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state_tensor)
        return q_values.argmax().item()

    def evaluate(self):
        obs, _ = self.test_env.reset()
        state = obs
        done = False
        total_reward = 0

        while not done:
            state_tensor = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(self.device)
            with torch.no_grad():
                action = self.q_net(state_tensor).argmax().item()
            next_obs, reward, terminated, truncated, _ = self.test_env.step(action)
            done = terminated or truncated
            total_reward += reward
            state = next_obs

        return total_reward

    def train(self):
        if len(self.replay_buffer) < self.replay_start_size:
            return 
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        self.train_count += 1

        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.from_numpy(np.array(states).astype(np.float32)).to(self.device)
        next_states = torch.from_numpy(np.array(next_states).astype(np.float32)).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)

        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + self.gamma * next_q * (1 - dones)

        loss = nn.MSELoss()(q_values, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.train_count % self.target_update_frequency == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

# --- 測試區（載入模型錄影片） ---

# 1. 建立 agent
agent = DQNAgent(env_name="CartPole-v1", args=args)

# 2. 載入 best_model
state_dict = torch.load("./results/best_model.pt", map_location=agent.device)
agent.q_net.load_state_dict(state_dict)
agent.target_net.load_state_dict(state_dict)  # 讓 target_net 同步

# 3. 讓 agent 不要探索
agent.epsilon = 0  

# 4. 重新建立一個錄影片用的環境
test_env = gym.make('CartPole-v1', render_mode='rgb_array', max_episode_steps=2000)
test_env = RecordVideo(test_env, video_folder="./eval_videos", episode_trigger=lambda x: True)
TOTAL_REWARDS = 0
# 5. 錄三個 episode
for episode in range(5):
    state = test_env.reset()[0]
    total_reward = 0
    steps = 0

    while True:
        action = agent.select_action(state)
        next_state, reward, done, truncated, _ = test_env.step(action)
        total_reward += reward
        state = next_state
        steps += 1

        if done or truncated or steps >= 1500:
            # print(steps)
            print(truncated, steps >= 1500, done)
            break

    print(f"Test Episode: {episode}, Reward: {total_reward}")
    TOTAL_REWARDS += total_reward
print(f"avg total rewards : {round(TOTAL_REWARDS/5, 4)}")
test_env.close()
print("錄製完成，影片保存在 ./eval_videos")
