import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import gymnasium as gym
import cv2
import ale_py
import os
from collections import deque
import wandb
import yaml
from types import SimpleNamespace

gym.register_envs(ale_py)

def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class PrioritizedReplayBuffer:
    """
    Proportional prioritized replay buffer with importance sampling.
    """
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=1e-6):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.pos = 0
        self.size = 0
        self.data = []
        self.priorities = []

    def add(self, state, action, reward, next_state, done):
        max_prio = max(self.priorities) if self.priorities else 1.0
        if self.size < self.capacity:
            self.data.append((state, action, reward, next_state, done))
            self.priorities.append(max_prio)
            self.size += 1
        else:
            self.data[self.pos] = (state, action, reward, next_state, done)
            self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def __len__(self):
        return self.size

    def sample(self, batch_size):
        if self.size == 0:
            raise ValueError("No samples in buffer.")
        prios = np.array(self.priorities[:self.size], dtype=np.float32)
        probs = prios ** self.alpha
        probs /= probs.sum()
        indices = np.random.choice(self.size, batch_size, p=probs)
        samples = [self.data[i] for i in indices]
        # increase beta toward 1
        self.beta = min(1.0, self.beta + self.beta_increment)
        # compute IS weights
        weights = (self.size * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        states, actions, rewards, next_states, dones = zip(*samples)
        return (
            np.stack(states),
            actions,
            rewards,
            np.stack(next_states),
            dones,
            indices,
            weights
        )

    def update_priorities(self, indices, priorities):
        for idx, prio in zip(indices, priorities):
            self.priorities[idx] = prio


class DuelingDQN(nn.Module):
    def __init__(self, input_channels, num_actions):
        super().__init__()
        # shared conv layers
        self.feature = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        # compute flatten dim
        dummy = torch.zeros(1, input_channels, 84, 84)
        n_flatten = self.feature(dummy).shape[1]
        # value stream
        self.value_stream = nn.Sequential(
            nn.Linear(n_flatten, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
        # advantage stream
        self.adv_stream = nn.Sequential(
            nn.Linear(n_flatten, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self, x):
        x = x / 255.0
        feats = self.feature(x)
        V = self.value_stream(feats)
        A = self.adv_stream(feats)
        Q = V + A - A.mean(dim=1, keepdim=True)
        return Q


class AtariPreprocessor:
    def __init__(self, frame_stack=4):
        self.frame_stack = frame_stack
        self.frames = deque(maxlen=frame_stack)

    def preprocess(self, obs):
        if obs.ndim == 3 and obs.shape[2] == 3:
            gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        else:
            gray = obs
        return cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)

    def reset(self, obs):
        frame = self.preprocess(obs)
        self.frames = deque([frame] * self.frame_stack, maxlen=self.frame_stack)
        return np.stack(self.frames, axis=0)

    def step(self, obs):
        frame = self.preprocess(obs)
        self.frames.append(frame)
        return np.stack(self.frames, axis=0)


class DQNAgent:
    def __init__(self, env_name, args):
        self.env = gym.make(env_name, render_mode="rgb_array")
        self.test_env = gym.make(env_name, render_mode="rgb_array")
        self.num_actions = self.env.action_space.n
        self.preprocessor = AtariPreprocessor(frame_stack=4)

        # hyperparams
        self.gamma = args.discount_factor
        self.n_step = getattr(args, 'n_step', 1)
        self.n_step_buffer = deque(maxlen=self.n_step)
        self.batch_size = args.batch_size
        self.epsilon = args.epsilon_start
        self.epsilon_decay = args.epsilon_decay
        self.epsilon_min = args.epsilon_min
        self.replay_start_size = args.replay_start_size
        self.train_per_step = args.train_per_step
        self.target_update_freq = args.target_update_frequency
        self.max_episode_steps = args.max_episode_steps
        
        # prioritized replay
        alpha = getattr(args, 'per_alpha', 0.6)
        beta_start = getattr(args, 'per_beta_start', 0.4)
        beta_frames = getattr(args, 'per_beta_frames', 100000)
        beta_inc = (1.0 - beta_start) / beta_frames
        self.memory = PrioritizedReplayBuffer(args.memory_size, alpha, beta_start, beta_inc)

        # networks
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)
        self.q_net = DuelingDQN(4, self.num_actions).to(self.device)
        self.q_net.apply(init_weights)
        self.target_net = DuelingDQN(4, self.num_actions).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=args.lr)

        # logging & save
        self.env_steps = 0
        self.train_steps = 0
        self.best_reward = -float('inf')
        self.save_dir = args.save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.num_actions)
        s = torch.from_numpy(state).unsqueeze(0).to(self.device).float()
        with torch.no_grad():
            return self.q_net(s).argmax().item()

    def _add_n_step(self, transition):
        self.n_step_buffer.append(transition)
        if len(self.n_step_buffer) < self.n_step:
            return
        R, next_s, done_n = 0, None, False
        for idx, (_, _, r, ns, d) in enumerate(self.n_step_buffer):
            R += (self.gamma ** idx) * r
            next_s = ns
            if d:
                done_n = True
                break
        s0, a0, _, _, _ = self.n_step_buffer[0]
        self.memory.add(s0, a0, R, next_s, done_n)

    def train(self):
        if len(self.memory) < max(self.replay_start_size, self.batch_size):
            return
        # decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.train_steps += 1
        # sample
        states, actions, rewards, next_states, dones, idxs, weights = self.memory.sample(self.batch_size)
        states = torch.from_numpy(states).to(self.device).float()
        next_states = torch.from_numpy(next_states).to(self.device).float()
        actions = torch.tensor(actions).to(self.device)
        rewards = torch.tensor(rewards).to(self.device)
        dones = torch.tensor(dones, dtype=torch.bool, device=self.device)
        is_weights = torch.tensor(weights).to(self.device)
        # current Q
        q_vals = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            not_done = (~dones).float()   # invert bool mask, then cast to float
            target_q  = rewards + (self.gamma**self.n_step) * next_q * not_done
        td_errors = target_q - q_vals
        loss = (is_weights * td_errors.pow(2)).mean()
        # optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # update priorities
        new_prios = td_errors.detach().abs().cpu().numpy() + 1e-6
        self.memory.update_priorities(idxs, new_prios)
        # sync target
        if self.train_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

    def run(self, episodes=1000):
        for ep in range(1, episodes+1):
            obs, _ = self.env.reset()
            state = self.preprocessor.reset(obs)
            done = False
            ep_reward = 0
            steps = 0
            while not done and steps < self.max_episode_steps:
                action = self.select_action(state)
                next_obs, reward, term, trunc, _ = self.env.step(action)
                done = term or trunc
                # reuse same preprocessor so frames accumulate
                next_state = self.preprocessor.step(next_obs)
                self._add_n_step((state, action, reward, next_state, done))
                for _ in range(self.train_per_step):
                    self.train()
                state = next_state
                ep_reward += reward
                self.env_steps += 1
                steps += 1
            # logs and checkpoint
            wandb.log({"episode": ep, "reward": ep_reward, "env_steps": self.env_steps, "epsilon": self.epsilon})
            if ep_reward > self.best_reward:
                self.best_reward = ep_reward
                torch.save(self.q_net.state_dict(), os.path.join(self.save_dir, 'best.pt'))
            if ep % 100 == 0:
                torch.save(self.q_net.state_dict(), os.path.join(self.save_dir, f'model_ep{ep}.pt'))

    def evaluate(self):
        obs, _ = self.test_env.reset()
        state = AtariPreprocessor().reset(obs)
        done = False
        tot = 0
        while not done:
            action = self.select_action(state)
            next_obs, reward, term, trunc, _ = self.test_env.step(action)
            done = term or trunc
            state = AtariPreprocessor().step(next_obs)
            tot += reward
        return tot

if __name__ == "__main__":
    def load_config(path):
        with open(path, 'r') as f:
            return SimpleNamespace(**yaml.safe_load(f))

    config = load_config('trainingcfg/task3_mstep_prior_dueling_hyper2.yaml')
    wandb.init(project='Final-Task3-boosteffic', name=config.wandb_run_name, save_code=True)
    agent = DQNAgent(env_name='ALE/Pong-v5', args=config)
    agent.run(episodes=10000)
