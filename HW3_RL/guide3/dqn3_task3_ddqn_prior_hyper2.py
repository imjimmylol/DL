import os
import random
from collections import deque

import cv2
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import yaml
from types import SimpleNamespace

# register ALE environments
import ale_py
gym.register_envs(ale_py)


def load_config(path="config.yaml"):
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
    return SimpleNamespace(**cfg)


# ---------------- Prioritized Replay Buffer ----------------
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, eps=1e-6):
        self.capacity = capacity
        self.alpha = alpha
        self.eps = eps
        self.pos = 0
        self.size = 0

        self.buffer = [None] * capacity
        self.priorities = np.zeros((capacity,), dtype=np.float32)

    def add(self, state, action, reward, next_state, done):
        max_prio = self.priorities.max() if self.size > 0 else 1.0
        self.buffer[self.pos] = (state, action, reward, next_state, done)
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, beta=0.4):
        if self.size == 0:
            raise ValueError("Buffer is empty")
        prios = self.priorities[:self.size] ** self.alpha
        probs = prios / prios.sum()

        indices = np.random.choice(self.size, batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        total = self.size
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = torch.tensor(weights, dtype=torch.float32, device=self._device)

        states, actions, rewards, next_states, dones = zip(*samples)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
            weights,
            indices
        )

    def update_priorities(self, indices, td_errors):
        for idx, err in zip(indices, td_errors):
            self.priorities[idx] = abs(err) + self.eps


# ---------------- DQN and Preprocessor ----------------
class DQN(nn.Module):
    def __init__(self, input_channels, num_actions):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self, x):
        return self.network(x / 255.0)


class AtariPreprocessor:
    def __init__(self, frame_stack=4):
        self.frame_stack = frame_stack
        self.frames = deque(maxlen=frame_stack)

    def preprocess(self, obs):
        if obs.ndim == 3 and obs.shape[2] == 3:
            gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        else:
            gray = obs
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return resized

    def reset(self, obs):
        frame = self.preprocess(obs)
        self.frames = deque([frame] * self.frame_stack, maxlen=self.frame_stack)
        return np.stack(self.frames, axis=0)

    def step(self, obs):
        frame = self.preprocess(obs)
        self.frames.append(frame)
        return np.stack(self.frames, axis=0)


# ---------------- DQN Agent with PER ----------------
class DQNAgent:
    def __init__(self, env_name="ALE/Pong-v5", args=None):
        # environments
        self.env = gym.make(env_name, render_mode="rgb_array")
        self.test_env = gym.make(env_name, render_mode="rgb_array")
        self.num_actions = self.env.action_space.n

        # prioritized buffer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        PrioritizedReplayBuffer._device = self.device
        self.replay_buffer = PrioritizedReplayBuffer(
            args.memory_size,
            alpha=args.per_alpha
        )
        self.beta_start  = args.per_beta_start
        self.beta_frames = args.per_beta_frames

        # preprocessor & nets
        self.preprocessor = AtariPreprocessor()
        self.q_net = DQN(input_channels=4, num_actions=self.num_actions).to(self.device)
        self.target_net = DQN(input_channels=4, num_actions=self.num_actions).to(self.device)
        self.q_net.apply(self.init_weights)
        self.target_net.load_state_dict(self.q_net.state_dict())

        # optimizer & hyperparams
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=args.lr)
        self.batch_size = args.batch_size
        self.gamma = args.discount_factor
        self.epsilon = args.epsilon_start
        self.epsilon_decay = args.epsilon_decay
        self.epsilon_min = args.epsilon_min

        # counters & logging
        self.env_count = 0
        self.train_count = 0
        self.best_reward = -float('inf')
        self.max_episode_steps      = args.max_episode_steps
        self.replay_start_size      = args.replay_start_size
        self.target_update_frequency = args.target_update_frequency
        self.train_per_step         = args.train_per_step
        self.save_dir               = args.save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.num_actions)
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state_tensor)
        return q_values.argmax().item()

    def run(self, episodes=10000):
        for ep in range(episodes):
            obs, _ = self.env.reset()
            state = self.preprocessor.reset(obs)
            done = False
            total_reward = 0
            step_count = 0

            while not done and step_count < self.max_episode_steps:
                action = self.select_action(state)
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                next_state = self.preprocessor.step(next_obs)
                self.replay_buffer.add(state, action, reward, next_state, done)

                for _ in range(self.train_per_step):
                    self.train()

                state = next_state
                total_reward += reward
                self.env_count += 1
                step_count += 1

                if self.env_count % 1000 == 0:
                    print(f"[Collect] Ep: {ep} Step: {step_count} SC: {self.env_count} UC: {self.train_count} Eps: {self.epsilon:.4f}")
                    wandb.log({"Episode": ep,
                               "Env Step Count": self.env_count,
                               "Update Count": self.train_count,
                               "Epsilon": self.epsilon})

            print(f"[Eval] Ep: {ep} Total Reward: {total_reward} SC: {self.env_count} UC: {self.train_count} Eps: {self.epsilon:.4f}")
            wandb.log({"Episode": ep,
                       "Total Reward": total_reward,
                       "Env Step Count": self.env_count,
                       "Update Count": self.train_count,
                       "Epsilon": self.epsilon})

            if ep % 100 == 0:
                path = os.path.join(self.save_dir, f"model_ep{ep}.pt")
                torch.save(self.q_net.state_dict(), path)
                print(f"Saved checkpoint to {path}")

            if ep % 20 == 0:
                eval_reward = self.evaluate()
                if eval_reward > self.best_reward:
                    self.best_reward = eval_reward
                    best_path = os.path.join(self.save_dir, "best_model.pt")
                    torch.save(self.q_net.state_dict(), best_path)
                    print(f"New best model with {eval_reward} saved to {best_path}")
                print(f"[TrueEval] Ep: {ep} Eval Reward: {eval_reward:.2f} SC: {self.env_count} UC: {self.train_count}")
                wandb.log({"Eval Reward": eval_reward,
                           "Env Step Count": self.env_count,
                           "Update Count": self.train_count})

    def evaluate(self):
        obs, _ = self.test_env.reset()
        state = self.preprocessor.reset(obs)
        done = False
        total_reward = 0
        while not done:
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            with torch.no_grad():
                action = self.q_net(state_tensor).argmax().item()
            next_obs, reward, terminated, truncated, _ = self.test_env.step(action)
            done = terminated or truncated
            total_reward += reward
            state = self.preprocessor.step(next_obs)
        return total_reward

    def train(self):
        if self.replay_buffer.size < self.replay_start_size:
            return
        # decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        self.train_count += 1

        # anneal beta
        frac = min(self.train_count / self.beta_frames, 1.0)
        beta = self.beta_start + frac * (1.0 - self.beta_start)

        # sample
        (states, actions, rewards, next_states, dones,
         weights, indices) = self.replay_buffer.sample(self.batch_size, beta)

        # to tensors
        states      = torch.from_numpy(states).to(self.device)
        next_states = torch.from_numpy(next_states).to(self.device)
        actions     = torch.tensor(actions,    device=self.device)
        rewards     = torch.tensor(rewards,    device=self.device)
        dones       = torch.tensor(dones,      device=self.device)

        # Q and target
        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_actions = self.q_net(next_states).argmax(1)
            next_q       = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q     = rewards + self.gamma * next_q * (1 - dones)

        # weighted MSE loss
        losses = (q_values - target_q).pow(2) * weights
        loss   = losses.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # update priorities
        td_errors = (q_values - target_q).abs().detach().cpu().numpy()
        self.replay_buffer.update_priorities(indices, td_errors)

        # sync target
        if self.train_count % self.target_update_frequency == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
        if self.train_count % 1000 == 0:
            print(f"[Train #{self.train_count}] Loss: {loss.item():.4f} Q mean: {q_values.mean().item():.3f} std: {q_values.std().item():.3f}")


if __name__ == "__main__":
    config = load_config("./trainingcfg/task3_ddqn_prior_hyper2.yaml")
    wandb.init(project="Final-Task3-prior-hyper", name=config.wandb_run_name, save_code=True)
    agent = DQNAgent(env_name="ALE/Pong-v5", args=config)
    agent.run(episodes=10000)
