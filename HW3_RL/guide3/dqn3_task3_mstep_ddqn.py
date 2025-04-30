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
    """
        Preprocessing the state input of DQN for Atari
    """
    def __init__(self, frame_stack=4):
        self.frame_stack = frame_stack
        self.frames = deque(maxlen=frame_stack)

    def preprocess(self, obs):
        if len(obs.shape) == 3 and obs.shape[2] == 3:
            gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        else:
            gray = obs
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return resized

    def reset(self, obs):
        frame = self.preprocess(obs)
        self.frames = deque([frame for _ in range(self.frame_stack)], maxlen=self.frame_stack)
        return np.stack(self.frames, axis=0)

    def step(self, obs):
        frame = self.preprocess(obs)
        self.frames.append(frame)
        return np.stack(self.frames, axis=0)

class DQNAgent:
    def __init__(self, env_name="CartPole-v1", args=None):
        # Environment setup
        self.env = gym.make(env_name, render_mode="rgb_array")
        self.test_env = gym.make(env_name, render_mode="rgb_array")
        self.num_actions = self.env.action_space.n

        # Multi-step return setup
        self.gamma = args.discount_factor
        self.n_step = getattr(args, 'n_step', 3)  # use `n_step` from config or default to 3
        self.n_step_buffer = deque(maxlen=self.n_step)

        # Replay buffer
        self.replay_buffer = deque(maxlen=args.memory_size)

        # Preprocessor
        self.preprocessor = AtariPreprocessor()

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)

        # Networks
        self.q_net = DQN(input_channels=4, num_actions=self.num_actions).to(self.device)
        self.q_net.apply(init_weights)
        self.target_net = DQN(input_channels=4, num_actions=self.num_actions).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=args.lr)

        # Hyperparameters
        self.batch_size = args.batch_size
        self.epsilon = args.epsilon_start
        self.epsilon_decay = args.epsilon_decay
        self.epsilon_min = args.epsilon_min
        self.env_count = 0
        self.train_count = 0
        self.best_reward = -21  # for Atari Pong
        self.max_episode_steps = args.max_episode_steps
        self.replay_start_size = args.replay_start_size
        self.target_update_frequency = args.target_update_frequency
        self.train_per_step = args.train_per_step

        # Checkpointing
        self.save_dir = args.save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        state_tensor = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state_tensor)
        return q_values.argmax().item()

    def _add_n_step(self, transition):
        """
        Add single transition to n-step buffer, and once full,
        compute the n-step return and store into replay_buffer.
        """
        self.n_step_buffer.append(transition)
        if len(self.n_step_buffer) < self.n_step:
            return

        # compute n-step return
        R, next_state, done_n = 0.0, None, False
        for idx, (_, _, r, n_s, d) in enumerate(self.n_step_buffer):
            R += (self.gamma ** idx) * r
            next_state = n_s
            if d:
                done_n = True
                break

        state_0, action_0, *_ = self.n_step_buffer[0]
        # push aggregated transition
        self.replay_buffer.append((state_0, action_0, R, next_state, done_n))

    def run(self, episodes=1000):
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

                # use multi-step buffer
                self._add_n_step((state, action, reward, next_state, done))

                # training steps
                for _ in range(self.train_per_step):
                    self.train()

                state = next_state
                total_reward += reward
                self.env_count += 1
                step_count += 1

                # periodic logging
                if self.env_count % 1000 == 0:
                    # print(f"[Collect] Ep: {ep} Step: {step_count} SC: {self.env_count} UC: {self.train_count} Eps: {self.epsilon:.4f}")
                    wandb.log({
                        "Episode": ep,
                        "Step Count": step_count,
                        "Env Step Count": self.env_count,
                        "Update Count": self.train_count,
                        "Epsilon": self.epsilon
                    })

            # end of episode logging & checkpoint
            # print(f"[Eval] Ep: {ep} Total Reward: {total_reward} SC: {self.env_count} UC: {self.train_count} Eps: {self.epsilon:.4f}")
            wandb.log({
                "Episode": ep,
                "Total Reward": total_reward,
                "Env Step Count": self.env_count,
                "Update Count": self.train_count,
                "Epsilon": self.epsilon
            })

            if ep % 100 == 0:
                chk = os.path.join(self.save_dir, f"model_ep{ep}.pt")
                torch.save(self.q_net.state_dict(), chk)
                print(f"Saved model checkpoint to {chk}")

            if ep % 20 == 0:
                eval_reward = self.evaluate()
                if eval_reward > self.best_reward:
                    self.best_reward = eval_reward
                    best_chk = os.path.join(self.save_dir, "best_model.pt")
                    torch.save(self.q_net.state_dict(), best_chk)
                    # print(f"Saved new best model to {best_chk} with reward {eval_reward}")
                # print(f"[TrueEval] Ep: {ep} Eval Reward: {eval_reward:.2f} SC: {self.env_count} UC: {self.train_count}")
                wandb.log({
                    "Env Step Count": self.env_count,
                    "Update Count": self.train_count,
                    "Eval Reward": eval_reward
                })

    def evaluate(self):
        obs, _ = self.test_env.reset()
        state = self.preprocessor.reset(obs)
        done = False
        total_reward = 0

        while not done:
            state_tensor = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(self.device)
            with torch.no_grad():
                action = self.q_net(state_tensor).argmax().item()
            next_obs, reward, terminated, truncated, _ = self.test_env.step(action)
            done = terminated or truncated
            total_reward += reward
            state = self.preprocessor.step(next_obs)

        return total_reward

    def train(self):
        # wait until enough data
        if len(self.replay_buffer) < max(self.replay_start_size, self.batch_size):
            return

        # epsilon decay & counter
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        self.train_count += 1

        # sample batch
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # to tensors
        states      = torch.from_numpy(np.array(states)).float().to(self.device)
        next_states = torch.from_numpy(np.array(next_states)).float().to(self.device)
        actions     = torch.tensor(actions, dtype=torch.int64).to(self.device)
        rewards     = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones       = torch.tensor(dones,   dtype=torch.float32).to(self.device)

        # current Q
        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # multi-step + Double DQN bootstrap target
        with torch.no_grad():
            # 1) select best next-action with online net
            next_actions = self.q_net(next_states).argmax(dim=1)
            # 2) evaluate that action with target net
            next_q = self.target_net(next_states) \
                          .gather(1, next_actions.unsqueeze(1)) \
                          .squeeze(1)

            gamma_n = self.gamma ** self.n_step
            target_q = rewards + gamma_n * next_q * (1 - dones)

        # compute loss & backprop
        loss = nn.MSELoss()(q_values, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # sync target network periodically
        if self.train_count % self.target_update_frequency == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        # optional logs
        if self.train_count % 1000 == 0:
            print(f"[Train #{self.train_count}] "
                  f"Loss: {loss.item():.4f}  "
                  f"Q mean: {q_values.mean().item():.3f}  "
                  f"std: {q_values.std().item():.3f}")

if __name__ == "__main__":
    def load_config(path="config.yaml"):
        with open(path, 'r') as f:
            cfg = yaml.safe_load(f)
        return SimpleNamespace(**cfg)

    config = load_config("./trainingcfg/task3_mstep_ddqn.yaml")
    # make sure your YAML includes: n_step: 3  (or your desired multi-step length)

    wandb.init(project="Final-Task3", name=config.wandb_run_name, save_code=True)

    agent = DQNAgent(args=config, env_name="ALE/Pong-v5")
    agent.run(episodes=10000)
