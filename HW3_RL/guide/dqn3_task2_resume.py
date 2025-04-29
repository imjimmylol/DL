# Spring 2025, 535507 Deep Learning
# Lab5: Value-based RL
# Contributors: Wei Hung and Alison Wen
# Instructor: Ping-Chun Hsieh

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
        Preprocesing the state input of DQN for Atari
    """    
    def __init__(self, frame_stack=4):
        self.frame_stack = frame_stack
        self.frames = deque(maxlen=frame_stack)

    # def preprocess(self, obs):
    #     gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    #     resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    #     return resized

    def preprocess(self, obs):
        if len(obs.shape) == 3 and obs.shape[2] == 3:
            # 如果是 RGB，就轉成灰階
            gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        else:
            # 如果已經是灰階（單通道），直接用
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


class PrioritizedReplayBuffer:
    """
        Prioritizing the samples in the replay memory by the Bellman error
        See the paper (Schaul et al., 2016) at https://arxiv.org/abs/1511.05952
    """ 
    def __init__(self, capacity, alpha=0.6, beta=0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0

    def add(self, transition, error):
        ########## YOUR CODE HERE (for Task 3) ########## 
                    
        ########## END OF YOUR CODE (for Task 3) ########## 
        return 
    def sample(self, batch_size):
        ########## YOUR CODE HERE (for Task 3) ########## 
                    
        ########## END OF YOUR CODE (for Task 3) ########## 
        return
    def update_priorities(self, indices, errors):
        ########## YOUR CODE HERE (for Task 3) ########## 
                    
        ########## END OF YOUR CODE (for Task 3) ########## 
        return
        

class DQNAgent:
    def __init__(self, env_name="CartPole-v1", args=None, load_checkpoint=None):
        self.env = gym.make(env_name, render_mode="rgb_array")
        self.test_env = gym.make(env_name, render_mode="rgb_array")
        self.num_actions = self.env.action_space.n
        self.state_dim = self.env.observation_space.shape[0]
        self.replay_buffer = deque(maxlen=args.memory_size)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)

        self.q_net = DQN(input_channels=4, num_actions=self.num_actions).to(self.device)
        self.q_net.apply(init_weights)
        self.target_net = DQN(input_channels=4, num_actions=self.num_actions).to(self.device)

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=args.lr)


        self.start_episode = 0
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

        if load_checkpoint is not None:
            self.q_net.load_state_dict(torch.load(load_checkpoint, map_location=self.device))
            self.target_net.load_state_dict(torch.load(load_checkpoint, map_location=self.device))
            print(f"Loaded checkpoint from {load_checkpoint}")
        else:
            self.target_net.load_state_dict(self.q_net.state_dict())

    def run(self, episodes=10000):
        for ep in range(self.start_episode, self.start_episode + episodes):
            obs, _ = self.env.reset()
            state = obs
            done = False
            total_reward = 0
            step_count = 0

            while not done and step_count < self.max_episode_steps:
                action = self.select_action(state)
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                next_state = next_obs

                self.replay_buffer.append((state, action, reward, next_state, done))

                for _ in range(self.train_per_step):
                    self.train()

                state = next_state
                total_reward += reward
                self.env_count += 1
                step_count += 1

                if self.env_count % 1000 == 0:
                    print(f"[Collect] Ep: {ep} Step: {step_count} SC: {self.env_count} UC: {self.train_count} Eps: {self.epsilon:.4f}")
                    wandb.log({
                        "Episode": ep,
                        "Step Count": step_count,
                        "Env Step Count": self.env_count,
                        "Update Count": self.train_count,
                        "Epsilon": self.epsilon
                    })
            print(f"[Eval] Ep: {ep} Total Reward: {total_reward} SC: {self.env_count} UC: {self.train_count} Eps: {self.epsilon:.4f}")
            wandb.log({
                "Episode": ep,
                "Total Reward": total_reward,
                "Env Step Count": self.env_count,
                "Update Count": self.train_count,
                "Epsilon": self.epsilon
            })

            if ep % 100 == 0:
                model_path = os.path.join(self.save_dir, f"model_ep{ep}.pt")
                torch.save(self.q_net.state_dict(), model_path)
                print(f"Saved model checkpoint to {model_path}")

            if ep % 20 == 0:
                eval_reward = self.evaluate()
                if eval_reward > self.best_reward:
                    self.best_reward = eval_reward
                    model_path = os.path.join(self.save_dir, "best_model.pt")
                    torch.save(self.q_net.state_dict(), model_path)
                    print(f"Saved new best model to {model_path} with reward {eval_reward}")
                print(f"[TrueEval] Ep: {ep} Eval Reward: {eval_reward:.2f} SC: {self.env_count} UC: {self.train_count}")
                wandb.log({
                    "Env Step Count": self.env_count,
                    "Update Count": self.train_count,
                    "Eval Reward": eval_reward
                })

    def resume(self, checkpoint_path):
        """
        Resume training from a given checkpoint.
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.q_net.load_state_dict(checkpoint['q_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.env_count = checkpoint['env_count']
        self.train_count = checkpoint['train_count']
        self.best_reward = checkpoint['best_reward']
        # Infer start_episode from env_count (approx)
        self.start_episode = checkpoint.get('episode', 0)
        print(f"Resumed training from {checkpoint_path}, starting at episode {self.start_episode}")


    # other methods: select_action(), evaluate(), train() are same





if __name__ == "__main__":
    def load_config(path="config.yaml"):
        with open(path, 'r') as f:
            cfg = yaml.safe_load(f)
        return SimpleNamespace(**cfg)

    config = load_config("./trainingcfg/task2.yaml")

    wandb.init(project="DLP-Lab5-DQN-pong", name=config.wandb_run_name, save_code=True)

    agent = DQNAgent(args=config, env_name="ALE/Pong-v5")

    # Add this to resume from a checkpoint
    resume_path = "./results/pong_attempt2/model_ep1100.pt"
    if os.path.exists(resume_path):
        agent.resume(resume_path)

    agent.run(episodes=10000)

