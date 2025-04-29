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
import argparse
import time
import yaml
from types import SimpleNamespace
import wandb

def load_config(path="config.yaml"):
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
    return SimpleNamespace(**cfg)


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

class DQN(nn.Module):
    """
        Design the architecture of your deep Q network
        - Input size is the same as the state dimension; the output size is the same as the number of actions
        - Feel free to change the architecture (e.g. number of hidden layers and the width of each hidden layer) as you like
        - Feel free to add any member variables/functions whenever needed
    """
    def __init__(self, state_dim, num_actions):
        super(DQN, self).__init__()
        # An example: 
        self.network = nn.Sequential(
           nn.Linear(state_dim, 64),
           nn.ReLU(),
           nn.Linear(64, 64),
           nn.ReLU(),
           nn.Linear(64, num_actions)
        )       
        ########## YOUR CODE HERE (5~10 lines) ##########

        
        ########## END OF YOUR CODE ##########

    def forward(self, x):
        return self.network(x)

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
        epsilon = 1e-6
        priority = np.abs(error) + epsilon  

        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition

        self.priorities[self.pos] = priority
        self.pos = (self.pos + 1) % self.capacity 
        ########## END OF YOUR CODE (for Task 3) ##########
        return 

    def sample(self, batch_size):
        ########## YOUR CODE HERE (for Task 3) ##########
        if len(self.buffer) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.pos]

        # Apply alpha when computing sampling probabilities
        probs = priorities ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        # Importance Sampling Weights
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max()  # Normalize for stability
        ########## END OF YOUR CODE (for Task 3) ##########
        return samples, indices, weights

    def update_priorities(self, indices, errors):
        ########## YOUR CODE HERE (for Task 3) ##########
        epsilon = 1e-6
        for idx, error in zip(indices, errors):
            priority = np.abs(error) + epsilon  # No alpha here!
            self.priorities[idx] = priority
        ########## END OF YOUR CODE (for Task 3) ##########
        return

class NStepReplayWrapper:
    def __init__(self, base_buffer, n, gamma):
        """
        base_buffer: an instance of PrioritizedReplayBuffer
        n: the number of steps for multi-step return
        gamma: discount factor
        """
        self.base = base_buffer
        self.n = n
        self.gamma = gamma
        self.deck = deque()

    def add(self, transition, td_error=None):
        """
        transition: (s_t, a_t, r_{t+1}, s_{t+1}, done_{t+1})
        td_error: optional; you can pass None and let base_buffer use a default max-priority
        """
        self.deck.append(transition)
        # Once we have n transitions, or if the last one is done, emit an n-step transition:
        if len(self.deck) >= self.n or transition[4]:
            R, discount = 0.0, 1.0
            for (_, _, r, _, _ ) in list(self.deck)[:self.n]:
                R += discount * r
                discount *= self.gamma

            s0, a0, _, _, _ = self.deck[0]
            _, _, _, sn, done_n = self.deck[min(self.n, len(self.deck)) - 1]
            n_step_transition = (s0, a0, R, sn, done_n)

            # use provided td_error if you have it, else None
            self.base.add(n_step_transition, td_error)

            # pop oldest
            self.deck.popleft()

        # if this was a terminal, flush remaining shorter sequences
        if transition[4]:
            while self.deck:
                # same logic as above but deck shorter than n
                R, discount = 0.0, 1.0
                for (_, _, r, _, _ ) in self.deck:
                    R += discount * r
                    discount *= self.gamma
                s0, a0, _, _, _ = self.deck[0]
                _, _, _, sn, done_n = self.deck[-1]
                self.base.add((s0, a0, R, sn, done_n), td_error)
                self.deck.popleft()

    def sample(self, *args, **kwargs):
        # just forward to prioritized buffer
        return self.base.sample(*args, **kwargs)

    def update_priorities(self, *args, **kwargs):
        return self.base.update_priorities(*args, **kwargs)



class DQNAgent:
    def __init__(self, env_name="CartPole-v1", args=None):
        self.env = gym.make(env_name, render_mode="rgb_array")
        self.test_env = gym.make(env_name, render_mode="rgb_array")
        self.num_actions = self.env.action_space.n
        self.state_dim = self.env.observation_space.shape[0]

        # 1) build a prioritized buffer
        prio = PrioritizedReplayBuffer(
            capacity=args.memory_size,
            alpha=0.6,
            beta=0.4
        )

        # 2) wrap it for n-step returns
        self.replay_buffer = NStepReplayWrapper(
            base_buffer=prio,
            n=args.n_step,
            gamma=args.discount_factor
        )


        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)

        self.q_net = DQN(num_actions=self.num_actions, state_dim=self.state_dim).to(self.device)
        self.q_net.apply(init_weights)
        self.target_net = DQN(num_actions=self.num_actions, state_dim=self.state_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=args.lr)

        self.batch_size = args.batch_size
        self.gamma = args.discount_factor
        self.epsilon = args.epsilon_start
        self.epsilon_decay = args.epsilon_decay
        self.epsilon_min = args.epsilon_min
        self.n_step = args.n_step
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

    def run(self, episodes=1000):
        for ep in range(episodes):
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

                # Compute initial error for new transitions
                if len(self.replay_buffer.buffer) > 0:
                    max_priority = self.replay_buffer.priorities.max()
                else:
                    max_priority = 1.0

                transition = (state, action, reward, next_state, done)
                self.replay_buffer.add(transition, error=max_priority)

                for _ in range(self.train_per_step):
                    self.train()

                state = next_state
                total_reward += reward
                self.env_count += 1
                step_count += 1

                if self.env_count % 1000 == 0:                 
                    print(f"[Collect] Ep: {ep} Step: {step_count} EnvStep: {self.env_count} UpdateCount: {self.train_count} Eps: {self.epsilon:.4f}")
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
                wandb.log({
                    "Env Step Count": self.env_count,
                    "Update Count": self.train_count,
                    "Eval Reward": eval_reward
                })

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
        if len(self.replay_buffer.buffer) < self.replay_start_size:
            return 
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        self.train_count += 1

        samples, indices, weights = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*samples)

        states = torch.from_numpy(np.array(states)).float().to(self.device)
        next_states = torch.from_numpy(np.array(next_states)).float().to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)
        weights = torch.tensor(weights, dtype=torch.float32).to(self.device)
        

        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_online = self.q_net(next_states)
            next_actions = next_q_online.argmax(1)
            next_q_target = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q = rewards + (self.gamma ** self.n_step) * next_q_target * (1 - dones)

        loss = (nn.functional.mse_loss(q_values, target_q, reduction='none') * weights).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        td_errors = (q_values - target_q).abs().detach().cpu().numpy()
        self.replay_buffer.update_priorities(indices, td_errors)

        if self.train_count % self.target_update_frequency == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        if self.train_count % 1000 == 0:
            wandb.log({
                "train/q_mean": q_values.mean().item(),
                "train/train_step": self.train_count,
                "train/env_step": self.env_count,
                "train/loss": loss.item()
            })
            print(f"[Train #{self.train_count}] Loss: {loss.item():.4f} Q mean: {q_values.mean().item():.3f} std: {q_values.std().item():.3f}")
if __name__ == "__main__":
    config = load_config("./trainingcfg/task1.yaml")

    wandb.init(project="RL-Task3-compare-cartpole", name=config.wandb_run_name, save_code=True)

    agent = DQNAgent(args=config)
    agent.run(episodes=5000)
