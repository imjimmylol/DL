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
    Prioritized replay with built-in n-step returns.
    Stores n-step transitions once the internal deque is full.
    """
    def __init__(self, capacity, alpha=0.6, beta=0.4, n_step=3, gamma=0.99):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.n_step = n_step

        # main storage
        self.buffer    = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos       = 0

        # for building n-step returns
        self.n_step_buffer = deque(maxlen=n_step)

    def _get_n_step_data(self):
        """Compute (R^{(n)}, next_state, done) from the n-step buffer."""
        R, next_state, done = 0.0, None, False
        for idx, (_, _, r, ns, d) in enumerate(self.n_step_buffer):
            R += (self.gamma**idx) * r
            if d:
                next_state = ns
                done = True
                break
            next_state = ns
        return R, next_state, done

    def add(self, transition, error):
        """
        transition: (state, action, reward, next_state, done)
        error:    initial TD-error estimate to seed the priority (e.g. max priority)
        """
        # push the raw 1-step transition
        self.n_step_buffer.append(transition)

        # only once we have n steps do we store a bundled n-step transition
        if len(self.n_step_buffer) < self.n_step:
            return

        # build the n-step transition
        state, action, _, _, _ = self.n_step_buffer[0]
        R_n, next_state, done_n = self._get_n_step_data()
        n_transition = (state, action, R_n, next_state, done_n)

        # priority = |error| + ε
        epsilon = 1e-6
        prio = abs(error) + epsilon

        if len(self.buffer) < self.capacity:
            self.buffer.append(n_transition)
        else:
            self.buffer[self.pos] = n_transition

        self.priorities[self.pos] = prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        # only use filled portion of priorities
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]

        probs = prios ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[i] for i in indices]

        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        return samples, indices, weights

    def update_priorities(self, indices, errors):
        epsilon = 1e-6
        for idx, err in zip(indices, errors):
            self.priorities[idx] = abs(err) + epsilon
class DQNAgent:
    def __init__(self, env_name="CartPole-v1", args=None):
        self.env = gym.make(env_name, render_mode="rgb_array")
        self.test_env = gym.make(env_name, render_mode="rgb_array")
        self.num_actions = self.env.action_space.n
        self.replay_buffer = PrioritizedReplayBuffer(
            capacity=args.memory_size,
            alpha=0.6,
            beta=0.4,
            n_step=args.n_step,       # e.g. 3 or 5
            gamma=args.discount_factor
        )
        self.n_step = args.n_step
        self.preprocessor = AtariPreprocessor()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)


        self.q_net = DQN(input_channels=4, num_actions=self.num_actions).to(self.device)
        self.q_net.apply(init_weights)
        self.target_net = DQN(input_channels=4, num_actions=self.num_actions).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=args.lr)

        self.batch_size = args.batch_size
        self.gamma = args.discount_factor
        self.epsilon = args.epsilon_start
        self.epsilon_decay = args.epsilon_decay
        self.epsilon_min = args.epsilon_min

        self.env_count = 0
        self.train_count = 0
        self.best_reward = -21  # Initilized to 0 for CartPole and to -21 for Pong
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
                state = self.preprocessor.reset(obs)
                done = False
                total_reward = 0
                step_count = 0
                while not done and step_count < self.max_episode_steps:
                    # 1) choose action
                    action = self.select_action(state)

                    # 2) step the env
                    next_obs, reward, terminated, truncated, _ = self.env.step(action)
                    done = terminated or truncated
                    next_state = self.preprocessor.step(next_obs)

                    # 3) compute initial priority for this new transition
                    #    (use max existing priority or 1.0 if buffer empty)
                    if len(self.replay_buffer.buffer) > 0:
                        max_prio = self.replay_buffer.priorities.max()
                    else:
                        max_prio = 1.0

                    # 4) add into prioritized replay (internally handles n-step)
                    transition = (state, action, reward, next_state, done)
                    self.replay_buffer.add(transition, error=max_prio)

                    # 5) optionally train
                    for _ in range(self.train_per_step):
                        self.train()

                    # 6) advance
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
                        ########## YOUR CODE HERE  ##########
                        # Add additional wandb logs for debugging if needed 
                        
                        ########## END OF YOUR CODE ##########   
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
                
    def evaluate(self):
        obs, _ = self.test_env.reset()
        state = self.preprocessor.reset(obs)
        done = False
        total_reward = 0

        while not done:
            state_tensor = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(self.device)
            with torch.no_grad():
                # print(state_tensor.shape)
                action = self.q_net(state_tensor).argmax().item()
            next_obs, reward, terminated, truncated, _ = self.test_env.step(action)
            done = terminated or truncated
            total_reward += reward
            state = self.preprocessor.step(next_obs)

        return total_reward


    def train(self):

        if len(self.replay_buffer.buffer) < self.replay_start_size:
            return 
        
        # Decay function for epsilin-greedy exploration
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        self.train_count += 1
       
        ########## YOUR CODE HERE (<5 lines) ##########
        # Sample a mini-batch of (s,a,r,s',done) from the replay buffer
        samples, indices, weights = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*samples)

      
            
        ########## END OF YOUR CODE ##########

        # Convert the states, actions, rewards, next_states, and dones into torch tensors
        # NOTE: Enable this part after you finish the mini-batch sampling
        states = torch.from_numpy(np.array(states)).float().to(self.device)
        next_states = torch.from_numpy(np.array(next_states)).float().to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)
        weights = torch.tensor(weights, dtype=torch.float32).to(self.device)
        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        ########## YOUR CODE HERE (~10 lines) ##########
        # Implement the loss function of DQN and the gradient updates 
        with torch.no_grad():
            # print(next_states.shape)
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
        
      
        ########## END OF YOUR CODE ##########  

        if self.train_count % self.target_update_frequency == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        # NOTE: Enable this part if "loss" is defined
        if self.train_count % 1000 == 0:
            wandb.log({
                "train/q_mean": q_values.mean().item(),
                "train/train_step": self.train_count,
                "train/env_step": self.env_count,
                "train/loss": loss.item()
            })
            print(f"[Train #{self.train_count}] Loss: {loss.item():.4f} Q mean: {q_values.mean().item():.3f} std: {q_values.std().item():.3f}")

if __name__ == "__main__":

    def load_config(path="config.yaml"):
        with open(path, 'r') as f:
            cfg = yaml.safe_load(f)
        return SimpleNamespace(**cfg)

    config = load_config("./trainingcfg/task2.yaml")

    wandb.init(project="RL-Task3-compare-Pong", name=config.wandb_run_name, save_code=True)

    agent = DQNAgent(args=config, env_name="ALE/Pong-v5")
    agent.run(episodes=10000)
