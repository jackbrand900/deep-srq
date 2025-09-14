import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class DQNAgent:
    def __init__(self, obs_dim, act_dim, lr=1e-3):
        self.q_net = DQN(obs_dim, act_dim)
        self.target_q_net = DQN(obs_dim, act_dim)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.replay = deque(maxlen=10000)
        self.batch_size = 32
        self.gamma = 0.99
        self.epsilon = 1.0

    def act(self, obs):
        if random.random() < self.epsilon:
            return random.randint(0, self.q_net.net[-1].out_features - 1)
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_net(obs_tensor)
        return torch.argmax(q_values).item()

    def store(self, *transition):
        self.replay.append(transition)

    def update(self):
        if len(self.replay) < self.batch_size:
            return

        batch = random.sample(self.replay, self.batch_size)
        obs, act, rew, next_obs, done = zip(*batch)

        obs = torch.tensor(obs, dtype=torch.float32)
        act = torch.tensor(act, dtype=torch.int64).unsqueeze(1)
        rew = torch.tensor(rew, dtype=torch.float32).unsqueeze(1)
        next_obs = torch.tensor(next_obs, dtype=torch.float32)
        done = torch.tensor(done, dtype=torch.float32).unsqueeze(1)

        q_vals = self.q_net(obs).gather(1, act)
        with torch.no_grad():
            max_next_q = self.target_q_net(next_obs).max(1, keepdim=True)[0]
            target = rew + self.gamma * max_next_q * (1 - done)

        loss = nn.functional.mse_loss(q_vals, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def soft_update(self, tau=0.01):
        for target_param, param in zip(self.target_q_net.parameters(), self.q_net.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
