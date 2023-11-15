import random

import torch
import torch.nn as nn
import torch.optim as optim


class DRLAgent(nn.Module):
    def __init__(self, input_shape, num_actions, learning_rate=1e-4, gamma=0.99):
        super(DRLAgent, self).__init__()
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.gamma = gamma

        # Define the policy network
        self.policy_network = nn.Sequential(
            nn.Linear(input_shape, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, state):
        return self.policy_network(state)

    def select_action(self, state, epsilon=0.1):
        if random.random() < epsilon:
            return random.randint(0, self.num_actions - 1)
        else:
            state_tensor = torch.tensor(state, dtype=torch.float).unsqueeze(0)
            action_probs = self.forward(state_tensor)
            return torch.argmax(action_probs).item()

    def update_policy(self, transitions, icm):
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*transitions)

        # state_batch = torch.tensor(state_batch, dtype=torch.float)
        # action_batch = torch.tensor(action_batch, dtype=torch.long)
        # reward_batch = torch.tensor(reward_batch, dtype=torch.float)
        # next_state_batch = torch.tensor(next_state_batch, dtype=torch.float)
        # done_batch = torch.tensor(done_batch, dtype=torch.float)
        state_batch = torch.stack([torch.tensor(s, dtype=torch.float) for s in state_batch])
        action_batch = torch.stack([torch.tensor(s, dtype=torch.float) for s in action_batch])
        reward_batch = torch.stack([torch.tensor(s, dtype=torch.float) for s in reward_batch])
        next_state_batch = torch.stack([torch.tensor(s, dtype=torch.float) for s in next_state_batch])
        done_batch = torch.stack([torch.tensor(s, dtype=torch.float) for s in done_batch])

        # Calculate intrinsic rewards using ICM for each transition
        intrinsic_rewards = []
        for state, action, next_state in zip(state_batch, action_batch, next_state_batch):
            state_features, next_state_features, _, forward_features_pred = icm(state, next_state, action)
            intrinsic_reward = icm.calculate_intrinsic_reward(next_state_features, forward_features_pred)
            intrinsic_rewards.append(intrinsic_reward)

        intrinsic_rewards = torch.tensor(intrinsic_rewards, dtype=torch.float)

        # Combine intrinsic and extrinsic rewards
        total_rewards = reward_batch + intrinsic_rewards

        # Compute Q values
        current_q_values = self.forward(state_batch).gather(1, action_batch.unsqueeze(-1)).squeeze(-1)
        next_q_values = self.forward(next_state_batch).max(1)[0]
        expected_q_values = total_rewards + self.gamma * next_q_values * (1 - done_batch)

        # Compute loss
        loss = nn.MSELoss()(current_q_values, expected_q_values.detach())

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
