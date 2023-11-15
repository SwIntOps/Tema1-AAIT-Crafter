import torch
import torch.nn as nn
import torch.nn.functional as f
from torch import optim


class ICM(nn.Module):
    def __init__(self, input_shape, num_actions, learning_rate=1e-4):
        super(ICM, self).__init__()
        self.num_actions = num_actions  # Store the number of actions
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_shape, 256),
            nn.ReLU()
        )

        # Inverse model: Predicts action from two consecutive states' features
        self.inverse_model = nn.Sequential(
            nn.Linear(672, 256),  # Adjusted to match the concatenated input size
            nn.ReLU(),
            nn.Linear(256, num_actions)
        )

        # Forward model: Predicts next state's features from current state's features and action
        self.forward_model = nn.Sequential(
            nn.Linear(256 + num_actions, 256),
            nn.ReLU(),
            nn.Linear(256, 256)
        )
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, state, next_state, action):
        state_features = self.feature_extractor(state)
        next_state_features = self.feature_extractor(next_state)

        combined_features = torch.cat([state_features, next_state_features], dim=1)

        print("Combined features shape:", combined_features.shape)

        # Predict action using inverse model
        inverse_action_pred = self.inverse_model(torch.transpose(combined_features, 1, 0))

        # Prepare action input for forward model
        action_one_hot = f.one_hot(action, num_classes=self.num_actions).float()
        forward_input = torch.cat([state_features, action_one_hot], dim=1)

        # Predict next state's features using forward model
        forward_features_pred = self.forward_model(forward_input)

        return state_features, next_state_features, inverse_action_pred, forward_features_pred

    @staticmethod
    def calculate_intrinsic_reward(next_state_features, forward_features_pred):
        # Intrinsic reward based on prediction error of forward model
        reward = f.mse_loss(forward_features_pred, next_state_features.detach(), reduction='none').sum(1)
        return reward

    def update(self, state, next_state, action):
        # Compute predictions
        state_features, next_state_features, predicted_action, forward_features_pred = self.forward(state, next_state,
                                                                                                    action)

        # Calculate losses
        inverse_loss = f.cross_entropy(predicted_action, action)
        forward_loss = f.mse_loss(forward_features_pred, next_state_features.detach())

        # Total loss
        total_loss = inverse_loss + forward_loss

        # Perform backpropagation
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return total_loss.item()
