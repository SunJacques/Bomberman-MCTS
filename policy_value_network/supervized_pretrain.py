"""
Train policy-value network on PureMCTS self-play data in supervised manner
"""


import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from policy_value_network.utils import BombermanDataset
from policy_value_network.network import BombermanCNN


def train_supervised(model, states, policies, values, batch_size=256, epochs=3, lr=0.005):
    """
    Train neural network on PureMCTS self-play data in supervised manner
    """
    dataset = BombermanDataset(
        torch.FloatTensor(states),
        torch.FloatTensor(policies),
        torch.FloatTensor(values)
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    policy_criterion = nn.CrossEntropyLoss()
    value_criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    history = {
        'loss': [], 
        'policy_loss': [], 
        'value_loss': [],
    }

    for epoch in range(epochs):
        epoch_loss = 0
        epoch_policy_loss = 0
        epoch_value_loss = 0

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch_states, batch_policies, batch_values in progress_bar:
            batch_states = batch_states.to(device)
            batch_policies = batch_policies.to(device)
            batch_values = batch_values.to(device)

            optimizer.zero_grad()
            policy_logits, value_preds = model(batch_states)

            policy_loss = policy_criterion(policy_logits, torch.argmax(batch_policies, dim=1))
            value_loss = value_criterion(value_preds, batch_values)
            loss = policy_loss + value_loss

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_policy_loss += policy_loss.item()
            epoch_value_loss += value_loss.item()

            progress_bar.set_postfix({
                'loss': epoch_loss / (progress_bar.n + 1),
                'policy_loss': epoch_policy_loss / (progress_bar.n + 1),
                'value_loss': epoch_value_loss / (progress_bar.n + 1)
            })

        avg_loss = epoch_loss / len(dataloader)

        history['loss'].append(avg_loss)
        history['policy_loss'].append(epoch_policy_loss / len(dataloader))
        history['value_loss'].append(epoch_value_loss / len(dataloader))
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

    return history


def load_self_play_data(filename="self_play_data.npz"):
    data = np.load(filename)
    return data["states"], data["policies"], data["values"]


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    data_file = "self_play_data.npz"
    save_folder = "policy_value_network/models"
    os.makedirs(save_folder, exist_ok=True)

    # Load training data 
    states, policies, values = load_self_play_data(data_file)
    print(f"Loaded {len(states)} training samples from {data_file}")

    # Initialize model
    model = BombermanCNN().to(device)

    # Train model
    history = train_supervised(model, states, policies, values, epochs=3)

    # Save model
    model_path = os.path.join(save_folder, "pretrained_model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Trained model saved at {model_path}")