"""Policy-value network self-play training à la AlphaZero"""


import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import multiprocessing
from tqdm import tqdm
from torch.utils.data import DataLoader
from engine.game import Game, Action
from mcts.train_mcts import MCTS
from policy_value_network.network import BombermanCNN
from policy_value_network.utils import encode_game_state, BombermanDataset
from policy_value_network.evaluate import evaluate_against_pure_mcts


def self_play_game_parallel(args):
    game_idx, model_path, num_simulations, temperature = args

    device = torch.device('cpu')
    model = BombermanCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    game = Game(num_players=2, width=11, height=9)
    game_history = []

    while not game.is_terminal():
        current_player = game.current_player_id
        mcts_agent = MCTS(model=model, player_id=current_player, num_simulations=num_simulations)

        state = encode_game_state(game, current_player)
        action_probs = mcts_agent.run(game)

        temp = temperature if game.turn < 20 else 0.5
        if temp == 0:
            action_idx = np.argmax(action_probs)
        else:
            action_probs_temp = action_probs ** (1 / temp)
            action_probs_temp /= np.sum(action_probs_temp)
            action_idx = np.random.choice(len(action_probs_temp), p=action_probs_temp)

        action = Action(action_idx)
        game_history.append((state, action_probs, current_player))
        game.apply_action(action)

    rankings = game.get_rankings()
    training_data = []

    for state, policy, player_id in game_history:
        value = 1.0 if rankings[0] == player_id else 0.0 if rankings[1] == player_id else 0.5
        training_data.append((state, policy, value))

    return training_data

def train_alphazero(model, num_iterations=10, games_per_iteration=25,
                   batch_size=64, epochs=5, num_simulations=100, temperature=1.0):

    save_dir = "policy_value_network/models"
    os.makedirs(save_dir, exist_ok=True)

    latest_model_path = os.path.join(save_dir, "latest_model.pth")
    if os.path.exists(latest_model_path):
        model.load_state_dict(torch.load(latest_model_path))
        print(f"Loaded model from {latest_model_path}")
    else:
        print("No latest model found. Training from scratch.")

    device = next(model.parameters()).device
    print(f"Training on device: {device}")

    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    policy_criterion = nn.KLDivLoss(reduction='batchmean')
    value_criterion = nn.MSELoss()

    for iteration in range(num_iterations):
        print(f"\nIteration {iteration+1}/{num_iterations}")

        temp_model_path = os.path.join(save_dir, "temp_training_model.pth")
        torch.save(model.state_dict(), temp_model_path)

        num_processes = min(4, multiprocessing.cpu_count())
        print(f"Using {num_processes} processes for self-play")

        args_list = [(i, temp_model_path, num_simulations, temperature)
                     for i in range(games_per_iteration)]

        training_data = []
        with multiprocessing.Pool(processes=num_processes) as pool:
            results = list(tqdm(
                pool.imap(self_play_game_parallel, args_list),
                total=games_per_iteration,
                desc="Self-play games"
            ))
            for game_data in results:
                training_data.extend(game_data)

        if os.path.exists(temp_model_path):
            os.remove(temp_model_path)

        print(f"Generated {len(training_data)} training examples")

        random.shuffle(training_data)
        states = np.array([item[0] for item in training_data])
        policies = np.array([item[1] for item in training_data])
        values = np.array([[item[2]] for item in training_data])

        dataset = BombermanDataset(
            torch.FloatTensor(states),
            torch.FloatTensor(policies),
            torch.FloatTensor(values)
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        model.train()
        total_loss = 0
        policy_loss_total = 0
        value_loss_total = 0

        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            progress_bar = tqdm(dataloader, desc="Training")

            epoch_total_loss = 0
            epoch_policy_loss = 0
            epoch_value_loss = 0
            batches = 0

            for batch_states, batch_policies, batch_values in progress_bar:
                batch_states = batch_states.to(device)
                batch_policies = batch_policies.to(device)
                batch_values = batch_values.to(device)

                optimizer.zero_grad()
                policy_logits, value_preds = model(batch_states)

                policy_loss = policy_criterion(
                    F.log_softmax(policy_logits, dim=1),
                    batch_policies
                )
                value_loss = value_criterion(value_preds, batch_values)
                loss = policy_loss + value_loss

                loss.backward()
                optimizer.step()

                batches += 1
                epoch_total_loss += loss.item()
                epoch_policy_loss += policy_loss.item()
                epoch_value_loss += value_loss.item()

                progress_bar.set_postfix({
                    'loss': epoch_total_loss / batches,
                    'policy_loss': epoch_policy_loss / batches,
                    'value_loss': epoch_value_loss / batches
                })

            total_loss += epoch_total_loss / batches
            policy_loss_total += epoch_policy_loss / batches
            value_loss_total += epoch_value_loss / batches

            avg_loss = epoch_total_loss / batches
            scheduler.step(avg_loss)

        avg_policy_loss = policy_loss_total / epochs
        avg_value_loss = value_loss_total / epochs
        avg_total_loss = total_loss / epochs

        print(f"Iteration {iteration+1} completed. Avg losses - Policy: {avg_policy_loss:.4f}, Value: {avg_value_loss:.4f}, Total: {avg_total_loss:.4f}")

        torch.save(model.state_dict(), latest_model_path)
        print(f"Updated latest model at {latest_model_path}")

        if (iteration + 1) % 2 == 0:
            evaluate_against_pure_mcts(model, num_games=10)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = BombermanCNN().to(device)
    try:
        model.load_state_dict(torch.load("bomberman_supervised.pth"))
        print("Loaded supervised learning model")
    except:
        print("Could not load supervised model. Starting with a fresh model.")

    train_alphazero(
        model=model,
        num_iterations=50,
        games_per_iteration=20,
        batch_size=64,
        num_simulations=50,
        temperature=1.0
    )