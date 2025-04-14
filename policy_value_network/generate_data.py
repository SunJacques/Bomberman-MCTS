"""
Generate PureMCTS self-play games for policy-value network pretraining
"""


import numpy as np
import multiprocessing
from functools import partial
from tqdm import tqdm
import os
from mcts.pure_mcts import PureMCTS
from engine.game import Game
from policy_value_network.utils import encode_game_state, action_to_policy


def generate_single_game(game_idx, num_simulations=100, max_depth=75):
    """Generate data from single self-play game"""
    game = Game(num_players=2, width=11, height=9)
    game_data = []

    while not game.is_terminal():
        current_player = game.current_player_id
        mcts_agent = PureMCTS(player_id=current_player,
                              num_simulations=num_simulations,
                              max_depth=max_depth)

        state = encode_game_state(game, current_player)
        legal_actions = game.get_legal_actions()
        action = mcts_agent.select_action(game)
        policy = action_to_policy(action, legal_actions)
        game_data.append((state, policy, current_player))
        game.apply_action(action)

    rankings = game.get_rankings()
    result_data = []
    for state, policy, player_id in game_data:
        value = 1.0 if rankings[0] == player_id else 0.0 if rankings[1] == player_id else 0.5
        result_data.append((state, policy, value))
    return result_data


def generate_self_play_data(num_games=10000, num_simulations=100, max_depth=75):
    """Generate self-play data"""
    num_processes = multiprocessing.cpu_count()
    print(f"Using {num_processes} CPU cores for generating {num_games} games")
    
    game_generator = partial(generate_single_game,
                             num_simulations=num_simulations,
                             max_depth=max_depth)

    all_data = []
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = list(tqdm(pool.imap(game_generator, range(num_games)),
                            total=num_games,
                            desc=f"Generating {num_games} games"))
        for game_data in results:
            all_data.extend(game_data)

    states = np.array([item[0] for item in all_data])
    policies = np.array([item[1] for item in all_data])
    values = np.array([[item[2]] for item in all_data])

    return states, policies, values


def save_self_play_data(states, policies, values, filename):
    """Save self-play data to compressed file"""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    np.savez_compressed(filename, states=states, policies=policies, values=values)
    print(f"Saved self-play data to {filename}")


if __name__ == "__main__":
    total_games = 8000  
    output_dir = "policy_value_network"
    output_filename = os.path.join(output_dir, "self_play_data.npz")
    
    print(f"Generating data from {total_games} self-play games...")
    states, policies, values = generate_self_play_data(
        num_games=total_games,
        num_simulations=100,
        max_depth=75
    )
    
    save_self_play_data(states, policies, values, output_filename)
    print(f"Data generation complete. Generated {len(states)} state-action pairs from {total_games} games.")