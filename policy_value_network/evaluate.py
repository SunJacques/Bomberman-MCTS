"""
Evaluate policy-value network against PureMCTS agent
"""


import os
import torch
import multiprocessing
from functools import partial
from tqdm import tqdm
from engine.game import Game
from mcts.pure_mcts import PureMCTS
from mcts.train_mcts import MCTS
from policy_value_network.utils import BombermanCNN


def evaluate_single_game(game_idx, model_path, player_id, num_simulations_alphazero=100, num_simulations_pure=100):
    """Evaluate single game against PureMCTS with model playing as specified player"""
    model = BombermanCNN().to('cpu')  
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    game = Game(num_players=2, width=11, height=9)
    while not game.is_terminal():
        current_player = game.current_player_id
        if current_player == player_id:  
            mcts_agent = MCTS(model=model,
                           player_id=current_player,
                           num_simulations=num_simulations_alphazero)
            action = mcts_agent.select_action(game, temperature=0)  
        else:  
            mcts_agent = PureMCTS(player_id=current_player,
                               num_simulations=num_simulations_pure)
            action = mcts_agent.select_action(game)
        game.apply_action(action)

    rankings = game.get_rankings()
    won = rankings[0] == player_id
    game_length = game.turn
    return won, game_length


def evaluate_against_pure_mcts(model, num_games=10, num_simulations_alphazero=100, num_simulations_pure=100):
    """
    Evaluate in parallel current model against PureMCTS 
    """
    print("\nEvaluating against Pure MCTS...")

    temp_model_path = "temp_eval_model.pth"
    torch.save(model.state_dict(), temp_model_path)

    num_processes = multiprocessing.cpu_count()
    print(f"Using {num_processes} CPU cores for evaluation")

    eval_as_player1 = partial(
        evaluate_single_game,
        model_path=temp_model_path,
        player_id=0,
        num_simulations_alphazero=num_simulations_alphazero,
        num_simulations_pure=num_simulations_pure
    )

    eval_as_player2 = partial(
        evaluate_single_game,
        model_path=temp_model_path,
        player_id=1,
        num_simulations_alphazero=num_simulations_alphazero,
        num_simulations_pure=num_simulations_pure
    )

    # Run evaluations in parallel
    with multiprocessing.Pool(processes=num_processes) as pool:
        results_player1 = list(tqdm(
            pool.imap(eval_as_player1, range(num_games)),
            total=num_games,
            desc="Model as Player 1"
        ))

        results_player2 = list(tqdm(
            pool.imap(eval_as_player2, range(num_games)),
            total=num_games,
            desc="Model as Player 2"
        ))

    # Clean up temporary model file
    if os.path.exists(temp_model_path):
        os.remove(temp_model_path)

    # Process results
    wins_as_player1 = sum(result[0] for result in results_player1)
    wins_as_player2 = sum(result[0] for result in results_player2)

    # Gather game lengths for completed games
    game_lengths = [result[1] for result in results_player1 + results_player2]

    # Calculate win rate
    total_games = num_games * 2
    total_wins = wins_as_player1 + wins_as_player2
    win_rate = total_wins / total_games

    # Calculate average game length
    avg_game_length = sum(game_lengths) / len(game_lengths) if game_lengths else 0

    print(f"Win rate: {win_rate:.2f} ({total_wins}/{total_games})")
    print(f"Wins as Player 1: {wins_as_player1}/{num_games}")
    print(f"Wins as Player 2: {wins_as_player2}/{num_games}")
    print(f"Average game length: {avg_game_length:.1f} turns")

    return win_rate, avg_game_length