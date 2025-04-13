import time
import random
import argparse
import os
from engine.game import Game
from engine.mcts import PureMCTS
from engine.render import GameRenderer

def main():
    parser = argparse.ArgumentParser(description='Bomberman with MCTS AI')
    parser.add_argument('--players', type=int, default=4, choices=[2, 3, 4],
                        help='Number of players (2-4)')
    parser.add_argument('--human', type=int, default=1, choices=[0, 1],
                        help='Number of human players (0-1)')
    parser.add_argument('--simulations', type=int, default=100,
                        help='Number of MCTS simulations per move')
    parser.add_argument('--render', action='store_true',
                        help='Render the game')
    parser.add_argument('--model_dir', type=str, default='models',
                        help='Directory containing trained models')
    parser.add_argument('--use_trained', action='store_true',
                        help='Use trained models for AI players')
    parser.add_argument('--fast', action='store_true',
                        help='Use fast mode for quicker gameplay')
    parser.add_argument('--max_depth', type=int, default=75,
                        help='Maximum search depth for MCTS')
    
    # Add a check to ensure render is enabled when human players are present
    args = parser.parse_args()
    
    # Force render mode if human players are involved
    if args.human > 0 and not args.render:
        print("Enabling render mode for human player")
        args.render = True
        
    # Create a new game instance with smaller grid (9x11)
    game = Game(num_players=args.players)
    
    # Create MCTS AI for computer players
    ai_players = []
    for i in range(args.human, args.players):
        model_path = f"{args.model_dir}/player{i}_latest.pkl"

        if args.use_trained and os.path.exists(model_path):
            print(f"Loading trained pure MCTS model for AI Player {i+1}")
            try:
                ai_players.append(PureMCTS.load(model_path))
            except Exception as e:
                print(f"Error loading model: {e}")
                print(f"Creating new pure MCTS for AI Player {i+1}")
                ai_players.append(PureMCTS(player_id=i, num_simulations=args.simulations, max_depth=args.max_depth))
        else:
            print(f"Creating pure MCTS for AI Player {i+1}")
            ai_players.append(PureMCTS(player_id=i, num_simulations=args.simulations, max_depth=args.max_depth))
        
    # Create renderer if needed
    renderer = GameRenderer(game) if args.render else None
    
    # Main game loop
    human_player_id = 0 if args.human > 0 else None
    
    while not game.is_terminal():
        if renderer:
            renderer.render()
            time.sleep(0.1)
        
        current_player = game.current_player_id
        
        if current_player == human_player_id:
            action = renderer.get_human_action()
        else:
            ai_index = current_player - args.human
            action = ai_players[ai_index].select_action(game, fast_mode=args.fast)
            
        game.apply_action(action)
    
    # Game over - show results
    if renderer:
        renderer.render()
        renderer.render_game_over(game.get_rankings())
    else:
        print("Game Over!")
        print("Final Rankings:", game.get_rankings())
        print("\nGame Statistics:")
        for player_id in range(game.num_players):
            survived = "Yes" if game.player_alive[player_id] else "No"
            print(f"Player {player_id + 1}: Boxes Destroyed={game.boxes_destroyed[player_id]}, "
                  f"Bomb Range={game.player_bomb_ranges[player_id]}, "
                  f"Bomb Count={game.player_bomb_counts[player_id]}, "
                  f"Survived={survived}")

if __name__ == "__main__":
    main()