import time
import random
import argparse
import os
from engine.game import Game
from engine.mcts import MCTS
from engine.render import GameRenderer

def main():
    parser = argparse.ArgumentParser(description='Bomberman with MCTS AI')
    parser.add_argument('--players', type=int, default=4, choices=[2, 3, 4],
                        help='Number of players (2-4)')
    parser.add_argument('--human', type=int, default=1, choices=[0, 1],
                        help='Number of human players (0-1)')
    parser.add_argument('--simulations', type=int, default=500,
                        help='Number of MCTS simulations per move')
    parser.add_argument('--render', action='store_true',
                        help='Render the game')
    parser.add_argument('--model_dir', type=str, default='models',
                        help='Directory containing trained models')
    parser.add_argument('--use_trained', action='store_true',
                        help='Use trained models for AI players')
    parser.add_argument('--fast', action='store_true',
                        help='Use fast mode for quicker gameplay')
    parser.add_argument('--max_depth', type=int, default=10,
                        help='Maximum search depth for MCTS')
    args = parser.parse_args()
    
    # Create a new game instance with smaller grid (9x11)
    game = Game(num_players=args.players)
    
    # Create MCTS AI for computer players
    ai_players = []
    for i in range(args.human, args.players):
        if args.use_trained and os.path.exists(f"{args.model_dir}/player{i}_latest.pkl"):
            print(f"Loading trained model for AI Player {i+1}")
            ai_players.append(MCTS.load(f"{args.model_dir}/player{i}_latest.pkl"))
        else:
            print(f"Creating new model for AI Player {i+1}")
            ai_players.append(MCTS(player_id=i, num_simulations=args.simulations, max_depth=args.max_depth))
    
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
            # Get move from human player (via UI in renderer)
            action = renderer.get_human_action()
        else:
            # Get move from appropriate AI
            ai_index = current_player - args.human
            # Use fast mode if specified
            action = ai_players[ai_index].select_action(game, fast_mode=args.fast)
            
        # Apply the selected action
        game.apply_action(action)
    
    # Game over - show results
    if renderer:
        renderer.render()
        renderer.render_game_over(game.get_rankings())
    else:
        print("Game Over!")
        print("Final Rankings:", game.get_rankings())
        
        # Display more detailed results
        print("\nGame Statistics:")
        for player_id in range(game.num_players):
            survived = "Yes" if game.player_alive[player_id] else "No"
            print(f"Player {player_id + 1}: Boxes Destroyed={game.boxes_destroyed[player_id]}, "
                  f"Bomb Range={game.player_bomb_ranges[player_id]}, "
                  f"Bomb Count={game.player_bomb_counts[player_id]}, "
                  f"Survived={survived}")

if __name__ == "__main__":
    main()
