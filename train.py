import argparse
import os
import time
import numpy as np
import pickle
from tqdm import tqdm
import multiprocessing
from engine.game import Game, Action
from mcts.pure_mcts import MCTS
from mcts.pure_mcts import PureMCTS  # Import PureMCTS class
from engine.render import GameRenderer

class TrainingStats:
    """Class to track training statistics."""
    def __init__(self):
        self.games_played = 0
        self.win_counts = {}  # Maps player_id to win count
        self.box_counts = {}  # Maps player_id to average boxes destroyed
        self.survival_rates = {}  # Maps player_id to survival rate
        self.game_lengths = []
        
    def update(self, game):
        """Update stats with a completed game."""
        self.games_played += 1
        
        # Track game length
        self.game_lengths.append(game.turn)
        
        # Get rankings
        rankings = game.get_rankings()
        winner = rankings[0]
        
        # Update win counts
        if winner not in self.win_counts:
            self.win_counts[winner] = 0
        self.win_counts[winner] += 1
        
        # Update box counts and survival rates
        for p_id in range(game.num_players):
            if p_id not in self.box_counts:
                self.box_counts[p_id] = []
            if p_id not in self.survival_rates:
                self.survival_rates[p_id] = []
                
            self.box_counts[p_id].append(game.boxes_destroyed[p_id])
            self.survival_rates[p_id].append(1 if game.player_alive[p_id] else 0)
    
    def get_summary(self):
        """Get summary statistics."""
        summary = {
            'games_played': self.games_played,
            'win_counts': self.win_counts,
            'win_rates': {p_id: count / self.games_played for p_id, count in self.win_counts.items()},
            'avg_boxes': {p_id: np.mean(counts) for p_id, counts in self.box_counts.items()},
            'survival_rates': {p_id: np.mean(rates) for p_id, rates in self.survival_rates.items()},
            'avg_game_length': np.mean(self.game_lengths)
        }
        return summary

def run_single_episode(args):
    """Run a single training episode."""
    episode_num, num_players, render_this_episode, render_speed, mcts_players_config, use_pure_mcts = args
    
    # Create a new game
    game = Game(num_players=num_players)
    
    # Create new MCTS instances for this episode
    mcts_players = []
    for config in mcts_players_config:
        # Determine which MCTS class to use
        mcts_class = PureMCTS if use_pure_mcts else MCTS
        
        # Create a new MCTS instance with the same parameters
        mcts = mcts_class(player_id=config['player_id'], 
                         num_simulations=config['num_simulations'],
                         max_depth=config['max_depth'])
        
        # Set additional parameters based on MCTS type
        if 'exploration_weight' in config:
            mcts.exploration_weight = config['exploration_weight']
            
        # Copy learned parameters if they exist
        if 'heatmap' in config:
            mcts.heatmap = config['heatmap'].copy()
        if 'action_stats' in config:
            mcts.action_stats = config['action_stats'].copy()
            
        mcts_players.append(mcts)
    
    # Create renderer if this is a visualization episode
    renderer = None
    if render_this_episode:
        renderer = GameRenderer(game)
        if hasattr(mcts_players[0], 'heatmap'):
            display_heatmap(mcts_players[0].heatmap, f"Player 1 Position Heatmap (Episode {episode_num})")
    
    # Main game loop
    while not game.is_terminal():
        if renderer:
            renderer.render()
            time.sleep(render_speed)
        
        # Get current player
        current_player = game.current_player_id
        
        # Get move from appropriate AI
        action = mcts_players[current_player].select_action(game)
        
        # Apply the action
        game.apply_action(action)
    
    # Render final state if needed
    if renderer:
        renderer.render()
        renderer.render_game_over(game.get_rankings(), training_mode=True)
    
    # Return game and updated MCTS players
    return game, mcts_players

def train_mcts(num_episodes=1000, num_players=4, render_interval=100, 
               simulations=500, save_interval=50, save_dir='models',
               render_speed=0.05, load_models=False, use_parallel=False, 
               num_processes=None, max_depth=10, use_pure_mcts=False):
    """Train MCTS agents through self-play."""
    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Determine number of processes for parallel training
    if use_parallel:
        if num_processes is None:
            num_processes = max(1, multiprocessing.cpu_count() - 1)
        print(f"Using {num_processes} processes for parallel training")
    
    # Initialize training stats
    stats = TrainingStats()
    
    # Create or load MCTS players
    mcts_players = []
    for i in range(num_players):
        model_path = f"{save_dir}/player{i}_latest.pkl"
        
        if load_models and os.path.exists(model_path):
            print(f"Loading existing model for Player {i+1}")
            if use_pure_mcts:
                mcts_players.append(PureMCTS.load(model_path))
            else:
                mcts_players.append(MCTS.load(model_path))
        else:
            print(f"Creating new model for Player {i+1}")
            # Use a smaller max_depth for faster simulations
            if use_pure_mcts:
                mcts_players.append(PureMCTS(player_id=i, num_simulations=simulations, max_depth=max_depth))
            else:
                mcts_players.append(MCTS(player_id=i, num_simulations=simulations, max_depth=max_depth))
    
    # Create progress bar
    pbar = tqdm(total=num_episodes, desc="Training Progress")
    
    # Prepare for parallel processing if enabled
    if use_parallel:
        pool = multiprocessing.Pool(processes=num_processes)
    
    # Run training episodes
    episode = 0
    while episode < num_episodes:
        # Determine batch size (number of episodes to run in parallel)
        batch_size = min(num_processes if use_parallel else 1, num_episodes - episode)
        
        if use_parallel:
            # Prepare arguments for parallel processing
            args_list = []
            for i in range(batch_size):
                current_episode = episode + i + 1
                render_this_episode = (current_episode % render_interval == 0)
                
                # Create a serializable representation of MCTS players
                mcts_configs = []
                for mcts in mcts_players:
                    config = {
                        'player_id': mcts.player_id,
                        'num_simulations': mcts.num_simulations,
                        'exploration_weight': mcts.exploration_weight,
                        'max_depth': mcts.max_depth,
                        'heatmap': mcts.heatmap.copy(),
                        'action_stats': mcts.action_stats.copy()
                    }
                    mcts_configs.append(config)
                
                args_list.append((current_episode, num_players, render_this_episode, render_speed, mcts_configs, use_pure_mcts))
            
            # Run episodes in parallel
            results = pool.map(run_single_episode, args_list)
            
            # Process results
            for i, (game, updated_mcts) in enumerate(results):
                current_episode = episode + i + 1
                
                # Update statistics
                stats.update(game)
                
                # Merge learned parameters from parallel processes back to main MCTS instances
                for j, mcts in enumerate(updated_mcts):
                    for pos in np.ndindex(mcts.heatmap.shape):
                        # Take the max value for heatmap positions
                        mcts_players[j].heatmap[pos] = max(mcts_players[j].heatmap[pos], mcts.heatmap[pos])
                    
                    # Merge action stats
                    for key, (visits, value) in mcts.action_stats.items():
                        if key in mcts_players[j].action_stats:
                            mcts_players[j].action_stats[key][0] += visits
                            mcts_players[j].action_stats[key][1] += value
                        else:
                            mcts_players[j].action_stats[key] = [visits, value]
                    
                    # Append rewards history
                    mcts_players[j].rewards_history.extend(mcts.rewards_history)
                
                # Save models periodically
                if current_episode % save_interval == 0:
                    save_training_state(current_episode, stats, mcts_players, save_dir, num_episodes)
                
                # Update progress bar
                pbar.update(1)
        else:
            # Run a single episode
            episode += 1
            render_this_episode = (episode % render_interval == 0)
            
            # Run the episode
            args = (episode, num_players, render_this_episode, render_speed, 
                    [{
                        'player_id': mcts.player_id,
                        'num_simulations': mcts.num_simulations,
                        'exploration_weight': mcts.exploration_weight,
                        'max_depth': mcts.max_depth,
                        'heatmap': mcts.heatmap.copy(),
                        'action_stats': mcts.action_stats.copy()
                    } for mcts in mcts_players], use_pure_mcts)
            
            game, updated_mcts = run_single_episode(args)
            
            # Update statistics
            stats.update(game)
            
            # Update MCTS players with learning from this episode
            for j, mcts in enumerate(updated_mcts):
                mcts_players[j].heatmap = mcts.heatmap.copy()
                mcts_players[j].action_stats.update(mcts.action_stats)
                mcts_players[j].rewards_history.extend(mcts.rewards_history)
            
            # Save models periodically
            if episode % save_interval == 0:
                save_training_state(episode, stats, mcts_players, save_dir, num_episodes)
            
            # Update progress bar
            pbar.update(1)
        
        # Move to the next batch of episodes
        episode += batch_size
    
    # Clean up
    if use_parallel:
        pool.close()
        pool.join()
    
    pbar.close()
    return stats, mcts_players

def save_training_state(episode, stats, mcts_players, save_dir, num_episodes):
    """Save training state and log progress."""
    # Calculate summary statistics
    summary = stats.get_summary()
    
    # Save stats
    with open(f"{save_dir}/training_stats_ep{episode}.pkl", "wb") as f:
        pickle.dump(stats, f)
    
    # Save models
    for i, mcts in enumerate(mcts_players):
        # Save latest model
        mcts.save(f"{save_dir}/player{i}_latest.pkl")
        
        # Also save a versioned copy
        mcts.save(f"{save_dir}/player{i}_ep{episode}.pkl")
    
    # Log progress
    win_rates = summary['win_rates']
    avg_boxes = summary['avg_boxes']
    
    log_msg = f"Episode {episode}/{num_episodes} - "
    log_msg += f"Win rates: {win_rates}, "
    log_msg += f"Avg boxes: {avg_boxes}, "
    log_msg += f"Avg game length: {summary['avg_game_length']:.1f}"
    
    tqdm.write(log_msg)

def display_heatmap(heatmap, title="Position Heatmap"):
    """Display the heatmap for visualization."""
    print(f"\n{title}:")
    print("-" * 30)
    
    # Use simple ASCII visualization
    for i in range(heatmap.shape[0]):
        for j in range(heatmap.shape[1]):
            value = heatmap[i, j]
            if value < 0.3:
                char = " "  # Low preference
            elif value < 0.5:
                char = "·"  # Medium-low preference
            elif value < 0.7:
                char = "+"  # Medium preference
            elif value < 0.9:
                char = "o"  # Medium-high preference
            else:
                char = "O"  # High preference
            print(char, end=" ")
        print()
    print("-" * 30)

def analyze_results(stats, mcts_players):
    """Analyze and print training results."""
    summary = stats.get_summary()
    
    print("\n===== Training Results =====")
    print(f"Total games played: {summary['games_played']}")
    print(f"Average game length: {summary['avg_game_length']:.1f} turns")
    
    print("\nWin Rates:")
    for player_id, rate in summary['win_rates'].items():
        print(f"Player {player_id+1}: {rate*100:.1f}%")
    
    print("\nAverage Boxes Destroyed:")
    for player_id, avg in summary['avg_boxes'].items():
        print(f"Player {player_id+1}: {avg:.1f}")
    
    print("\nSurvival Rates:")
    for player_id, rate in summary['survival_rates'].items():
        print(f"Player {player_id+1}: {rate*100:.1f}%")
    
    # Display reward history for each player
    print("\nReward History Summary:")
    for i, mcts in enumerate(mcts_players):
        rewards = mcts.rewards_history
        if rewards:
            print(f"Player {i+1}: Avg reward={np.mean(rewards):.1f}, Min={min(rewards):.1f}, Max={max(rewards):.1f}")
    
    # Display heatmaps
    for i, mcts in enumerate(mcts_players):
        display_heatmap(mcts.heatmap, f"Player {i+1} Final Position Heatmap")

def main():
    parser = argparse.ArgumentParser(description='Train Bomberman MCTS AI')
    parser.add_argument('--episodes', type=int, default=100,
                        help='Number of training episodes')
    parser.add_argument('--players', type=int, default=2, choices=[2, 3, 4],
                        help='Number of players (2-4)')
    parser.add_argument('--simulations', type=int, default=500,
                        help='Number of MCTS simulations per move')
    parser.add_argument('--render_interval', type=int, default=20,
                        help='Render every N episodes (0 to disable)')
    parser.add_argument('--save_interval', type=int, default=20,
                        help='Save models every N episodes')
    parser.add_argument('--render_speed', type=float, default=0.05,
                        help='Speed of rendering (seconds per frame)')
    parser.add_argument('--save_dir', type=str, default='models',
                        help='Directory to save models')
    parser.add_argument('--load', action='store_true',
                        help='Load existing models if available')
    parser.add_argument('--parallel', action='store_true',
                        help='Use parallel processing for faster training')
    parser.add_argument('--processes', type=int, default=None,
                        help='Number of processes for parallel training')
    parser.add_argument('--max_depth', type=int, default=10,
                        help='Maximum simulation depth for MCTS')
    parser.add_argument('--pure_mcts', action='store_true',
                        help='Use pure MCTS (traditional UCT) instead of enhanced MCTS')
    
    args = parser.parse_args()
    
    # Train the MCTS agents
    stats, mcts_players = train_mcts(
        num_episodes=args.episodes,
        num_players=args.players,
        render_interval=args.render_interval,
        simulations=args.simulations,
        save_interval=args.save_interval,
        save_dir=args.save_dir,
        render_speed=args.render_speed,
        load_models=args.load,
        use_parallel=args.parallel,
        num_processes=args.processes,
        max_depth=args.max_depth,
        use_pure_mcts=args.pure_mcts  # Pass the new parameter
    )
    
    # Print results
    analyze_results(stats, mcts_players)

if __name__ == "__main__":
    main()