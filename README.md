# Bomberman with Monte Carlo Tree Search

This project implements a Bomberman game with AI players powered by Monte Carlo Tree Search (MCTS).

## Game Overview

The game is played on a 9×11 grid with 2 to 4 players. The grid consists of:
- **Floors**: Passable by players
- **Walls**: Indestructible, arranged in a fixed pattern
- **Boxes**: Destructible, may contain power-ups

Players start in opposite corners, and their goal is to eliminate opponents using bombs while avoiding explosions.

### Game Mechanics

- Players can move (left, right, up, down, or stay still)
- Players can place bombs that explode after 8 turns, creating chain reactions
- Bomb explosions destroy boxes (revealing power-ups) and eliminate players caught in the blast
- Players cannot move onto walls, boxes, or bombs
- Power-ups grant extra bombs or increased explosion range
- The game ends after 200 turns or when all but one player is eliminated

## Implementation Details

The game is implemented with the following components:

1. **Game Engine**: Manages game state, rules, and mechanics
2. **MCTS AI**: Makes decisions for AI players using Monte Carlo Tree Search
3. **Console Renderer**: Visualizes the game state in the terminal

### Monte Carlo Tree Search

The AI players use a Single-Player MCTS approach, which:
- Predicts opponent moves and simulates possible game states
- Uses Upper Confidence Bound (UCT) formula for node selection
- Incorporates a sophisticated reward system that factors in:
  - Box destruction
  - Power-up collection
  - Survival priority
  - Strategic positioning

## Requirements

- Python 3.8+
- NumPy
- tqdm (for training progress bars)
- matplotlib (for optional visualization)

## Installation

1. Clone the repository:
```
git clone https://github.com/yourusername/bomberman-mcts.git
cd bomberman-mcts
```

2. Install dependencies:
```
pip install -r requirements.txt
```

## Usage

### Playing the Game

Run the game with the following command:

```
python main.py [options]
```

#### Command-line Options for Playing

- `--players N`: Number of players (2-4), default: 4
- `--human N`: Number of human players (0-1), default: 1
- `--simulations N`: Number of MCTS simulations per move, default: 500
- `--render`: Enable rendering (required for human players)
- `--model_dir DIR`: Directory containing trained models, default: 'models'
- `--use_trained`: Use trained models for AI players instead of default ones
- `--fast`: Use fast mode for quicker gameplay
- `--max_depth N`: Maximum search depth for MCTS, default: 10

### Training Mode

To train AI players through self-play, run:

```
python train.py [options]
```

#### Command-line Options for Training

- `--episodes N`: Number of training episodes, default: 100
- `--players N`: Number of players (2-4), default: 2
- `--simulations N`: Number of MCTS simulations per move, default: 500
- `--render_interval N`: Render every N episodes (0 to disable), default: 20
- `--save_interval N`: Save models every N episodes, default: 20
- `--render_speed S`: Speed of rendering (seconds per frame), default: 0.05
- `--save_dir DIR`: Directory to save models, default: 'models'
- `--load`: Load existing models if available
- `--parallel`: Use parallel processing for faster training
- `--processes N`: Number of processes for parallel training
- `--max_depth N`: Maximum simulation depth for MCTS, default: 10

### Examples

- Play a 4-player game with 1 human player (default):
```
python main.py
```

- Watch AI-only match with 3 players in fast mode:
```
python main.py --players 3 --human 0 --render --fast
```

- Play against 3 trained AI players:
```
python main.py --players 4 --use_trained --render
```

- Train AI players for 200 episodes with 4 players using parallel processing:
```
python train.py --episodes 200 --players 4 --render_interval 20 --parallel
```

- Continue training from previously saved models:
```
python train.py --episodes 100 --load --parallel
```

## Performance Optimizations

The implementation includes several optimizations for faster training and gameplay:

1. **Parallel Processing**: Train multiple episodes simultaneously
2. **Memory Management**: Automatically clean up memory to prevent excessive usage
3. **Adaptive Exploration**: Dynamically adjust exploration vs. exploitation balance
4. **Action Filtering**: Learn and prioritize promising actions
5. **State Simplification**: Simplified state representation for faster hashing
6. **Fast Mode**: Reduced simulation depth and quicker decision making
7. **Optimized Grid Size**: 9x11 grid for faster game progression

## Controls (Human Player)

- `w`: Move up
- `a`: Move left
- `s`: Move down
- `d`: Move right
- `b`: Place bomb
- `Space`: Stay still

## Learning Implementation

The MCTS implementation incorporates several learning mechanisms:

1. **Position Heatmaps**: The AI learns which board positions are advantageous
2. **Action Statistics**: Tracks the success of different actions in similar states
3. **Reward History**: Maintains a history of rewards to track improvement
4. **Model Persistence**: Models can be saved and loaded to continue training
5. **Action Priors**: Builds a database of promising actions for similar states

During training, the following statistics are tracked:
- Win rates for each player
- Average boxes destroyed
- Survival rates
- Game lengths
- Reward metrics

## Game Strategies

1. **Box Destruction**: Destroy boxes to find power-ups
2. **Power-up Collection**: More bombs and larger explosion range mean more power
3. **Opponent Trapping**: Place bombs to limit opponent movement
4. **Safe Distance**: Stay away from bombs and anticipate chain reactions
5. **Strategic Positioning**: Use walls for protection and trap opponents in corners

## License

MIT License

## Acknowledgments

This project was inspired by the classic Bomberman game and implements MCTS as described in academic literature. 