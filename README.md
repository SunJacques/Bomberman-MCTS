# Bomberman with Monte Carlo Tree Search

A Bomberman game implementation with AI players powered by Monte Carlo Tree Search (MCTS).

## Setup

1. Clone the repository
2. Install the requirements:
```
pip install -r requirements.txt
```

## Running the Game

### Play Mode

Run the game with the following command:

```
python main.py [options]
```

#### Command-line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--players N` | Number of players (2-4) | 4 |
| `--human N` | Number of human players (0-1) | 1 |
| `--simulations N` | MCTS simulations per move | 500 |
| `--render` | Enable rendering (required for human players) | enabled for human players |
| `--model_dir DIR` | Directory with trained models | 'models' |
| `--use_trained` | Use trained models for AI players | disabled |
| `--pure_mcts` | Use simple MCTS (traditional UCT) | disabled |
| `--fast` | Enable fast mode for quicker gameplay | disabled |
| `--max_depth N` | Maximum search depth for MCTS | 10 |

### Training Mode

To train AI players through self-play:

```
python train.py [options]
```

#### Training Options

| Option | Description | Default |
|--------|-------------|---------|
| `--episodes N` | Number of training episodes | 100 |
| `--players N` | Number of players (2-4) | 2 |
| `--simulations N` | MCTS simulations per move | 500 |
| `--render_interval N` | Render every N episodes (0 to disable) | 20 |
| `--save_interval N` | Save models every N episodes | 20 |
| `--render_speed S` | Speed of rendering (seconds per frame) | 0.05 |
| `--save_dir DIR` | Directory to save models | 'models' |
| `--load` | Load existing models if available | disabled |
| `--parallel` | Use parallel processing for faster training | disabled |
| `--processes N` | Number of processes for parallel training | CPU count - 1 |
| `--max_depth N` | Maximum simulation depth for MCTS | 10 |


## Examples

### Play Mode Examples

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

- Play with pure MCTS bots (no training):
```
python main.py --players 4 --human 1 --pure_mcts --render
```

### Training Examples

- Train AI players for 200 episodes with 4 players using parallel processing:
```
python train.py --episodes 200 --players 4 --render_interval 20 --parallel
```

- Continue training from previously saved models:
```
python train.py --episodes 100 --load --parallel
```

- Train pure MCTS models (faster but less sophisticated):
```
python train.py --episodes 50 --pure_mcts --parallel
```

## Game Controls (Human Player)

- `w`: Move up
- `a`: Move left
- `s`: Move down
- `d`: Move right
- `q`: Stay still
- `b`: Place bomb

## Game Rules

- Players start in opposite corners of a 9×11 grid
- The grid contains floors (passable), walls (indestructible), and boxes (destructible)
- Bombs explode after 8 turns, destroying boxes and eliminating players
- Boxes may contain power-ups (extra bombs or increased explosion range)
- The game ends after 200 turns or when all but one player is eliminated
- The objective is to be the last player standing and destroy as many boxes as possible
