import numpy as np
from torch.utils.data import Dataset


def encode_game_state(game, player_id):
    """
    Encode game state for neural network
    """
    height, width = game.height, game.width

    # Initialize state tensor
    state = np.zeros((17, height, width), dtype=np.float32)

    # One-hot encode board state (6 channels)
    for i in range(height):
        for j in range(width):
            cell_type = game.grid[i, j]
            state[cell_type, i, j] = 1.0

    # Encode bomb information (2 channels)
    bomb_timer_channel = 6
    bomb_range_channel = 7
    for x, y, timer, bomb_range, _ in game.bombs:
        state[bomb_timer_channel, x, y] = timer / 8.0  
        state[bomb_range_channel, x, y] = bomb_range / 6.0  

    # Encode player positions (2 channels)
    for p_id in range(2):  
        if game.player_alive[p_id]:
            x, y = game.player_positions[p_id]
            state[8 + p_id, x, y] = 1.0

    # Encode player bomb counts and ranges (4 channels)
    for p_id in range(2):  
        bomb_count = game.player_bomb_counts[p_id]
        bomb_range = game.player_bomb_ranges[p_id]

        # Broadcast values across entire board
        state[10 + p_id] = np.full((height, width), bomb_count / 5.0)  
        state[12 + p_id] = np.full((height, width), bomb_range / 6.0)  

    # Encode player alive status (2 channels)
    for p_id in range(2):
        state[14 + p_id] = np.full((height, width), float(game.player_alive[p_id]))

    # Encode whose turn it is (1 channel)
    state[16] = np.full((height, width), float(game.current_player_id == player_id))

    return state


def action_to_policy(action, legal_actions):
    """
    Convert action to policy vector
    """
    policy = np.zeros(6, dtype=np.float32)  

    # Set policy to 0 for illegal actions
    for a in legal_actions:
        policy[a.value] = 0.0

    # Set policy to 1 for chosen action
    policy[action.value] = 1.0

    return policy


class BombermanDataset(Dataset):
    """
    Create dataset for for neural network
    """
    def __init__(self, states, policies, values):
        self.states = states
        self.policies = policies
        self.values = values

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.policies[idx], self.values[idx]