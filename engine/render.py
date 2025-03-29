import os
import sys
import time
import numpy as np
from engine.game import Action, CellType

class GameRenderer:
    """Simple console-based renderer for the Bomberman game."""
    
    def __init__(self, game):
        self.game = game
        self.cell_chars = {
            CellType.FLOOR.value: ' ',
            CellType.WALL.value: '▓',
            CellType.BOX.value: '▒',
            CellType.BOMB.value: '*',
            CellType.POWERUP_BOMB.value: 'B',
            CellType.POWERUP_RANGE.value: 'R'
        }
        self.player_chars = ['1', '2', '3', '4']
        self.colors = {
            'reset': '\033[0m',
            'black': '\033[30m',
            'red': '\033[31m',
            'green': '\033[32m',
            'yellow': '\033[33m',
            'blue': '\033[34m',
            'magenta': '\033[35m',
            'cyan': '\033[36m',
            'white': '\033[37m',
            'bg_black': '\033[40m',
            'bg_red': '\033[41m',
            'bg_green': '\033[42m',
            'bg_yellow': '\033[43m',
            'bg_blue': '\033[44m',
            'bg_magenta': '\033[45m',
            'bg_cyan': '\033[46m',
            'bg_white': '\033[47m'
        }
        self.player_colors = [
            self.colors['red'],
            self.colors['blue'],
            self.colors['green'],
            self.colors['yellow']
        ]
        
    def clear_screen(self):
        """Clear the console screen."""
        os.system('cls' if os.name == 'nt' else 'clear')
        
    def render(self):
        """Render the current game state to the console."""
        self.clear_screen()
        
        # Print game information
        print(f"Turn: {self.game.turn}/{self.game.max_turns}")
        print(f"Boxes Remaining: {self.game.remaining_boxes}")
        print(f"Current Player: {self.game.current_player_id + 1}")
        print()
        
        # Create a grid for rendering
        render_grid = np.full((self.game.height, self.game.width), ' ', dtype=str)
        
        # Fill grid with cell types
        for i in range(self.game.height):
            for j in range(self.game.width):
                cell_type = self.game.grid[i, j]
                render_grid[i, j] = self.cell_chars[cell_type]
        
        # Add bombs to the grid
        for x, y, timer, _, _ in self.game.bombs:
            render_grid[x, y] = str(timer)
        
        # Add players to the grid
        for player_id, (x, y) in enumerate(self.game.player_positions):
            if self.game.player_alive[player_id]:
                render_grid[x, y] = self.player_chars[player_id]
        
        # Render the grid
        print("+" + "-" * (self.game.width * 2 - 1) + "+")
        for i in range(self.game.height):
            print("|", end="")
            for j in range(self.game.width):
                cell = render_grid[i, j]
                
                # Determine color based on cell content
                color = self.colors['reset']
                
                if cell in self.player_chars:
                    player_id = self.player_chars.index(cell)
                    color = self.player_colors[player_id]
                elif cell in ['B', 'R']:  # Power-ups
                    color = self.colors['cyan']
                elif cell == '▓':  # Wall
                    color = self.colors['white']
                elif cell == '▒':  # Box
                    color = self.colors['yellow']
                elif cell in '12345678':  # Bomb timer
                    color = self.colors['red']
                
                print(f"{color}{cell}{self.colors['reset']} ", end="")
            print("|")
        print("+" + "-" * (self.game.width * 2 - 1) + "+")
        
        # Print player information
        print("\nPlayers:")
        for player_id in range(self.game.num_players):
            status = "ALIVE" if self.game.player_alive[player_id] else "DEAD"
            color = self.player_colors[player_id]
            print(f"{color}Player {player_id + 1}{self.colors['reset']}: Bombs={self.game.player_bomb_counts[player_id]}, "
                  f"Range={self.game.player_bomb_ranges[player_id]}, "
                  f"Boxes Destroyed={self.game.boxes_destroyed[player_id]}, "
                  f"Status={status}")
        
        print("\nLegal Actions:", [a.name for a in self.game.get_legal_actions()])
    
    def render_game_over(self, rankings, training_mode=False):
        """Render game over screen with rankings."""
        print("\n===== GAME OVER =====")
        print("RANKINGS:")
        for i, player_id in enumerate(rankings):
            print(f"{i+1}. Player {player_id + 1}")
            
        # Only wait for input if not in training mode
        if not training_mode:
            input("\nPress Enter to exit...")
    
    def get_human_action(self):
        """Get action input from a human player."""
        action_map = {
            'w': Action.UP,
            'a': Action.LEFT,
            's': Action.DOWN,
            'd': Action.RIGHT,
            'q': Action.STAY,
            'b': Action.BOMB
        }
        
        legal_actions = self.game.get_legal_actions()
        
        while True:
            print("\nEnter move (w=up, a=left, s=down, d=right, space=stay, b=bomb): ", end="")
            sys.stdout.flush()  # Ensure prompt is displayed
            
            try:
                key = input().lower().strip()
                if key in action_map:
                    action = action_map[key]
                    if action in legal_actions:
                        return action
                    else:
                        print(f"Illegal action: {action.name}")
                else:
                    print("Invalid key. Please try again.")
            except (KeyboardInterrupt, EOFError):
                print("\nExiting...")
                sys.exit(0) 