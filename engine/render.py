import os
import sys
import time
import numpy as np
from engine.game import Action, CellType

class GameRenderer:
    """Improved console-based renderer for the Bomberman game.
    """

    def __init__(self, game):
        self.game = game
        # Define block dimensions: height is 3 and width is 7
        self.block_height = 3
        self.block_width = 7
        self.center_row = self.block_height // 2  # 1
        self.center_col = self.block_width // 2    # 3
        
        self.cell_chars = {
            CellType.FLOOR.value: ' ',
            CellType.WALL.value: '▓',
            CellType.BOX.value: '▒',
            CellType.BOMB.value: '*',
            CellType.POWERUP_BOMB.value: 'B',
            CellType.POWERUP_RANGE.value: 'R'
        }
        # Define player symbols
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

    def clear_screen(self):
        """Clear the console screen."""
        os.system('cls' if os.name == 'nt' else 'clear')

    def render(self):
        """Render the current game state with each cell as a 3x7 block.
           If a cell contains a power-up, only the center shows the power-up symbol.
           When a player has placed a bomb on the same cell, the player is drawn at (1,2)
           and the bomb countdown at (1,4).
        """
        self.clear_screen()

        # Print game information.
        print(f"Turn: {self.game.turn}/{self.game.max_turns}")
        print(f"Boxes Remaining: {self.game.remaining_boxes}")
        print(f"Current Player: {self.game.current_player_id + 1}\n")

        # Build a list of strings (each string is one line of output).
        render_lines = []

        # Create a 2D list of block strings for each cell.
        # Each cell block is a list of self.block_height strings.
        cell_blocks = [[None for _ in range(self.game.width)] for _ in range(self.game.height)]
        for i in range(self.game.height):
            for j in range(self.game.width):
                # Get the base symbol for the cell type.
                cell_type = self.game.grid[i, j]
                base_char = self.cell_chars.get(cell_type, '?')
                # Choose a base color for the cell.
                if base_char in ['▓']:
                    cell_color = self.colors['white']
                elif base_char in ['▒']:
                    cell_color = self.colors['white']
                else:
                    cell_color = self.colors['reset']

                # Special handling for power-ups:
                # Instead of filling the entire block with the power-up character,
                # use the floor (empty) background and later draw the power-up symbol in the center.
                if cell_type in [CellType.POWERUP_BOMB.value, CellType.POWERUP_RANGE.value]:
                    powerup_symbol = self.cell_chars[cell_type]
                    # Set background to floor.
                    base_char = self.cell_chars[CellType.FLOOR.value]
                    cell_color = self.colors['reset']
                else:
                    powerup_symbol = None

                # Check for overrides: player (always yellow), bomb (red), or power-up (cyan).
                player_symbol = None
                bomb_symbol = None
                player_color = self.colors['yellow']
                bomb_color = self.colors['red']
                powerup_color = self.colors['green']
                # Check if a player is present.
                for player_id, (px, py) in enumerate(self.game.player_positions):
                    if (px, py) == (i, j) and self.game.player_alive[player_id]:
                        player_symbol = self.player_chars[player_id]
                        break
                # Check for a bomb on this cell.
                for bomb in self.game.bombs:
                    bx, by, timer, _, _ = bomb
                    if (bx, by) == (i, j):
                        bomb_symbol = str(timer)
                        break
                # If no player or bomb override and this is a power-up cell, use power-up override.
                if player_symbol is None and bomb_symbol is None and powerup_symbol is not None:
                    # Use the power-up symbol as override.
                    override_symbol = powerup_symbol
                    override_color = powerup_color
                else:
                    override_symbol = None
                    override_color = self.colors['reset']
                    # If there's a player override or bomb override, choose accordingly.
                    if player_symbol is not None:
                        override_symbol = player_symbol
                        override_color = player_color
                    elif bomb_symbol is not None:
                        override_symbol = bomb_symbol
                        override_color = bomb_color

                # Build the block for this cell.
                block = []
                for r in range(self.block_height):
                    row_chars = ""
                    for c in range(self.block_width):
                        # If both a player and a bomb override exist, use specific positions.
                        if player_symbol is not None and bomb_symbol is not None:
                            if r == self.center_row and c == self.center_col - 1:
                                row_chars += f"{player_color}{player_symbol}{self.colors['reset']}"
                            elif r == self.center_row and c == self.center_col + 1:
                                row_chars += f"{bomb_color}{bomb_symbol}{self.colors['reset']}"
                            else:
                                row_chars += f"{cell_color}{base_char}{self.colors['reset']}"
                        else:
                            # If an override exists, draw it in the center.
                            if override_symbol is not None and r == self.center_row and c == self.center_col:
                                row_chars += f"{override_color}{override_symbol}{self.colors['reset']}"
                            else:
                                row_chars += f"{cell_color}{base_char}{self.colors['reset']}"
                    block.append(row_chars)
                cell_blocks[i][j] = block

        # Assemble the blocks row by row.
        for i in range(self.game.height):
            for r in range(self.block_height):
                line = ""
                for j in range(self.game.width):
                    line += cell_blocks[i][j][r]
                render_lines.append(line)

        # Optionally add a border around the grid.
        total_width = self.game.width * self.block_width
        border_line = "+" + "-" * total_width + "+"
        print(border_line)
        for line in render_lines:
            print("|" + line + "|")
        print(border_line)

        # Print player information.
        print("\nPlayers:")
        for player_id in range(self.game.num_players):
            status = "ALIVE" if self.game.player_alive[player_id] else "DEAD"
            print(f"{self.colors['yellow']}Player {player_id + 1}{self.colors['reset']}: Bombs={self.game.player_bomb_counts[player_id]}, "
                  f"Range={self.game.player_bomb_ranges[player_id]}, "
                  f"Boxes Destroyed={self.game.boxes_destroyed[player_id]}, "
                  f"Status={status}")

        print("\nLegal Actions:", [a.name for a in self.game.get_legal_actions()])

    def render_game_over(self, rankings, training_mode=False):
        """Render game over screen with rankings."""
        print("\n===== GAME OVER =====")
        print("RANKINGS:")
        for i, player_id in enumerate(rankings):
            print(f"{i + 1}. Player {player_id + 1}")

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
            ' ': Action.STAY,
            'b': Action.BOMB
        }

        legal_actions = self.game.get_legal_actions()

        while True:
            print("\nEnter move (w=up, a=left, s=down, d=right, q=stay, b=bomb): ", end="")
            sys.stdout.flush()

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