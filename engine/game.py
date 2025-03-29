import numpy as np
import random
from enum import Enum
from collections import deque

class CellType(Enum):
    FLOOR = 0
    WALL = 1
    BOX = 2
    BOMB = 3
    POWERUP_BOMB = 4
    POWERUP_RANGE = 5

class Action(Enum):
    STAY = 0
    UP = 1
    RIGHT = 2
    DOWN = 3
    LEFT = 4
    BOMB = 5

class Game:
    def __init__(self, num_players=4, width=9, height=11, max_turns=200):
        self.width = width
        self.height = height
        self.num_players = num_players
        self.current_player_id = 0
        self.turn = 0
        self.max_turns = max_turns
        self.remaining_boxes = 0
        
        # Initialize grid and players
        self.initialize_grid()
        self.initialize_players()
        
        # Bomb tracking
        self.bombs = []  # List of (x, y, timer, range, player_id)
        
        # Game state
        self.is_over = False
        self.boxes_destroyed = [0] * num_players
    
    def initialize_grid(self):
        """Initialize the game grid with walls, floors, and boxes."""
        # Create empty grid filled with floors
        self.grid = np.zeros((self.height, self.width), dtype=int)
        
        # Place walls in a fixed pattern (every 2nd cell in both dimensions)
        for i in range(1, self.height, 2):
            for j in range(1, self.width, 2):
                self.grid[i, j] = CellType.WALL.value
        
        # Keep corners free for players
        player_corners = [
            (0, 0), (0, 1), (1, 0),  # Top-left
            (0, self.width-1), (0, self.width-2), (1, self.width-1),  # Top-right
            (self.height-1, 0), (self.height-2, 0), (self.height-1, 1),  # Bottom-left
            (self.height-1, self.width-1), (self.height-1, self.width-2), (self.height-2, self.width-1)  # Bottom-right
        ]
        
        # Place boxes randomly but symmetrically
        box_positions = []
        for i in range(self.height):
            for j in range(self.width):
                if self.grid[i, j] == CellType.FLOOR.value and (i, j) not in player_corners:
                    if random.random() < 0.75:  # 75% chance of a box
                        box_positions.append((i, j))
        
        # Ensure symmetry
        for i, j in box_positions:
            self.grid[i, j] = CellType.BOX.value
            # Symmetric position
            sym_i = self.height - 1 - i
            sym_j = self.width - 1 - j
            if (sym_i, sym_j) not in player_corners:
                self.grid[sym_i, sym_j] = CellType.BOX.value
        
        # Count boxes
        self.remaining_boxes = np.sum(self.grid == CellType.BOX.value)
    
    def initialize_players(self):
        """Initialize player positions and attributes."""
        self.player_positions = [(0, 0)] * self.num_players
        self.player_alive = [True] * self.num_players
        self.player_bomb_counts = [1] * self.num_players  # Start with 1 bomb
        self.player_bomb_ranges = [2] * self.num_players  # Start with range 2
        
        # Player starting corners
        corners = [
            (1, 1),                    # Top-left
            (1, self.width - 2),       # Top-right
            (self.height - 2, 1),      # Bottom-left
            (self.height - 2, self.width - 2)  # Bottom-right
        ]
        
        # For 2 players, place them in opposite corners (top-left and bottom-right)
        if self.num_players == 2:
            self.player_positions = [corners[0], corners[3]]
        else:
            # For 3 or 4 players, use consecutive corners
            for i in range(self.num_players):
                self.player_positions[i] = corners[i]
    
    def get_legal_actions(self, player_id=None):
        """Get legal actions for the current player or specified player."""
        if player_id is None:
            player_id = self.current_player_id
        
        if not self.player_alive[player_id]:
            return [Action.STAY]
        
        legal_actions = [Action.STAY]
        x, y = self.player_positions[player_id]
        
        # Check movement in four directions
        for action in [Action.UP, Action.RIGHT, Action.DOWN, Action.LEFT]:
            new_x, new_y = self._get_new_position(x, y, action)
            if self._is_passable(new_x, new_y):
                legal_actions.append(action)
        
        # Check if player can place a bomb
        bombs_placed = sum(1 for _, _, _, _, p_id in self.bombs if p_id == player_id)
        if bombs_placed < self.player_bomb_counts[player_id]:
            # Check if there's already a bomb at this position
            if not any((x, y) == (bx, by) for bx, by, _, _, _ in self.bombs):
                legal_actions.append(Action.BOMB)
        
        return legal_actions
    
    def _get_new_position(self, x, y, action):
        """Get new position after an action."""
        if action == Action.UP:
            return x - 1, y
        elif action == Action.RIGHT:
            return x, y + 1
        elif action == Action.DOWN:
            return x + 1, y
        elif action == Action.LEFT:
            return x, y - 1
        return x, y  # STAY
    
    def _is_passable(self, x, y):
        """Check if a position is passable."""
        # Check bounds
        if x < 0 or x >= self.height or y < 0 or y >= self.width:
            return False
        
        # Check cell content
        cell = self.grid[x, y]
        if cell == CellType.WALL.value or cell == CellType.BOX.value:
            return False
        
        # Check for bombs
        for bx, by, _, _, _ in self.bombs:
            if (x, y) == (bx, by):
                return False
        
        return True
    
    def apply_action(self, action):
        """Apply an action for the current player and update game state."""
        player_id = self.current_player_id
        
        if self.player_alive[player_id]:
            x, y = self.player_positions[player_id]
            
            if action == Action.BOMB:
                # Place a bomb
                bomb_range = self.player_bomb_ranges[player_id]
                self.bombs.append((x, y, 8, bomb_range, player_id))  # 8 turns until explosion
            elif action != Action.STAY:
                # Move player
                new_x, new_y = self._get_new_position(x, y, action)
                if self._is_passable(new_x, new_y):
                    self.player_positions[player_id] = (new_x, new_y)
                    
                    # Check for power-ups
                    if self.grid[new_x, new_y] == CellType.POWERUP_BOMB.value:
                        self.player_bomb_counts[player_id] += 1
                        self.grid[new_x, new_y] = CellType.FLOOR.value
                    elif self.grid[new_x, new_y] == CellType.POWERUP_RANGE.value:
                        self.player_bomb_ranges[player_id] += 1
                        self.grid[new_x, new_y] = CellType.FLOOR.value
        
        # Move to next player
        self.current_player_id = (self.current_player_id + 1) % self.num_players
        
        # If completed a round, update bombs and check for explosions
        if self.current_player_id == 0:
            self.turn += 1
            self._update_bombs()
            self._check_game_over()
    
    def _update_bombs(self):
        """Update bomb timers and handle explosions."""
        # Decrease bomb timers
        for i in range(len(self.bombs)):
            x, y, timer, bomb_range, player_id = self.bombs[i]
            self.bombs[i] = (x, y, timer - 1, bomb_range, player_id)
        
        # Check for bombs that need to explode
        exploding_bombs = [i for i, (_, _, timer, _, _) in enumerate(self.bombs) if timer <= 0]
        
        # Process explosions
        while exploding_bombs:
            bomb_idx = exploding_bombs.pop(0)
            if bomb_idx >= len(self.bombs):  # Safety check for chain reactions
                continue
                
            x, y, _, bomb_range, player_id = self.bombs[bomb_idx]
            
            # Remove this bomb
            self.bombs.pop(bomb_idx)
            
            # Adjust indices for remaining bombs in the exploding list
            exploding_bombs = [idx if idx < bomb_idx else idx - 1 for idx in exploding_bombs]
            
            # Process explosion in four directions
            self._process_explosion(x, y, player_id)
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                for r in range(1, bomb_range + 1):
                    nx, ny = x + dx * r, y + dy * r
                    if not self._process_explosion(nx, ny, player_id):
                        break  # Stop in this direction if hit wall or box
            
            # Check for chain reactions (other bombs caught in the explosion)
            for i, (bx, by, btimer, brange, bplayer) in enumerate(self.bombs):
                if self._is_in_explosion(bx, by, x, y, bomb_range):
                    if i not in exploding_bombs:
                        exploding_bombs.append(i)
    
    def _process_explosion(self, x, y, player_id):
        """Process explosion at a position. Returns False if explosion stops at this cell."""
        # Check bounds
        if x < 0 or x >= self.height or y < 0 or y >= self.width:
            return False
        
        # Check cell
        cell = self.grid[x, y]
        
        if cell == CellType.WALL.value:
            return False  # Wall stops explosion
            
        elif cell == CellType.BOX.value:
            # Destroy box and possibly spawn power-up
            self.grid[x, y] = CellType.FLOOR.value
            self.remaining_boxes -= 1
            self.boxes_destroyed[player_id] += 1
            
            # 40% chance to spawn a power-up
            if random.random() < 0.4:
                # 50% chance each for bomb count or range power-up
                power_up = CellType.POWERUP_BOMB if random.random() < 0.5 else CellType.POWERUP_RANGE
                self.grid[x, y] = power_up.value
                
            return False  # Box stops explosion
        
        # Check for players at this position
        for p_id, (px, py) in enumerate(self.player_positions):
            if (px, py) == (x, y) and self.player_alive[p_id]:
                self.player_alive[p_id] = False  # Player caught in explosion
        
        return True  # Explosion continues
    
    def _is_in_explosion(self, bx, by, x, y, bomb_range):
        """Check if bomb at (bx, by) is caught in explosion from (x, y) with range bomb_range."""
        # Check if bomb is at the explosion center
        if (bx, by) == (x, y):
            return True
            
        # Check if bomb is in one of the four directions
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            for r in range(1, bomb_range + 1):
                nx, ny = x + dx * r, y + dy * r
                if (nx, ny) == (bx, by):
                    return True
                # Stop checking in this direction if hit wall or box
                if nx < 0 or nx >= self.height or ny < 0 or ny >= self.width:
                    break
                if self.grid[nx, ny] == CellType.WALL.value or self.grid[nx, ny] == CellType.BOX.value:
                    break
                    
        return False
    
    def _check_game_over(self):
        """Check if the game is over."""
        # Game ends after max_turns
        if self.turn >= self.max_turns:
            self.is_over = True
            return
            
        # Game ends if only one player is alive
        alive_count = sum(self.player_alive)
        if alive_count <= 1 and self.num_players > 1:
            self.is_over = True
            return
            
        # Game ends if all boxes are destroyed (plus 20 extra turns)
        if self.remaining_boxes == 0 and self.turn >= 20:
            self.is_over = True
            return
    
    def is_terminal(self):
        """Return True if the game is over."""
        return self.is_over
    
    def get_rankings(self):
        """Get player rankings based on survival and boxes destroyed."""
        # First by survival (alive players rank higher)
        # Then by boxes destroyed
        rankings = []
        for p_id in range(self.num_players):
            rankings.append((p_id, self.player_alive[p_id], self.boxes_destroyed[p_id]))
        
        # Sort: first by alive status (descending), then by boxes destroyed (descending)
        rankings.sort(key=lambda x: (x[1], x[2]), reverse=True)
        return [p_id for p_id, _, _ in rankings]
    
    def copy(self):
        """Create a deep copy of the game state."""
        game_copy = Game(num_players=self.num_players, width=self.width, 
                        height=self.height, max_turns=self.max_turns)
        
        # Copy grid
        game_copy.grid = np.copy(self.grid)
        
        # Copy player data
        game_copy.player_positions = self.player_positions.copy()
        game_copy.player_bomb_ranges = self.player_bomb_ranges.copy()
        game_copy.player_bomb_counts = self.player_bomb_counts.copy()
        game_copy.player_alive = self.player_alive.copy()
        
        # Copy bomb data
        game_copy.bombs = self.bombs.copy()
        
        # Copy game state
        game_copy.current_player_id = self.current_player_id
        game_copy.turn = self.turn
        game_copy.is_over = self.is_over
        game_copy.remaining_boxes = self.remaining_boxes
        game_copy.boxes_destroyed = self.boxes_destroyed.copy()
        
        return game_copy
    
    def is_survivable(self, player_id):
        """Check if a player can survive the current bomb situation using BFS."""
        if not self.player_alive[player_id]:
            return False
            
        x, y = self.player_positions[player_id]
        
        # If no bombs, it's survivable
        if not self.bombs:
            return True
            
        # Find minimum bomb timer
        min_timer = min(timer for _, _, timer, _, _ in self.bombs)
        
        # BFS to find safe position
        visited = set()
        queue = deque([(x, y, 0)])  # (x, y, steps)
        
        while queue:
            cx, cy, steps = queue.popleft()
            
            # If we've moved enough steps to outlast the bombs
            if steps > min_timer:
                return True
                
            # Check if current position is safe after min_timer
            is_safe = True
            for bx, by, timer, bomb_range, _ in self.bombs:
                if timer <= steps and self._is_in_explosion(cx, cy, bx, by, bomb_range):
                    is_safe = False
                    break
                    
            if is_safe:
                return True
                
            # Try moving in all directions
            for dx, dy in [(0, 0), (0, 1), (1, 0), (0, -1), (-1, 0)]:
                nx, ny = cx + dx, cy + dy
                if (nx, ny) not in visited and self._is_passable(nx, ny):
                    visited.add((nx, ny))
                    queue.append((nx, ny, steps + 1))
                    
        return False
    
    def estimated_bombs(self):
        """Estimate the score based on future bomb explosions."""
        game_copy = self.copy()
        
        # Fast forward until all bombs explode
        bombs_left = len(game_copy.bombs)
        while bombs_left > 0:
            # Save the current player
            saved_player = game_copy.current_player_id
            
            # Make all players take STAY action for one round
            for _ in range(game_copy.num_players):
                game_copy.apply_action(Action.STAY)
                
            # Check if bombs exploded
            bombs_left = len(game_copy.bombs)
        
        # Return the boxes destroyed after bombs explode
        return game_copy.boxes_destroyed
    
    def can_kill(self, player_id, enemy_id):
        """Check if player can trap enemy within two turns."""
        if not self.player_alive[player_id] or not self.player_alive[enemy_id]:
            return False
            
        px, py = self.player_positions[player_id]
        ex, ey = self.player_positions[enemy_id]
        
        # If too far away, can't trap
        if abs(px - ex) + abs(py - ey) > 6:  # Manhattan distance
            return False
            
        # Simplified check: if enemy is near a bomb with timer <= 3, they might be trapped
        for bx, by, timer, bomb_range, _ in self.bombs:
            if timer <= 3 and abs(bx - ex) + abs(by - ey) <= 2:
                # Check if enemy has few escape routes
                escape_routes = 0
                for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    nx, ny = ex + dx, ey + dy
                    if self._is_passable(nx, ny):
                        escape_routes += 1
                        
                if escape_routes <= 1:
                    return True
                    
        return False 