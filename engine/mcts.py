import math
import random
import numpy as np
import pickle
import os
from collections import deque
from engine.game import Action, CellType

class SimpleNode:
    def __init__(self, game_state, player_id, parent=None, action=None):
        self.game_state = game_state
        self.player_id = player_id
        self.parent = parent
        self.action = action
        self.children = {}
        self.visits = 0
        self.value = 0.0
        self.max_value = float('-inf')  # Stores the maximum simulation result seen

    def is_fully_expanded(self):
        return len(self.children) == len(self.game_state.get_legal_actions())

    def expand(self):
        legal_actions = self.game_state.get_legal_actions()
        for action in legal_actions:
            if action not in self.children:
                next_state = self.game_state.copy()
                next_state.apply_action(action)
                child = SimpleNode(next_state, self.player_id, parent=self, action=action)
                self.children[action] = child
                return child
        return None

    def best_child(self, c=1.0):
        """
        Use UCT (Upper Confidence Bound for Trees) for selection.
        """
        return max(
            self.children.values(),
            key=lambda child: (child.value / child.visits if child.visits > 0 else 0) +
                              c * math.sqrt(math.log(self.visits) / child.visits if child.visits > 0 else float('inf'))
        )

    def update(self, reward):
        """Update statistics: visits, cumulative value, and max simulation result."""
        self.visits += 1
        self.value += reward
        self.max_value = max(self.max_value, reward)

    def best_action(self):
        """Choose the action with highest max value that doesn't lead to death."""
        # If no children are present, choose from legal actions
        if not self.children:
            legal_actions = self.game_state.get_legal_actions()
            # Filter out bomb placement if it would trap the agent
            if Action.BOMB in legal_actions and PureMCTS.would_bomb_trap_agent(self.game_state, self.player_id):
                legal_actions = [a for a in legal_actions if a != Action.BOMB]
            return legal_actions[0] if legal_actions else Action.STAY

        # Filter out actions that lead to certain death
        safe_actions = {
            a: child for a, child in self.children.items()
            if not PureMCTS.would_die_if_action_taken_static(self.game_state, self.player_id, a)
        }

        # Extra caution for staying in the same place if that action is risky
        if Action.STAY in safe_actions and PureMCTS.is_stay_risky(self.game_state, self.player_id):
            del safe_actions[Action.STAY]

        # If no safe actions remain, pick the action that maximizes survival time
        if not safe_actions:
            return PureMCTS.get_best_survival_action(self.game_state, self.player_id)

        # Select the action from the safe set which has the highest maximum simulation reward
        return max(
            safe_actions.items(),
            key=lambda item: item[1].max_value
        )[0]


class PureMCTS:
    def __init__(self, player_id, num_simulations=100, max_depth=50):
        self.player_id = player_id
        self.num_simulations = num_simulations
        self.max_depth = max_depth
        # Attributes needed for compatibility with training code
        self.exploration_weight = 1.0
        self.heatmap = np.ones((9, 11)) * 0.5
        self.action_stats = {}
        self.rewards_history = []

    def select_action(self, game, fast_mode=False):
        # Early filtering for performance
        legal_actions = game.get_legal_actions(self.player_id)
        if Action.BOMB in legal_actions and PureMCTS.would_bomb_trap_agent(game, self.player_id):
            legal_actions = [a for a in legal_actions if a != Action.BOMB]
            
        # If only one action is possible after filtering, return it immediately
        if len(legal_actions) == 1:
            return legal_actions[0]
            
        root = SimpleNode(game.copy(), self.player_id)

        # Check if we're already in a risky situation before running simulations
        if PureMCTS.is_current_position_risky(game, self.player_id):
            return PureMCTS.get_best_escape_action(game, self.player_id)

        for _ in range(self.num_simulations):
            node = root
            depth = 0

            while not node.game_state.is_terminal() and node.is_fully_expanded() and depth < self.max_depth:
                # Use UCT selection with exploration constant c=1.0
                node = node.best_child(c=1.0)
                depth += 1

            if not node.game_state.is_terminal() and not node.is_fully_expanded():
                node = node.expand()

            reward = self.simulate(node.game_state)

            while node is not None:
                node.update(reward)
                node = node.parent

        return root.best_action()

    def simulate(self, state):
        sim_state = state.copy()
        depth = 0

        while not sim_state.is_terminal() and depth < self.max_depth:
            legal = sim_state.get_legal_actions(self.player_id)
            if not legal:
                break
            
            # Use a simple, fast safety check during rollout
            safe_actions = []
            player_pos = sim_state.player_positions[self.player_id]
            
            for action in legal:
                # Lightweight checks during simulation for performance
                nx, ny = self._get_new_position_from_action(player_pos, action)
                
                # Skip if the resulting position is in immediate danger
                if not self._is_position_immediately_dangerous(sim_state, nx, ny):
                    safe_actions.append(action)
            
            # Choose from safe actions if available, otherwise random
            if safe_actions:
                action = random.choice(safe_actions)
            else:
                action = random.choice(legal)
                
            sim_state.apply_action(action)
            depth += 1

        return self.evaluate(sim_state)
        
    @staticmethod
    def _get_new_position_from_action(pos, action):
        """Helper to get new position after an action."""
        x, y = pos
        if action == Action.UP:
            return x - 1, y
        elif action == Action.RIGHT:
            return x, y + 1
        elif action == Action.DOWN:
            return x + 1, y
        elif action == Action.LEFT:
            return x, y - 1
        return x, y  # STAY or BOMB
        
    @staticmethod
    def _is_position_immediately_dangerous(state, x, y):
        """Fast check if a position is in immediate danger (1-2 turns)."""
        # Check if position is out of bounds
        if not (0 <= x < state.height and 0 <= y < state.width):
            return True
            
        # Check if position has a wall, box, or bomb
        cell = state.grid[x, y]
        if cell == 1 or cell == 2:  # Wall or box
            return True
            
        # Check if there's a bomb at this position
        for bx, by, _, _, _ in state.bombs:
            if (x, y) == (bx, by):
                return True
                
        # Check for bombs about to explode
        for bx, by, timer, brange, _ in state.bombs:
            if timer <= 2 and PureMCTS._in_explosion_radius(x, y, bx, by, brange, state):
                return True
                
        return False

    def evaluate(self, state):
        # If the agent is dead in the simulation, return very negative value
        if not state.player_alive[self.player_id]:
            return -10000
            
        # Compute the reward
        boxes_reward = state.boxes_destroyed[self.player_id]
        num_bombs_reward = state.player_bomb_counts[self.player_id]
        bomb_range_reward = state.player_bomb_ranges[self.player_id]
        reward_total = boxes_reward + num_bombs_reward + bomb_range_reward
        
        reward_component = reward_total * 50
        
        # Get player position
        player_pos = state.player_positions[self.player_id]
        
        # Calculate box_dist as average distance to boxes
        box_positions = np.argwhere(state.grid == CellType.BOX.value)
        if len(box_positions) > 0:  # Only calculate if there are boxes
            box_dist = sum(
                abs(pos[0] - player_pos[0]) + abs(pos[1] - player_pos[1]) 
                for pos in box_positions
            ) / len(box_positions)  # Average distance
        else:
            box_dist = 0
        
        # Calculate player_dist as minimum distance to another player
        player_dist = float('inf')
        for p_id, pos in enumerate(state.player_positions):
            if p_id != self.player_id and state.player_alive[p_id]:
                dist = abs(pos[0] - player_pos[0]) + abs(pos[1] - player_pos[1])
                player_dist = min(player_dist, dist)
        
        # If no other players alive, set player_dist to 0
        if player_dist == float('inf'):
            player_dist = 0
        
        # Calculate centre_dist as distance to center of the board
        center_x, center_y = state.height // 2, state.width // 2
        centre_dist = abs(player_pos[0] - center_x) + abs(player_pos[1] - center_y)
        
        # Decide which distance metric to use based on remaining boxes
        if state.remaining_boxes > 20:
            # When many boxes remain, prioritize center distance
            distance_penalty = centre_dist + player_dist
        else:
            # When fewer boxes remain, prioritize box distance
            distance_penalty = box_dist + player_dist
        
        # Compute final value - negative weight for distances, positive for winning
        value = reward_component - distance_penalty 
        return value

    @staticmethod
    def would_bomb_trap_agent(state, player_id):
        """Check if placing a bomb would trap the agent with no escape path"""
        x, y = state.player_positions[player_id]
        test_state = state.copy()
        
        # Add our bomb with timer 8
        bomb_range = test_state.player_bomb_ranges[player_id]
        test_state.bombs.append((x, y, 8, bomb_range, player_id))
        
        # Simulate the agent's movement for 8 turns or until safety is found
        queue = deque([(test_state, (x, y), 0)])  # (state, position, steps)
        visited = set([(x, y, 0)])  # (x, y, steps)
        
        while queue:
            current_state, (cx, cy), steps = queue.popleft()
            
            # If we've already moved 8 steps and are still alive, we can escape
            if steps >= 8:
                return False  # Not trapped
            
            # Check if this position will be safe after all bombs explode
            future_state = current_state.copy()
            # Fast-forward until bombs with timer <= (8-steps) explode
            for _ in range(8-steps):
                future_state._update_bombs()
            
            # If we would survive at this position, we're safe
            future_x, future_y = future_state.player_positions[player_id]
            if future_state.player_alive[player_id] and (future_x, future_y) == (cx, cy):
                return False  # Not trapped
            
            # Try moving in all four directions
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nx, ny = cx + dx, cy + dy
                
                # Skip if already visited or not passable
                if (nx, ny, steps+1) in visited or not test_state._is_passable(nx, ny):
                    continue
                
                # Simulate moving to this position
                new_state = current_state.copy()
                original_pos = new_state.player_positions[player_id]
                new_state.player_positions[player_id] = (nx, ny)
                
                # Check if we'd die immediately from any exploding bombs
                new_state._update_bombs()
                if not new_state.player_alive[player_id]:
                    continue
                
                visited.add((nx, ny, steps+1))
                queue.append((new_state, (nx, ny), steps+1))
            
            # Also try staying in place (might be safe to wait)
            if (cx, cy, steps+1) not in visited:
                new_state = current_state.copy()
                new_state._update_bombs()
                if new_state.player_alive[player_id]:
                    visited.add((cx, cy, steps+1))
                    queue.append((new_state, (cx, cy), steps+1))
        
        # If we explored all possible paths and found no escape
        return True  # Trapped

    @staticmethod
    def is_current_position_deadly(state, player_id):
        """Check if the current position is already deadly (bombs about to explode)"""
        if not state.player_alive[player_id]:
            return True
            
        x, y = state.player_positions[player_id]
        # Check if any bomb with timer 0 or 1 will hit us
        for bx, by, timer, brange, _ in state.bombs:
            if timer <= 1 and PureMCTS._in_explosion_radius(x, y, bx, by, brange, state):
                return True
        return False
        
    @staticmethod
    def is_current_position_risky(state, player_id):
        """More conservative check if position is risky (near bombs with timer ≤ 3)"""
        if not state.player_alive[player_id]:
            return True
            
        x, y = state.player_positions[player_id]
        
        # Check if we're within blast radius of any bomb
        for bx, by, timer, brange, _ in state.bombs:
            if timer <= 3 and PureMCTS._in_explosion_radius(x, y, bx, by, brange, state):
                # For bombs with timer 2-3, check if we have enough escape routes
                escape_routes = 0
                for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    nx, ny = x + dx, y + dy
                    if PureMCTS._is_valid_move(state, nx, ny):
                        if not PureMCTS._in_explosion_radius(nx, ny, bx, by, brange, state):
                            escape_routes += 1
                
                # If we have fewer than 2 escape routes, consider it risky
                if escape_routes < 2:
                    return True
                    
                # For bombs with timer <= 1, always consider risky
                if timer <= 1:
                    return True
        
        return False
        
    @staticmethod
    def _is_valid_move(state, x, y):
        """Check if a position is a valid move."""
        # Check bounds
        if not (0 <= x < state.height and 0 <= y < state.width):
            return False
            
        # Check for walls, boxes, or bombs
        if state.grid[x, y] == 1 or state.grid[x, y] == 2:
            return False
            
        for bx, by, _, _, _ in state.bombs:
            if (x, y) == (bx, by):
                return False
                
        return True
        
    @staticmethod
    def is_stay_risky(state, player_id):
        """Special check to see if STAY action is risky."""
        if not state.player_alive[player_id]:
            return True
            
        x, y = state.player_positions[player_id]
        
        # More conservative for STAY - check bombs with timer ≤ 4
        for bx, by, timer, brange, _ in state.bombs:
            if timer <= 4 and PureMCTS._in_explosion_radius(x, y, bx, by, brange, state):
                return True
                
        return False

    @staticmethod
    def get_best_survival_action(state, player_id):
        """When all actions lead to death, find the one that prolongs life the most"""
        legal_actions = state.get_legal_actions()
        if not legal_actions:
            return Action.STAY
            
        # For each action, determine how many turns we can survive
        best_action = legal_actions[0]
        max_survival_time = 0
        
        for action in legal_actions:
            test_state = state.copy()
            test_state.apply_action(action)
            
            # Remove other players
            for pid in range(test_state.num_players):
                if pid != player_id:
                    test_state.player_alive[pid] = False
            
            # Count how many turns we can survive
            survival_time = PureMCTS._count_survival_turns(test_state, player_id)
            
            if survival_time > max_survival_time:
                max_survival_time = survival_time
                best_action = action
                
        return best_action
        
    @staticmethod
    def get_best_escape_action(state, player_id):
        """Find the best action to escape from risky situations."""
        legal_actions = state.get_legal_actions()
        if not legal_actions:
            return Action.STAY
            
        player_pos = state.player_positions[player_id]
        x, y = player_pos
        
        # Evaluate each action based on safety
        best_action = None
        best_safety_score = float('-inf')
        
        for action in legal_actions:
            # Skip bomb placement if it would trap us (comprehensive check)
            if action == Action.BOMB and PureMCTS.would_bomb_trap_agent(state, player_id):
                continue
            
            # Skip STAY if in blast radius of any bomb
            if action == Action.STAY and any(
                PureMCTS._in_explosion_radius(x, y, bx, by, brange, state)
                for bx, by, timer, brange, _ in state.bombs
            ):
                continue
            
            # Get new position for movement actions
            nx, ny = PureMCTS._get_new_position_from_action(player_pos, action)
            
            # Calculate safety score based on distance from bombs
            safety_score = 0
            
            # Heavy penalty for positions in bomb blast radius
            for bx, by, timer, brange, _ in state.bombs:
                if PureMCTS._in_explosion_radius(nx, ny, bx, by, brange, state):
                    # Penalty based on timer - shorter timer = bigger penalty
                    safety_score -= (10 / max(1, timer))
                else:
                    # Bonus for being outside explosion radius
                    safety_score += 5
            
            # Bonus for moving away from bombs (Manhattan distance)
            for bx, by, _, _, _ in state.bombs:
                curr_dist = abs(x - bx) + abs(y - by)
                new_dist = abs(nx - bx) + abs(ny - by)
                if new_dist > curr_dist:
                    safety_score += 3
            
            if best_action is None or safety_score > best_safety_score:
                best_safety_score = safety_score
                best_action = action
        
        # If no safe action found, use survival action
        if best_action is None:
            return PureMCTS.get_best_survival_action(state, player_id)
            
        return best_action

    @staticmethod
    def _count_survival_turns(state, player_id, max_simulation=16):
        """Count how many turns the player can survive"""
        if not state.player_alive[player_id]:
            return 0
            
        turns = 0
        sim_state = state.copy()
        
        while turns < max_simulation:
            # Try each possible action and pick the one that keeps us alive
            legal_actions = sim_state.get_legal_actions(player_id)
            if not legal_actions:
                break
                
            stayed_alive = False
            for action in legal_actions:
                # Using only lightweight checks during simulation for performance
                test_state = sim_state.copy()
                test_state.apply_action(action)
                
                # Skip other players' turns
                for _ in range(test_state.num_players - 1):
                    test_state.apply_action(Action.STAY)
                    
                if test_state.player_alive[player_id]:
                    sim_state = test_state
                    stayed_alive = True
                    break
                    
            if not stayed_alive:
                break
                
            turns += 1
            
        return turns

    @staticmethod
    def would_die_if_action_taken(state, player_id, action):
        # Special case for bomb placement - check if it would trap us
        if action == Action.BOMB and PureMCTS.would_bomb_trap_agent(state, player_id):
            return True
            
        test_state = state.copy()
        test_state.apply_action(action)

        # Special handling for STAY action - be more conservative
        if action == Action.STAY:
            # Get the new position after action
            x, y = test_state.player_positions[player_id]
            
            # For STAY, check bombs with timer <= 2
            for bx, by, timer, brange, _ in test_state.bombs:
                if timer <= 2 and PureMCTS._in_explosion_radius(x, y, bx, by, brange, test_state):
                    # Extra conservative - if staying in a bomb blast radius with timer <= 2, consider it deadly
                    return True

        # Remove other players from play (but keep their bombs)
        for pid in range(test_state.num_players):
            if pid != player_id:
                test_state.player_alive[pid] = False

        return not PureMCTS._can_survive(test_state, player_id)

    @staticmethod
    def _can_survive(state, player_id, max_depth=32):
        if not state.player_alive[player_id]:
            return False

        start_pos = state.player_positions[player_id]
        
        # Create a more comprehensive state tracking mechanism
        # Include serialized bomb state to avoid duplicate paths
        def get_bombs_state(bombs):
            return tuple(sorted((x, y, t) for x, y, t, _, _ in bombs))
            
        visited = set()
        queue = deque()
        
        initial_state = state.copy()
        initial_bombs = get_bombs_state(initial_state.bombs)
        queue.append((initial_state, 0, start_pos, initial_bombs))
        
        # Track position, depth, AND bomb state
        visited.add((start_pos, 0, initial_bombs))

        while queue:
            current_state, depth, (x, y), bombs_state = queue.popleft()

            if not current_state.player_alive[player_id]:
                continue

            # Consider survived if lived long enough
            if depth > max_depth:
                return True

            # Check if current position is safe from any immediate explosions
            bombs = current_state.bombs.copy()
            is_safe = True
            
            # Check bombs with timer <= 1
            for bx, by, timer, brange, _ in bombs:
                if timer <= 1 and PureMCTS._in_explosion_radius(x, y, bx, by, brange, current_state):
                    is_safe = False
                    break
                    
            # If we're in a safe position and no bombs are about to explode, we survived
            if is_safe and all(timer > 1 for _, _, timer, _, _ in bombs):
                return True

            # Try all possible actions from here
            actions = current_state.get_legal_actions(player_id)
            
            # Sort actions by safety (actions that move away from bombs first)
            def action_safety_score(action):
                nx, ny = PureMCTS._get_new_position_from_action((x, y), action)
                score = 0
                for bx, by, timer, brange, _ in bombs:
                    if PureMCTS._in_explosion_radius(nx, ny, bx, by, brange, current_state):
                        # Strong penalty for moving into explosion radius
                        score -= 100
                        # Penalty inversely proportional to bomb timer
                        score -= (10 / max(1, timer))
                    else:
                        # Bonus for being outside explosion radius
                        score += 10
                # Bonus for moving away from bombs (Manhattan distance)
                for bx, by, _, _, _ in bombs:
                    curr_dist = abs(x - bx) + abs(y - by)
                    new_dist = abs(nx - bx) + abs(ny - by)
                    if new_dist > curr_dist:
                        score += 5
                return score
                
            # Sort actions by safety score
            actions.sort(key=action_safety_score, reverse=True)
            
            for action in actions:
                next_state = current_state.copy()
                next_state.apply_action(action)

                # Skip other players' turns (but keep their bombs)
                for _ in range(next_state.num_players - 1):
                    next_state.apply_action(Action.STAY)

                new_pos = next_state.player_positions[player_id]
                new_bombs_state = get_bombs_state(next_state.bombs)
                key = (new_pos, depth + 1, new_bombs_state)

                if key not in visited:
                    visited.add(key)
                    queue.append((next_state, depth + 1, new_pos, new_bombs_state))

        return False  # Explored all paths, no survivable outcome

    @staticmethod
    def _in_explosion_radius(x, y, bx, by, bomb_range, state):
        # Check if directly on bomb
        if (x, y) == (bx, by):
            return True
            
        # Check in each of the four directions
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            for r in range(1, bomb_range + 1):
                nx, ny = bx + dx * r, by + dy * r
                
                # Check bounds
                if not (0 <= nx < state.height and 0 <= ny < state.width):
                    break
                    
                # Check if blocked by wall or box
                if state.grid[nx, ny] == 1 or state.grid[nx, ny] == 2:
                    break
                    
                # Check if player is in this position
                if (nx, ny) == (x, y):
                    return True
                    
        return False

    would_die_if_action_taken_static = would_die_if_action_taken

    def save(self, filepath):
        """Save the PureMCTS model to a file."""
        # Create a dictionary with the model parameters
        model_data = {
            'player_id': self.player_id,
            'num_simulations': self.num_simulations,
            'max_depth': self.max_depth,
            'exploration_weight': self.exploration_weight,
            'heatmap': self.heatmap,
            'action_stats': self.action_stats,
            'rewards_history': self.rewards_history
        }
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save to file
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    @classmethod
    def load(cls, filepath):
        """Load a PureMCTS model from a file."""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            # Create a new instance
            mcts = cls(
                player_id=model_data['player_id'],
                num_simulations=model_data.get('num_simulations', 100),
                max_depth=model_data.get('max_depth', 50)
            )
            
            # Load attributes for compatibility
            if 'exploration_weight' in model_data:
                mcts.exploration_weight = model_data['exploration_weight']
            if 'heatmap' in model_data:
                mcts.heatmap = model_data['heatmap']
            if 'action_stats' in model_data:
                mcts.action_stats = model_data['action_stats']
            if 'rewards_history' in model_data:
                mcts.rewards_history = model_data['rewards_history']
            
            return mcts
        except Exception as e:
            print(f"Error loading model from {filepath}: {e}")
            # Return a new instance with default values
            return cls(player_id=0, num_simulations=100, max_depth=50)