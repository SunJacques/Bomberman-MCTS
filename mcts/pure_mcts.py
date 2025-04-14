import math
import random
import numpy as np
import pickle
import os
from collections import deque, Counter
import concurrent.futures
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
        self.max_value = float('-inf')  

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
        Uses UCT for selection
        """
        return max(
            self.children.values(),
            key=lambda child: (child.value / child.visits if child.visits > 0 else 0) +
                              c * math.sqrt(math.log(self.visits) / child.visits if child.visits > 0 else float('inf'))
        )

    def update(self, reward):
        self.visits += 1
        self.value += reward
        self.max_value = max(self.max_value, reward)

    def best_action(self):
        # If no children are present, choose from legal actions
        if not self.children:
            legal_actions = self.game_state.get_legal_actions()
            safe_actions = [a for a in legal_actions if PureMCTS.is_safe_action(self.game_state, self.player_id, a)]
            
            # If no safe actions, pick any legal action
            if not safe_actions:
                return legal_actions[0] if legal_actions else Action.STAY
            return safe_actions[0]
        
        # Filter actions by safety
        safe_children = {
            a: child for a, child in self.children.items() 
            if PureMCTS.is_safe_action(self.game_state, self.player_id, a)
        }
        
        # Further filter to avoid actions that could lead to being trapped
        trap_free_children = {}
        for a, child in safe_children.items():
            # Check if action could lead to being trapped
            test_state = self.game_state.copy()
            
            # Save original player
            original_player = test_state.current_player_id
            
            # Set current player to our player and apply the action
            test_state.current_player_id = self.player_id
            test_state.apply_action(a)
            
            # Restore original player
            test_state.current_player_id = original_player
            
            if not PureMCTS.can_be_trapped_by_enemy(test_state, self.player_id):
                trap_free_children[a] = child
        
        # Use trap-free actions if possible
        if trap_free_children:
            return max(
                trap_free_children.items(),
                key=lambda item: item[1].max_value
            )[0]
        
        # If no trap-free actions use safe actions
        if safe_children:
            return max(
                safe_children.items(),
                key=lambda item: item[1].max_value
            )[0]
        
        # If no safe actions pick any action with highest value
        return max(
            self.children.items(),
            key=lambda item: item[1].max_value
        )[0]


class PureMCTS:
    def __init__(self, player_id, num_simulations=100, max_depth=50):
        self.player_id = player_id
        self.num_simulations = num_simulations
        self.max_depth = max_depth
        self.exploration_weight = 1.0
        self.heatmap = np.ones((9, 11)) * 0.5
        self.action_stats = {}
        self.rewards_history = []

    def select_action(self, game, fast_mode=False):
        legal_actions = game.get_legal_actions(self.player_id)
        
        # Filter actions for safety
        safe_actions = [a for a in legal_actions if PureMCTS.is_safe_action(game, self.player_id, a)]
        
        # If no safe actions return any legal action
        if not safe_actions:
            return legal_actions[0] if legal_actions else Action.STAY
        
        # If only one safe action return it immediately
        if len(safe_actions) == 1:
            return safe_actions[0]
        
        # Further filter to avoid actions that lead to being trapped
        trap_free_actions = []
        for a in safe_actions:
            test_state = game.copy()
            original_player = test_state.current_player_id
            test_state.current_player_id = self.player_id
            test_state.apply_action(a)
            test_state.current_player_id = original_player
            
            if not PureMCTS.can_be_trapped_by_enemy(test_state, self.player_id):
                trap_free_actions.append(a)
        
        # If we have trap-free actions only consider those
        if trap_free_actions:
            safe_actions = trap_free_actions
        
        root = SimpleNode(game.copy(), self.player_id)
        
        # Run MCTS simulations
        for _ in range(self.num_simulations):
            node = root
            depth = 0
            
            while not node.game_state.is_terminal() and node.is_fully_expanded() and depth < self.max_depth:
                node = node.best_child(c=1.0)
                depth += 1
            
            if not node.game_state.is_terminal() and not node.is_fully_expanded():
                node = node.expand()
            
            reward = self.simulate(node.game_state)
            
            while node is not None:
                node.update(reward)
                node = node.parent
        
        # Choose the best action from the filtered actions
        best_action = None
        best_value = float('-inf')
        
        for action in safe_actions:
            if action in root.children:
                child = root.children[action]
                value = child.max_value if child.visits > 0 else float('-inf')
                
                if value > best_value:
                    best_value = value
                    best_action = action
        
        # If no child was expanded or all have negative value pick first safe action
        if best_action is None:
            best_action = safe_actions[0]
        
        return best_action
    
    def select_action_parallel(self, game, num_workers=4):
        """
        Run multiple independent MCTS searches in parallel and aggregate the results 

        """
        legal_actions = game.get_legal_actions(self.player_id)
        
        # Early returns for trivial cases
        safe_actions = [a for a in legal_actions if PureMCTS.is_safe_action(game, self.player_id, a)]
        if not safe_actions:
            return legal_actions[0] if legal_actions else Action.STAY
        if len(safe_actions) == 1:
            return safe_actions[0]
            
        # Divide simulations among workers
        sims_per_worker = max(1, self.num_simulations // num_workers)
        
        def worker_task(game_copy):
            """Task function for each worker process"""
            # Create a new MCTS instance with reduced simulation count
            worker_mcts = PureMCTS(
                player_id=self.player_id,
                num_simulations=sims_per_worker,
                max_depth=self.max_depth
            )
            # Run standard MCTS and return the selected action
            return worker_mcts.select_action(game_copy)
        
        # Create copies of the game state for each worker
        game_copies = [game.copy() for _ in range(num_workers)]
        
        # Run parallel searches 
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(worker_task, game_copies))
        
        # Count votes for each action and choose the most common one
        action_votes = Counter(results)
        best_action = action_votes.most_common(1)[0][0]
        
        return best_action
    
    def simulate(self, state):
        sim_state = state.copy()
        depth = 0
        
        while not sim_state.is_terminal() and depth < self.max_depth:
            legal = sim_state.get_legal_actions(self.player_id)
            if not legal:
                break
            
            # Filter actions for safety using the simplified check
            safe_actions = [a for a in legal if PureMCTS.is_safe_action_simple(sim_state, self.player_id, a)]
            
            # If no safe actions use any legal action
            if not safe_actions:
                action = random.choice(legal)
            else:
                # Favor BOMB during simulation
                bomb_available = Action.BOMB in safe_actions
                if bomb_available and random.random() < 0.5: 
                    action = Action.BOMB
                else:
                    action = random.choice(safe_actions)
            
            sim_state.apply_action(action)
            depth += 1
        
        return self.evaluate(sim_state)
    
    def evaluate(self, state):
        if not state.player_alive[self.player_id]:
            return -1000
            
        boxes_reward = state.boxes_destroyed[self.player_id]
        num_bombs_reward = state.player_bomb_counts[self.player_id]
        bomb_range_reward = state.player_bomb_ranges[self.player_id]
        reward_total = 5 * boxes_reward + num_bombs_reward + bomb_range_reward
        
        reward_component = reward_total * 50
        
        player_pos = state.player_positions[self.player_id]
        
        box_positions = np.argwhere(state.grid == CellType.BOX.value)
        if len(box_positions) > 0: 
            box_dist = sum(
                abs(pos[0] - player_pos[0]) + abs(pos[1] - player_pos[1]) 
                for pos in box_positions
            ) / len(box_positions)  
        else:
            box_dist = 0
        
        player_dist = float('inf')
        for p_id, pos in enumerate(state.player_positions):
            if p_id != self.player_id and state.player_alive[p_id]:
                dist = abs(pos[0] - player_pos[0]) + abs(pos[1] - player_pos[1])
                player_dist = min(player_dist, dist)
        if player_dist == float('inf'):
            player_dist = 0

        center_x, center_y = state.height // 2, state.width // 2
        centre_dist = abs(player_pos[0] - center_x) + abs(player_pos[1] - center_y)
        
        if state.remaining_boxes > 10:
            distance_penalty = centre_dist + player_dist
        else:
            distance_penalty = box_dist + player_dist
        
        estimated_future_box_reward = self.estimate_future_boxes_destroyed(state)
    
        value = reward_component - distance_penalty + estimated_future_box_reward
        return value
    
    def estimate_future_boxes_destroyed(self, state):
        """
        Estimate future box destruction from existing bombs
        """
        future_state = state.copy()
        
        # Make all players "vanish" 
        for p_id in range(future_state.num_players):
            future_state.player_alive[p_id] = False
        
        # Track boxes destroyed per turn with decay
        gamma = 0.95
        total_future_reward = 0
        
        # Simulate 8 future turns
        for d in range(8):
            # Count boxes before update
            boxes_before = np.sum(future_state.grid == CellType.BOX.value)
            
            # Update bombs only 
            future_state._update_bombs()
            
            # Count boxes after update
            boxes_after = np.sum(future_state.grid == CellType.BOX.value)
            
            # Calculate boxes destroyed in this turn
            boxes_destroyed_this_turn = boxes_before - boxes_after
            
            # Add reward
            total_future_reward += boxes_destroyed_this_turn * (gamma ** (d + 1))
        
        return total_future_reward * 50
    
    @staticmethod
    def _in_explosion_radius(x, y, bx, by, bomb_range, state):
        """Check if a position is within the explosion radius of a bomb"""
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
                if state.grid[nx, ny] == CellType.WALL.value or state.grid[nx, ny] == CellType.BOX.value:
                    break
                    
                # Check if player is in this position
                if (nx, ny) == (x, y):
                    return True
                    
        return False
    
    @staticmethod
    def is_safe_action_simple(state, player_id, action):
        """
        Simplified version of is_safe_action for use during simulation
        """
        # Get current position
        x, y = state.player_positions[player_id]
        
        # Get new position after action
        new_x, new_y = x, y
        if action == Action.UP:
            new_x -= 1
        elif action == Action.RIGHT:
            new_y += 1
        elif action == Action.DOWN:
            new_x += 1
        elif action == Action.LEFT:
            new_y -= 1
        
        # Check if position is out of bounds
        if not (0 <= new_x < state.height and 0 <= new_y < state.width):
            return False
        
        # Check if position has a wall or box
        cell = state.grid[new_x, new_y]
        if cell == CellType.WALL.value or cell == CellType.BOX.value:
            return False
        
        # Check if there's a bomb at this position
        for bx, by, _, _, _ in state.bombs:
            if (new_x, new_y) == (bx, by):
                return False
                
        # Check for bombs about to explode
        for bx, by, timer, bomb_range, _ in state.bombs:
            if timer <= 2 and PureMCTS._in_explosion_radius(new_x, new_y, bx, by, bomb_range, state):
                return False
        
        # Don't place bombs if there are bombs with timer <= 3 nearby
        if action == Action.BOMB:
            for bx, by, timer, bomb_range, _ in state.bombs:
                if timer <= 3 and PureMCTS._in_explosion_radius(x, y, bx, by, bomb_range, state):
                    return False
        
        return True
    
    @staticmethod
    def is_safe_action(state, player_id, action, max_depth=8):
        """
        Uses 8-turns deep BFS to find a safe sequence of actions
        """
        # Create a copy of the state
        test_state = state.copy()
        
        # Save the current player ID
        original_player = test_state.current_player_id
        
        # Set current player to our player and apply the action
        test_state.current_player_id = player_id
        test_state.apply_action(action)
        
        # Complete the round (make other players take STAY actions)
        while test_state.current_player_id != 0:
            test_state.apply_action(Action.STAY)
        
        # If already dead after the action return False
        if not test_state.player_alive[player_id]:
            return False
        
        # BFS queue
        queue = deque([(test_state, 1)])
        
        # Track visited states to avoid cycles
        visited = set()
        
        while queue:
            current_state, depth = queue.popleft()
            
            # If player is dead skip this state
            if not current_state.player_alive[player_id]:
                continue
            
            # If we've survived for max_depth turns consider it safe
            if depth >= max_depth:
                return True
            
            # Get player position
            x, y = current_state.player_positions[player_id]
            
            # Create a state key for visit tracking
            bomb_state = tuple((bx, by, timer) for bx, by, timer, _, _ in sorted(current_state.bombs))
            state_key = ((x, y), depth, bomb_state)
            
            if state_key in visited:
                continue
            
            visited.add(state_key)
            
            # If no bombs left it's safe
            if not current_state.bombs:
                return True
            
            # Try all possible moves from here
            legal_actions = current_state.get_legal_actions(player_id)
            for next_action in legal_actions:
                # Skip BOMB action
                if next_action == Action.BOMB:
                    continue
                
                # Create new state for next step
                next_state = current_state.copy()
                
                # Apply the action and advance through all players' turns
                next_state.current_player_id = player_id
                next_state.apply_action(next_action)
                
                # Complete the round 
                while next_state.current_player_id != 0:
                    next_state.apply_action(Action.STAY)
                
                # Add to queue for next depth
                queue.append((next_state, depth + 1))
        
        # If we've exhausted all possibilities without finding safety
        return False
    
    @staticmethod
    def can_be_trapped_by_enemy(state, player_id, max_trap_turns=2, max_kill_turns=10):
        """
        Check if any enemy player can trap and eventually kill the agent within the specified turns
        """
        if not state.player_alive[player_id]:
            return False
            
        player_pos = state.player_positions[player_id]
        player_x, player_y = player_pos
        
        # Check each enemy
        for enemy_id in range(state.num_players):
            if enemy_id != player_id and state.player_alive[enemy_id]:
                enemy_pos = state.player_positions[enemy_id]
                enemy_x, enemy_y = enemy_pos
                
                # If enemy is too far they can't trap in max_trap_turns
                manhattan_dist = abs(player_x - enemy_x) + abs(player_y - enemy_y)
                if manhattan_dist > max_trap_turns * 2:  # assuming maximum of 2 moves per turn
                    continue
                
                # Check if player is already near a bomb with few escape routes
                near_bomb = False
                for bx, by, timer, _, _ in state.bombs:
                    if abs(bx - player_x) + abs(by - player_y) <= 2 and timer <= 3:
                        near_bomb = True
                        break
                        
                if near_bomb:
                    # Count escape routes
                    escape_routes = 0
                    for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                        nx, ny = player_x + dx, player_y + dy
                        if 0 <= nx < state.height and 0 <= ny < state.width:
                            cell = state.grid[nx, ny]
                            if cell != CellType.WALL.value and cell != CellType.BOX.value:
                                has_bomb = any((nx, ny) == (bx, by) for bx, by, _, _, _ in state.bombs)
                                if not has_bomb:
                                    escape_routes += 1
                    
                    # If player has limited escape routes they might be trapped
                    if escape_routes <= 1 and not state.is_survivable(player_id):
                        return True
                
                # Find positions where the enemy could place bombs within max_trap_turns
                potential_bomb_positions = []
                
                # Use BFS to find all positions the enemy could reach in max_trap_turns
                visited = set([enemy_pos])
                queue = deque([(enemy_pos, 0)])  
                
                while queue:
                    (ex, ey), steps = queue.popleft()
                    
                    if steps <= max_trap_turns:
                        # Position where the enemy could place a bomb
                        potential_bomb_positions.append((ex, ey))
                        
                        # Explore adjacent cells if we have steps left
                        if steps < max_trap_turns:
                            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                                nx, ny = ex + dx, ey + dy
                                if (nx, ny) not in visited and 0 <= nx < state.height and 0 <= ny < state.width:
                                    if state.grid[nx, ny] == CellType.FLOOR.value:
                                        bomb_here = any((nx, ny) == (bx, by) for bx, by, _, _, _ in state.bombs)
                                        if not bomb_here:
                                            visited.add((nx, ny))
                                            queue.append(((nx, ny), steps + 1))
                
                # For each potential bomb position check if placing a bomb there would trap the agent
                bomb_range = state.player_bomb_ranges[enemy_id]
                for bx, by in potential_bomb_positions:
                    # Skip if too far from player to affect them
                    if abs(bx - player_x) + abs(by - player_y) > bomb_range + max_kill_turns:
                        continue
                        
                    # Create a copy of the state and add a bomb
                    test_state = state.copy()
                    test_state.bombs.append((bx, by, 8, bomb_range, enemy_id))  # 8 turns until explosion
                    
                    # Check if the agent can find safety
                    if not test_state.is_survivable(player_id):
                        return True
        
        return False
    
    def save(self, filepath):
        """Save a PureMCTS model to file"""
        model_data = {
            'player_id': self.player_id,
            'num_simulations': self.num_simulations,
            'max_depth': self.max_depth,
            'exploration_weight': self.exploration_weight,
            'heatmap': self.heatmap,
            'action_stats': self.action_stats,
            'rewards_history': self.rewards_history
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    @classmethod
    def load(cls, filepath):
        """Load a PureMCTS model from a file"""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            mcts = cls(
                player_id=model_data['player_id'],
                num_simulations=model_data.get('num_simulations', 100),
                max_depth=model_data.get('max_depth', 50)
            )
            
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
            return cls(player_id=0, num_simulations=100, max_depth=50)