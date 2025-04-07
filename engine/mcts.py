import math
import random
import numpy as np
import pickle
import os
from engine.game import Action, CellType

class Node:
    """Node in the MCTS tree."""
    def __init__(self, game_state, parent=None, action=None):
        self.game_state = game_state
        self.parent = parent
        self.action = action  # Action that led to this state
        self.children = {}  # Map from action to Node
        self.visits = 0
        self.value = 0.0
        self.max_value = float('-inf')
        self.player_id = game_state.current_player_id
        
    def is_fully_expanded(self):
        """Check if all possible actions have been expanded."""
        legal_actions = self.game_state.get_legal_actions()
        return len(self.children) == len(legal_actions)
        
    def select_child(self, exploration_weight=1.0):
        """Select a child node using the UCT formula."""
        # If not fully visited, choose a random unexpanded action
        if not self.is_fully_expanded():
            legal_actions = self.game_state.get_legal_actions()
            unexpanded = [a for a in legal_actions if a not in self.children]
            return self.expand(random.choice(unexpanded))
            
        # Use UCT to select the best child
        log_visits = math.log(self.visits)
        
        def uct(child):
            exploit = child.value / child.visits
            explore = exploration_weight * math.sqrt(log_visits / child.visits)
            return exploit + explore
            
        return max(self.children.values(), key=uct)
        
    def expand(self, action):
        """Expand the tree by adding a new child node."""
        if action in self.children:
            return self.children[action]
            
        # Create a copy of the game state
        child_state = self.game_state.copy()
        
        # Apply the action
        child_state.apply_action(action)
        
        # Create a new child node
        child = Node(child_state, parent=self, action=action)
        self.children[action] = child
        
        return child
        
    def update(self, reward):
        """Update node statistics with a new reward."""
        self.visits += 1
        self.value += reward
        self.max_value = max(self.max_value, reward)
        
    def get_best_action(self, use_max=False):
        """Get the best action based on visit count or value."""
        if not self.children:
            # If no children, return STAY as a default
            return Action.STAY
            
        if use_max:
            # Choose action with the highest max value
            return max(self.children.items(), key=lambda x: x[1].max_value)[0]
        else:
            # Choose action with the highest visit count
            return max(self.children.items(), key=lambda x: x[1].visits)[0]

class MCTS:
    """Monte Carlo Tree Search implementation."""
    def __init__(self, player_id, num_simulations=500, exploration_weight=1.0, max_depth=10):
        self.player_id = player_id
        self.num_simulations = num_simulations
        self.exploration_weight = exploration_weight
        self.max_depth = max_depth
        self.discount_factor = 0.95 
        
        # Learning parameters
        self.rewards_history = []  # Track rewards history
        self.action_stats = {}  # Maps (state_hash, action) to (visits, value)
        self.position_weights = np.ones((9, 11)) * 0.5  # Default position weights (note height, width swap)
        
        # MCTS can learn which positions are favorable over time
        self.heatmap = np.ones((9, 11)) * 0.5  # Initial neutral heatmap (note height, width swap)
        
        # Performance optimization
        self.action_priors = {}  # For quickly filtering unlikely actions
        
        # Adaptive parameters (can change during training)
        self.adaptive_exploration = True
        self.min_exploration_weight = 0.5
    
    def select_action(self, game, fast_mode=False):
        """Run MCTS and return the best action."""
        # Create a copy of the game state to avoid modifying the original
        root = Node(game.copy())
        
        # In fast mode, reduce simulations for quicker decision making
        simulations = self.num_simulations // 2 if fast_mode else self.num_simulations
        
        # Run simulations
        for i in range(simulations):
            # Selection and expansion
            node = self._select_and_expand(root)
            
            # Simulation
            reward = self._simulate(node, fast_mode)
            
            # Backpropagation
            self._backpropagate(node, reward)
            
            # Update action stats
            if i % 5 == 0:  # Only update periodically for speed
                self._update_action_stats(node, reward)
        
        # Return the best action
        best_action = root.get_best_action(use_max=True)
        
        # Update learning based on chosen action
        if best_action != Action.STAY and best_action != Action.BOMB:
            # Update position heatmap for movement actions
            x, y = game.player_positions[self.player_id]
            new_x, new_y = game._get_new_position(x, y, best_action)
            if 0 <= new_x < game.height and 0 <= new_y < game.width:
                # Slightly increase probability of choosing this position again
                self.heatmap[new_x, new_y] += 0.01
                # Normalize to keep values in reasonable range
                self.heatmap = np.clip(self.heatmap, 0.1, 1.0)
        
        return best_action
    
    def _select_and_expand(self, node):
        """Select a node to simulate from and expand if necessary."""
        # Quick terminal check to avoid unnecessary computation
        if node.game_state.is_terminal():
            return node
            
        while not node.game_state.is_terminal():
            if not node.is_fully_expanded():
                legal_actions = node.game_state.get_legal_actions()
                
                # Prune legal actions if we have prior knowledge
                if self.adaptive_exploration:
                    filtered_actions = self._prune_actions(node.game_state, legal_actions)
                    if filtered_actions:
                        legal_actions = filtered_actions
                
                unexpanded = [a for a in legal_actions if a not in node.children]
                if unexpanded:
                    # Prefer actions that lead toward boxes or away from bombs
                    return node.expand(self._choose_promising_action(node.game_state, unexpanded))
            
            # Adaptive exploration weight
            if self.adaptive_exploration:
                # Use a smaller exploration weight as we get deeper
                depth_factor = max(self.min_exploration_weight, 
                                  1.0 - 0.05 * len(self._get_ancestors(node)))
                exploration = self.exploration_weight * depth_factor
            else:
                exploration = self.exploration_weight
                
            node = node.select_child(exploration)
        
        return node
    
    def _get_ancestors(self, node):
        """Get number of ancestors for a node."""
        ancestors = []
        while node.parent is not None:
            ancestors.append(node.parent)
            node = node.parent
        return ancestors
    
    def _prune_actions(self, state, legal_actions):
        """Filter legal actions based on learned experience."""
        state_hash = self._get_state_hash(state)
        
        if state_hash in self.action_priors and random.random() < 0.7:
            # Get the top 2 actions by prior value
            priors = self.action_priors[state_hash]
            if len(priors) > 0:
                # Filter to top 2 or all if fewer
                top_actions = [a for a, _ in priors[:min(2, len(priors))]]
                filtered = [a for a in legal_actions if a in top_actions]
                if filtered:
                    return filtered
        
        return legal_actions
    
    def _choose_promising_action(self, state, actions):
        """Choose a promising action for expansion."""
        # First check if there's a bomb action and player is near a box
        if Action.BOMB in actions:
            x, y = state.player_positions[self.player_id]
            near_box = False
            
            # Check if there's a box in adjacent cells
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if (0 <= nx < state.height and 0 <= ny < state.width and 
                    state.grid[nx, ny] == CellType.BOX.value):
                    near_box = True
                    break
            
            if near_box:
                return Action.BOMB
        
        # Otherwise choose randomly but prefer movement over staying
        move_actions = [a for a in actions if a != Action.STAY]
        if move_actions and random.random() < 0.8:
            return random.choice(move_actions)
        
        return random.choice(actions)
    
    def _simulate(self, node, fast_mode=False):
        """Simulate a random game from the node and return the reward."""
        # If terminal state, evaluate and return
        if node.game_state.is_terminal():
            return self._evaluate(node.game_state)
        
        # Create a copy for simulation
        state = node.game_state.copy()
        depth = 0
        
        # Initialize cumulative reward
        cumulative_reward = 0
        
        # Adjust max_depth based on fast_mode
        max_depth = self.max_depth // 2 if fast_mode else self.max_depth
        
        # Simulate random actions until terminal or max depth
        while not state.is_terminal() and depth < max_depth:
            # Get the current player
            current_player = state.current_player_id
            
            # Choose a random action, but bias toward survival for self
            action = self._choose_smart_action(state, current_player, fast_mode)
            
            # Apply the action
            state.apply_action(action)
            
            # Calculate intermediate reward
            if current_player == self.player_id:
                step_reward = self._calculate_step_reward(state, depth)
                cumulative_reward += step_reward * (self.discount_factor ** depth)
            
            depth += 1
            
            # Early termination for speed
            if fast_mode and depth > max_depth // 2 and random.random() < 0.5:
                break
        
        # Final evaluation
        final_reward = self._evaluate(state)
        
        # Combine cumulative and final rewards
        total_reward = cumulative_reward + final_reward
        
        # Only store rewards occasionally to save memory
        if len(self.rewards_history) < 10000 and random.random() < 0.1:
            self.rewards_history.append(total_reward)
        
        return total_reward
    
    def _choose_smart_action(self, state, player_id, fast_mode=False):
        """Choose a somewhat smart action for simulation."""
        legal_actions = state.get_legal_actions()
        
        # If only one legal action, return it
        if len(legal_actions) == 1:
            return legal_actions[0]
        
        # If this is the MCTS player, be smarter
        if player_id == self.player_id:
            # Significantly increased chance to place bombs near boxes
            if random.random() < 0.5:  # Increased from 0.4 to 0.7
                x, y = state.player_positions[player_id]
                near_box = False
                
                # Check if there's a box in adjacent cells
                for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    nx, ny = x + dx, y + dy
                    if (0 <= nx < state.height and 0 <= ny < state.width and 
                        state.grid[nx, ny] == CellType.BOX.value):
                        near_box = True
                        break
                
                # If near box and can place bomb, do so with higher probability
                if near_box and Action.BOMB in legal_actions:
                    return Action.BOMB
            
            # Move toward boxes if not already next to one
            if random.random() < 0.6:  # Add probability for box-seeking behavior
                x, y = state.player_positions[player_id]
                closest_box = None
                min_dist = float('inf')
                
                # Find the closest box
                for i in range(state.height):
                    for j in range(state.width):
                        if state.grid[i, j] == CellType.BOX.value:
                            dist = abs(x - i) + abs(y - j)
                            if dist < min_dist:
                                min_dist = dist
                                closest_box = (i, j)
                
                # If found a box, try to move toward it
                if closest_box and min_dist > 1:
                    box_x, box_y = closest_box
                    # Determine direction to move
                    if x < box_x and Action.DOWN in legal_actions:
                        return Action.DOWN
                    elif x > box_x and Action.UP in legal_actions:
                        return Action.UP
                    elif y < box_y and Action.RIGHT in legal_actions:
                        return Action.RIGHT
                    elif y > box_y and Action.LEFT in legal_actions:
                        return Action.LEFT
            
            # In fast mode, prefer random movement
            if fast_mode:
                move_actions = [a for a in legal_actions if a not in [Action.STAY, Action.BOMB]]
                if move_actions:
                    return random.choice(move_actions)
            
            # Rarely stay still
            if Action.STAY in legal_actions and random.random() < 0.1:  # Reduced from default
                return Action.STAY
            
            # Use learned heatmap for movement actions
            move_actions = [a for a in legal_actions if a not in [Action.STAY, Action.BOMB]]
            if move_actions and random.random() < 0.6:  # 70% chance to use heatmap
                x, y = state.player_positions[player_id]
                action_scores = []
                
                for action in move_actions:
                    nx, ny = state._get_new_position(x, y, action)
                    if 0 <= nx < state.height and 0 <= ny < state.width:
                        score = self.heatmap[nx, ny]
                        action_scores.append((action, score))
                
                if action_scores:
                    # Choose action with probability proportional to score
                    total_score = sum(score for _, score in action_scores)
                    if total_score > 0:
                        choice = random.random() * total_score
                        cumsum = 0
                        for action, score in action_scores:
                            cumsum += score
                            if cumsum >= choice:
                                return action
        
        # For opponents, in fast mode just pick randomly
        if fast_mode:
            return random.choice(legal_actions)
            
        # For opponents, use a simpler strategy
        # Prefer movement and bombing over staying
        action_weights = {a: 1.0 for a in legal_actions}
        if Action.STAY in action_weights:
            action_weights[Action.STAY] = 0.5  # Lower probability for staying
        
        if Action.BOMB in action_weights:
            # Check if there are boxes nearby to bomb
            x, y = state.player_positions[player_id]
            box_nearby = False
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if (0 <= nx < state.height and 0 <= ny < state.width and 
                    state.grid[nx, ny] == CellType.BOX.value):
                    box_nearby = True
                    break
            
            if box_nearby:
                action_weights[Action.BOMB] = 2.0  # Higher probability for bomb if box nearby
        
        # Choose action based on weights
        total_weight = sum(action_weights.values())
        choice = random.random() * total_weight
        cumsum = 0
        for action, weight in action_weights.items():
            cumsum += weight
            if cumsum >= choice:
                return action
        
        # Default to random action
        return random.choice(legal_actions)
    
    def _update_action_stats(self, node, reward):
        """Update action statistics for learning."""
        # For now, we'll use a simple state hash based on player positions and bomb locations
        # In a real implementation, you would want a more sophisticated state representation
        if node.action is not None and node.parent is not None:
            parent_state = node.parent.game_state
            state_hash = self._get_state_hash(parent_state)
            action = node.action
            
            key = (state_hash, action)
            if key not in self.action_stats:
                self.action_stats[key] = [0, 0.0]  # [visits, total_value]
            
            self.action_stats[key][0] += 1  # Increment visits
            self.action_stats[key][1] += reward  # Add reward
            
            # Update action priors for this state
            if state_hash not in self.action_priors:
                self.action_priors[state_hash] = []
            
            # Calculate average value for this action
            avg_value = self.action_stats[key][1] / self.action_stats[key][0]
            
            # Update the action prior list
            prior_exists = False
            for i, (a, v) in enumerate(self.action_priors[state_hash]):
                if a == action:
                    # Replace with new value
                    self.action_priors[state_hash][i] = (action, avg_value)
                    prior_exists = True
                    break
            
            if not prior_exists:
                self.action_priors[state_hash].append((action, avg_value))
            
            # Sort by value
            self.action_priors[state_hash].sort(key=lambda x: x[1], reverse=True)
            
            # Limit the size of the list to save memory
            self.action_priors[state_hash] = self.action_priors[state_hash][:3]
    
    def _get_state_hash(self, state):
        """Create a simple hash for a game state."""
        # Hash based on player positions and bomb locations
        # This is a simplification - a real implementation would need a better state representation
        player_pos = tuple(state.player_positions)
        bomb_info = tuple((x, y, t) for x, y, t, _, _ in state.bombs)
        
        # Try to be more general by simplifying the hash:
        # Include only the nearest bombs and group bomb timers in ranges
        simplified_bombs = []
        if bomb_info:
            for x, y, t in bomb_info[:3]:  # Consider only up to 3 bombs
                # Group bomb timers: 0-2, 3-5, 6-8
                timer_group = t // 3
                simplified_bombs.append((x, y, timer_group))
        
        return hash((player_pos, tuple(simplified_bombs)))
    
    def _calculate_step_reward(self, state, depth):
        """Calculate intermediate reward for a step."""
        # Base reward
        reward = 0
        
        # Check if the player is alive
        if not state.player_alive[self.player_id]:
            return -1000  # Heavy penalty for death
        
        # Reward for destroying boxes - SIGNIFICANTLY INCREASED
        boxes_destroyed = state.boxes_destroyed[self.player_id]
        reward += boxes_destroyed * 2.0  # Increased from 1.0 to 5.0
        
        # Reward for power-ups - INCREASED
        reward += 3.4 * min(2,state.player_bomb_counts[self.player_id]) + \
                  1.7 * min(4,state.player_bomb_counts[self.player_id]) + \
                  0.7 * state.player_bomb_counts[self.player_id] # Increased from 2 to 5
        reward += 0.9 * min(5,state.player_bomb_ranges[self.player_id]) + 0.4 * state.player_bomb_ranges[self.player_id] # Increased from 3 to 7
        
        # Distance-based rewards (encourage movement toward boxes) - IMPROVED
        x, y = state.player_positions[self.player_id]
        min_box_dist = float('inf')
        box_count = 0
        total_box_dist = 0
        
        for i in range(state.height):
            for j in range(state.width):
                if state.grid[i, j] == CellType.BOX.value:
                    box_count += 1
                    dist = abs(x - i) + abs(y - j)  # Manhattan distance
                    total_box_dist += dist
                    min_box_dist = min(min_box_dist, dist)
        
        avg_box_dist = total_box_dist / max(1, box_count)
        
        # if min_box_dist != float('inf'):
        #     # Higher reward for being close to boxes
        #     reward += 10.0 / (1.0 + min_box_dist)  # Increased from 5.0 to 10.0
            
        #     # Add higher reward for placing bombs near boxes
        #     if min_box_dist <= 1 and Action.BOMB in state.get_legal_actions():
        #         reward += 15.0  # Significant bonus for bombing when next to boxes
        
        # Distance to other players penalty - encourages pursuing other players
        player_dist = 0
        player_count = 0
        for p_id, (px, py) in enumerate(state.player_positions):
            if p_id != self.player_id and state.player_alive[p_id]:
                player_count += 1
                player_dist += abs(x - px) + abs(y - py)
        
        if player_count > 0:
            avg_player_dist = player_dist / player_count
            # Penalty for being far from other players (encourages combat)
            reward -= 0.05 * avg_player_dist
        
        # Center or box targeting based on game state
        center_x, center_y = state.height // 2, state.width // 2
        distance_to_center = abs(x - center_x) + abs(y - center_y)
        
        if state.remaining_boxes > 20:
            # Early game: encourage moving to center
            reward -= 0.04 * distance_to_center
        else:
            # Late game: focus on remaining boxes
            reward -= 0.1 * avg_box_dist
        
        # Add position-based reward using the heatmap
        reward += self.heatmap[x, y] * 2.0  # Reward based on position preference
        
        # Reward for placing bombs (to encourage more bombing)
        bomb_placed = False
        for bx, by, _, _, p_id in state.bombs:
            if p_id == self.player_id:
                bomb_placed = True
                break
        
        if bomb_placed:
            reward += 5.0  # Bonus for having active bombs
        
        # # Penalize being too close to bombs - adjusted to be less risk-averse
        # for bx, by, timer, bomb_range, _ in state.bombs:
        #     dist = abs(x - bx) + abs(y - by)
        #     if dist <= bomb_range and timer <= 2:  # Only penalize immediate threats
        #         reward -= (5.0 / (timer + 1))  # Reduced from 10.0 to 5.0
        
        # Penalize staying in the same spot for too long
        # This encourages movement and exploration
        if Action.STAY in state.get_legal_actions() and random.random() < 0.7:
            reward -= 1.0  # Penalty for staying still
        
        return reward
    
    def _evaluate(self, state):
        """Evaluate a terminal state."""
        if not state.player_alive[self.player_id]:
            return -1000  # Heavy penalty for death
        
        reward = 0
        
        # Reward for destroying boxes - SIGNIFICANTLY INCREASED
        boxes_destroyed = state.boxes_destroyed[self.player_id]
        reward += boxes_destroyed * 50.0  # Increased from 10.0 to 50.0
        
        # Reward for power-ups - INCREASED
        reward += (state.player_bomb_counts[self.player_id] - 1) * 40  # Increased from 20 to 40
        reward += (state.player_bomb_ranges[self.player_id] - 2) * 60  # Increased from 30 to 60
        
        # Reward for being alive at the end
        if state.player_alive[self.player_id]:
            reward += 500
            
            # Extra reward for winning
            rankings = state.get_rankings()
            if rankings[0] == self.player_id:
                reward += 1000
                
                # Additional bonus for winning with many boxes destroyed
                if boxes_destroyed > 5:
                    reward += boxes_destroyed * 30  # Extra bonus for aggressive winners
        
        # Consider future bombs if not terminal due to player elimination
        estimated_bombs = state.estimated_bombs()
        expected_bombs = estimated_bombs[self.player_id]
        reward += expected_bombs * 10  # Increased from 5 to 10
        
        # Calculate distance to boxes for distance-based reward
        x, y = state.player_positions[self.player_id]
        box_dist_sum = 0
        box_count = 0
        
        for i in range(state.height):
            for j in range(state.width):
                if state.grid[i, j] == CellType.BOX.value:
                    dist = abs(x - i) + abs(y - j)  # Manhattan distance
                    box_dist_sum += dist
                    box_count += 1
        
        # Calculate average distance to boxes
        avg_box_dist = box_dist_sum / max(1, box_count)
        
        # Higher reward for being close to boxes at the end
        if box_count > 0:
            reward += (300 - box_dist_sum / box_count) * 1.0  # Increased from 200 to 300
        
        # Distance to other players penalty
        player_dist = 0
        player_count = 0
        for p_id, (px, py) in enumerate(state.player_positions):
            if p_id != self.player_id and state.player_alive[p_id]:
                player_count += 1
                player_dist += abs(x - px) + abs(y - py)
        
        if player_count > 0:
            avg_player_dist = player_dist / player_count
            # Penalty for being far from other players (encourages combat)
            reward -= 0.05 * avg_player_dist
        
        # Center or box targeting based on game state
        center_x, center_y = state.height // 2, state.width // 2
        distance_to_center = abs(x - center_x) + abs(y - center_y)
        
        if state.remaining_boxes > 20:
            # Early game: encourage moving to center
            reward -= 0.04 * distance_to_center
        else:
            # Late game: focus on remaining boxes
            reward -= 0.1 * avg_box_dist
        
        return reward
    
    def _backpropagate(self, node, reward):
        """Backpropagate the reward up the tree."""
        while node is not None:
            node.update(reward)
            node = node.parent
    
    def clean_memory(self):
        """Clean up memory by reducing the size of storage."""
        # Limit rewards history
        if len(self.rewards_history) > 1000:
            # Keep only the most recent rewards
            self.rewards_history = self.rewards_history[-1000:]
        
        # Limit action_stats
        if len(self.action_stats) > 5000:
            # Convert to sorted list
            items = list(self.action_stats.items())
            # Sort by visits, keep most visited
            items.sort(key=lambda x: x[1][0], reverse=True)
            # Keep top 5000
            self.action_stats = dict(items[:5000])
        
        # Limit action_priors
        if len(self.action_priors) > 1000:
            # Keep only 1000 random items
            keys = list(self.action_priors.keys())
            random.shuffle(keys)
            self.action_priors = {k: self.action_priors[k] for k in keys[:1000]}
    
    def save(self, filepath):
        """Save the MCTS model to a file."""
        # Clean memory before saving
        self.clean_memory()
        
        # Create a dictionary with the model parameters
        model_data = {
            'player_id': self.player_id,
            'num_simulations': self.num_simulations,
            'exploration_weight': self.exploration_weight,
            'max_depth': self.max_depth,
            'discount_factor': self.discount_factor,
            'heatmap': self.heatmap,
            'action_stats': self.action_stats,
            'rewards_history': self.rewards_history,
            'action_priors': self.action_priors
        }
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save to file
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    @classmethod
    def load(cls, filepath):
        """Load an MCTS model from a file."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # Create a new instance
        mcts = cls(
            player_id=model_data['player_id'],
            num_simulations=model_data['num_simulations'],
            exploration_weight=model_data['exploration_weight'],
            max_depth=model_data['max_depth']
        )
        
        # Load learned parameters
        mcts.discount_factor = model_data['discount_factor']
        mcts.heatmap = model_data['heatmap']
        mcts.action_stats = model_data['action_stats']
        mcts.rewards_history = model_data['rewards_history']
        
        # Load action priors if available (for backward compatibility)
        if 'action_priors' in model_data:
            mcts.action_priors = model_data['action_priors']
        
        return mcts 