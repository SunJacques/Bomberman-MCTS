"""
AlphaZero-style MCTS for self-supervized learning
"""

import math
import numpy as np
import torch
import torch.nn.functional as F
from engine.game import Action
from policy_value_network.utils import encode_game_state


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Node:
    def __init__(self, game_state, player_id, prior=0, parent=None, action=None):
        self.game_state = game_state
        self.player_id = player_id
        self.parent = parent
        self.action = action
        self.children = {}

        self.visit_count = 0
        self.value_sum = 0
        self.prior = prior  

    def expanded(self):
        return len(self.children) > 0

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def select_child(self, c_puct=1.0):
        """
        Select child according to PUCT formula from AlphaZero
        """
        best_score = -float('inf')
        best_action = None
        best_child = None

        sum_visits = sum(child.visit_count for child in self.children.values())
        sum_visits = max(sum_visits, 1)  # Avoid division by zero

        for action, child in self.children.items():
            q_value = child.value()
            u_value = c_puct * child.prior * math.sqrt(sum_visits) / (1 + child.visit_count)
            score = q_value + u_value

            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        return best_action, best_child


class MCTS:
    def __init__(self, model, player_id, num_simulations=100, c_puct=1.0):
        self.model = model
        self.player_id = player_id
        self.num_simulations = num_simulations
        self.c_puct = c_puct

    def run(self, game):
        root = Node(game.copy(), self.player_id)

        for _ in range(self.num_simulations):
            node = root
            search_path = [node]

            while node.expanded():
                action, node = node.select_child(self.c_puct)
                search_path.append(node)

            game_state = node.game_state.copy()

            if not game_state.is_terminal():
                policy, value = self.evaluate_state(game_state)

                # Add noise to the policy at the root
                if node == root:
                    policy = self.add_dirichlet_noise(policy)

                legal_actions = game_state.get_legal_actions()
                for action in legal_actions:
                    if action not in node.children:
                        next_state = game_state.copy()
                        next_state.apply_action(action)

                        # Create child node with prior probability from policy
                        child = Node(
                            game_state=next_state,
                            player_id=self.player_id,
                            prior=policy[action.value],
                            parent=node,
                            action=action
                        )
                        node.children[action] = child
            else:
                rankings = game_state.get_rankings()
                if rankings[0] == self.player_id:  
                    value = 1.0
                elif rankings[1] == self.player_id:  
                    value = 0.0
                else:  
                    value = 0.5

            for node in reversed(search_path):
                node.visit_count += 1
                node.value_sum += value
                value = 1 - value

        action_probs = np.zeros(6)  
        for action, child in root.children.items():
            action_probs[action.value] = child.visit_count

        if np.sum(action_probs) > 0:
            action_probs = action_probs / np.sum(action_probs)

        return action_probs

    def evaluate_state(self, game_state):
        """Evaluate state using neural network"""
        state = encode_game_state(game_state, self.player_id)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)

        # Inference
        self.model.eval()
        with torch.no_grad():
            policy_logits, value = self.model(state_tensor)
            policy = F.softmax(policy_logits, dim=1).squeeze(0).cpu().numpy()
            value = value.item()

        return policy, value

    def add_dirichlet_noise(self, policy, alpha=0.3, epsilon=0.25):
        """Add Dirichlet noise to policy for exploration the root"""
        noise = np.random.dirichlet([alpha] * len(policy))
        return (1 - epsilon) * policy + epsilon * noise

    def select_action(self, game, temperature=1.0):
        """Select action based on MCTS search results"""
        action_probs = self.run(game)

        # Apply temperature to control exploration/exploitation
        if temperature == 0:
            # Greedy selection
            action_idx = np.argmax(action_probs)
            action = Action(action_idx)
        else:
            # Sample from distribution
            action_probs = action_probs ** (1 / temperature)
            action_probs = action_probs / np.sum(action_probs)
            action_idx = np.random.choice(len(action_probs), p=action_probs)
            action = Action(action_idx)

        return action