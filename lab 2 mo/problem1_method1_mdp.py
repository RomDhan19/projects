"""
Method 1: Марковский процесс принятия решения (Markov Decision Process - MDP)

This method is used for finding optimal decision strategies in stochastic systems
where the outcome of an action depends on the current state and probabilistic
transitions between states.
"""


class MDP:
    """
    Implementation of Markov Decision Process.
    
    Parameters:
    - states: list of states
    - actions: list of available actions
    - transition_probs: transition probabilities P(s'|s,a) [state][action][next_state]
    - rewards: rewards R(s,a,s') [state][action][next_state]
    - discount: discount factor (gamma)
    """
    
    def __init__(self, states, actions, transition_probs, rewards, discount=0.9):
        self.states = states
        self.actions = actions
        self.transition_probs = transition_probs
        self.rewards = rewards
        self.discount = discount
        self.num_states = len(states)
        self.num_actions = len(actions)
    
    def value_iteration(self, theta=1e-6, max_iterations=1000):
        """
        Value iteration algorithm for finding optimal policy.
        
        Returns:
        - values: optimal state values (list)
        - policy: optimal policy (dict: state -> action)
        """
        V = [0.0] * self.num_states
        
        for iteration in range(max_iterations):
            V_prev = V[:]
            
            for s in range(self.num_states):
                # Calculate Q-values for all actions
                q_values = []
                for a in range(self.num_actions):
                    q = 0.0
                    for s_next in range(self.num_states):
                        prob = self.transition_probs[s][a][s_next]
                        reward = self.rewards[s][a][s_next]
                        q += prob * (reward + self.discount * V_prev[s_next])
                    q_values.append(q)
                
                # Choose action with maximum Q-value
                V[s] = max(q_values)
            
            # Check convergence
            max_diff = max(abs(V[i] - V_prev[i]) for i in range(self.num_states))
            if max_diff < theta:
                break
        
        # Extract optimal policy
        policy = {}
        for s in range(self.num_states):
            q_values = []
            for a in range(self.num_actions):
                q = 0.0
                for s_next in range(self.num_states):
                    prob = self.transition_probs[s][a][s_next]
                    reward = self.rewards[s][a][s_next]
                    q += prob * (reward + self.discount * V[s_next])
                q_values.append(q)
            policy[s] = q_values.index(max(q_values))
        
        return V, policy


def create_inventory_mdp():
    """
    Create MDP example for inventory management problem.
    
    Problem: Warehouse inventory management
    - State 0: low inventory
    - State 1: medium inventory
    - State 2: high inventory
    - Action 0: don't order
    - Action 1: order
    """
    states = [0, 1, 2]
    actions = [0, 1]
    
    # Transition probabilities [state][action][next_state]
    transition_probs = [
        # State 0 (low inventory)
        [
            [0.7, 0.3, 0.0],  # Action 0: don't order
            [0.0, 0.8, 0.2]   # Action 1: order
        ],
        # State 1 (medium inventory)
        [
            [0.2, 0.6, 0.2],  # Action 0: don't order
            [0.0, 0.0, 1.0]   # Action 1: order
        ],
        # State 2 (high inventory)
        [
            [0.0, 0.3, 0.7],  # Action 0: don't order
            [0.0, 0.0, 1.0]   # Action 1: order
        ]
    ]
    
    # Rewards [state][action][next_state]
    rewards = [
        # State 0
        [
            [-10, -10, 0],  # Action 0: penalty for low inventory
            [-2, -2, -2]    # Action 1: order cost
        ],
        # State 1
        [
            [0, 0, 0],      # Action 0: no reward/penalty
            [-2, -2, -2]    # Action 1: order cost
        ],
        # State 2
        [
            [5, 5, 5],      # Action 0: reward for good inventory
            [-2, -2, -2]    # Action 1: order cost
        ]
    ]
    
    return MDP(states, actions, transition_probs, rewards, discount=0.9)

