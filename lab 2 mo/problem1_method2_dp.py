"""
Method 2: Dynamic Programming (Deterministic Optimization)

Alternative method for solving multi-stage optimization problems.
This method is suitable when the problem can be modeled deterministically
with known stage transitions and costs.

"""


class DynamicProgramming:
    """
    Implementation of Dynamic Programming for deterministic multi-stage optimization.
    
    This method solves problems by breaking them into stages and solving
    backwards from the terminal stage.
    """
    
    def __init__(self, stages, states, transition_costs, terminal_rewards):
        """
        Initialize the Dynamic Programming problem.
        
        Parameters:
        - stages: list of stage indices
        - states: list of possible states
        - transition_costs: dict {(stage, state, action, next_state): cost}
        - terminal_rewards: dict {state: reward} - rewards at final stage
        """
        self.stages = stages
        self.states = states
        self.transition_costs = transition_costs
        self.terminal_rewards = terminal_rewards
        self.num_stages = len(stages)
    
    def solve(self):
        """
        Solve using backward induction (dynamic programming).
        
        Returns:
        - optimal_values: optimal value for each state at each stage (dict)
        - optimal_policy: optimal action for each state at each stage (dict)
        """
        optimal_values = {}
        optimal_policy = {}
        
        # Initialize terminal stage
        for state in self.states:
            optimal_values[(self.num_stages - 1, state)] = self.terminal_rewards.get(state, 0.0)
        
        # Backward induction
        for stage in range(self.num_stages - 2, -1, -1):
            for state in self.states:
                best_value = float('-inf')
                best_action = None
                
                # Find all possible transitions from current state
                possible_transitions = {}
                for (s, st, action, next_st), cost in self.transition_costs.items():
                    if s == stage and st == state:
                        if action not in possible_transitions:
                            possible_transitions[action] = []
                        possible_transitions[action].append((next_st, cost))
                
                # For each possible action, calculate expected value
                for action, transitions in possible_transitions.items():
                    if len(transitions) == 0:
                        continue
                    
                    # Calculate value for this action (average if multiple transitions)
                    total_value = 0.0
                    for next_state, cost in transitions:
                        next_value = optimal_values.get((stage + 1, next_state), 0.0)
                        total_value += next_value - cost
                    
                    avg_value = total_value / len(transitions)
                    if avg_value > best_value:
                        best_value = avg_value
                        best_action = action
                
                if best_action is not None:
                    optimal_values[(stage, state)] = best_value
                    optimal_policy[(stage, state)] = best_action
        
        return optimal_values, optimal_policy


def create_inventory_dp():
    """
    Create Dynamic Programming example for inventory management problem.
    
    This is a deterministic version of the inventory problem over multiple periods.
    """
    stages = [0, 1, 2, 3]  # 4 time periods
    states = [0, 1, 2]  # 0=low, 1=medium, 2=high inventory
    
    # Transition costs: (stage, state, action, next_state): cost
    # Action 0 = don't order, Action 1 = order
    transition_costs = {}
    
    # Define transitions for stage 0
    transition_costs[(0, 0, 0, 0)] = 10  # low -> low (penalty)
    transition_costs[(0, 0, 0, 1)] = 5   # low -> medium
    transition_costs[(0, 0, 1, 1)] = 2   # order: low -> medium (order cost)
    transition_costs[(0, 0, 1, 2)] = 2   # order: low -> high
    
    transition_costs[(0, 1, 0, 0)] = 3   # medium -> low
    transition_costs[(0, 1, 0, 1)] = 0   # medium -> medium
    transition_costs[(0, 1, 0, 2)] = -2  # medium -> high (reward)
    transition_costs[(0, 1, 1, 2)] = 2   # order: medium -> high
    
    transition_costs[(0, 2, 0, 1)] = 0   # high -> medium
    transition_costs[(0, 2, 0, 2)] = -5  # high -> high (reward)
    transition_costs[(0, 2, 1, 2)] = 2   # order: high -> high
    
    # Copy structure for stages 1 and 2
    for stage in [1, 2]:
        for state in states:
            for action in [0, 1]:
                for next_state in states:
                    if (0, state, action, next_state) in transition_costs:
                        transition_costs[(stage, state, action, next_state)] = \
                            transition_costs[(0, state, action, next_state)]
    
    # Terminal rewards
    terminal_rewards = {
        0: 0,   # No reward for low inventory
        1: 5,   # Small reward for medium
        2: 10   # Good reward for high inventory
    }
    
    return DynamicProgramming(stages, states, transition_costs, terminal_rewards)

