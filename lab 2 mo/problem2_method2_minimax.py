"""
Method 2: Minimax Algorithm (Pure Strategies)

Alternative method for game theory problems.
Finds optimal pure strategies using the minimax principle.

"""


class Minimax:
    """
    Implementation of Minimax algorithm for two-player zero-sum games.
    
    This method finds the optimal pure strategy (deterministic choice)
    using the minimax principle.
    """
    
    def __init__(self, payoff_matrix):
        """
        Initialize the Minimax game solver.
        
        Parameters:
        - payoff_matrix: list of lists, payoff_matrix[i][j] is payoff for player 1
                         when player 1 chooses row i and player 2 chooses column j
        """
        self.payoff_matrix = payoff_matrix
        self.m = len(payoff_matrix)  # number of rows (player 1 strategies)
        self.n = len(payoff_matrix[0])  # number of columns (player 2 strategies)
    
    def solve(self):
        """
        Solve the game using minimax principle.
        
        Returns:
        - player1_strategy: optimal strategy index for player 1 (int)
        - player2_strategy: optimal strategy index for player 2 (int)
        - game_value: value of the game (float)
        """
        # Player 1 wants to maximize the minimum payoff (maximin)
        maxmin_value = float('-inf')
        player1_strategy = 0
        
        for i in range(self.m):
            # Find minimum payoff for player 1 when choosing strategy i
            min_payoff = min(self.payoff_matrix[i][j] for j in range(self.n))
            
            # Choose strategy that maximizes the minimum payoff
            if min_payoff > maxmin_value:
                maxmin_value = min_payoff
                player1_strategy = i
        
        # Player 2 wants to minimize the maximum payoff (minimax)
        minimax_value = float('inf')
        player2_strategy = 0
        
        for j in range(self.n):
            # Find maximum payoff for player 1 when player 2 chooses strategy j
            max_payoff = max(self.payoff_matrix[i][j] for i in range(self.m))
            
            # Choose strategy that minimizes the maximum payoff
            if max_payoff < minimax_value:
                minimax_value = max_payoff
                player2_strategy = j
        
        # Game value (average of maxmin and minimax)
        game_value = (maxmin_value + minimax_value) / 2
        
        return player1_strategy, player2_strategy, game_value
    
    def is_saddle_point(self):
        """
        Check if the game has a saddle point (pure strategy equilibrium).
        
        Returns:
        - has_saddle: True if saddle point exists
        - saddle_point: tuple (row, col) if exists, None otherwise
        """
        player1_strategy, player2_strategy, game_value = self.solve()
        
        # Check if this is actually a saddle point
        maxmin = max(min(self.payoff_matrix[i][j] for j in range(self.n)) 
                    for i in range(self.m))
        minimax = min(max(self.payoff_matrix[i][j] for i in range(self.m)) 
                     for j in range(self.n))
        
        if abs(maxmin - minimax) < 1e-6:
            return True, (player1_strategy, player2_strategy)
        else:
            return False, None


def create_investment_minimax():
    """
    Create Minimax example for investment decision under uncertainty.
    """
    # Payoff matrix: rows = investment strategies, columns = market conditions
    payoff_matrix = [
        [-2, 2, 3],   # Conservative strategy
        [-5, 5, 7],   # Moderate strategy
        [-10, 8, 15], # Aggressive strategy
    ]
    
    return Minimax(payoff_matrix)

