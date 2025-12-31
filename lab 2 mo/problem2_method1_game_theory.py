"""
Method 1: Поиск цены игры по платежной матрице (Game Theory - Mixed Strategies)

This method finds optimal mixed strategies for players in a zero-sum matrix game.
In mixed strategies, players choose actions randomly according to probability distributions.

"""


class GameTheory:
    """
    Implementation of game theory methods for finding optimal mixed strategies.
    
    Parameters:
    - payoff_matrix: payoff matrix of the game (from player 1's perspective)
                    list of lists: payoff_matrix[i][j] is payoff for player 1
    """
    
    def __init__(self, payoff_matrix):
        self.payoff_matrix = payoff_matrix
        self.m = len(payoff_matrix)  # number of rows (player 1 strategies)
        self.n = len(payoff_matrix[0])  # number of columns (player 2 strategies)
    
    def solve_game(self):
        """
        Solve the game for both players using iterative gradient method.
        
        Returns:
        - strategy1: optimal mixed strategy for player 1 (dict)
        - strategy2: optimal mixed strategy for player 2 (dict)
        - value: game value (float)
        """
        # Solve for player 1 (maximin)
        strategy1, value1 = self._solve_player1()
        
        # Solve for player 2 (minimax)
        strategy2, value2 = self._solve_player2()
        
        # Game value (average of both solutions)
        value = (value1 + value2) / 2
        
        return strategy1, strategy2, value
    
    def _solve_player1(self):
        """
        Solve for player 1 using gradient ascent.
        """
        # Initial uniform distribution
        p = [1.0 / self.m] * self.m
        learning_rate = 0.1
        
        for iteration in range(500):
            # Calculate minimum payoff for each opponent strategy
            payoffs = []
            for j in range(self.n):
                payoff = sum(p[i] * self.payoff_matrix[i][j] for i in range(self.m))
                payoffs.append(payoff)
            
            min_payoff = min(payoffs)
            
            # Calculate gradients
            gradients = []
            for i in range(self.m):
                avg_payoff = sum(self.payoff_matrix[i][j] for j in range(self.n)) / self.n
                gradient = avg_payoff - min_payoff
                gradients.append(gradient)
            
            # Update strategy (gradient ascent)
            new_p = []
            for i in range(self.m):
                new_val = p[i] + learning_rate * gradients[i]
                new_p.append(max(0.0, new_val))
            
            # Normalize
            total = sum(new_p)
            if total > 1e-10:
                new_p = [x / total for x in new_p]
            else:
                new_p = [1.0 / self.m] * self.m
            
            # Check convergence
            if iteration > 50:
                max_diff = max(abs(new_p[i] - p[i]) for i in range(self.m))
                if max_diff < 1e-6:
                    break
            
            p = new_p
            learning_rate *= 0.999
        
        # Calculate game value
        value = min(sum(p[i] * self.payoff_matrix[i][j] for i in range(self.m)) 
                   for j in range(self.n))
        
        strategy = {i: p[i] for i in range(self.m)}
        return strategy, value
    
    def _solve_player2(self):
        """
        Solve for player 2 using gradient descent.
        """
        q = [1.0 / self.n] * self.n
        learning_rate = 0.1
        
        for iteration in range(500):
            # Calculate maximum payoff for each player 1 strategy
            payoffs = []
            for i in range(self.m):
                payoff = sum(q[j] * self.payoff_matrix[i][j] for j in range(self.n))
                payoffs.append(payoff)
            
            max_payoff = max(payoffs)
            
            # Calculate gradients
            gradients = []
            for j in range(self.n):
                avg_payoff = sum(self.payoff_matrix[i][j] for i in range(self.m)) / self.m
                gradient = max_payoff - avg_payoff
                gradients.append(gradient)
            
            # Update strategy (gradient descent for minimization)
            new_q = []
            for j in range(self.n):
                new_val = q[j] - learning_rate * gradients[j]
                new_q.append(max(0.0, new_val))
            
            # Normalize
            total = sum(new_q)
            if total > 1e-10:
                new_q = [x / total for x in new_q]
            else:
                new_q = [1.0 / self.n] * self.n
            
            if iteration > 50:
                max_diff = max(abs(new_q[j] - q[j]) for j in range(self.n))
                if max_diff < 1e-6:
                    break
            
            q = new_q
            learning_rate *= 0.999
        
        value = max(sum(q[j] * self.payoff_matrix[i][j] for j in range(self.n)) 
                   for i in range(self.m))
        
        strategy = {j: q[j] for j in range(self.n)}
        return strategy, value
    
    def expected_payoff(self, strategy1, strategy2):
        """
        Calculate expected payoff for given mixed strategies.
        
        Parameters:
        - strategy1: mixed strategy for player 1 (dict: action -> probability)
        - strategy2: mixed strategy for player 2 (dict: action -> probability)
        
        Returns:
        - expected_value: expected payoff for player 1 (float)
        """
        expected = 0.0
        for i, p_i in strategy1.items():
            for j, q_j in strategy2.items():
                expected += p_i * q_j * self.payoff_matrix[i][j]
        return expected


def create_investment_game():
    """
    Create game theory example for investment decision problem.
    
    Problem: Investment strategy selection under market uncertainty
    Player 1 (investor) chooses investment strategy
    Player 2 (nature/market) chooses market condition
    """
    # Payoff matrix: rows = investment strategies, columns = market conditions
    payoff_matrix = [
        [-2, 2, 3],   # Conservative strategy
        [-5, 5, 7],   # Moderate strategy
        [-10, 8, 15], # Aggressive strategy
    ]
    
    return GameTheory(payoff_matrix)

