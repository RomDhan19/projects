"""
Examples of using methods to solve optimization strategy problems.

Student: Торши Р.
"""

from problem1_method1_mdp import MDP, create_inventory_mdp
from problem1_method2_dp import DynamicProgramming, create_inventory_dp
from problem2_method1_game_theory import GameTheory, create_investment_game
from problem2_method2_minimax import Minimax, create_investment_minimax


def problem1_inventory_management():
    """
    Problem 1: Inventory Management Optimization
    
    Task: Find optimal ordering strategy for warehouse inventory management
    to maximize profit while minimizing costs.
    """
    print("=" * 80)
    print("PROBLEM 1: Inventory Management Optimization")
    print("=" * 80)
    
    # Method 1: MDP (Марковский процесс принятия решения)
    print("\n" + "-" * 80)
    print("METHOD 1: Марковский процесс принятия решения (Markov Decision Process)")
    print("-" * 80)
    print("\nApproach: Stochastic optimization with probabilistic state transitions\n")
    
    mdp = create_inventory_mdp()
    
    print("Problem Parameters:")
    print(f"- States: {mdp.states} (0=low inventory, 1=medium, 2=high)")
    print(f"- Actions: {mdp.actions} (0=don't order, 1=order)")
    print(f"- Discount factor: {mdp.discount}\n")
    
    values, policy = mdp.value_iteration()
    
    print("Solution using MDP:")
    state_names = ["Low inventory", "Medium inventory", "High inventory"]
    action_names = ["Don't order", "Order"]
    
    print("\nOptimal state values:")
    for i, state_name in enumerate(state_names):
        print(f"  {state_name} (state {i}): {values[i]:.4f}")
    
    print("\nOptimal policy:")
    for i, state_name in enumerate(state_names):
        action = policy[i]
        print(f"  {state_name}: {action_names[action]}")
    
    # Method 2: Dynamic Programming
    print("\n" + "-" * 80)
    print("METHOD 2: Dynamic Programming (Deterministic Optimization)")
    print("-" * 80)
    print("\nApproach: Deterministic multi-stage optimization with backward induction\n")
    
    dp = create_inventory_dp()
    
    print("Problem Parameters:")
    print(f"- Stages: {dp.num_stages} time periods")
    print(f"- States: {dp.states} (0=low, 1=medium, 2=high inventory)\n")
    
    optimal_values, optimal_policy = dp.solve()
    
    print("Solution using Dynamic Programming:")
    print("\nOptimal values at each stage (sample):")
    for stage in [0, dp.num_stages - 1]:
        print(f"\n  Stage {stage}:")
        for state in dp.states:
            value = optimal_values.get((stage, state), 0.0)
            action = optimal_policy.get((stage, state), None)
            action_str = action_names[action] if action is not None else "N/A"
            print(f"    State {state} ({state_names[state]}): value = {value:.4f}, action = {action_str}")
    
    print("\n" + "=" * 80 + "\n")


def problem2_investment_strategy():
    """
    Problem 2: Investment Strategy Optimization
    
    Task: Find optimal investment strategy under market uncertainty
    considering different market conditions.
    """
    print("=" * 80)
    print("PROBLEM 2: Investment Strategy Optimization")
    print("=" * 80)
    
    # Method 1: Game Theory (Поиск цены игры)
    print("\n" + "-" * 80)
    print("METHOD 1: Поиск цены игры по платежной матрице (Game Theory - Mixed Strategies)")
    print("-" * 80)
    print("\nApproach: Finding optimal mixed strategies using game theory\n")
    
    game = create_investment_game()
    
    print("Problem Parameters:")
    print("Payoff matrix (payoffs for investor):")
    print("       Market: Recession  Stable  Growth")
    strategy_names = ["Conservative", "Moderate", "Aggressive"]
    market_names = ["Recession", "Stable", "Growth"]
    
    for i in range(game.m):
        row_str = " ".join(f"{game.payoff_matrix[i][j]:8.1f}" for j in range(game.n))
        print(f"  {strategy_names[i]:12s}: {row_str}")
    print()
    
    strategy1, strategy2, value = game.solve_game()
    
    print("Solution using Game Theory (Mixed Strategies):")
    print(f"\nGame value: {value:.4f}")
    
    print("\nOptimal mixed strategy for Investor:")
    for action, prob in strategy1.items():
        if prob > 1e-6:
            print(f"  {strategy_names[action]}: probability {prob:.4f}")
    
    print("\nOptimal mixed strategy for Market (Nature):")
    for action, prob in strategy2.items():
        if prob > 1e-6:
            print(f"  {market_names[action]}: probability {prob:.4f}")
    
    expected = game.expected_payoff(strategy1, strategy2)
    print(f"\nExpected payoff with optimal strategies: {expected:.4f}")
    
    # Method 2: Minimax
    print("\n" + "-" * 80)
    print("METHOD 2: Minimax Algorithm (Pure Strategies)")
    print("-" * 80)
    print("\nApproach: Finding optimal pure strategies using minimax principle\n")
    
    minimax = create_investment_minimax()
    
    print("Same payoff matrix as above.\n")
    
    player1_strategy, player2_strategy, game_value = minimax.solve()
    has_saddle, saddle_point = minimax.is_saddle_point()
    
    print("Solution using Minimax:")
    print(f"\nOptimal strategy for Investor: {strategy_names[player1_strategy]}")
    print(f"Optimal strategy for Market: {market_names[player2_strategy]}")
    print(f"Game value: {game_value:.4f}")
    
    if has_saddle:
        print(f"\nSaddle point exists at: ({player1_strategy}, {player2_strategy})")
        print("This is a pure strategy equilibrium.")
    else:
        print("\nNo saddle point exists. Mixed strategies may be better.")
        maxmin = max(min(minimax.payoff_matrix[i][j] for j in range(minimax.n)) 
                    for i in range(minimax.m))
        minimax_val = min(max(minimax.payoff_matrix[i][j] for i in range(minimax.m)) 
                         for j in range(minimax.n))
        print(f"Maxmin value: {maxmin:.4f}")
        print(f"Minimax value: {minimax_val:.4f}")
        print(f"Difference: {abs(maxmin - minimax_val):.4f}")
    
    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    print("\n")
    print("Laboratory Work 2")
    print("Student: Торши Р.")
    print("All implementations in pure Python without external libraries")
    print("\n")
    
    # Solve Problem 1 with two methods
    problem1_inventory_management()
    
    # Solve Problem 2 with two methods
    problem2_investment_strategy()
    
    print("\nAll examples completed successfully!\n")

