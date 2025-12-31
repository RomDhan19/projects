# Analysis of Methods for Optimization Strategy Problems

**Student:** Торши Р.

---

## Problem 1: Inventory Management Optimization

### Method 1: Марковский процесс принятия решения (Markov Decision Process - MDP)

#### Method Description

Markov Decision Process (MDP) is a mathematical model for decision-making in stochastic systems where the outcome of an action depends on the current state and probabilistic transitions between states.

**Key Components:**
- **States (S)**: Discrete set of possible system states
- **Actions (A)**: Set of actions available to the agent in each state
- **Transition Probabilities (P)**: P(s'|s,a) - probability of transitioning from state s to state s' when taking action a
- **Rewards (R)**: R(s,a,s') - reward for transitioning from state s to state s' with action a
- **Discount Factor (γ)**: Accounts for the importance of future rewards

**Algorithm:** Value Iteration - iteratively updates state values until convergence.

#### Why This Method is Suitable

MDP is optimally suitable for inventory management problems because:

1. **Stochastic Nature**: Inventory levels change probabilistically due to uncertain demand
2. **Multi-period Decisions**: Need to consider long-term consequences of ordering decisions
3. **State-dependent Strategy**: Optimal ordering policy depends on current inventory level
4. **Known Probabilities**: Historical data allows estimation of transition probabilities

#### Alternative Methods

1. **Dynamic Programming (DP)**
   - Simpler for deterministic problems
   - Does not account for uncertainty properly
   - Less flexible for stochastic systems

2. **Reinforcement Learning**
   - Used when probabilities are unknown
   - Requires learning from data
   - More complex implementation

3. **Monte Carlo Methods**
   - Good for complex systems
   - Computationally intensive
   - Useful for policy evaluation

#### Advantages

- Guaranteed optimal solution
- Accounts for uncertainty and stochasticity
- Multi-period optimization
- Flexible modeling of various systems
- Well-established theory

#### Disadvantages

- Requires knowledge of transition probabilities
- Computational complexity grows with state space size
- Requires discretization for continuous states

---

### Method 2: Dynamic Programming (Deterministic Optimization)

#### Method Description

Dynamic Programming solves multi-stage optimization problems by breaking them into stages and solving backwards from the terminal stage. It uses the principle of optimality: an optimal policy has the property that whatever the initial state and decision are, the remaining decisions must constitute an optimal policy with regard to the state resulting from the first decision.

**Algorithm:** Backward Induction - starts from the final stage and works backwards, storing optimal values and policies for each state at each stage.

#### Why This Method is Suitable

Dynamic Programming is suitable for inventory management when:

1. **Deterministic Transitions**: When state transitions can be modeled deterministically
2. **Multi-stage Problem**: Problem can be decomposed into stages (time periods)
3. **Optimal Substructure**: Optimal solution contains optimal solutions to subproblems
4. **Finite Horizon**: Problem has a finite number of stages

#### Alternative Methods

1. **MDP (Markov Decision Process)**
   - Better for stochastic problems
   - More realistic for uncertain demand
   - Handles probabilities explicitly

2. **Linear Programming**
   - For simpler one-stage problems
   - Cannot handle multi-stage easily
   - Less intuitive for dynamic problems

3. **Greedy Algorithms**
   - Simpler but not optimal
   - May work for simple cases
   - Does not consider future stages

#### Advantages

- Guaranteed optimal solution
- Efficient for problems with optimal substructure
- Clear and intuitive algorithm
- No probability estimation needed (deterministic version)

#### Disadvantages

- Assumes deterministic transitions
- May not reflect real-world uncertainty
- Curse of dimensionality for large state spaces
- Less realistic than stochastic models

---

## Problem 2: Investment Strategy Optimization

### Method 1: Поиск цены игры по платежной матрице (Game Theory - Mixed Strategies)

#### Method Description

This method finds optimal mixed strategies for players in a zero-sum matrix game. In mixed strategies, players choose actions randomly according to probability distributions. The method uses iterative gradient-based optimization to find Nash equilibrium in mixed strategies.

**Key Concepts:**
- **Payoff Matrix**: A[i,j] - payoff for player 1 when choosing row i and player 2 chooses column j
- **Mixed Strategy**: Probability distribution over pure strategies
- **Game Value**: Expected payoff when using optimal strategies
- **Nash Equilibrium**: Strategy profile where no player can improve by unilaterally changing strategy

**Algorithm:** Iterative gradient method - starts with uniform distribution and iteratively updates probabilities based on gradients.

#### Why This Method is Suitable

Game Theory with mixed strategies is optimal for investment problems because:

1. **Strategic Interaction**: Investment outcome depends on market conditions (nature as opponent)
2. **Uncertainty about Opponent**: No information about probabilities of market conditions
3. **Zero-sum Nature**: Investor's gain equals nature's loss (suitable approximation)
4. **Guaranteed Solution**: Nash equilibrium guarantees certain payoff
5. **Optimal for Uncertainty**: Best approach when probabilities are unknown

#### Alternative Methods

1. **Minimax (Pure Strategies)**
   - Simpler implementation
   - May give worse results than mixed strategies
   - Not optimal in general case

2. **Expected Utility Theory**
   - Requires probability estimates
   - More subjective
   - Less robust without good estimates

3. **Robust Optimization**
   - Good for worst-case scenarios
   - May be too conservative
   - Different philosophical approach

#### Advantages

- Guaranteed optimal solution (Nash equilibrium)
- Accounts for strategic interaction
- Finds guaranteed result (maximin solution)
- No need for probability estimates of opponent actions
- Well-established mathematical theory
- Robust to uncertainty

#### Disadvantages

- Assumes rational opponents
- Primarily for zero-sum games
- May be difficult to determine payoff matrix
- Computational complexity grows with matrix size

---

### Method 2: Minimax Algorithm (Pure Strategies)

#### Method Description

Minimax algorithm finds optimal pure strategies using the minimax principle. Player 1 maximizes the minimum payoff (maximin), while Player 2 minimizes the maximum payoff (minimax). When maximin equals minimax, a saddle point exists, indicating a pure strategy equilibrium.

**Algorithm:**
1. For each strategy of Player 1, find minimum payoff (worst-case scenario)
2. Choose strategy that maximizes this minimum (maximin)
3. For each strategy of Player 2, find maximum payoff for Player 1
4. Choose strategy that minimizes this maximum (minimax)

#### Why This Method is Suitable

Minimax is suitable when:

1. **Simplicity Required**: Easier to implement and understand than mixed strategies
2. **Saddle Point Exists**: When pure strategy equilibrium exists
3. **Deterministic Choice**: When mixed strategies are not practical
4. **Quick Solution**: Faster computation than iterative methods
5. **Educational Purpose**: Good for understanding game theory basics

#### Alternative Methods

1. **Mixed Strategy Game Theory**
   - Better optimal solution
   - Works even without saddle points
   - More sophisticated approach

2. **Dominance Method**
   - Simplifies the matrix
   - Only applicable to some games
   - Does not always find solution

3. **Nash Equilibrium (Pure)**
   - Simpler concept
   - May not exist in all games
   - Less powerful than mixed strategies

#### Advantages

- Simple to implement and understand
- Fast computation
- No need for iterative optimization
- Works well when saddle point exists
- Deterministic solution (no randomness)

#### Disadvantages

- May not find optimal solution (when mixed strategies are better)
- Requires saddle point for optimality
- Less powerful than mixed strategies
- May give worse results than mixed strategy approach

---

## Comparison and Recommendations

### Problem 1: Inventory Management

**MDP vs Dynamic Programming:**

- Use **MDP** when:
  - Demand is uncertain and probabilistic
  - Historical data provides transition probabilities
  - Realistic modeling is required
  
- Use **Dynamic Programming** when:
  - Problem can be modeled deterministically
  - Simpler solution is acceptable
  - Educational/illustrative purposes

**Recommendation:** MDP is generally better for real-world inventory problems due to uncertainty, but DP provides a simpler deterministic alternative.

### Problem 2: Investment Strategy

**Game Theory (Mixed) vs Minimax (Pure):**

- Use **Game Theory (Mixed Strategies)** when:
  - Probabilities are unknown
  - Guaranteed optimal solution is needed
  - Worst-case protection is important
  
- Use **Minimax (Pure Strategies)** when:
  - Saddle point exists
  - Simpler solution is preferred
  - Pure strategy is more practical

**Recommendation:** Mixed strategies generally provide better results, but minimax is useful when a saddle point exists or when simplicity is important.

---

## Conclusions

Both problems demonstrate different approaches to optimization under uncertainty:

1. **Problem 1** shows the difference between stochastic (MDP) and deterministic (DP) approaches
2. **Problem 2** shows the difference between mixed strategies (more powerful) and pure strategies (simpler)

The choice of method depends on:
- Nature of uncertainty (probabilistic vs strategic)
- Available information (known vs unknown probabilities)
- Problem requirements (optimality vs simplicity)
- Computational constraints

Each method has its place and provides valuable insights into optimization strategy problems.

