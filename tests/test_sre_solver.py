import numpy as np
from utils.sre_solver import approximate_sre


def run_test(Q1: np.ndarray, Q2: np.ndarray, epsilon: float):
    """Helper to run and display a single test case."""
    pi1, pi2 = approximate_sre(Q1, Q2, epsilon=epsilon)

    # Compute expected payoffs
    payoff1 = pi1 @ Q1 @ pi2
    payoff2 = pi1 @ Q2 @ pi2

    print("\n=== Test Case ===")
    print("Q1:\n", Q1)
    print("Q2:\n", Q2)
    print(f"epsilon = {epsilon}")
    print("Approximate SRE Agent 1:", pi1)
    print("Approximate SRE Agent 2:", pi2)
    print("Agent 1 Expected Payoff:", payoff1)
    print("Agent 2 Expected Payoff:", payoff2)


def test_matching_pennies():
    # Zero-sum game: Matching Pennies
    Q1 = np.array([[1, -1],
                   [-1, 1]])
    Q2 = -Q1
    run_test(Q1, Q2, epsilon=0)


def test_coordination_game():
    # Coordination game (both agents rewarded for same choice)
    Q1 = np.array([[5, 0],
                   [0, 5]])
    Q2 = Q1.copy()
    run_test(Q1, Q2, epsilon=0)


def test_random_game(seed: int = 42):
    # Random payoff matrices
    rng = np.random.default_rng(seed)
    Q1 = rng.integers(low=-5, high=6, size=(3, 3))
    Q2 = rng.integers(low=-5, high=6, size=(3, 3))
    run_test(Q1, Q2, epsilon=0.3)


if __name__ == "__main__":
    test_matching_pennies()
    test_coordination_game()
    test_random_game()
