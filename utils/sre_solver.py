import numpy as np
from typing import Tuple

def approximate_sre(Q1: np.ndarray, Q2: np.ndarray, epsilon: float = 0.1,
                    max_iter: int = 100, tol: float = 1e-5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Approximate Strategically Robust Equilibrium (SRE) strategies
    for a 2-player normal-form game using a Nash-DQN-style solver.

    Args:
        Q1: Payoff matrix for agent 1, shape (n_actions_1, n_actions_2).
        Q2: Payoff matrix for agent 2, shape (n_actions_1, n_actions_2).
        epsilon: Robustness parameter (radius of Wasserstein ball).
        max_iter: Maximum iterations for approximation.
        tol: Convergence tolerance.

    Returns:
        pi1: Mixed strategy for agent 1 (np.array of shape (n_actions_1,))
        pi2: Mixed strategy for agent 2 (np.array of shape (n_actions_2,))
    """

    n1, n2 = Q1.shape
    # Start with uniform strategies
    pi1 = np.ones(n1) / n1
    pi2 = np.ones(n2) / n2

    for _ in range(max_iter):
        old_pi1, old_pi2 = pi1.copy(), pi2.copy()

        # Compute expected utilities
        u1 = Q1 @ pi2
        u2 = Q2.T @ pi1

        # Robust adjustment: perturb opponent’s strategy within L1-ball of radius epsilon
        # Here: simple projection (shrink mass to worst-case distribution)
        worst_pi2 = project_to_l1_ball(pi2, epsilon)
        worst_pi1 = project_to_l1_ball(pi1, epsilon)

        # Best responses against worst-case opponent
        br1 = np.zeros_like(pi1)
        br1[np.argmax(Q1 @ worst_pi2)] = 1.0

        br2 = np.zeros_like(pi2)
        br2[np.argmax(Q2.T @ worst_pi1)] = 1.0

        # Update with learning rate (like fictitious play)
        eta = 0.5
        pi1 = (1 - eta) * pi1 + eta * br1
        pi2 = (1 - eta) * pi2 + eta * br2

        # Convergence check
        if np.linalg.norm(pi1 - old_pi1, 1) < tol and np.linalg.norm(pi2 - old_pi2, 1) < tol:
            break

    return pi1, pi2


def project_to_l1_ball(p: np.ndarray, epsilon: float) -> np.ndarray:
    """
    Project a probability vector onto an L1 ball of radius epsilon.
    This simulates the Wasserstein ball perturbation.
    """
    q = p.copy()
    # Worst-case: push probability mass towards most harmful actions
    idx = np.argmax(p)
    q = (1 - epsilon) * p
    q[idx] += epsilon
    return q / q.sum()
