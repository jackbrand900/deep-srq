import numpy as np
import cvxpy as cp
from scipy.optimize import linprog

class StrategicallyRobustEquilibria:
    def __init__(self, U1, U2, D1=None, D2=None):
        """
        Initialize the SRE solver with payoff matrices and distance matrices.

        Parameters:
        U1: numpy array of shape (K1, K2) - Player 1's payoff matrix
        U2: numpy array of shape (K2, K1) - Player 2's payoff matrix
        D1: numpy array of shape (K1, K1) - Distance matrix for Player 1's actions (default: total variation)
        D2: numpy array of shape (K2, K2) - Distance matrix for Player 2's actions (default: total variation)
        """
        self.U1 = U1
        self.U2 = U2
        self.K1, self.K2 = U1.shape

        # Default distance matrices (total variation)
        if D1 is None:
            D1 = 1 - np.eye(self.K1)
        if D2 is None:
            D2 = 1 - np.eye(self.K2)
        self.D1 = D1
        self.D2 = D2

        # Precompute projection matrices
        self.Pi_x1, self.Pi_y1 = self._compute_projection_matrices(self.K1)
        self.Pi_x2, self.Pi_y2 = self._compute_projection_matrices(self.K2)

        # Vectorized distance matrices
        self.d1 = self.D1.flatten()
        self.d2 = self.D2.flatten()

    def _compute_projection_matrices(self, n):
        """Compute projection matrices for vectorized formulation."""
        I = np.eye(n)
        ones = np.ones(n)
        Pi_x = np.kron(I, ones.T)  # n^2 x n
        Pi_y = np.kron(ones.T, I)  # n^2 x n
        return Pi_x, Pi_y

    def robust_best_response(self, U, p_opponent, epsilon, Pi_x, Pi_y, d):
        """
        Compute the strategically robust best response for a player.

        Parameters:
        U: numpy array - The player's payoff matrix
        p_opponent: numpy array - Opponent's strategy
        epsilon: float - Robustness level
        Pi_x, Pi_y: numpy arrays - Projection matrices
        d: numpy array - Vectorized distance matrix

        Returns:
        p: numpy array - Robust best response strategy
        xi: numpy array - Auxiliary variable
        lam: float - Lambda value
        """
        K = U.shape[0]  # Number of actions for this player

        # Variables
        p = cp.Variable(K)
        xi = cp.Variable(K)
        lam = cp.Variable(nonneg=True)

        # Constraints
        constraints = [
            cp.sum(p) == 1,
            p >= 0,
            Pi_x.T @ xi - lam * d - Pi_y.T @ (U.T @ p) <= 0
        ]

        # Objective
        objective = cp.Maximize(p_opponent @ xi - epsilon * lam)

        # Solve
        prob = cp.Problem(objective, constraints)
        prob.solve()

        return p.value, xi.value, lam.value

    def find_sre(self, epsilon, max_iter=100, tol=1e-6):
        """
        Find strategically robust equilibrium using iterative best responses.

        Parameters:
        epsilon: float - Robustness level
        max_iter: int - Maximum number of iterations
        tol: float - Convergence tolerance

        Returns:
        p1: numpy array - Player 1's SRE strategy
        p2: numpy array - Player 2's SRE strategy
        """
        # Initialize with uniform strategies
        p1 = np.ones(self.K1) / self.K1
        p2 = np.ones(self.K2) / self.K2

        for i in range(max_iter):
            # Player 1's robust best response to p2
            p1_new, _, _ = self.robust_best_response(
                self.U1, p2, epsilon, self.Pi_x2, self.Pi_y2, self.d2
            )

            # Player 2's robust best response to p1
            p2_new, _, _ = self.robust_best_response(
                self.U2, p1, epsilon, self.Pi_x1, self.Pi_y1, self.d1
            )

            # Check convergence
            if np.linalg.norm(p1_new - p1) < tol and np.linalg.norm(p2_new - p2) < tol:
                return p1_new, p2_new

            p1, p2 = p1_new, p2_new

        print("Warning: Did not converge within max iterations")
        return p1, p2

# Example usage
if __name__ == "__main__":
    # Coordination game example
    U1 = np.array([[1, 0], [0, 1]])
    U2 = np.array([[1, 0], [0, 1]])

    # Initialize solver
    sre_solver = StrategicallyRobustEquilibria(U1, U2)

    # Find SRE for different robustness levels
    for epsilon in [0, 0.2, 0.5, 1.0]:
        p1, p2 = sre_solver.find_sre(epsilon)
        print(f"ε = {epsilon}: p1 = {p1}, p2 = {p2}")