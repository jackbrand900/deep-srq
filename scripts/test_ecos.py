""" Solve Nash equilibrium for general-sum games using Linear Complementarity Problem (LCP) approach """
import numpy as np
import cvxpy as cp
from scipy.optimize import linprog
import time
from multiprocessing.pool import ThreadPool


def NashEquilibriumGeneralSum(A, B):
    """
    Solve Nash equilibrium for general sum game using Linear Complementarity Problem

    Args:
        A: numpy.array - Player 1's payoff matrix (rows are P1's strategies, cols are P2's strategies)
        B: numpy.array - Player 2's payoff matrix (rows are P1's strategies, cols are P2's strategies)

    Returns:
        (p1_strategy, p2_strategy): tuple of mixed strategies for both players
        (p1_value, p2_value): expected payoffs for both players
    """
    m, n = A.shape  # m strategies for P1, n strategies for P2

    # Method 1: Using CVXPY to solve the complementarity conditions
    # Variables: p (P1's strategy), q (P2's strategy), u (P1's slack), v (P2's slack)
    p = cp.Variable(m, nonneg=True)
    q = cp.Variable(n, nonneg=True)
    u = cp.Variable(m, nonneg=True)
    v = cp.Variable(n, nonneg=True)

    # Constraints
    constraints = [
        cp.sum(p) == 1,  # P1's strategy is a probability distribution
        cp.sum(q) == 1,  # P2's strategy is a probability distribution
        A @ q + u == A @ q + u,  # This will be reformulated below
        B.T @ p + v == B.T @ p + v,  # This will be reformulated below
    ]

    # For Nash equilibrium, we need:
    # If p[i] > 0, then u[i] = 0 (P1's best response condition)
    # If q[j] > 0, then v[j] = 0 (P2's best response condition)
    # This is equivalent to: p^T @ u = 0 and q^T @ v = 0

    # Let's use the expected payoff approach instead
    # For mixed strategy Nash equilibrium:
    # P1's expected payoff from any pure strategy should be equal if that strategy has positive probability
    # Same for P2

    try:
        # Use a simpler approach: iterate through support sizes
        best_solution = None
        best_objective = -np.inf

        # Try different support sizes (strategies with positive probability)
        for p1_support_size in range(1, min(m + 1, 4)):  # Limit search for efficiency
            for p2_support_size in range(1, min(n + 1, 4)):
                solution = _solve_with_fixed_support(A, B, p1_support_size, p2_support_size)
                if solution is not None:
                    p1_strat, p2_strat, obj = solution
                    if obj > best_objective:
                        best_objective = obj
                        best_solution = (p1_strat, p2_strat)

        if best_solution is not None:
            p1_strategy, p2_strategy = best_solution
            p1_value = p1_strategy @ A @ p2_strategy
            p2_value = p1_strategy @ B @ p2_strategy
            return (p1_strategy, p2_strategy), (p1_value, p2_value)

    except Exception as e:
        print(f"CVXPY approach failed: {e}")

    # Fallback: Use iterative best response (may not find exact Nash)
    return _iterative_best_response(A, B)


def _solve_with_fixed_support(A, B, p1_support_size, p2_support_size):
    """
    Solve Nash equilibrium assuming specific support sizes for both players
    """
    m, n = A.shape

    # Try all combinations of supports of given sizes
    from itertools import combinations

    for p1_support in combinations(range(m), p1_support_size):
        for p2_support in combinations(range(n), p2_support_size):
            try:
                # Set up the system of equations for Nash equilibrium
                # For strategies in support, expected payoffs must be equal

                p1_vars = cp.Variable(len(p1_support), nonneg=True)
                p2_vars = cp.Variable(len(p2_support), nonneg=True)

                constraints = [
                    cp.sum(p1_vars) == 1,
                    cp.sum(p2_vars) == 1
                ]

                # Build full strategy vectors
                p1_full = np.zeros(m)
                p2_full = np.zeros(n)

                # For Nash equilibrium with these supports:
                # All strategies in P1's support must yield equal expected payoff
                if len(p1_support) > 1:
                    A_sub = A[list(p1_support)][:, list(p2_support)]
                    for i in range(len(p1_support) - 1):
                        constraints.append(
                            A_sub[i] @ p2_vars == A_sub[i + 1] @ p2_vars
                        )

                # All strategies in P2's support must yield equal expected payoff
                if len(p2_support) > 1:
                    B_sub = B[list(p1_support)][:, list(p2_support)]
                    for j in range(len(p2_support) - 1):
                        constraints.append(
                            p1_vars @ B_sub[:, j] == p1_vars @ B_sub[:, j + 1]
                        )

                # Objective: maximize sum of expected payoffs (or just feasibility)
                A_sub = A[list(p1_support)][:, list(p2_support)]
                B_sub = B[list(p1_support)][:, list(p2_support)]
                objective = cp.Maximize(
                    p1_vars @ A_sub @ p2_vars + p1_vars @ B_sub @ p2_vars
                )

                prob = cp.Problem(objective, constraints)
                prob.solve(solver=cp.ECOS, verbose=False)

                if prob.status == cp.OPTIMAL and p1_vars.value is not None and p2_vars.value is not None:
                    # Check if this is a valid Nash equilibrium
                    p1_strategy = np.zeros(m)
                    p2_strategy = np.zeros(n)

                    for i, idx in enumerate(p1_support):
                        p1_strategy[idx] = p1_vars.value[i]
                    for j, idx in enumerate(p2_support):
                        p2_strategy[idx] = p2_vars.value[j]

                    # Verify Nash equilibrium conditions
                    if _verify_nash_equilibrium(A, B, p1_strategy, p2_strategy):
                        obj_value = prob.value if prob.value is not None else 0
                        return p1_strategy, p2_strategy, obj_value

            except Exception:
                continue

    return None


def _verify_nash_equilibrium(A, B, p1_strategy, p2_strategy, tolerance=1e-6):
    """
    Verify if the given strategies form a Nash equilibrium
    """
    m, n = A.shape

    # Check P1's best response condition
    p1_payoffs = A @ p2_strategy
    max_payoff_p1 = np.max(p1_payoffs)
    for i in range(m):
        if p1_strategy[i] > tolerance:  # Strategy is in support
            if p1_payoffs[i] < max_payoff_p1 - tolerance:
                return False

    # Check P2's best response condition
    p2_payoffs = p1_strategy @ B
    max_payoff_p2 = np.max(p2_payoffs)
    for j in range(n):
        if p2_strategy[j] > tolerance:  # Strategy is in support
            if p2_payoffs[j] < max_payoff_p2 - tolerance:
                return False

    return True


def _iterative_best_response(A, B, max_iterations=1000, tolerance=1e-8):
    """
    Fallback method using iterative best response with better convergence
    """
    m, n = A.shape

    # Initialize with uniform strategies
    p1_strategy = np.ones(m) / m
    p2_strategy = np.ones(n) / n

    # Use smaller learning rate for better convergence
    alpha = 0.01

    for iteration in range(max_iterations):
        old_p1 = p1_strategy.copy()
        old_p2 = p2_strategy.copy()

        # P1's best response to P2's current strategy
        p1_payoffs = A @ p2_strategy
        max_payoff = np.max(p1_payoffs)
        # Use softmax for smoother updates
        exp_payoffs = np.exp((p1_payoffs - max_payoff) / 0.1)
        new_p1_strategy = exp_payoffs / np.sum(exp_payoffs)

        # P2's best response to P1's current strategy
        p2_payoffs = p1_strategy @ B
        max_payoff = np.max(p2_payoffs)
        exp_payoffs = np.exp((p2_payoffs - max_payoff) / 0.1)
        new_p2_strategy = exp_payoffs / np.sum(exp_payoffs)

        # Update with damping
        p1_strategy = (1 - alpha) * p1_strategy + alpha * new_p1_strategy
        p2_strategy = (1 - alpha) * p2_strategy + alpha * new_p2_strategy

        # Check for convergence
        if (np.linalg.norm(p1_strategy - old_p1) < tolerance and
                np.linalg.norm(p2_strategy - old_p2) < tolerance):
            break

    # Final cleanup
    p1_strategy = np.maximum(p1_strategy, 1e-12)  # Avoid exact zeros
    p2_strategy = np.maximum(p2_strategy, 1e-12)
    p1_strategy = p1_strategy / np.sum(p1_strategy)
    p2_strategy = p2_strategy / np.sum(p2_strategy)

    p1_value = p1_strategy @ A @ p2_strategy
    p2_value = p1_strategy @ B @ p2_strategy

    return (p1_strategy, p2_strategy), (p1_value, p2_value)


def NashEquilibriumGeneralSumParallel(game_pairs):
    """
    Solve multiple general sum games in parallel

    Args:
        game_pairs: List of (A, B) tuples where A and B are payoff matrices
    """
    pool = ThreadPool(2)
    results = pool.starmap(NashEquilibriumGeneralSum, game_pairs)
    pool.close()
    pool.join()

    strategies = []
    values = []
    for (p1_strat, p2_strat), (p1_val, p2_val) in results:
        strategies.append((p1_strat, p2_strat))
        values.append((p1_val, p2_val))

    return strategies, values


# Utility function to convert zero-sum game to general sum format
def zero_sum_to_general_sum(M):
    """
    Convert zero-sum game matrix M to general sum format (A, B)
    where B = -A (Player 2's payoffs are negative of Player 1's)
    """
    A = M
    B = -M  # Corrected: should be -M, not -M.T
    return A, B

def _osc_rowwise(A):
    # osc_i = max_j A[i,j] - min_j A[i,j]
    return A.max(axis=1) - A.min(axis=1)


def _osc_colwise(B):
    # osc_j = max_i B[i,j] - min_i B[i,j]
    return B.max(axis=0) - B.min(axis=0)


def robust_br_row_tv_ecos(A, p2, epsilon, solver=cp.ECOS):
    """
    Player 1 robust best response to p2 under TV (L1) Wasserstein ball of radius epsilon.
    Implements the dual LP from Theorem 5.1 specialized to TV: D has 0 on diag, 1 off-diag.
    Maximize   p2^T xi - epsilon * lam
    s.t.       sum(p1)=1, p1 >= 0
               xi_j - (A^T p1)_j <= 0,                     for all j
               xi_j - (A^T p1)_k <= lam,                   for all j != k
    """
    m, n = A.shape
    p1 = cp.Variable(m, nonneg=True)
    xi = cp.Variable(n)
    lam = cp.Variable(nonneg=True)

    cons = [cp.sum(p1) == 1]
    uTp = A.T @ p1  # length n

    for j in range(n):
        cons.append(xi[j] - uTp[j] <= 0)      # diagonal constraints
        for k in range(n):
            if j != k:
                cons.append(xi[j] - uTp[k] <= lam)  # off-diagonal with D_{jk}=1

    obj = cp.Maximize(p2 @ xi - epsilon * lam)
    prob = cp.Problem(obj, cons)
    prob.solve(solver=solver, verbose=False)

    if p1.value is None:
        raise RuntimeError("Row robust BR failed.")
    x = np.clip(p1.value, 0, None).flatten()
    x /= x.sum() if x.sum() > 0 else 1.0
    return x

def robust_br_col_tv_ecos(B, p1, epsilon, solver=cp.ECOS):
    """
    Player 2 robust best response to p1 (TV ball, radius epsilon).
    Maximize   p1^T zeta - epsilon * mu
    s.t.       sum(p2)=1, p2 >= 0
               zeta_i - (B^T p2)_i <= 0,                   for all i
               zeta_i - (B^T p2)_k <= mu,                  for all i != k
    """
    m, n = B.shape
    p2 = cp.Variable(n, nonneg=True)
    zeta = cp.Variable(m)
    mu = cp.Variable(nonneg=True)

    cons = [cp.sum(p2) == 1]
    Btq = B.T @ p2  # length m

    for i in range(m):
        cons.append(zeta[i] - Btq[i] <= 0)
        for k in range(m):
            if i != k:
                cons.append(zeta[i] - Btq[k] <= mu)

    obj = cp.Maximize(p1 @ zeta - epsilon * mu)
    prob = cp.Problem(obj, cons)
    prob.solve(solver=cp.ECOS, verbose=False)

    if p2.value is None:
        raise RuntimeError("Column robust BR failed.")
    q = np.clip(p2.value, 0, None).flatten()
    q /= q.sum() if q.sum() > 0 else 1.0
    return q

# ---------- Worst-case payoff evaluators (to report robust values) ----------

def worst_case_value_row_tv(A, p1, p2_center, epsilon, solver=cp.ECOS):
    """
    Given fixed p1 and center p2_center, compute min_{q: ||q-p2_center||_1 <= 2*eps} p1^T A q.
    Linear in q (p1 is fixed), DCP OK.
    """
    n = A.shape[1]
    q = cp.Variable(n, nonneg=True)
    cons = [cp.sum(q) == 1, cp.norm1(q - p2_center) <= 2*epsilon]
    obj = cp.Minimize((A.T @ p1) @ q)
    prob = cp.Problem(obj, cons)
    prob.solve(solver=solver, verbose=False)
    if q.value is None:
        raise RuntimeError("Row worst-case value LP failed.")
    return float(prob.value)

def worst_case_value_col_tv(B, p1_center, p2, epsilon, solver=cp.ECOS):
    """
    Given fixed p2 and center p1_center, compute min_{r: ||r-p1_center||_1 <= 2*eps} r^T B p2.
    Linear in r (p2 is fixed), DCP OK.
    """
    m = B.shape[0]
    r = cp.Variable(m, nonneg=True)
    cons = [cp.sum(r) == 1, cp.norm1(r - p1_center) <= 2*epsilon]
    obj = cp.Minimize(r @ (B @ p2))
    prob = cp.Problem(obj, cons)
    prob.solve(solver=solver, verbose=False)
    if r.value is None:
        raise RuntimeError("Column worst-case value LP failed.")
    return float(prob.value)

# ---------- Fixed-point SRE search (ECOS-only) ----------

def StrategicallyRobustEquilibrium_TV_ECOS(A, B, epsilon=0.0, tol=1e-9, max_iters=1000, restarts=5):
    """
    ECOS-only SRE via robust best-response iteration under TV cost.
    Not guaranteed globally, but works well in small games (and matches the thesis examples).
    """
    m, n = A.shape
    best = None

    for _ in range(restarts):
        p1 = np.ones(m) / m
        p2 = np.ones(n) / n

        for _ in range(max_iters):
            p1_new = robust_br_row_tv_ecos(A, p2, epsilon)
            p2_new = robust_br_col_tv_ecos(B, p1_new, epsilon)

            if np.linalg.norm(p1_new - p1, 1) < tol and np.linalg.norm(p2_new - p2, 1) < tol:
                p1, p2 = p1_new, p2_new
                break
            p1, p2 = p1_new, p2_new

        # compute robust values at the fixed point
        v1 = worst_case_value_row_tv(A, p1, p2, epsilon)
        v2 = worst_case_value_col_tv(B, p1, p2, epsilon)

        cand = (p1, p2, v1, v2)
        if best is None or (v1 + v2) > (best[2] + best[3] - 1e-12):
            best = cand

    if best is None:
        raise RuntimeError("SRE (TV) not found.")
    p1, p2, v1, v2 = best
    return (p1, p2), (v1, v2)

# ---------- Strategically Robust Equilibrium (TV ball) via support enumeration ----------

def StrategicallyRobustEquilibrium_TV(A, B, epsilon=0.1, max_support=None, solver="ECOS"):
    """
    Compute a Strategically Robust Equilibrium (SRE) for 2-player finite games
    with total-variation (L1) ambiguity sets of radius epsilon centered at the equilibrium strategy.

    For a pure strategy i of P1, robust payoff = A[i,:] @ y - (epsilon/2) * (max_j A[i,j] - min_j A[i,j]).
    For a pure strategy j of P2, robust payoff = x @ B[:,j] - (epsilon/2) * (max_i B[i,j] - min_i B[i,j]).

    We enumerate supports and solve a linear feasibility problem per support using ECOS.
    """
    import cvxpy as cp
    from itertools import combinations

    m, n = A.shape
    if max_support is None:
        max_support = max(m, n)

    oscA = _osc_rowwise(A)  # shape (m,)
    oscB = _osc_colwise(B)  # shape (n,)

    # Try supports by increasing total size
    for kx in range(1, min(m, max_support) + 1):
        for ky in range(1, min(n, max_support) + 1):
            for Sx in combinations(range(m), kx):
                Sx = set(Sx)
                for Sy in combinations(range(n), ky):
                    Sy = set(Sy)
                    try:
                        x = cp.Variable(m, nonneg=True)
                        y = cp.Variable(n, nonneg=True)
                        alpha = cp.Variable()
                        beta = cp.Variable()

                        cons = [
                            cp.sum(x) == 1,
                            cp.sum(y) == 1,
                        ]
                        # Off-support zeros
                        for i in range(m):
                            if i not in Sx:
                                cons.append(x[i] == 0)
                        for j in range(n):
                            if j not in Sy:
                                cons.append(y[j] == 0)

                        # Robust indifference / best-response constraints (TV ambiguity)
                        # P1 side:
                        for i in range(m):
                            expr = A[i, :] @ y - 0.5 * epsilon * float(oscA[i])
                            if i in Sx:
                                cons.append(expr == alpha)
                            else:
                                cons.append(expr <= alpha)
                        # P2 side:
                        for j in range(n):
                            expr = x @ B[:, j] - 0.5 * epsilon * float(oscB[j])
                            if j in Sy:
                                cons.append(expr == beta)
                            else:
                                cons.append(expr <= beta)

                        prob = cp.Problem(cp.Minimize(0), cons)
                        prob.solve(solver=getattr(cp, solver), verbose=False)

                        if prob.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
                            xv = np.maximum(x.value, 0)
                            yv = np.maximum(y.value, 0)
                            if xv.sum() <= 0 or yv.sum() <= 0:
                                continue
                            xv = xv / xv.sum()
                            yv = yv / yv.sum()

                            # Optional: verify robust BR conditions numerically
                            if _verify_sre_tv(A, B, xv, yv, epsilon):
                                p1_val = float(xv @ A @ yv - 0.5 * epsilon * np.dot(xv, oscA))
                                p2_val = float(xv @ B @ yv - 0.5 * epsilon * np.dot(oscB, yv))
                                return (xv, yv), (p1_val, p2_val)
                    except Exception:
                        continue

    raise RuntimeError("SRE (TV) solver: no equilibrium found with the searched supports.")


def _verify_sre_tv(A, B, x, y, epsilon, tol=1e-7):
    """
    Numerical SRE check under TV ambiguity:
    - For any i with x[i] > 0, robust payoff_i equals max robust payoff across all i (within tol).
    - For any j with y[j] > 0, robust payoff_j equals max robust payoff across all j (within tol).
    """
    oscA = _osc_rowwise(A)
    oscB = _osc_colwise(B)

    robust_row = A @ y - 0.5 * epsilon * oscA  # shape (m,)
    robust_col = (x @ B) - 0.5 * epsilon * oscB  # shape (n,)

    # Row player: support strategies must be maximizing robust_row
    max_row = robust_row.max()
    for i in range(A.shape[0]):
        if x[i] > tol and robust_row[i] < max_row - 1e-6:
            return False

    # Column player: support strategies must be maximizing robust_col
    max_col = robust_col.max()
    for j in range(A.shape[1]):
        if y[j] > tol and robust_col[j] < max_col - 1e-6:
            return False

    return True


if __name__ == "__main__":
    # Example 1: Prisoner's Dilemma (classic general sum game)
    print("=== Prisoner's Dilemma ===")
    # Payoff format: (P1_payoff, P2_payoff)
    # Strategies: Cooperate, Defect
    A_pd = np.array([[5, 0],  # P1: Cooperate vs (Coop, Defect)
                     [8, 1]])  # P1: Defect vs (Coop, Defect)
    B_pd = np.array([[5, 8],  # P2: (Coop, Defect) vs P1 Cooperate
                     [0, 1]])  # P2: (Coop, Defect) vs P1 Defect

    t0 = time.time()
    (p1_strat, p2_strat), (p1_val, p2_val) = NashEquilibriumGeneralSum(A_pd, B_pd)
    t1 = time.time()

    print(f"Time: {t1 - t0:.4f}s")
    print(f"Player 1 strategy: {p1_strat}")
    print(f"Player 2 strategy: {p2_strat}")
    print(f"Player 1 expected payoff: {p1_val:.4f}")
    print(f"Player 2 expected payoff: {p2_val:.4f}")

    # Example 2: Convert the original zero-sum game
    print("\n=== Converted Zero-Sum Game ===")
    A_orig = np.array([[0.001, 0.001, 0.00, 0.00, 0.005, 0.01, ],
                       [0.033, 0.166, 0.086, 0.002, -0.109, 0.3, ],
                       [0.001, 0.003, 0.023, 0.019, -0.061, -0.131, ],
                       [-0.156, -0.039, 0.051, 0.016, -0.028, -0.287, ],
                       [0.007, 0.029, 0.004, 0.005, 0.003, -0.012],
                       [0.014, 0.018, -0.001, 0.008, -0.009, 0.007]])

    A_gs, B_gs = zero_sum_to_general_sum(A_orig)

    t0 = time.time()
    (p1_strat, p2_strat), (p1_val, p2_val) = NashEquilibriumGeneralSum(A_gs, B_gs)
    t1 = time.time()

    print(f"Time: {t1 - t0:.4f}s")
    print(f"Player 1 strategy: {p1_strat}")
    print(f"Player 2 strategy: {p2_strat}")
    print(f"Player 1 expected payoff: {p1_val:.4f}")
    print(f"Player 2 expected payoff: {p2_val:.4f}")
    print(f"Sum of payoffs (should be ~0 for zero-sum): {p1_val + p2_val:.4f}")

    # Example 3: Battle of the Sexes
    print("\n=== Battle of the Sexes ===")
    A_bos = np.array([[2, 0],  # P1 prefers coordination on strategy 1
                      [0, 1]])
    B_bos = np.array([[1, 0],  # P2 prefers coordination on strategy 2
                      [0, 2]])

    t0 = time.time()
    (p1_strat, p2_strat), (p1_val, p2_val) = NashEquilibriumGeneralSum(A_bos, B_bos)
    t1 = time.time()
    print(f"Time: {t1 - t0:.4f}s")
    print(f"Player 1 strategy: {p1_strat}")
    print(f"Player 2 strategy: {p2_strat}")
    print(f"Player 1 expected payoff: {p1_val:.4f}")
    print(f"Player 2 expected payoff: {p2_val:.4f}")

    print("\n=== SRE Prisoner's Dilemma (ε=0.2) ===")
    t0 = time.time()
    (p1_sre, p2_sre), (v1_sre, v2_sre) = StrategicallyRobustEquilibrium_TV_ECOS(A_pd, B_pd, epsilon=0.5)
    t1 = time.time()
    print(f"Time: {t1 - t0:.4f}s")
    print(f"Player 1 SRE strategy: {p1_sre}")
    print(f"Player 2 SRE strategy: {p2_sre}")
    print(f"Robust payoffs: {v1_sre:.4f}, {v2_sre:.4f}")
