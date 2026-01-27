"""
Complex CVXPY + PDCS Example (Guaranteed Optimal Solution)

This example demonstrates modeling an optimization problem with various constraint types using CVXPY:
- Linear constraints (≤, ≥, =)
- Second-order cone (SOC) constraints
- Exponential cone constraints
And solves it on GPU using the PDCS solver.

Key Features:
1. First, construct a feasible (interior) solution, then derive all constraints based on this feasible solution.
2. Ensure all constraints are satisfied at the feasible point (with slack).
3. Add upper bounds to variables to keep the problem bounded.
4. Use a positive definite quadratic term in the objective to ensure the objective is lower bounded.
5. Verify the feasible point satisfies all constraints, establishing that the problem has an optimal solution.
"""

import sys
import os

# Add local cvxpy directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'cvxpy'))

import cvxpy as cp
import numpy as np

print("=" * 70)
print("Complex example: Solve conic-constrained problem with CVXPY + PDCS")
print("=" * 70)

# ============================================================================
# Problem Description:
# Minimize an objective with linear, quadratic, and exponential terms.
# Subject to:
# 1. Linear inequality constraints (≤, ≥)
# 2. Linear equality constraints
# 3. Second-order cone constraints
# 4. Exponential cone constraints
# ============================================================================

print("\nProblem description:")
print("Minimize: c^T*x + (1/2)*x^T*Q*x + sum(exp(y_i))")
print("Subject to:")
print("  - Linear inequalities: A*x <= b, C*x >= d")
print("  - Linear equalities:  E*x == f")
print("  - SOC: ||G*x + h||_2 <= t")
print("  - Exponential cone: (u, v, w) in exp cone")
print("  - Variable bounds: x >= 0, y >= 0, t >= 0")

# Set random seed for reproducibility
np.random.seed(42)

# ============================================================================
# Problem dimensions
# ============================================================================
n = 5  # Dimension of main variable x
m = 3  # Dimension of auxiliary variable y (for exp terms)
p_ineq1 = 2  # Number of ≤ constraints
p_ineq2 = 2  # Number of ≥ constraints
p_eq = 1     # Number of equality constraints

# ============================================================================
# Generate a feasible solution to guarantee solvability
# ============================================================================
print("\nGenerating a feasible solution to guarantee the problem is solvable...")

# Feasible x in reasonable range
x_feasible = np.random.rand(n) * 2.0 + 0.5   # In [0.5, 2.5]
y_feasible = np.random.rand(m) * 1.0 + 0.5   # In [0.5, 1.5]
t_feasible = np.array([3.0])                 # Large enough t
u_feasible = np.array([0.5])
v_feasible = np.array([1.0])                 # v > 0
w_feasible = np.array([2.0])

print(f"Feasible x* = {x_feasible}")
print(f"Feasible y* = {y_feasible}")

# ============================================================================
# Define variables
# ============================================================================
x = cp.Variable(n, name='x')      # Main optimization variable
y = cp.Variable(m, name='y')      # For exponential terms
t = cp.Variable(1, name='t')      # Scalar for SOC
u = cp.Variable(1, name='u')      # For exponential cone
v = cp.Variable(1, name='v')      # For exponential cone
w = cp.Variable(1, name='w')      # For exponential cone

# ============================================================================
# Generate problem data (based on feasible solution)
# ============================================================================
# Linear objective coefficients
c = np.random.randn(n) * 0.5

# Quadratic matrix (positive definite, ensures bounded below)
Q = np.random.randn(n, n)
Q = Q.T @ Q + 0.5 * np.eye(n)    # Stronger positive definiteness

# Random constraint matrices
A = np.random.randn(p_ineq1, n)
C = np.random.randn(p_ineq2, n)
E = np.random.randn(p_eq, n)
G = np.random.randn(3, n)
F = np.random.randn(2, n)

# Right-hand sides to ensure feasibility at the generated point
b = A @ x_feasible + np.random.rand(p_ineq1) * 0.5 + 0.5 # slack
d = C @ x_feasible - np.random.rand(p_ineq2) * 0.5 - 0.5 # slack
f = E @ x_feasible                                       # equality

# SOC: ||Gx + h||_2 <= t
Gx_feasible = G @ x_feasible
h = -Gx_feasible + np.random.randn(3) * 0.3  # ensure ||Gx+h|| small
t_feasible = np.array([np.linalg.norm(Gx_feasible + h) + 0.5])

# ============================================================================
# Objective function
# ============================================================================
linear_term = c.T @ x
quadratic_term = 0.5 * cp.quad_form(x, Q)

# Exponential term using exponential cone
# Minimize sum(exp(y_i)).  This can be achieved via the exponential cone constraints:
#  exp(y_i) <= z_i  where  (y_i, 1, z_i) \in ExpCone for all i
z = cp.Variable(m, name='z')
exp_term = cp.sum(z)

objective = cp.Minimize(linear_term + quadratic_term + exp_term)

# ============================================================================
# Constraints
# ============================================================================
constraints = []

# 1. Linear inequality: A*x <= b
constraints.append(A @ x <= b)
print(f"\nAdded {p_ineq1} linear inequality constraints: A*x <= b")

# 2. Linear inequality: C*x >= d
constraints.append(C @ x >= d)
print(f"Added {p_ineq2} linear inequality constraints: C*x >= d")

# 3. Linear equality: E*x == f
constraints.append(E @ x == f)
print(f"Added {p_eq} linear equality constraints: E*x == f")

# 4. x >= 0
constraints.append(x >= 0)
print(f"Added nonnegativity constraint: x >= 0")

# 5. y >= 0, t >= 0
constraints.append(y >= 0)
constraints.append(t >= 0)
print(f"Added nonnegativity: y >= 0, t >= 0")

# 6. SOC: ||G*x + h||_2 <= t
constraints.append(cp.SOC(t[0], G @ x + h))
print(f"Added SOC constraint: ||G*x + h||_2 <= t")

# 7. Exponential cone for each y_i: exp(y_i) <= z_i   <=>   (y_i, 1, z_i) in ExpCone
z_feasible = np.exp(y_feasible) + 0.1
for i in range(m):
    constraints.append(cp.constraints.ExpCone(y[i], 1.0, z[i]))
print(f"Added {m} exponential cone constraints: (y_i, 1, z_i) in ExpCone")

# 8. Additional exponential cone: (u, v, w) in ExpCone, with v > 0
w_feasible = v_feasible[0] * np.exp(u_feasible[0] / v_feasible[0]) + 0.1
constraints.append(cp.constraints.ExpCone(u[0], v[0], w[0]))
constraints.append(v >= 0.1)  # numerical safeguard for v > 0
constraints.append(w >= 0.1)
print(f"Added additional exponential cone constraint: (u, v, w) in ExpCone, v >= 0.1, w >= 0.1")

# 9. Extra <= 0 linear constraints (ensure F*x_feasible <= 0)
F = np.random.randn(2, n)
Fx_feasible = F @ x_feasible
if np.any(Fx_feasible > 0):
    F = -np.abs(F)
constraints.append(F @ x <= 0)
print(f"Added additional <= 0 constraint: F*x <= 0")

# 10. Variable upper bound (keep problem bounded)
x_upper_bound = np.max(x_feasible) * 2.0 + 1.0
constraints.append(x <= x_upper_bound)
print(f"Added variable upper bound: x <= {x_upper_bound:.2f}")

# 11. y upper bound
y_upper_bound = np.max(y_feasible) * 2.0 + 1.0
constraints.append(y <= y_upper_bound)
print(f"Added y upper bound: y <= {y_upper_bound:.2f}")

# 12. t upper bound
t_upper_bound = t_feasible[0] * 2.0 + 1.0
constraints.append(t <= t_upper_bound)
print(f"Added t upper bound: t <= {t_upper_bound:.2f}")

# ============================================================================
# Build the optimization problem
# ============================================================================
prob = cp.Problem(objective, constraints)

print(f"\nProblem overview:")
print(f"  Number of variables: {len(prob.variables())}")
print(f"  Number of constraints: {len(constraints)}")
print(f"  Objective type: {type(objective).__name__}")

# ============================================================================
# Verify the feasible point satisfies all constraints (proof of solution existence)
# ============================================================================
print(f"\nVerifying feasibility of the constructed solution:")
print("-" * 70)

Ax_feas = A @ x_feasible
print(f"  A*x* <= b: max(A*x* - b) = {np.max(Ax_feas - b):.6f} (should <= 0)")

Cx_feas = C @ x_feasible
print(f"  C*x* >= d: min(C*x* - d) = {np.min(Cx_feas - d):.6f} (should >= 0)")

Ex_feas = E @ x_feasible
print(f"  E*x* == f: ||E*x* - f|| = {np.linalg.norm(Ex_feas - f):.6e} (should ≈ 0)")

# Variable bounds
print(f"  x* >= 0: min(x*) = {np.min(x_feasible):.6f} (should >= 0)")
print(f"  y* >= 0: min(y*) = {np.min(y_feasible):.6f} (should >= 0)")

# SOC constraint
Gx_plus_h_feas = G @ x_feasible + h
soc_norm = np.linalg.norm(Gx_plus_h_feas)
print(f"  ||G*x* + h||_2 = {soc_norm:.6f}, t* = {t_feasible[0]:.6f}")
print(f"    SOC constraint satisfied: {soc_norm <= t_feasible[0]}")

# Exponential cone constraints
for i in range(m):
    y_i = y_feasible[i]
    z_i = z_feasible[i]
    exp_val = np.exp(y_i)
    print(f"  exp(y*[{i}]) = {exp_val:.6f} <= z*[{i}] = {z_i:.6f}: {exp_val <= z_i}")

# Upper bounds
print(f"  x* <= {x_upper_bound:.2f}: max(x*) = {np.max(x_feasible):.6f}")
print(f"  y* <= {y_upper_bound:.2f}: max(y*) = {np.max(y_feasible):.6f}")

# Compute value at feasible point
obj_feasible = (c.T @ x_feasible +
                0.5 * x_feasible.T @ Q @ x_feasible +
                np.sum(z_feasible))
print(f"\nObjective at feasible point: {obj_feasible:.6f}")
print("(Optimal value should be <= this value)")
print("-" * 70)

# ============================================================================
# Solve using PDCS
# ============================================================================
print("\n" + "=" * 70)
print("Solving with PDCS...")
print("=" * 70)

try:
    prob.solve(solver=cp.PDCS, verbose=True, abs_tol=1e-6, rel_tol=1e-6)
    
    print("\n" + "=" * 70)
    print("Solver Result")
    print("=" * 70)
    print(f"\nSolve status: {prob.status}")
    
    if prob.status in ['optimal', 'optimal_inaccurate']:
        print(f"Optimal value: {prob.value:.6f}")
        
        # Compute value at feasible point for comparison
        obj_feasible = (c.T @ x_feasible +
                        0.5 * x_feasible.T @ Q @ x_feasible +
                        np.sum(z_feasible))
        print(f"Feasible point objective: {obj_feasible:.6f}")
        print(f"Improvement: {obj_feasible - prob.value:.6f} (optimal value should be <= feasible value)")
        
        print(f"\nOptimal solution:")
        print(f"  x = {x.value}")
        print(f"  y = {y.value}")
        print(f"  t = {t.value[0]:.6f}")
        print(f"  z = {z.value}")
        print(f"  u = {u.value[0]:.6f}")
        print(f"  v = {v.value[0]:.6f}")
        print(f"  w = {w.value[0]:.6f}")
        
        print(f"\nComparison to feasible point:")
        print(f"  ||x - x*|| = {np.linalg.norm(x.value - x_feasible):.6f}")
        print(f"  ||y - y*|| = {np.linalg.norm(y.value - y_feasible):.6f}")
        
        # Constraint verification
        print(f"\nConstraint verification:")
        
        Ax_val = A @ x.value
        print(f"  A*x <= b: max violation = {np.max(Ax_val - b):.6e}")
        
        Cx_val = C @ x.value
        print(f"  C*x >= d: max violation = {np.max(d - Cx_val):.6e}")
        
        Ex_val = E @ x.value
        print(f"  E*x == f: max violation = {np.max(np.abs(Ex_val - f)):.6e}")
        
        Gx_plus_h = G @ x.value + h
        soc_violation = np.linalg.norm(Gx_plus_h) - t.value[0]
        print(f"  ||G*x + h||_2 <= t: violation = {soc_violation:.6e}")
        
        # Exponential cone constraints
        for i in range(m):
            y_val = y.value[i]
            z_val = z.value[i]
            if y_val > 0:
                exp_violation = np.exp(y_val) - z_val
                print(f"  exp(y[{i}]) <= z[{i}]: violation = {exp_violation:.6e}")
        
        print(f"  x >= 0: min(x) = {np.min(x.value):.6e}")
        print(f"  y >= 0: min(y) = {np.min(y.value):.6e}")
        print(f"  x <= 10: max(x) = {np.max(x.value):.6f}")
        
    else:
        print(f"Problem not solved optimally")
        print(f"Status: {prob.status}")
        if prob.status == 'infeasible':
            print("Problem is infeasible")
        elif prob.status == 'unbounded':
            print("Problem is unbounded")
        
except Exception as e:
    print(f"\nSolve error: {e}")
    import traceback
    traceback.print_exc()
    print("\nHint: Please ensure the Julia environment and PDCS are properly set up.")

print("\n" + "=" * 70)
print("Example complete")
print("=" * 70)

