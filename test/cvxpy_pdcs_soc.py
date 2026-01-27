"""
Simple CVXPY + PDCS Example (Second-Order Cone Only)

This example demonstrates modeling an optimization problem with ONLY second-order cone 
(SOC) constraints using CVXPY and solves it on GPU using the PDCS solver.

Key Features:
1. Only second-order cone constraints: ||G*x + h||_2 <= t
2. Construct a feasible solution to guarantee the problem has an optimal solution
3. Add variable bounds to keep the problem bounded
4. Verify the feasible point satisfies all constraints (primal feasible)
"""

import sys
import os

# Add local cvxpy directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'cvxpy'))

import cvxpy as cp
import numpy as np

print("=" * 70)
print("Simple example: Solve SOC cone problem with CVXPY + PDCS")
print("=" * 70)

# ============================================================================
# Problem Description:
# Minimize a linear-quadratic objective
# Subject to:
# 1. Multiple second-order cone constraints: ||G_i*x + h_i||_2 <= t_i
# 2. Variable bounds to ensure boundedness
# ============================================================================

print("\nProblem description:")
print("Minimize: c^T*x + sum(c_t * t_i)")
print("Subject to:")
print("  - SOC constraints: ||G_i*x + h_i||_2 <= t_i for i = 1, ..., n_soc")
print("  - Variable bounds: x in [x_min, x_max], t_i >= t_min")

# Set random seed for reproducibility
np.random.seed(42)

# ============================================================================
# Problem dimensions
# ============================================================================
n = 5  # Dimension of main variable x
n_soc = 3  # Number of SOC constraints

# ============================================================================
# Generate a feasible solution to guarantee solvability
# ============================================================================
print("\nGenerating a feasible solution to guarantee the problem is solvable...")

# Generate feasible x in a bounded range
x_min = -2.0
x_max = 2.0
x_feasible = np.random.rand(n) * (x_max - x_min) + x_min  # In [x_min, x_max]

# Generate SOC constraint matrices and vectors
G_list = []
h_list = []
t_feasible = np.zeros(n_soc)

# For each SOC constraint: ||G_i*x + h_i||_2 <= t_i
for i in range(n_soc):
    # Random matrix G_i (each SOC constraint has dimension 3)
    G_i = np.random.randn(3, n) * 0.5
    G_list.append(G_i)
    
    # Compute G_i*x_feasible
    Gx_feasible = G_i @ x_feasible
    
    # Generate h_i such that ||G_i*x + h_i|| is small
    h_i = -Gx_feasible + np.random.randn(3) * 0.3
    h_list.append(h_i)
    
    # Set t_i to be larger than ||G_i*x + h_i|| with slack
    t_feasible[i] = np.linalg.norm(Gx_feasible + h_i) + 0.5

t_min = 0.1
t_max = np.max(t_feasible) * 2.0 + 1.0

print(f"Feasible x* = {x_feasible}")
print(f"Feasible t* = {t_feasible}")

# Verify SOC constraints are satisfied
print(f"\nVerifying SOC constraint feasibility:")
for i in range(n_soc):
    soc_norm = np.linalg.norm(G_list[i] @ x_feasible + h_list[i])
    print(f"  SOC {i}: ||G_{i}*x* + h_{i}||_2 = {soc_norm:.6f} <= t*[{i}] = {t_feasible[i]:.6f}: {soc_norm <= t_feasible[i]}")

# ============================================================================
# Define variables
# ============================================================================
x = cp.Variable(n, name='x')      # Main optimization variable
t = cp.Variable(n_soc, name='t')  # Scalars for SOC constraints

# ============================================================================
# Generate problem data (based on feasible solution)
# ============================================================================
# Linear objective coefficients
c = np.random.randn(n) * 0.5
c_t = np.random.randn(n_soc) * 0.1

# ============================================================================
# Objective function
# ============================================================================
# Linear terms only
linear_term_x = c.T @ x
linear_term_t = c_t.T @ t

objective = cp.Minimize(linear_term_x + linear_term_t)

# ============================================================================
# Constraints
# ============================================================================
constraints = []

# 1. SOC constraints: ||G_i*x + h_i||_2 <= t_i for each i
for i in range(n_soc):
    constraints.append(cp.SOC(t[i], G_list[i] @ x + h_list[i]))
print(f"\nAdded {n_soc} SOC constraints: ||G_i*x + h_i||_2 <= t_i")

# 2. Lower bounds on t (must be >= 0 for SOC to be well-defined)
# t_min already defined above
constraints.append(t >= t_min)
print(f"Added lower bound: t >= {t_min}")

# 3. Upper bounds on t (keep problem bounded)
# t_max already defined above
constraints.append(t <= t_max)
print(f"Added upper bound: t <= {t_max:.2f}")

# 4. Bounds on x (keep problem bounded)
# x_min, x_max already defined above
constraints.append(x >= x_min)
constraints.append(x <= x_max)
print(f"Added bounds on x: {x_min:.2f} <= x <= {x_max:.2f}")

# ============================================================================
# Build the optimization problem
# ============================================================================
prob = cp.Problem(objective, constraints)

print(f"\nProblem overview:")
print(f"  Number of variables: {len(prob.variables())}")
print(f"  Number of constraints: {len(constraints)}")
print(f"  Objective type: {type(objective).__name__}")

# ============================================================================
# Verify the feasible point satisfies all constraints
# ============================================================================
print(f"\nVerifying feasibility of the constructed solution:")
print("-" * 70)

# Check SOC constraints
for i in range(n_soc):
    soc_norm = np.linalg.norm(G_list[i] @ x_feasible + h_list[i])
    print(f"  SOC {i}: ||G_{i}*x* + h_{i}||_2 = {soc_norm:.6f} <= t*[{i}] = {t_feasible[i]:.6f}: {soc_norm <= t_feasible[i]}")

# Check bounds
print(f"  x* >= {x_min:.2f}: min(x*) = {np.min(x_feasible):.6f}")
print(f"  x* <= {x_max:.2f}: max(x*) = {np.max(x_feasible):.6f}")
print(f"  t* >= {t_min}: min(t*) = {np.min(t_feasible):.6f}")
print(f"  t* <= {t_max:.2f}: max(t*) = {np.max(t_feasible):.6f}")

# Compute objective at feasible point
linear_obj_x = c.T @ x_feasible
linear_obj_t = c_t.T @ t_feasible
obj_feasible = linear_obj_x + linear_obj_t

print(f"\nObjective at feasible point: {obj_feasible:.6f}")
print(f"  Linear part (x): {linear_obj_x:.6f}")
print(f"  Linear part (t): {linear_obj_t:.6f}")
print("(Optimal value should be <= this value)")
print("\nPrimal feasibility: ✓ (verified above)")
print("-" * 70)

# ============================================================================
# Solve using PDCS
# ============================================================================
print("\n" + "=" * 70)
print("Solving with PDCS...")
print("=" * 70)

try:
    prob.solve(solver=cp.PDCS, verbose=True)
    
    print("\n" + "=" * 70)
    print("Solver Result")
    print("=" * 70)
    print(f"\nSolve status: {prob.status}")
    
    if prob.status in ['optimal', 'optimal_inaccurate']:
        print(f"Optimal value: {prob.value:.6f}")
        
        # Compute value at feasible point for comparison
        linear_obj_x = c.T @ x_feasible
        linear_obj_t = c_t.T @ t_feasible
        obj_feasible = linear_obj_x + linear_obj_t
        
        print(f"Feasible point objective: {obj_feasible:.6f}")
        print(f"Improvement: {obj_feasible - prob.value:.6f} (optimal value should be <= feasible value)")
        
        print(f"\nOptimal solution:")
        print(f"  x = {x.value}")
        print(f"  t = {t.value}")
        
        print(f"\nComparison to feasible point:")
        print(f"  ||x - x*|| = {np.linalg.norm(x.value - x_feasible):.6f}")
        print(f"  ||t - t*|| = {np.linalg.norm(t.value - t_feasible):.6f}")
        
        # Constraint verification
        print(f"\nConstraint verification:")
        
        # SOC constraints
        for i in range(n_soc):
            soc_norm = np.linalg.norm(G_list[i] @ x.value + h_list[i])
            violation = soc_norm - t.value[i]
            print(f"  SOC {i}: ||G_{i}*x + h_{i}||_2 = {soc_norm:.6f} <= t[{i}] = {t.value[i]:.6f}, violation = {violation:.6e}")
        
        print(f"  x >= {x_min:.2f}: min(x) = {np.min(x.value):.6f}")
        print(f"  x <= {x_max:.2f}: max(x) = {np.max(x.value):.6f}")
        print(f"  t >= {t_min}: min(t) = {np.min(t.value):.6e}")
        print(f"  t <= {t_max:.2f}: max(t) = {np.max(t.value):.6f}")
        
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
    print("\nHint: Please ensure the Julia environment and Clarabel are properly set up.")

print("\n" + "=" * 70)
print("Example complete")
print("=" * 70)

