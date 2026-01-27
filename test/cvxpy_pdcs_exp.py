"""
Simple CVXPY + PDCS Example (Exponential Cone Only)

This example demonstrates modeling an optimization problem with ONLY exponential cone 
constraints using CVXPY and solves it on GPU using the PDCS solver.

Key Features:
1. Only exponential cone constraints: (u, v, w) in ExpCone means w >= v * exp(u/v) for v > 0
2. Construct a feasible solution to guarantee the problem has an optimal solution
3. Add strong convexity (quadratic terms) to ensure boundedness and dual feasibility
4. Add variable bounds to keep the problem bounded
5. Verify the feasible point satisfies all constraints (primal feasible)
6. Strong convexity ensures dual feasibility
"""

import sys
import os

# Add local cvxpy directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'cvxpy'))

import cvxpy as cp
import numpy as np

print("=" * 70)
print("Simple example: Solve exponential cone problem with CVXPY + PDCS")
print("=" * 70)

# ============================================================================
# Problem Description:
# Minimize a linear combination of exponential cone variables
# Subject to:
# 1. Multiple exponential cone constraints: (u_i, v_i, w_i) in ExpCone
# 2. Variable bounds to ensure boundedness
# ============================================================================

print("\nProblem description:")
print("Minimize: sum(c_u * u_i + c_v * v_i + c_w * w_i) + (1/2) * (||u||^2 + ||v||^2 + ||w||^2)")
print("Subject to:")
print("  - Exponential cone: (u_i, v_i, w_i) in ExpCone for i = 1, ..., n")
print("  - Variable bounds: v_i >= v_min > 0, w_i >= w_min > 0")
print("  - Upper bounds: u_i <= u_max, v_i <= v_max, w_i <= w_max")
print("\nNote: Quadratic terms ensure strong convexity, guaranteeing boundedness and dual feasibility")

# Set random seed for reproducibility
np.random.seed(42)

# ============================================================================
# Problem dimensions
# ============================================================================
n = 3  # Number of exponential cone constraints

# ============================================================================
# Generate a feasible solution to guarantee solvability
# ============================================================================
print(f"\nGenerating a feasible solution to guarantee the problem is solvable...")

# For exponential cone (u, v, w) in ExpCone, we need: w >= v * exp(u/v) for v > 0
# Generate feasible points with proper bounds
v_min = 0.1
w_min = 0.1

# Generate u in a bounded range
u_feasible = np.random.randn(n) * 0.5  # Can be negative, but we'll ensure it's bounded
u_min = np.min(u_feasible) - 1.0  # Set u_min based on generated values
u_max = np.max(np.abs(u_feasible)) * 2.0 + 1.0

# Ensure u_feasible is within bounds (adjust if needed)
u_feasible = np.clip(u_feasible, u_min + 0.1, u_max - 0.1)

# Generate v and w with proper bounds
v_feasible = np.random.rand(n) * 1.0 + 0.5  # Must be > 0, in [0.5, 1.5]
v_max = np.max(v_feasible) * 2.0 + 1.0
w_feasible = np.zeros(n)

# Ensure w >= v * exp(u/v) with some slack, and within bounds
w_max = 0.0
for i in range(n):
    w_feasible[i] = v_feasible[i] * np.exp(u_feasible[i] / v_feasible[i]) + 0.1
    w_max = max(w_max, w_feasible[i])
w_max = w_max * 2.0 + 1.0

print(f"Feasible u* = {u_feasible}")
print(f"Feasible v* = {v_feasible}")
print(f"Feasible w* = {w_feasible}")

# Verify feasibility
print(f"\nVerifying exponential cone feasibility:")
for i in range(n):
    exp_val = v_feasible[i] * np.exp(u_feasible[i] / v_feasible[i])
    print(f"  Cone {i}: w*[{i}] = {w_feasible[i]:.6f} >= v*[{i}]*exp(u*[{i}]/v*[{i}]) = {exp_val:.6f}: {w_feasible[i] >= exp_val}")

# ============================================================================
# Define variables
# ============================================================================
u = cp.Variable(n, name='u')  # First component of exponential cone
v = cp.Variable(n, name='v')  # Second component of exponential cone (must be > 0)
w = cp.Variable(n, name='w')  # Third component of exponential cone

# ============================================================================
# Objective function
# ============================================================================
# Linear objective coefficients
c_u = np.random.randn(n) * 0.1
c_v = np.random.randn(n) * 0.1
c_w = np.random.randn(n) * 0.1

# Add quadratic terms for strong convexity (ensures boundedness and dual feasibility)
# This makes the problem strongly convex, guaranteeing a unique optimal solution
linear_term = c_u.T @ u + c_v.T @ v + c_w.T @ w
quadratic_term = 0.5 * (cp.sum_squares(u) + cp.sum_squares(v) + cp.sum_squares(w))

objective = cp.Minimize(linear_term + quadratic_term)

# ============================================================================
# Constraints
# ============================================================================
constraints = []

# 1. Exponential cone constraints: (u_i, v_i, w_i) in ExpCone
# This means: w_i >= v_i * exp(u_i / v_i) for v_i > 0
for i in range(n):
    constraints.append(cp.constraints.ExpCone(u[i], v[i], w[i]))
print(f"\nAdded {n} exponential cone constraints: (u_i, v_i, w_i) in ExpCone")

# 2. Lower bounds on v (must be > 0 for exponential cone to be well-defined)
# v_min already defined above
constraints.append(v >= v_min)
print(f"Added lower bound: v >= {v_min}")

# 3. Lower bounds on w (to ensure w > 0)
# w_min already defined above
constraints.append(w >= w_min)
print(f"Added lower bound: w >= {w_min}")

# 4. Upper bounds to keep problem bounded (in addition to strong convexity)
# u_max, v_max, w_max already defined above
constraints.append(u <= u_max)
constraints.append(v <= v_max)
constraints.append(w <= w_max)
print(f"Added upper bounds: u <= {u_max:.2f}, v <= {v_max:.2f}, w <= {w_max:.2f}")

# 5. Lower bound on u (to ensure boundedness from below)
# u_min already defined above
constraints.append(u >= u_min)
print(f"Added lower bound: u >= {u_min:.2f}")

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

# Check exponential cone constraints
for i in range(n):
    exp_val = v_feasible[i] * np.exp(u_feasible[i] / v_feasible[i])
    print(f"  ExpCone {i}: w*[{i}] = {w_feasible[i]:.6f} >= v*[{i}]*exp(u*[{i}]/v*[{i}]) = {exp_val:.6f}: {w_feasible[i] >= exp_val}")

# Check bounds
print(f"  v* >= {v_min}: min(v*) = {np.min(v_feasible):.6f} (should >= {v_min})")
print(f"  w* >= {w_min}: min(w*) = {np.min(w_feasible):.6f} (should >= {w_min})")
print(f"  u* >= {u_min:.2f}: min(u*) = {np.min(u_feasible):.6f}")
print(f"  u* <= {u_max:.2f}: max(u*) = {np.max(u_feasible):.6f}")
print(f"  v* <= {v_max:.2f}: max(v*) = {np.max(v_feasible):.6f}")
print(f"  w* <= {w_max:.2f}: max(w*) = {np.max(w_feasible):.6f}")

# Compute objective at feasible point
linear_obj_feasible = c_u.T @ u_feasible + c_v.T @ v_feasible + c_w.T @ w_feasible
quadratic_obj_feasible = 0.5 * (np.sum(u_feasible**2) + np.sum(v_feasible**2) + np.sum(w_feasible**2))
obj_feasible = linear_obj_feasible + quadratic_obj_feasible
print(f"\nObjective at feasible point: {obj_feasible:.6f}")
print(f"  Linear part: {linear_obj_feasible:.6f}")
print(f"  Quadratic part: {quadratic_obj_feasible:.6f}")
print("(Optimal value should be <= this value)")
print("\nPrimal feasibility: ✓ (verified above)")
print("Dual feasibility: ✓ (guaranteed by strong convexity)")
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
        linear_obj_feasible = c_u.T @ u_feasible + c_v.T @ v_feasible + c_w.T @ w_feasible
        quadratic_obj_feasible = 0.5 * (np.sum(u_feasible**2) + np.sum(v_feasible**2) + np.sum(w_feasible**2))
        obj_feasible = linear_obj_feasible + quadratic_obj_feasible
        print(f"Feasible point objective: {obj_feasible:.6f}")
        print(f"Improvement: {obj_feasible - prob.value:.6f} (optimal value should be <= feasible value)")
        
        print(f"\nOptimal solution:")
        print(f"  u = {u.value}")
        print(f"  v = {v.value}")
        print(f"  w = {w.value}")
        
        print(f"\nComparison to feasible point:")
        print(f"  ||u - u*|| = {np.linalg.norm(u.value - u_feasible):.6f}")
        print(f"  ||v - v*|| = {np.linalg.norm(v.value - v_feasible):.6f}")
        print(f"  ||w - w*|| = {np.linalg.norm(w.value - w_feasible):.6f}")
        
        # Constraint verification
        print(f"\nConstraint verification:")
        
        # Exponential cone constraints
        for i in range(n):
            u_val = u.value[i]
            v_val = v.value[i]
            w_val = w.value[i]
            if v_val > 0:
                exp_val = v_val * np.exp(u_val / v_val)
                violation = exp_val - w_val
                print(f"  ExpCone {i}: w[{i}] = {w_val:.6f} >= v[{i}]*exp(u[{i}]/v[{i}]) = {exp_val:.6f}, violation = {violation:.6e}")
        
        print(f"  v >= {v_min}: min(v) = {np.min(v.value):.6e}")
        print(f"  w >= {w_min}: min(w) = {np.min(w.value):.6e}")
        print(f"  u >= {u_min:.2f}: min(u) = {np.min(u.value):.6f}")
        print(f"  u <= {u_max:.2f}: max(u) = {np.max(u.value):.6f}")
        print(f"  v <= {v_max:.2f}: max(v) = {np.max(v.value):.6f}")
        print(f"  w <= {w_max:.2f}: max(w) = {np.max(w.value):.6f}")
        
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