"""Solve a small SOC model through the maintained CVXPY parameter bridge."""

from pathlib import Path
import sys

import cvxpy as cp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] /
                      "src/pdcs_gpu/cvxpy_wrapper"))
from pdcs_cvxpy import register

register()
t = cp.Variable()
problem = cp.Problem(cp.Minimize(t), [cp.SOC(t, cp.Constant([3.0, 4.0]))])
problem.solve(
    solver=cp.PDCS,
    abs_tol=1e-6,
    rel_tol=1e-6,
    time_limit_secs=60.0,
    use_adaptive_step=False,
    use_scaling=True,
    rescaling_method="ruiz_pock_chambolle",
    max_inner_iter=100000,
    verbose=True,
)
print(f"status={problem.status}, objective={problem.value} (expected 5)")
