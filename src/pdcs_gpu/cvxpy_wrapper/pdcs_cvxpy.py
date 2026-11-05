"""PDCS's maintained CVXPY parameter bridge.

Call ``register()`` before constructing/solving CVXPY problems to use this
checkout's adapter with ``problem.solve(solver=cp.PDCS, **options)``.
CVXPY's upstream adapter supplies canonicalization and result inversion.
"""

import importlib.util
import math
import numbers
import os
import warnings

import numpy as np

import cvxpy.settings as s
from cvxpy.reductions.solvers import defines
from cvxpy.reductions.solvers.conic_solvers.pdcs_conif import PDCS as _UpstreamPDCS


_BOOL_OPTIONS = {
    "scalar_cone_rescaling", "use_adaptive_diagonal_scalar_rescaling",
    "use_preconditioner", "use_adaptive_restart", "use_adaptive_step_size_weight",
    "use_aggressive", "use_accelerated", "use_resolving", "use_restart",
    "use_adaptive_step", "use_weighted_average", "use_reflection", "use_halpern",
    "use_current_restart_candidate", "use_mean_restart_candidate",
    "use_kkt_restart", "use_duality_gap_restart",
}
_OPTIONAL_BOOL_OPTIONS = {"adaptive_projection_tolerance", "use_inline_halpern"}
_INT_OPTIONS = {
    "max_outer_iter", "max_inner_iter", "print_freq", "check_terminate_freq",
    "kkt_restart_freq", "duality_gap_restart_freq",
}
_FLOAT_OPTIONS = {
    "abs_tol", "rel_tol", "time_limit",
    "eps_primal_infeasible_low_acc", "eps_dual_infeasible_low_acc",
    "eps_primal_infeasible_high_acc", "eps_dual_infeasible_high_acc",
}
_SYMBOL_OPTIONS = {
    "rescaling_method": {
        "none", "ruiz", "pock_chambolle", "ruiz_pock_chambolle",
        "ruiz_pock_chambolle_scalar_cone",
    },
    "method": {"average"},
}
_ALIASES = {
    "time_limit_secs": "time_limit",
    "logfile": "logfile_name",
    "use_scaling": "use_preconditioner",
    "pdcs_method": "method",
}
_OPTIONS = (
    _BOOL_OPTIONS | _OPTIONAL_BOOL_OPTIONS | _INT_OPTIONS | _FLOAT_OPTIONS
    | set(_SYMBOL_OPTIONS) | {"logfile_name", "julia_env"}
)
_CACHE_KEY = "PDCS_CHECKOUT_WARM_START"


def _option_value(name, value):
    """Validate Python values before importing Julia or allocating GPU data."""
    if name in _OPTIONAL_BOOL_OPTIONS and value is None:
        return None
    if name in _BOOL_OPTIONS | _OPTIONAL_BOOL_OPTIONS:
        if not isinstance(value, (bool, np.bool_)):
            raise TypeError(f"PDCS option {name!r} must be a bool"
                            + (" or None" if name in _OPTIONAL_BOOL_OPTIONS else ""))
        return bool(value)
    if name in _INT_OPTIONS:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
            raise TypeError(f"PDCS option {name!r} must be an integer")
        if not 0 < value <= np.iinfo(np.int64).max:
            raise ValueError(f"PDCS option {name!r} must be a positive Int64 integer")
        return int(value)
    if name in _FLOAT_OPTIONS:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
            raise TypeError(f"PDCS option {name!r} must be a real number")
        value = float(value)
        if value < 0 or math.isnan(value) or (name != "time_limit" and not math.isfinite(value)):
            raise ValueError(f"PDCS option {name!r} must be nonnegative"
                             + (" and finite" if name != "time_limit" else ""))
        return value
    if name in _SYMBOL_OPTIONS:
        if name == "method" and value == "halpern":
            raise ValueError("PDCS method='halpern' is an unsupported legacy path; "
                             "use pdcs_method='average' with use_halpern instead")
        if not isinstance(value, str) or value not in _SYMBOL_OPTIONS[name]:
            choices = ", ".join(sorted(_SYMBOL_OPTIONS[name]))
            raise ValueError(f"PDCS option {name!r} must be one of: {choices}")
        return value
    if value is None:
        return None
    if not isinstance(value, (str, os.PathLike)):
        raise TypeError(f"PDCS option {name!r} must be a path string or None")
    value = os.fspath(value)
    if not isinstance(value, str):
        raise TypeError(f"PDCS option {name!r} must be a text path, not bytes")
    # Preserve the old wrapper's sentinels without evaluating strings as Julia.
    if (name == "logfile_name" and value == "nothing") or (
        name == "julia_env" and value == "placeholder"
    ):
        return None
    return value


def normalize_solver_options(solver_opts, verbose=False):
    """Return Julia constructor options and an optional Julia project path.

    Unspecified algorithm settings are left to the Julia constructor. Keep the
    historical CVXPY time limit (1000 seconds) and CVXPY's verbosity convention.
    """
    options = {}
    origins = {}
    for key, value in solver_opts.items():
        name = _ALIASES.get(key, key)
        if name not in _OPTIONS:
            raise ValueError(f"Unknown or unsupported PDCS option: {key!r}")
        value = _option_value(name, value)
        if name in options and options[name] != value:
            raise ValueError(f"Conflicting PDCS options: {origins[name]!r} and {key!r}")
        options[name] = value
        origins[name] = key
    if "use_accelerated" in options:
        warnings.warn("use_accelerated is deprecated and ignored by PDCS",
                      DeprecationWarning, stacklevel=2)
    julia_env = options.pop("julia_env", None)
    options.setdefault("time_limit", 1000.0)
    options["verbose"] = int(bool(verbose))
    return options, julia_env


def _load_julia(julia_env):
    from juliacall import Main as jl

    if julia_env is not None:
        jl.seval("using Pkg")
        jl.Pkg.activate(julia_env)
    jl.seval("using CUDA, CUDA.CUSPARSE; using PDCS: PDCS_GPU")
    return jl


class PDCS(_UpstreamPDCS):
    """CVXPY PDCS adapter with validated, typed Julia keyword forwarding."""

    def import_solver(self):
        import cupy  # noqa: F401

        # Do not initialize Julia during CVXPY's installed-solver discovery.
        if importlib.util.find_spec("juliacall") is None:
            raise ImportError("PDCS requires juliacall")

    def solve_via_data(self, data, warm_start, verbose, solver_opts, solver_cache=None):
        options, julia_env = normalize_solver_options(solver_opts, verbose)
        if s.P in data:
            raise ValueError("PDCS does not support a quadratic objective directly")
        cones = data[self.DIMS]
        if any(getattr(cones, name, ()) for name in ("psd", "p3d", "pnd")):
            raise ValueError("PDCS supports only zero, nonnegative, SOC and exponential cones")

        import cupy
        from cupyx.scipy.sparse import csr_matrix

        jl = _load_julia(julia_env)
        for key in _SYMBOL_OPTIONS.keys() & options.keys():
            options[key] = jl.Symbol(options[key])

        # These owners remain alive for the entire solve: Julia's pointer views
        # do not own the CuPy allocations. Use dedicated CSR storage because the
        # Julia bridge converts its indices to one-based indexing in place.
        c_owner = cupy.array(data[s.C], dtype=cupy.float64)
        b_owner = cupy.array(data[s.B], dtype=cupy.float64)
        G_owner = csr_matrix(data[s.A], dtype=cupy.float64, copy=True)
        m, n = G_owner.shape
        if max(m, n, G_owner.nnz) >= np.iinfo(np.int32).max:
            raise ValueError("The CVXPY CuPy bridge requires dimensions and nnz "
                             "below the Int32 limit; use the Julia direct API "
                             "for Int64 sparse matrices")
        G_owner.indices = G_owner.indices.astype(cupy.int32, copy=False)
        G_owner.indptr = G_owner.indptr.astype(cupy.int32, copy=False)
        c = jl.PDCS_GPU.cupy_to_cuvector(jl.Float64, int(c_owner.data.ptr), c_owner.size)
        b = jl.PDCS_GPU.cupy_to_cuvector(jl.Float64, int(b_owner.data.ptr), b_owner.size)
        G = jl.PDCS_GPU.cupy_to_cucsrmat(
            jl.Float64, int(G_owner.data.data.ptr), int(G_owner.indices.data.ptr),
            int(G_owner.indptr.data.ptr), m, n, G_owner.nnz,
        )
        signature = (m, n, cones.zero, cones.nonneg, tuple(cones.soc), cones.exp, julia_env)
        cached = solver_cache.get(_CACHE_KEY) if solver_cache is not None else None
        options["warm_start"] = bool(warm_start and cached and cached[0] == signature)
        if options["warm_start"]:
            # The Julia solve scales/mutates its initial vectors.
            options["primal_sol"] = jl.copy(cached[1])
            options["dual_sol"] = jl.copy(cached[2])

        int_vector = jl.seval("Vector{Int}")
        negate = jl.seval("-")
        solver = jl.PDCS_GPU.PDCS_GPU_Solver(
            n=n, m=m, nb=n, c=c, G=negate(G), h=negate(b),
            mGzero=int(cones.zero), mGnonnegative=int(cones.nonneg),
            socG=int_vector(list(cones.soc)), expG=int(cones.exp), dual_expG=0,
            bl=jl.CUDA.fill(-math.inf, n), bu=jl.CUDA.fill(math.inf, n),
            soc_x=int_vector([]), **options,
        )
        result = jl.PDCS_GPU.solve_with_solver(solver)
        if solver_cache is not None:
            solver_cache.pop(_CACHE_KEY, None)
            if str(result.info.exit_status) == "optimal":
                solver_cache[_CACHE_KEY] = (
                    signature,
                    result.x.recovered_primal.primal_sol.x,
                    result.y.recovered_dual.dual_sol.y,
                )
        return result


def register():
    """Use this adapter for ``cp.PDCS`` in the current Python process.

    Register before the first solve so existing Problem solving-chain caches
    cannot retain the upstream adapter. No site-packages files are modified.
    """
    solver = PDCS()
    defines.SOLVER_MAP_CONIC[solver.name()] = solver
    installed = solver.is_installed()
    for names in (defines.INSTALLED_SOLVERS, defines.INSTALLED_CONIC_SOLVERS):
        if installed and solver.name() not in names:
            names.append(solver.name())
        elif not installed and solver.name() in names:
            names.remove(solver.name())
    return solver
