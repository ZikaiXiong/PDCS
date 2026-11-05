"""Run with a Python environment containing CVXPY (no GPU is needed)."""

import copy
import math
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import cvxpy as cp
from cvxpy.reductions.solvers import defines
import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] /
                      "src/pdcs_gpu/cvxpy_wrapper"))
import pdcs_cvxpy


class Symbol(str):
    """Distinguish Julia symbols from Python strings at the constructor boundary."""


class Backend:
    """Record Julia constructor calls after real CVXPY canonicalization."""

    def __init__(self):
        self.storage = {}
        self.calls = []
        self.cupy = types.ModuleType("cupy")
        self.cupy.float64 = np.float64
        self.cupy.int32 = np.int32
        self.cupy.array = self.array
        self.sparse = types.ModuleType("cupyx.scipy.sparse")
        self.sparse.csr_matrix = self.csr_matrix
        self.julia = types.SimpleNamespace(
            Float64=float, Symbol=Symbol, copy=np.copy, seval=self.seval,
            CUDA=types.SimpleNamespace(fill=lambda value, n: np.full(n, value)),
            PDCS_GPU=types.SimpleNamespace(
                cupy_to_cuvector=lambda dtype, ptr, size: self.storage[ptr],
                cupy_to_cucsrmat=self.julia_csr,
                PDCS_GPU_Solver=lambda **kwargs: kwargs,
                solve_with_solver=self.solve,
            ),
        )

    def array(self, values, dtype=None):
        values = np.array(values, dtype=dtype, copy=True)
        pointer = id(values)
        self.storage[pointer] = values
        return types.SimpleNamespace(data=types.SimpleNamespace(ptr=pointer),
                                     size=values.size, dtype=values.dtype,
                                     astype=lambda dtype, copy=False: self.array(values, dtype))

    def csr_matrix(self, values, dtype=None, copy=True):
        matrix = sp.csr_matrix(values, dtype=dtype, copy=copy)
        return types.SimpleNamespace(
            data=self.array(matrix.data), indices=self.array(matrix.indices),
            indptr=self.array(matrix.indptr), shape=matrix.shape, nnz=matrix.nnz,
        )

    def julia_csr(self, dtype, data, indices, indptr, m, n, nnz):
        return sp.csr_matrix((self.storage[data], self.storage[indices],
                              self.storage[indptr]), shape=(m, n)).copy()

    def seval(self, source):
        if source == "Vector{Int}":
            return lambda values: np.array(values, dtype=np.int64)
        if source == "-":
            return lambda value: -value
        raise AssertionError(f"Unexpected Julia code evaluation: {source}")

    def solve(self, kwargs):
        self.calls.append(kwargs)
        x = np.ones(kwargs["n"])
        y = np.ones(kwargs["m"])
        return types.SimpleNamespace(
            info=types.SimpleNamespace(exit_status="optimal", pObj=float(kwargs["c"] @ x),
                                       iter=1, time=0.0),
            x=types.SimpleNamespace(recovered_primal=types.SimpleNamespace(
                primal_sol=types.SimpleNamespace(x=x))),
            y=types.SimpleNamespace(recovered_dual=types.SimpleNamespace(
                dual_sol=types.SimpleNamespace(y=y))),
        )


class OptionTests(unittest.TestCase):
    def test_defaults_are_delegated_and_input_is_unchanged(self):
        original = {"use_adaptive_step": np.bool_(False), "max_inner_iter": np.int64(7)}
        before = copy.deepcopy(original)
        options, env = pdcs_cvxpy.normalize_solver_options(original)
        self.assertEqual(original, before)
        self.assertIs(options["use_adaptive_step"], False)
        self.assertIs(type(options["max_inner_iter"]), int)
        self.assertNotIn("scalar_cone_rescaling", options)
        self.assertNotIn("use_restart", options)
        self.assertEqual(options["time_limit"], 1000.0)
        self.assertEqual(options["verbose"], 0)
        self.assertIsNone(env)

    def test_aliases_and_conflicts(self):
        for first, second, value, other in (
            ("time_limit_secs", "time_limit", 4, 5),
            ("logfile", "logfile_name", "a.log", "b.log"),
            ("use_scaling", "use_preconditioner", False, True),
        ):
            with self.subTest(first=first):
                options, _ = pdcs_cvxpy.normalize_solver_options({first: value, second: value})
                self.assertEqual(options[second], value)
                self.assertNotIn(first, options)
                with self.assertRaisesRegex(ValueError, "Conflicting"):
                    pdcs_cvxpy.normalize_solver_options({first: value, second: other})

    def test_bad_options_fail_before_gpu_import(self):
        for options, error in (
            ({"use_adpative_step": False}, ValueError),
            ({"socG": [3]}, ValueError),
            ({"sparse_index_type": "int64"}, ValueError),
            ({"use_adaptive_step": "false"}, TypeError),
            ({"max_inner_iter": True}, TypeError),
            ({"max_inner_iter": 2.5}, TypeError),
            ({"print_freq": 0}, ValueError),
            ({"max_inner_iter": 2**64}, ValueError),
            ({"abs_tol": -1.0}, ValueError),
            ({"rel_tol": float("nan")}, ValueError),
            ({"abs_tol": float("inf")}, ValueError),
            ({"time_limit": False}, TypeError),
            ({"rescaling_method": "invalid"}, ValueError),
            ({"method": "average_no_restart"}, ValueError),
            ({"pdcs_method": "halpern"}, ValueError),
            ({"logfile": 2}, TypeError),
        ):
            with self.subTest(options=options), self.assertRaises(error):
                pdcs_cvxpy.PDCS().solve_via_data({}, False, False, options)

    def test_paths_none_and_infinity(self):
        path = 'results/a "quoted" $name\\test.log'
        options, env = pdcs_cvxpy.normalize_solver_options({
            "logfile": path, "julia_env": Path('project with "quotes"'),
            "time_limit": math.inf, "adaptive_projection_tolerance": None,
            "use_inline_halpern": None,
        })
        self.assertEqual(options["logfile_name"], path)
        self.assertEqual(env, 'project with "quotes"')
        self.assertEqual(options["time_limit"], math.inf)
        self.assertIsNone(options["adaptive_projection_tolerance"])
        self.assertIsNone(options["use_inline_halpern"])
        options, env = pdcs_cvxpy.normalize_solver_options({
            "logfile": "nothing", "julia_env": "placeholder",
        })
        self.assertIsNone(options["logfile_name"])
        self.assertIsNone(env)

    def test_deprecated_option_is_explicit(self):
        with self.assertWarns(DeprecationWarning):
            options, _ = pdcs_cvxpy.normalize_solver_options({"use_accelerated": True})
        self.assertTrue(options["use_accelerated"])

    def test_project_path_is_passed_as_data(self):
        from unittest.mock import Mock
        jl = Mock()
        path = 'project with "quotes" and $name'
        with patch.dict(sys.modules, {"juliacall": types.SimpleNamespace(Main=jl)}):
            self.assertIs(pdcs_cvxpy._load_julia(path), jl)
        jl.Pkg.activate.assert_called_once_with(path)
        self.assertEqual([call.args[0] for call in jl.seval.call_args_list], [
            "using Pkg", "using CUDA, CUDA.CUSPARSE; using PDCS: PDCS_GPU",
        ])


class CVXPYForwardingTests(unittest.TestCase):
    def setUp(self):
        self.backend = Backend()
        self.old_solver = defines.SOLVER_MAP_CONIC[cp.PDCS]
        self.old_installed = defines.INSTALLED_SOLVERS[:]
        self.old_conic = defines.INSTALLED_CONIC_SOLVERS[:]
        self.modules = patch.dict(sys.modules, {
            "cupy": self.backend.cupy,
            "cupyx": types.ModuleType("cupyx"),
            "cupyx.scipy": types.ModuleType("cupyx.scipy"),
            "cupyx.scipy.sparse": self.backend.sparse,
        })
        self.modules.start()
        self.loader = patch.object(pdcs_cvxpy, "_load_julia", return_value=self.backend.julia)
        self.load_julia = self.loader.start()
        self.importer = patch.object(pdcs_cvxpy.PDCS, "import_solver")
        self.importer.start()
        pdcs_cvxpy.register()
        x = cp.Variable()
        self.problem = cp.Problem(cp.Minimize(x), [x >= 1])

    def tearDown(self):
        self.importer.stop()
        self.loader.stop()
        self.modules.stop()
        defines.SOLVER_MAP_CONIC[cp.PDCS] = self.old_solver
        defines.INSTALLED_SOLVERS[:] = self.old_installed
        defines.INSTALLED_CONIC_SOLVERS[:] = self.old_conic

    def test_real_problem_forwards_algorithm_options(self):
        settings = {
            "use_adaptive_step": False,
            "scalar_cone_rescaling": True,
            "use_adaptive_diagonal_scalar_rescaling": False,
            "adaptive_projection_tolerance": False,
            "use_restart": False, "use_reflection": False,
            "use_halpern": True, "use_inline_halpern": None,
            "use_current_restart_candidate": False,
            "use_mean_restart_candidate": False,
            "max_outer_iter": np.int64(3), "max_inner_iter": 9,
            "check_terminate_freq": 2,
            "eps_primal_infeasible_low_acc": 1e-8,
            "eps_dual_infeasible_low_acc": 2e-8,
            "eps_primal_infeasible_high_acc": 3e-12,
            "eps_dual_infeasible_high_acc": 4e-12,
            "abs_tol": 1e-7, "rel_tol": 2e-7,
        }
        before = copy.deepcopy(settings)
        value = self.problem.solve(solver=cp.PDCS, warm_start=False, **settings)
        self.assertEqual(value, 1.0)
        self.assertEqual(settings, before)
        actual = self.backend.calls[-1]
        for key, expected in settings.items():
            with self.subTest(key=key):
                self.assertEqual(actual[key], expected)
        self.assertIs(actual["use_adaptive_step"], False)
        self.assertIs(type(actual["max_outer_iter"]), int)
        self.assertFalse(actual["warm_start"])
        self.assertNotIn("rsocG", actual)
        self.assertNotIn("rsoc_x", actual)
        np.testing.assert_allclose(actual["G"].toarray(), [[1.0]])
        np.testing.assert_allclose(actual["h"], [1.0])

    def test_scaling_symbols_paths_and_aliases_reach_constructor(self):
        filename = 'a "quoted" $name.log'
        self.problem.solve(
            solver=cp.PDCS, use_scaling=False, rescaling_method="ruiz",
            pdcs_method="average", time_limit_secs=12, logfile=filename,
            julia_env=Path('project with "quotes"'),
        )
        actual = self.backend.calls[-1]
        self.assertIs(actual["use_preconditioner"], False)
        self.assertIsInstance(actual["rescaling_method"], Symbol)
        self.assertEqual(actual["rescaling_method"], "ruiz")
        self.assertIsInstance(actual["method"], Symbol)
        self.assertEqual(actual["method"], "average")
        self.assertEqual(actual["time_limit"], 12.0)
        self.assertIs(type(actual["time_limit"]), float)
        self.assertEqual(actual["logfile_name"], filename)
        self.load_julia.assert_called_with('project with "quotes"')

    def test_warm_start_and_options_are_refreshed_between_solves(self):
        self.problem.solve(solver=cp.PDCS, use_adaptive_step=True)
        self.assertFalse(self.backend.calls[-1]["warm_start"])
        cached_x = self.problem._solver_cache[pdcs_cvxpy._CACHE_KEY][1]
        self.problem.solve(solver=cp.PDCS, use_adaptive_step=False)
        actual = self.backend.calls[-1]
        self.assertTrue(actual["warm_start"])
        self.assertFalse(actual["use_adaptive_step"])
        np.testing.assert_array_equal(actual["primal_sol"], [1.0])
        self.assertIsNot(actual["primal_sol"], cached_x)
        self.problem.solve(solver=cp.PDCS, warm_start=False)
        self.assertFalse(self.backend.calls[-1]["warm_start"])
        self.assertNotIn("primal_sol", self.backend.calls[-1])

    def test_unknown_problem_option_is_not_silently_ignored(self):
        with self.assertRaisesRegex(ValueError, "use_adpative_step"):
            self.problem.solve(solver=cp.PDCS, use_adpative_step=False)
        self.assertEqual(self.backend.calls, [])
        self.load_julia.assert_not_called()

    def test_cone_dimensions_are_supplied_by_cvxpy(self):
        x = cp.Variable(2)
        t = cp.Variable()
        problem = cp.Problem(cp.Minimize(t), [
            x[0] == 1, x >= 0, cp.SOC(t, x), cp.ExpCone(x[0], 1, t),
        ])
        problem.solve(solver=cp.PDCS, use_inline_halpern=False)
        actual = self.backend.calls[-1]
        self.assertEqual(actual["mGzero"], 1)
        self.assertEqual(actual["mGnonnegative"], 2)
        np.testing.assert_array_equal(actual["socG"], [3])
        self.assertEqual(actual["expG"], 1)
        self.assertEqual(actual["dual_expG"], 0)
        self.assertIs(actual["use_inline_halpern"], False)


if __name__ == "__main__":
    unittest.main()
