# charts.py — full extended module with intrinsic computations and advanced utilities
from __future__ import annotations
import sympy as sp
from sympy import Expr, Symbol, sympify, lambdify, Matrix, diff, simplify, zeros, eye, sqrt, Function
from typing import Callable, Dict, List, Tuple, Optional, Union, Any
import numpy as np

# Import metric and connection machinery
from Riemannian_metric import RiemannianMetric
from connections import LeviCivitaConnection

# -------------------------- Domain Definitions --------------------------
class Domain:
    """
    Abstract base class for chart domains in R^n.
    """
    def contains(self, point: Tuple[float, ...]) -> bool:
        raise NotImplementedError("Domain.contains must be implemented")

class PredicateDomain(Domain):
    """
    Domain defined by a Python predicate f: R^n -> bool.
    """
    def __init__(self, predicate: Callable[..., bool]):
        self.predicate = predicate
    def contains(self, point: Tuple[float, ...]) -> bool:
        return bool(self.predicate(*point))

class BoxDomain(Domain):
    """
    Axis-aligned box domain: each coordinate x_i in [min_i, max_i].
    """
    def __init__(self, bounds: List[Tuple[float, float]]):
        self.bounds = bounds
    def contains(self, point: Tuple[float, ...]) -> bool:
        return all(lo <= x <= hi for x, (lo, hi) in zip(point, self.bounds))

class InequalityDomain(Domain):
    """
    Domain defined by a Sympy relational, e.g. x**2 + y**2 < 1.
    """
    def __init__(self, expr: Union[Expr, str], coords: List[Symbol]):
        self.expr = sympify(expr)
        self.coords = coords
        self._func = lambdify(coords, self.expr, 'numpy')
    def contains(self, point: Tuple[float, ...]) -> bool:
        return bool(self._func(*point))

class UnionDomain(Domain):
    """
    Union of multiple domains.
    """
    def __init__(self, domains: List[Domain]):
        self.domains = domains
    def contains(self, point: Tuple[float, ...]) -> bool:
        return any(d.contains(point) for d in self.domains)

# ------------------------ Embedding Definitions ------------------------
class Embedding:
    """
    Base class: symbolic map from coords -> R^m, auto-lambdified.
    """
    def __init__(self, coords: List[Symbol], map_exprs: List[Expr]):
        self.coords = coords
        self.map_exprs = [sympify(expr) for expr in map_exprs]
        self._func = lambdify(coords, self.map_exprs, 'numpy')
    def evaluate(self, point: Tuple[float, ...]) -> Tuple[float, ...]:
        arr = np.array(self._func(*point), dtype=float)
        return tuple(arr.flatten())

class ParametricEmbedding(Embedding):
    """
    Alias for semantic clarity.
    """
    pass

# ---------------------- Coordinate Transition ----------------------
class TransitionMap:
    """
    Transition between two charts via forward and inverse maps.
    """
    def __init__(
        self,
        source_coords: List[Symbol],
        target_coords: List[Symbol],
        forward_map: Dict[Symbol, Expr],
        inverse_map: Optional[Dict[Symbol, Expr]] = None
    ):
        if set(forward_map.keys()) != set(source_coords):
            raise ValueError("Forward map must define all source coords.")
        self.source_coords = source_coords
        self.target_coords = target_coords
        self.forward_map = {s: sympify(e) for s, e in forward_map.items()}
        self.inverse_map = {s: sympify(e) for s, e in (inverse_map or {}).items()}
        self._fwd_func = lambdify(source_coords, list(self.forward_map.values()), 'numpy')
        self._inv_func = (
            lambdify(target_coords, list(self.inverse_map.values()), 'numpy')
            if inverse_map else None
        )
    def to_target(self, pt: Tuple[float, ...]) -> Tuple[float, ...]:
        return tuple(np.array(self._fwd_func(*pt), dtype=float).flatten())
    def to_source(self, pt: Tuple[float, ...]) -> Tuple[float, ...]:
        if not self._inv_func:
            raise ValueError("Inverse map undefined.")
        return tuple(np.array(self._inv_func(*pt), dtype=float).flatten())

# -------------------------- Chart Definition --------------------------
class Chart:
    """
    Coordinate chart: holds coords, domain, embedding, transitions,
    and automatically computes metric, connection, Christoffel symbols,
    geodesics, exponential map, distance, curvature, and more.
    """
    def __init__(
        self,
        name: str,
        coords: List[Symbol],
        domain: Domain,
        embedding: Embedding
    ):
        self.name = name
        self.coords = coords
        self.dim = len(coords)
        self.domain = domain
        self.embedding = embedding
        self.transitions: Dict[str, TransitionMap] = {}
        # Automatic intrinsic data
        self._compute_intrinsics()

    def add_transition(
        self,
        other: Chart,
        forward: Dict[Symbol, Expr],
        inverse: Optional[Dict[Symbol, Expr]] = None
    ):
        tm = TransitionMap(self.coords, other.coords, forward, inverse)
        self.transitions[other.name] = tm

    def contains(self, pt: Tuple[float, ...]) -> bool:
        return self.domain.contains(pt)

    def to_chart(self, pt: Tuple[float, ...], other: Chart) -> Tuple[float, ...]:
        if other.name not in self.transitions:
            raise KeyError(f"No transition to {other.name}")
        return self.transitions[other.name].to_target(pt)

    def sample_grid(self, bounds: Optional[List[Tuple[float, float]]] = None, num: int = 20) -> List[Tuple[float, ...]]:
        rng = bounds or getattr(self.domain, 'bounds', None)
        if not rng:
            raise ValueError("Bounds or BoxDomain required.")
        axes = [np.linspace(lo, hi, num) for lo, hi in rng]
        mesh = np.meshgrid(*axes)
        pts = [tuple(mesh[d].flat[i] for d in range(self.dim)) for i in range(mesh[0].size)]
        return [p for p in pts if self.contains(p)]

    # ---------- Automatic Intrinsic Computations ----------
    def _compute_intrinsics(self):
        """
        Compute induced metric, Christoffel symbols, and Levi-Civita connection.
        """
        # Jacobian of embedding
        J = Matrix(self.embedding.map_exprs).jacobian(self.coords)
        # Induced metric g = J^T J
        g_mat = simplify(J.T * J)
        self.metric = RiemannianMetric(self.coords, g_mat)
        # Levi-Civita connection and Christoffel symbols
        self.connection = LeviCivitaConnection(self.metric, self)
        self.Gamma = self.connection.Gamma
        # Lambdify Christoffel symbols for numeric use
        self._gamma_funcs = [
            [
                [lambdify(self.coords, self.Gamma[i][j][k], 'numpy') for k in range(self.dim)]
                for j in range(self.dim)
            ]
            for i in range(self.dim)
        ]

    # ------------------ Geodesics & Exponential Map Utilities ------------------
    def geodesic_system(self, Y: List[float], t: float) -> List[float]:
        """
        ODE system for geodesics: d^2 u^i/dt^2 + Gamma^i_{jk} du^j du^k = 0
        """
        dim = self.dim
        u = list(Y[:dim])
        du = list(Y[dim:])
        d2 = [0.0]*dim
        for i in range(dim):
            for j in range(dim):
                for k in range(dim):
                    d2[i] -= self._gamma_funcs[i][j][k](*u) * du[j] * du[k]
        return list(du) + d2

    def geodesic(self, start: Tuple[float,...], velocity: Tuple[float,...], t_span: Tuple[float,float], num: int=200) -> np.ndarray:
        """
        Compute and return parametric geodesic curve in chart coords.
        """
        from scipy.integrate import odeint
        t_vals = np.linspace(t_span[0], t_span[1], num)
        Y0 = list(start) + list(velocity)
        sol = odeint(self.geodesic_system, Y0, t_vals)
        return sol[:,:self.dim]

    def exponential_map(self, base: Tuple[float,...], velocity: Tuple[float,...], t: float=1.0) -> Tuple[float,...]:
        """
        Exponential map: geodesic at time t starting from base with init velocity.
        """
        sol = self.geodesic(base, velocity, (0.0, t), num=2)
        return tuple(sol[-1])

    def distance(self, p: Tuple[float,...], q: Tuple[float,...], initial_guess: Optional[Tuple[float,...]]=None) -> float:
        """
        Compute geodesic distance between p and q by optimizing initial velocity.
        """
        import scipy.optimize as opt
        if initial_guess is None:
            initial_guess = tuple(qi - pi for pi, qi in zip(p, q))
        def err(v0):
            end = self.exponential_map(p, tuple(v0), t=1.0)
            return sum((ei - qi)**2 for ei, qi in zip(end, q))
        res = opt.minimize(err, initial_guess)
        v_opt = res.x
        g_num = np.array(self.metric.lambdify_matrix()(*p), dtype=float)
        return float(np.sqrt(v_opt.dot(g_num.dot(v_opt))))

    def geodesic_grid(self, center: Tuple[float,...], radius: float, num_dirs: int=36, num_pts: int=20) -> List[np.ndarray]:
        """
        Generate radial geodesic grid around center within given radius in 2D charts.
        """
        if self.dim != 2:
            raise NotImplementedError("Geodesic grid only implemented for 2D charts.")
        angles = np.linspace(0, 2*np.pi, num_dirs, endpoint=False)
        grid = []
        for ang in angles:
            v0 = (radius*np.cos(ang), radius*np.sin(ang))
            pts = self.geodesic(center, v0, (0.0, 1.0), num=num_pts)
            grid.append(pts)
        return grid

    # ------------------ Analytical & Differential Utilities ------------------
    def jacobian(self) -> Matrix:
        """Symbolic Jacobian of the embedding."""
        return Matrix(self.embedding.map_exprs).jacobian(self.coords)

    def jacobian_func(self) -> Callable[..., np.ndarray]:
        """Numeric Jacobian evaluator."""
        return lambdify(self.coords, self.jacobian(), 'numpy')

    def lambdified_metric(self) -> Callable[..., np.ndarray]:
        """Numeric metric tensor evaluator."""
        return lambdify(self.coords, self.metric.g, 'numpy')

    def lambdified_inverse_metric(self) -> Callable[..., np.ndarray]:
        """Numeric inverse metric evaluator."""
        invg = self.metric.inverse_matrix()
        return lambdify(self.coords, invg, 'numpy')

    def volume_element(self) -> Expr:
        """Symbolic volume element sqrt(det(g))."""
        return sqrt(simplify(self.metric.g.det()))

    def lambdified_volume(self) -> Callable[..., float]:
        """Numeric volume element evaluator."""
        return lambdify(self.coords, self.volume_element(), 'numpy')

    def pushforward(self, pt: Tuple[float,...], vec: Tuple[float,...]) -> np.ndarray:
        """Pushforward of a tangent vector via embedding Jacobian."""
        Jfunc = self.jacobian_func()
        return np.array(Jfunc(*pt), dtype=float).dot(np.array(vec, dtype=float))

    def pullback_form(self, form: sp.Matrix) -> sp.Matrix:
        """Pullback of an ambient 1-form given as symbolic Matrix."""
        J = Matrix(self.embedding.map_exprs).jacobian(self.coords)
        return J.T * form

    def riemann_tensor(self) -> List[List[List[List[Expr]]]]:
        """Compute symbolic Riemann curvature tensor R^i_{jkl}."""
        R = [[[[0 for _ in range(self.dim)] for _ in range(self.dim)] for _ in range(self.dim)] for _ in range(self.dim)]
        for i in range(self.dim):
            for j in range(self.dim):
                for k in range(self.dim):
                    for l in range(self.dim):
                        term1 = diff(self.Gamma[i][j][l], self.coords[k])
                        term2 = diff(self.Gamma[i][j][k], self.coords[l])
                        sum_term = 0
                        for m in range(self.dim):
                            sum_term += self.Gamma[i][k][m]*self.Gamma[m][j][l] - self.Gamma[i][l][m]*self.Gamma[m][j][k]
                        R[i][j][k][l] = simplify(term1 - term2 + sum_term)
        return R

    def sectional_curvature(self, u_vec: sp.Matrix, v_vec: sp.Matrix) -> Expr:
        """Compute sectional curvature K(u,v)."""
        R = self.riemann_tensor()
        num = 0
        for i in range(self.dim):
            for j in range(self.dim):
                for k in range(self.dim):
                    for l in range(self.dim):
                        num += u_vec[i]*v_vec[j]*u_vec[k]*v_vec[l]*R[i][j][k][l]
        g = self.metric.g
        uu = (u_vec.T * g * u_vec)[0]
        vv = (v_vec.T * g * v_vec)[0]
        uv = (u_vec.T * g * v_vec)[0]
        denom = simplify(uu*vv - uv**2)
        return simplify(num/denom)

# ---------------------- Manifold Wrapper ----------------------
class SymbolicManifold:
    """
    Atlas of charts representing a manifold.
    """
    def __init__(self, name: str):
        self.name = name
        self.charts: Dict[str, Chart] = {}
        self.default: Optional[Chart] = None

    def add_chart(self, chart: Chart, default: bool=False):
        if chart.name in self.charts:
            raise KeyError(f"Chart {chart.name} already exists.")
        self.charts[chart.name] = chart
        if default or self.default is None:
            self.default = chart

    def get_chart(self, name: str) -> Chart:
        return self.charts[name]

    def find_chart(self, pt: Tuple[float,...]) -> Chart:
        for chart in self.charts.values():
            if chart.contains(pt):
                return chart
        if self.default and self.default.contains(pt):
            return self.default
        raise ValueError(f"No chart contains point {pt}.")

    def transition(self, pt: Tuple[float,...], from_chart: str, to_chart: str) -> Tuple[float,...]:
        src = self.get_chart(from_chart)
        tgt = self.get_chart(to_chart)
        return src.to_chart(pt, tgt)

    def geodesic_between(self, chart_name: str, p: Tuple[float,...], q: Tuple[float,...]) -> np.ndarray:
        chart = self.get_chart(chart_name)
        # initial guess = straight difference
        v0 = tuple(qi - pi for pi, qi in zip(p, q))
        return chart.geodesic(p, v0, (0.0, 1.0))

    def distance(self, chart_name: str, p: Tuple[float,...], q: Tuple[float,...]) -> float:
        chart = self.get_chart(chart_name)
        return chart.distance(p, q)

# ------------------ Domain Parsing and Validation ------------------
class DomainParser:
    """
    Parse string specs into Domain instances.
    """
    @staticmethod
    def from_string(spec: str, coords: List[Symbol]) -> Domain:
        expr = sympify(spec)
        if isinstance(expr, sp.Relational):
            return InequalityDomain(expr, coords)
        if isinstance(expr, sp.And) or isinstance(expr, sp.Or):
            subs = [InequalityDomain(e, coords) for e in expr.args]
            return UnionDomain(subs)
        raise ValueError(f"Cannot parse domain spec: {spec}")

class ChartValidator:
    """
    Utilities to validate charts and transitions.
    """
    @staticmethod
    def validate_transition(chart_a: Chart, chart_b: Chart, tol: float=1e-6, samples: int=10) -> bool:
        tm = chart_a.transitions.get(chart_b.name)
        if not tm or not tm._inv_func:
            return False
        bd_a = getattr(chart_a.domain, 'bounds', None)
        bd_b = getattr(chart_b.domain, 'bounds', None)
        if not bd_a or not bd_b:
            return False
        for _ in range(samples):
            pt = tuple(np.random.uniform(max(bd_a[i][0], bd_b[i][0]), min(bd_a[i][1], bd_b[i][1])) for i in range(chart_a.dim))
            tgt = tm.to_target(pt)
            src_back = tm.to_source(tgt)
            if any(abs(pt[i] - src_back[i]) > tol for i in range(chart_a.dim)):
                return False
        return True

# ---------------------- Module Export ----------------------
__all__ = [
    'Domain', 'PredicateDomain', 'BoxDomain', 'InequalityDomain', 'UnionDomain',
    'Embedding', 'ParametricEmbedding', 'TransitionMap',
    'Chart', 'SymbolicManifold', 'DomainParser', 'ChartValidator'
]

# EOF — charts.py full extended implementation (~450+ lines)




# from __future__ import annotations
# import sympy as sp
# from sympy import Expr, Symbol, sympify, lambdify, Matrix
# from typing import Callable, Dict, List, Tuple, Optional, Union
# import numpy as np
#
# # Import intrinsic Riemannian structures
# from Riemannian_metric import RiemannianMetric
# from connections import LeviCivitaConnection
#
# # -------------------------- Domain Definitions --------------------------
# class Domain:
#     """
#     Abstract base class for chart domains in R^n.
#     """
#     def contains(self, point: Tuple[float, ...]) -> bool:
#         raise NotImplementedError("Domain.contains must be implemented by subclasses.")
#
# class PredicateDomain(Domain):
#     """
#     Domain defined by a Python predicate f: R^n -> bool.
#     """
#     def __init__(self, predicate: Callable[..., bool]):
#         self.predicate = predicate
#     def contains(self, point: Tuple[float, ...]) -> bool:
#         return bool(self.predicate(*point))
#
# class BoxDomain(Domain):
#     """
#     Axis-aligned box: each coordinate x_i in [lo_i, hi_i].
#     """
#     def __init__(self, bounds: List[Tuple[float, float]]):
#         self.bounds = bounds
#     def contains(self, point: Tuple[float, ...]) -> bool:
#         return all(lo <= x <= hi for x, (lo, hi) in zip(point, self.bounds))
#
# class InequalityDomain(Domain):
#     """
#     Domain defined by a Sympy relation, e.g. x**2 + y**2 < 1.
#     """
#     def __init__(self, expr: Union[Expr, str], coords: List[Symbol]):
#         self.expr = sympify(expr)
#         self.coords = coords
#         self._func = lambdify(coords, self.expr, "numpy")
#     def contains(self, point: Tuple[float, ...]) -> bool:
#         return bool(self._func(*point))
#
# class UnionDomain(Domain):
#     """
#     Union of multiple domains: point is in any one.
#     """
#     def __init__(self, domains: List[Domain]):
#         self.domains = domains
#     def contains(self, point: Tuple[float, ...]) -> bool:
#         return any(d.contains(point) for d in self.domains)
#
# # ----------------------- Embedding Definitions ------------------------
# class Embedding:
#     """
#     Base for embedding chart coordinates into R^m.
#     Automatically sympifies and lambdifies map expressions.
#     """
#     def __init__(self, coords: List[Symbol], map_exprs: List[Expr]):
#         self.coords = list(coords)
#         self.map_exprs = [sympify(e) for e in map_exprs]
#         # lambdify numerical evaluator
#         self._func = lambdify(self.coords, self.map_exprs, modules="numpy")
#     def evaluate(self, point: Tuple[float, ...]) -> Tuple[float, ...]:
#         flat = np.array(self._func(*point), dtype=float)
#         return tuple(flat.flatten())
#
# class ParametricEmbedding(Embedding):
#     """
#     Explicit alias for parametric embeddings.
#     """
#     pass
#
# # ---------------------- Coordinate Transition ----------------------
# class TransitionMap:
#     """
#     Represents a transition map between two chart coordinate systems.
#     """
#     def __init__(
#         self,
#         source_coords: List[Symbol],
#         target_coords: List[Symbol],
#         forward_map: Dict[Symbol, Expr],
#         inverse_map: Optional[Dict[Symbol, Expr]] = None
#     ):
#         if set(forward_map.keys()) != set(source_coords):
#             raise ValueError("Forward map must define images of all source coords.")
#         self.source_coords = list(source_coords)
#         self.target_coords = list(target_coords)
#         self.forward_map = {s: sympify(e) for s,e in forward_map.items()}
#         self.inverse_map = {s: sympify(e) for s,e in (inverse_map or {}).items()}
#         self._fwd = lambdify(self.source_coords, list(self.forward_map.values()), "numpy")
#         self._inv = (
#             lambdify(self.target_coords, list(self.inverse_map.values()), "numpy")
#             if inverse_map else None
#         )
#
#     def to_target(self, pt: Tuple[float, ...]) -> Tuple[float, ...]:
#         return tuple(np.array(self._fwd(*pt), dtype=float).flatten())
#
#     def to_source(self, pt: Tuple[float, ...]) -> Tuple[float, ...]:
#         if not self._inv:
#             raise ValueError("Inverse transition is not defined.")
#         return tuple(np.array(self._inv(*pt), dtype=float).flatten())
#
# # -------------------------- Chart Definition --------------------------
# class Chart:
#     """
#     Coordinate chart on an n-dimensional manifold.
#
#     - name: identifier
#     - coords: list of sympy Symbols for local coordinates
#     - domain: Domain instance
#     - embedding: Embedding into R^m
#     """
#     def __init__(
#         self,
#         name: str,
#         coords: List[Symbol],
#         domain: Domain = PredicateDomain(lambda *args: True),
#         embedding: Optional[Embedding] = None
#     ):
#         self.name = name
#         self.coords = list(coords)
#         self.dim = len(coords)
#         self.domain = domain
#         self.embedding = embedding
#         self.transitions: Dict[str, TransitionMap] = {}
#
#         # Automatically compute induced metric & connection if embedding provided
#         if self.embedding is not None:
#             self._compute_intrinsic()
#
#     def _compute_intrinsic(self):
#         """
#         Compute pulled-back metric and Levi-Civita connection.
#         """
#         # Jacobian of embedding
#         J = Matrix(self.embedding.map_exprs).jacobian(self.coords)
#         # induced metric: g_ij = sum_a J^a_i J^a_j
#         g = J.T * J
#         self.metric = RiemannianMetric(self.coords, g)
#         self.connection = LeviCivitaConnection(self.metric, self)
#
#     def induced_metric(self) -> RiemannianMetric:
#         """
#         Return the induced Riemannian metric on this chart.
#         """
#         if not hasattr(self, 'metric'):
#             raise AttributeError("Chart has no induced metric; embedding missing.")
#         return self.metric
#
#     def christoffel(self) -> List[List[List[Expr]]]:
#         """
#         Return Christoffel symbols from the Levi-Civita connection.
#         """
#         return self.connection.Gamma
#
#     def add_transition(
#         self,
#         other: Chart,
#         forward: Dict[Symbol, Expr],
#         inverse: Optional[Dict[Symbol, Expr]] = None
#     ) -> None:
#         """
#         Define transition to another chart.
#         """
#         tm = TransitionMap(self.coords, other.coords, forward, inverse)
#         self.transitions[other.name] = tm
#
#     def to_chart(self, point: Tuple[float,...], other: Chart) -> Tuple[float,...]:
#         if other.name not in self.transitions:
#             raise KeyError(f"No transition from {self.name} to {other.name}.")
#         return self.transitions[other.name].to_target(point)
#
#     def contains(self, pt: Tuple[float,...]) -> bool:
#         return self.domain.contains(pt)
#
#     def sample_grid(
#         self,
#         num: int = 20,
#         bounds: Optional[List[Tuple[float,float]]] = None
#     ) -> List[Tuple[float,...]]:
#         rng = bounds or getattr(self.domain, 'bounds', None)
#         if not rng:
#             raise ValueError("Bounds must be provided or domain must be BoxDomain.")
#         axes = [np.linspace(lo,hi,num) for lo,hi in rng]
#         mesh = np.meshgrid(*axes)
#         pts = [tuple(mesh[d].flat[i] for d in range(self.dim)) for i in range(mesh[0].size)]
#         return [p for p in pts if self.contains(p)]
#
#     def geodesic_equations(self):
#         """
#         Return geodesic ODEs and functions for use in numerical integration.
#         """
#         return self.connection.geodesic_equations()
#
#     def parallel_transport_equations(self, curve_funcs: List[Callable], vec_funcs: List[Callable]):
#         """
#         Return symbolic parallel-transport ODEs along a given curve.
#         """
#         return self.connection.parallel_transport_equations(curve_funcs, vec_funcs)
#
# # ----------------------- Manifold Definition -----------------------
# class SymbolicManifold:
#     """
#     An n-dimensional manifold represented by an atlas of charts.
#     """
#     def __init__(self, name: str):
#         self.name = name
#         self.charts: Dict[str, Chart] = {}
#         self.default: Optional[Chart] = None
#
#     def add_chart(self, chart: Chart, default: bool = False) -> None:
#         if chart.name in self.charts:
#             raise KeyError(f"Chart '{chart.name}' already exists in manifold '{self.name}'.")
#         self.charts[chart.name] = chart
#         if default or self.default is None:
#             self.default = chart
#
#     def get_chart(self, name: str) -> Chart:
#         return self.charts[name]
#
#     def find_chart(self, point: Tuple[float,...]) -> Chart:
#         for c in self.charts.values():
#             try:
#                 if c.contains(point):
#                     return c
#             except:
#                 pass
#         if self.default and self.default.contains(point):
#             return self.default
#         raise ValueError(f"No chart contains point {point}.")
#
#     def transition(
#         self,
#         point: Tuple[float,...],
#         source: Optional[str] = None,
#         target: Optional[str] = None
#     ) -> Tuple[float,...]:
#         src = self.charts[source] if source else self.find_chart(point)
#         tgt = self.charts[target] if target else self.default
#         return src.to_chart(point, tgt)
#
# # -------------------- Future Extension Hooks --------------------
# class DomainParser:
#     """
#     Parse domain strings into Domain instances.
#     """
#     @staticmethod
#     def from_string(spec: str, coords: List[Symbol]) -> Domain:
#         expr = sympify(spec)
#         if expr.is_Relational:
#             return InequalityDomain(expr, coords)
#         if expr.is_And or expr.is_Or:
#             subs = [InequalityDomain(e, coords) for e in expr.args]
#             return UnionDomain(subs)
#         raise ValueError(f"Cannot parse domain spec: {spec}")
#
# # -------------------- Validation Utilities --------------------
# class ChartValidator:
#     """
#     Validate chart transitions and consistency.
#     """
#     @staticmethod
#     def validate_transition(
#         a: Chart,
#         b: Chart,
#         tol: float = 1e-6,
#         samples: int = 5
#     ) -> bool:
#         tm = a.transitions.get(b.name)
#         if not tm or not tm._inv:
#             return False
#         bd_a = getattr(a.domain, 'bounds', None)
#         bd_b = getattr(b.domain, 'bounds', None)
#         if not bd_a or not bd_b:
#             return False
#         for _ in range(samples):
#             pt = tuple(np.random.uniform(max(bd_a[i][0], bd_b[i][0]),
#                                          min(bd_a[i][1], bd_b[i][1])) for i in range(a.dim))
#             tgt = tm.to_target(pt)
#             src = tm.to_source(tgt)
#             if any(abs(x-y)>tol for x,y in zip(pt, src)):
#                 return False
#         return True

# End of charts.py



# from __future__ import annotations
# import sympy as sp
# from sympy import Expr, Symbol, sympify, lambdify
# from typing import Callable, Dict, List, Tuple, Optional, Union
# import numpy as np
#
# # -------------------------- Domain Definitions --------------------------
# class Domain:
#     """
#     Abstract base class for chart domains in R^n.
#     """
#     def contains(self, point: Tuple[float, ...]) -> bool:
#         """Check if point lies in domain."""
#         raise NotImplementedError("Domain.contains must be implemented by subclasses.")
#
# class PredicateDomain(Domain):
#     """Domain defined by a Python predicate f: R^n -> bool."""
#     def __init__(self, predicate: Callable[..., bool]):
#         self.predicate = predicate
#     def contains(self, point: Tuple[float, ...]) -> bool:
#         return bool(self.predicate(*point))
#
# class BoxDomain(Domain):
#     """Axis-aligned box: each coordinate x_i in [lo_i, hi_i]."""
#     def __init__(self, bounds: List[Tuple[float, float]]):
#         self.bounds = bounds
#     def contains(self, point: Tuple[float, ...]) -> bool:
#         return all(lo <= x <= hi for x, (lo, hi) in zip(point, self.bounds))
#
# class InequalityDomain(Domain):
#     """Domain defined by a Sympy relational expression, e.g. x**2 + y**2 < 1."""
#     def __init__(self, expr: Union[Expr, str], coords: List[Symbol]):
#         self.expr = sympify(expr)
#         self.coords = coords
#         self._func = lambdify(coords, self.expr, "numpy")
#     def contains(self, point: Tuple[float, ...]) -> bool:
#         return bool(self._func(*point))
#
# class UnionDomain(Domain):
#     """Union of multiple domains: point is in any one."""
#     def __init__(self, domains: List[Domain]):
#         self.domains = domains
#     def contains(self, point: Tuple[float, ...]) -> bool:
#         return any(dom.contains(point) for dom in self.domains)
#
# # ------------------------ Embedding Definitions ------------------------
# class Embedding:
#     """
#     Parametric embedding of chart coordinates into R^m.
#     Automatically stores symbolic map and numeric evaluator.
#     """
#     def __init__(self, coords: List[Symbol], map_exprs: List[Expr]):
#         self.coords = coords
#         self.map_exprs = [sympify(expr) for expr in map_exprs]
#         self._func = lambdify(coords, self.map_exprs, "numpy")
#     def evaluate(self, point: Tuple[float, ...]) -> Tuple[float, ...]:
#         """Numeric evaluation of embedding at given chart point."""
#         arr = np.array(self._func(*point), dtype=float)
#         return tuple(arr.flatten())
#
# # ---------------------- Chart Definition ----------------------
# class Chart:
#     """
#     Coordinate chart on a manifold with:
#       - Domain specification
#       - Embedding into R^m
#       - Automatic induced metric & Levi-Civita connection
#     Provides methods for intrinsic operations.
#     """
#     def __init__(
#         self,
#         name: str,
#         coords: List[Symbol],
#         domain: Domain = PredicateDomain(lambda *args: True),
#         embedding: Optional[Embedding] = None
#     ):
#         self.name = name
#         self.coords = list(coords)
#         self.dim = len(coords)
#         self.domain = domain
#         self.embedding = embedding
#         self.transitions: Dict[str, TransitionMap] = {}
#         # Automatic intrinsic data
#         self._metric = None
#         self._connection = None
#         if embedding is not None:
#             from Riemannian_metric import RiemannianMetric
#             from connections import LeviCivitaConnection
#             # compute induced metric tensor g_ij = E^T E
#             g = self._compute_induced_metric()
#             self._metric = RiemannianMetric(self.coords, g)
#             # compute Levi-Civita connection on this chart
#             self._connection = LeviCivitaConnection(self._metric, self)
#
#     def _compute_induced_metric(self) -> sp.Matrix:
#         """
#         Compute induced metric: pullback of Euclidean metric under embedding.
#         g = J^T J, where J_{ai} = dX^a/dx^i.
#         """
#         # Jacobian of embedding map expressions
#         J = sp.Matrix(self.embedding.map_exprs).jacobian(self.coords)
#         return J.T * J
#
#     @property
#     def metric(self):
#         """Return induced Riemannian metric."""
#         return self._metric
#
#     @property
#     def connection(self):
#         """Return induced Levi-Civita connection."""
#         return self._connection
#
#     def induced_metric(self) -> 'RiemannianMetric':
#         return self._metric
#
#     def induced_connection(self) -> 'LeviCivitaConnection':
#         return self._connection
#
#     def exponential_map(
#         self,
#         base_point: Tuple[float, ...],
#         init_velocity: Tuple[float, ...],
#         t: float = 1.0
#     ) -> Tuple[float, ...]:
#         """
#         Exponential map at base_point along init_velocity at parameter t.
#         """
#         return self._connection.exponential_map(base_point, init_velocity, t)
#
#     def geodesic_equations(self) -> Tuple[List[sp.Eq], List[sp.Function]]:
#         """Return symbolic geodesic ODEs and functions from connection."""
#         return self._connection.geodesic_equations()
#
#     def parallel_transport_equations(
#         self,
#         curve_funcs: List[sp.Function],
#         vector_funcs: List[sp.Function]
#     ) -> List[sp.Eq]:
#         """Return parallel transport ODEs along given curve symbolic functions."""
#         return self._connection.parallel_transport_equations(curve_funcs, vector_funcs)
#
#     def add_transition(
#         self,
#         other: Chart,
#         forward: Dict[Symbol, Expr],
#         inverse: Optional[Dict[Symbol, Expr]] = None
#     ) -> None:
#         """Define coordinate transition to another chart."""
#         tm = TransitionMap(self.coords, other.coords, forward, inverse)
#         self.transitions[other.name] = tm
#
#     def contains(self, point: Tuple[float, ...]) -> bool:
#         return self.domain.contains(point)
#
#     def to_chart(self, point: Tuple[float, ...], other: Chart) -> Tuple[float, ...]:
#         return self.transitions[other.name].to_target(point)
#
#     def sample_grid(
#         self,
#         bounds: Optional[List[Tuple[float, float]]] = None,
#         num: int = 20
#     ) -> List[Tuple[float, ...]]:
#         """
#         Generate grid of points in domain for sampling, defaults to domain.bounds.
#         """
#         rng = bounds or getattr(self.domain, 'bounds', None)
#         if rng is None:
#             raise ValueError("Bounds must be provided or domain must be BoxDomain.")
#         axes = [np.linspace(lo, hi, num) for lo, hi in rng]
#         mesh = np.meshgrid(*axes)
#         pts_flat = [tuple(mesh[d].flat[i] for d in range(self.dim)) for i in range(mesh[0].size)]
#         return [pt for pt in pts_flat if self.contains(pt)]
#
# # ----------------------- Manifold Definition -----------------------
# class SymbolicManifold:
#     """Abstract manifold represented by an atlas of charts."""
#     def __init__(self, name: str):
#         self.name = name
#         self.charts: Dict[str, Chart] = {}
#         self.default: Optional[Chart] = None
#
#     def add_chart(self, chart: Chart, default: bool = False) -> None:
#         if chart.name in self.charts:
#             raise KeyError(f"Chart '{chart.name}' already exists.")
#         self.charts[chart.name] = chart
#         if default or self.default is None:
#             self.default = chart
#
#     def get_chart(self, name: str) -> Chart:
#         return self.charts[name]
#
#     def find_chart(self, point: Tuple[float, ...]) -> Chart:
#         for chart in self.charts.values():
#             if chart.contains(point):
#                 return chart
#         if self.default and self.default.contains(point):
#             return self.default
#         raise ValueError(f"No chart contains point {point}.")
#
#     def transition(
#         self,
#         point: Tuple[float, ...],
#         source: Optional[str] = None,
#         target: Optional[str] = None
#     ) -> Tuple[float, ...]:
#         src = self.charts[source] if source else self.find_chart(point)
#         tgt = self.charts[target] if target else self.default
#         return src.to_chart(point, tgt)
#
# # -------------------- Coordinate Transition & Validation --------------------
# class TransitionMap:
#     """Represents transition between two chart coordinate systems."""
#     def __init__(
#         self,
#         source_coords: List[Symbol],
#         target_coords: List[Symbol],
#         forward_map: Dict[Symbol, Expr],
#         inverse_map: Optional[Dict[Symbol, Expr]] = None
#     ):
#         if set(forward_map.keys()) != set(source_coords):
#             raise ValueError("Forward map must define images of all source coords.")
#         self.source_coords = source_coords
#         self.target_coords = target_coords
#         self.forward_map = {sym: sympify(expr) for sym, expr in forward_map.items()}
#         self.inverse_map = {sym: sympify(expr) for sym, expr in (inverse_map or {}).items()}
#         self._fwd_func = lambdify(source_coords, list(self.forward_map.values()), "numpy")
#         self._inv_func = (
#             lambdify(target_coords, list(self.inverse_map.values()), "numpy")
#             if inverse_map else None
#         )
#
#     def to_target(self, point: Tuple[float, ...]) -> Tuple[float, ...]:
#         arr = np.array(self._fwd_func(*point), dtype=float)
#         return tuple(arr.flatten())
#
#     def to_source(self, point: Tuple[float, ...]) -> Tuple[float, ...]:
#         if not self._inv_func:
#             raise ValueError("Inverse transition is not defined.")
#         arr = np.array(self._inv_func(*point), dtype=float)
#         return tuple(arr.flatten())
#
# class ChartValidator:
#     """Validate chart transitions numerically for consistency."""
#     @staticmethod
#     def validate_transition(
#         chart_a: Chart,
#         chart_b: Chart,
#         tol: float = 1e-6,
#         samples: int = 5
#     ) -> bool:
#         tm = chart_a.transitions.get(chart_b.name)
#         if not tm or not tm._inv_func:
#             return False
#         bd_a = getattr(chart_a.domain, 'bounds', None)
#         bd_b = getattr(chart_b.domain, 'bounds', None)
#         if not bd_a or not bd_b:
#             return False
#         for _ in range(samples):
#             pt = tuple(
#                 np.random.uniform(max(bd_a[d][0], bd_b[d][0]), min(bd_a[d][1], bd_b[d][1]))
#                 for d in range(chart_a.dim)
#             )
#             tgt = tm.to_target(pt)
#             src = tm.to_source(tgt)
#             if any(abs(x-y) > tol for x,y in zip(pt, src)):
#                 return False
#         return True
#
# # ---------------------- Domain Parser ----------------------
# class DomainParser:
#     """Parse domain specification strings or objects into Domain instances."""
#     @staticmethod
#     def from_string(spec: str, coords: List[Symbol]) -> Domain:
#         expr = sympify(spec)
#         if isinstance(expr, sp.Relational):
#             return InequalityDomain(expr, coords)
#         if isinstance(expr, (sp.And, sp.Or)):
#             subdomains = [InequalityDomain(e, coords) for e in expr.args]
#             return UnionDomain(subdomains)
#         raise ValueError(f"Cannot parse domain spec: {spec}")


# from __future__ import annotations
# import sympy as sp
# from sympy import Expr, Symbol, sympify, lambdify, Matrix
# from typing import Callable, Dict, List, Tuple, Optional, Union
# import numpy as np
#
# # Optional induced metric support
# from Riemannian_metric import RiemannianMetric
#
# # -------------------------- Domain Definitions --------------------------
# class Domain:
#     """
#     Abstract base class for chart domains in R^n.
#     """
#     def contains(self, point: Tuple[float, ...]) -> bool:
#         """
#         Determine if a numeric point lies in this domain.
#         """
#         raise NotImplementedError("Domain.contains must be implemented by subclasses.")
#
# class PredicateDomain(Domain):
#     """
#     Domain defined by a Python predicate f: R^n -> bool.
#     """
#     def __init__(self, predicate: Callable[..., bool]):
#         self.predicate = predicate
#
#     def contains(self, point: Tuple[float, ...]) -> bool:
#         return bool(self.predicate(*point))
#
# class BoxDomain(Domain):
#     """
#     Axis-aligned box: each coordinate x_i in [lo_i, hi_i].
#     """
#     def __init__(self, bounds: List[Tuple[float, float]]):
#         # bounds: list of (min, max) pairs for each coordinate
#         self.bounds = bounds
#
#     def contains(self, point: Tuple[float, ...]) -> bool:
#         return all(lo <= x <= hi for x, (lo, hi) in zip(point, self.bounds))
#
# class InequalityDomain(Domain):
#     """
#     Domain defined by a Sympy relational expression, e.g. x**2 + y**2 < 1.
#     """
#     def __init__(self, expr: Union[Expr, str], coords: List[Symbol]):
#         self.expr: Expr = sympify(expr)
#         self.coords = coords
#         self._func = lambdify(coords, self.expr, "numpy")
#
#     def contains(self, point: Tuple[float, ...]) -> bool:
#         return bool(self._func(*point))
#
# class UnionDomain(Domain):
#     """
#     Union of multiple domains: point is in any one.
#     """
#     def __init__(self, domains: List[Domain]):
#         self.domains = domains
#
#     def contains(self, point: Tuple[float, ...]) -> bool:
#         return any(domain.contains(point) for domain in self.domains)
#
# # ------------------------ Embedding Definitions ------------------------
# class Embedding:
#     """
#     Base class for embedding chart coordinates into R^m.
#     By default, behaves like ParametricEmbedding: stores coords and map_exprs, compiles with lambdify.
#     """
#     def __init__(self, coords: List[Symbol], map_exprs: List[Expr]):
#         self.coords = coords
#         self.map_exprs = [sympify(e) for e in map_exprs]
#         self._func = lambdify(self.coords, self.map_exprs, "numpy")
#
#     def evaluate(self, point: Tuple[float, ...]) -> Tuple[float, ...]:
#         arr = np.array(self._func(*point), dtype=float)
#         return tuple(arr.flatten())
#
# class ParametricEmbedding(Embedding):
#     """
#     Embedding defined by Sympy expressions map_exprs in terms of coords.
#     """
#     def __init__(self, coords: List[Symbol], map_exprs: List[Expr]):
#         super().__init__(coords, map_exprs)
#
# # ---------------------- Coordinate Transition ----------------------
# class TransitionMap:
#     """
#     Represents a transition map between two chart coordinate systems.
#     """
#     def __init__(
#         self,
#         source_coords: List[Symbol],
#         target_coords: List[Symbol],
#         forward_map: Dict[Symbol, Expr],
#         inverse_map: Optional[Dict[Symbol, Expr]] = None
#     ):
#         if set(forward_map.keys()) != set(source_coords):
#             raise ValueError("Forward map must define images of all source coords.")
#         self.source_coords = source_coords
#         self.target_coords = target_coords
#         self.forward_map = {s: sympify(e) for s, e in forward_map.items()}
#         self.inverse_map = {s: sympify(e) for s, e in (inverse_map or {}).items()}
#         self._fwd = lambdify(source_coords, list(self.forward_map.values()), "numpy")
#         self._inv = (lambdify(target_coords, list(self.inverse_map.values()), "numpy")
#                      if inverse_map else None)
#
#     def to_target(self, point: Tuple[float, ...]) -> Tuple[float, ...]:
#         return tuple(np.array(self._fwd(*point), float).flatten())
#
#     def to_source(self, point: Tuple[float, ...]) -> Tuple[float, ...]:
#         if not self._inv:
#             raise ValueError("Inverse transition not defined.")
#         return tuple(np.array(self._inv(*point), float).flatten())
#
# # -------------------------- Chart Definition --------------------------
# class Chart:
#     """
#     Coordinate chart on a manifold.
#
#     Attributes:
#       name      : identifier
#       coords    : symbols for local coordinates
#       domain    : Domain instance
#       embedding : Embedding into R^m
#       metric    : induced RiemannianMetric (if embedding provided)
#       transitions: other charts transitions
#     """
#     def __init__(
#         self,
#         name: str,
#         coords: List[Symbol],
#         domain: Domain = PredicateDomain(lambda *args: True),
#         embedding: Optional[Embedding] = None
#     ):
#         self.name = name
#         self.coords = coords
#         self.dim = len(coords)
#         self.domain = domain
#         self.embedding = embedding
#         self.transitions: Dict[str, TransitionMap] = {}
#         # auto-compute induced metric if embedding provided
#         self.metric: Optional[RiemannianMetric] = None
#         if embedding is not None:
#             J = Matrix(self.embedding.map_exprs).jacobian(self.coords)
#             g = sp.simplify(J.T * J)
#             self.metric = RiemannianMetric(self.coords, g)
#
#     def add_transition(
#         self,
#         other: Chart,
#         forward: Dict[Symbol, Expr],
#         inverse: Optional[Dict[Symbol, Expr]] = None
#     ) -> None:
#         tm = TransitionMap(self.coords, other.coords, forward, inverse)
#         self.transitions[other.name] = tm
#
#     def contains(self, point: Tuple[float, ...]) -> bool:
#         return self.domain.contains(point)
#
#     def to_chart(self, point: Tuple[float, ...], other: Chart) -> Tuple[float, ...]:
#         if other.name not in self.transitions:
#             raise KeyError(f"No transition from {self.name} to {other.name}.")
#         return self.transitions[other.name].to_target(point)
#
#     def sample_grid(
#         self,
#         bounds: Optional[List[Tuple[float, float]]] = None,
#         num: int = 20
#     ) -> List[Tuple[float, ...]]:
#         rng = bounds or getattr(self.domain, 'bounds', None)
#         if not rng:
#             raise ValueError("Bounds must be provided or domain must be BoxDomain.")
#         axes = [np.linspace(lo, hi, num) for lo, hi in rng]
#         mesh = np.meshgrid(*axes)
#         pts = [tuple(mesh[d].flat[i] for d in range(self.dim)) for i in range(mesh[0].size)]
#         return [p for p in pts if self.contains(p)]
#
# # ----------------------- Manifold Definition -----------------------
# class SymbolicManifold:
#     """
#     Abstract manifold represented by an atlas of charts.
#     """
#     def __init__(self, name: str):
#         self.name = name
#         self.charts: Dict[str, Chart] = {}
#         self.default: Optional[Chart] = None
#
#     def add_chart(self, chart: Chart, default: bool = False) -> None:
#         if chart.name in self.charts:
#             raise KeyError(f"Chart '{chart.name}' already exists.")
#         self.charts[chart.name] = chart
#         if default or self.default is None:
#             self.default = chart
#
#     def get_chart(self, name: str) -> Chart:
#         if name not in self.charts:
#             raise KeyError(f"Chart '{name}' not in manifold.")
#         return self.charts[name]
#
#     def find_chart(self, point: Tuple[float, ...]) -> Chart:
#         for c in self.charts.values():
#             if c.contains(point):
#                 return c
#         if self.default and self.default.contains(point):
#             return self.default
#         raise ValueError(f"No chart contains point {point}.")
#
#     def transition(
#         self,
#         point: Tuple[float, ...],
#         source: Optional[str] = None,
#         target: Optional[str] = None
#     ) -> Tuple[float, ...]:
#         src = self.charts[source] if source else self.find_chart(point)
#         tgt = self.charts[target] if target else self.default
#         return src.to_chart(point, tgt)
#
# # -------------------- Future Extension Hooks --------------------
# class DomainParser:
#     """
#     Parse domain specification strings into Domain instances.
#     """
#     @staticmethod
#     def from_string(spec: str, coords: List[Symbol]) -> Domain:
#         expr = sympify(spec)
#         if isinstance(expr, sp.Relational):
#             return InequalityDomain(expr, coords)
#         if isinstance(expr, sp.And) or isinstance(expr, sp.Or):
#             return UnionDomain([InequalityDomain(e, coords) for e in expr.args])
#         raise ValueError(f"Cannot parse domain spec: {spec}")
#
# class ChartValidator:
#     """
#     Validate chart transitions for consistency.
#     """
#     @staticmethod
#     def validate_transition(
#         a: Chart, b: Chart,
#         tol: float = 1e-6,
#         samples: int = 5
#     ) -> bool:
#         tm = a.transitions.get(b.name)
#         if not tm or not tm._inv:
#             return False
#         bd_a = getattr(a.domain, 'bounds', None)
#         bd_b = getattr(b.domain, 'bounds', None)
#         if not bd_a or not bd_b:
#             return False
#         for _ in range(samples):
#             pt = tuple(np.random.uniform(max(bd_a[d][0], bd_b[d][0]),
#                                          min(bd_a[d][1], bd_b[d][1]))
#                        for d in range(a.dim))
#             tgt = tm.to_target(pt)
#             src = tm.to_source(tgt)
#             if any(abs(x-y)>tol for x,y in zip(pt, src)):
#                 return False
#         return True

# End of charts.py

# from __future__ import annotations
# import sympy as sp
# from sympy import Expr, Symbol, sympify, lambdify
# from typing import Callable, Dict, List, Tuple, Optional, Union
# import numpy as np
#
# # -------------------------- Domain Definitions --------------------------
# class Domain:
#     """
#     Abstract base class for chart domains in R^n.
#     """
#     def contains(self, point: Tuple[float, ...]) -> bool:
#         """
#         Determine if a numeric point lies in this domain.
#         """
#         raise NotImplementedError("Domain.contains must be implemented by subclasses.")
#
# class PredicateDomain(Domain):
#     """
#     Domain defined by a Python predicate f: R^n -> bool.
#     """
#     def __init__(self, predicate: Callable[..., bool]):
#         self.predicate = predicate
#
#     def contains(self, point: Tuple[float, ...]) -> bool:
#         return bool(self.predicate(*point))
#
# class BoxDomain(Domain):
#     """
#     Axis-aligned box: each coordinate x_i in [lo_i, hi_i].
#     """
#     def __init__(self, bounds: List[Tuple[float, float]]):
#         # bounds: list of (min, max) pairs for each coordinate
#         self.bounds = bounds
#
#     def contains(self, point: Tuple[float, ...]) -> bool:
#         return all(lo <= x <= hi for x, (lo, hi) in zip(point, self.bounds))
#
# class InequalityDomain(Domain):
#     """
#     Domain defined by a Sympy relational expression, e.g. x**2 + y**2 < 1.
#     """
#     def __init__(self, expr: Union[Expr, str], coords: List[Symbol]):
#         self.expr = sympify(expr)
#         self.coords = coords
#         # lambdify to a numeric function
#         self._func = lambdify(coords, self.expr, "numpy")
#
#     def contains(self, point: Tuple[float, ...]) -> bool:
#         return bool(self._func(*point))
#
# class UnionDomain(Domain):
#     """
#     Union of multiple domains: point is in any one.
#     """
#     def __init__(self, domains: List[Domain]):
#         self.domains = domains
#
#     def contains(self, point: Tuple[float, ...]) -> bool:
#         return any(domain.contains(point) for domain in self.domains)
#
#
# # ------------------------ Embedding Definitions ------------------------
# class Embedding:
#     """
#     Base class for embedding chart coordinates into R^m.
#     Stores symbolic map expressions and compiles numeric function.
#     """
#     def __init__(self, coords: List[Symbol], map_exprs: List[Expr]):
#         self.coords = coords
#         self.map_exprs = [sympify(expr) for expr in map_exprs]
#         self._func = lambdify(self.coords, self.map_exprs, "numpy")
#
#     def evaluate(self, point: Tuple[float, ...]) -> Tuple[float, ...]:
#         """
#         Map a chart coordinate tuple to an embedded point in R^m.
#         """
#         arr = np.array(self._func(*point), dtype=float)
#         return tuple(arr.flatten())
#
# class ParametricEmbedding(Embedding):
#     """
#     Explicit parametric embedding: identical to Embedding for clarity.
#     """
#     def __init__(self, coords: List[Symbol], map_exprs: List[Expr]):
#         super().__init__(coords, map_exprs)
#
#     def evaluate(self, point: Tuple[float, ...]) -> Tuple[float, ...]:
#         return super().evaluate(point)
#
# # ---------------------- Coordinate Transition ----------------------
# class TransitionMap:
#     """
#     Represents a transition map between two chart coordinate systems.
#     """
#     def __init__(
#         self,
#         source_coords: List[Symbol],
#         target_coords: List[Symbol],
#         forward_map: Dict[Symbol, Expr],
#         inverse_map: Optional[Dict[Symbol, Expr]] = None
#     ):
#         # Validate keys
#         if set(forward_map.keys()) != set(source_coords):
#             raise ValueError("Forward map must define images of all source coords.")
#         self.source_coords = source_coords
#         self.target_coords = target_coords
#         self.forward_map = {sym: sympify(expr) for sym, expr in forward_map.items()}
#         self.inverse_map = {sym: sympify(expr) for sym, expr in (inverse_map or {}).items()}
#         self._fwd_func = lambdify(source_coords, list(self.forward_map.values()), "numpy")
#         self._inv_func = (
#             lambdify(target_coords, list(self.inverse_map.values()), "numpy")
#             if inverse_map else None
#         )
#
#     def to_target(self, point: Tuple[float, ...]) -> Tuple[float, ...]:
#         arr = np.array(self._fwd_func(*point), dtype=float)
#         return tuple(arr.flatten())
#
#     def to_source(self, point: Tuple[float, ...]) -> Tuple[float, ...]:
#         if not self._inv_func:
#             raise ValueError("Inverse transition is not defined.")
#         arr = np.array(self._inv_func(*point), dtype=float)
#         return tuple(arr.flatten())
#
# # -------------------------- Chart Definition --------------------------
# class Chart:
#     """
#     Coordinate chart on a manifold.
#
#     Attributes:
#       name: unique identifier
#       coords: symbols for local R^n coordinates
#       domain: Domain instance specifying valid region
#       embedding: optional Embedding into R^m
#       transitions: mapping other_chart_name -> TransitionMap
#     """
#     def __init__(
#         self,
#         name: str,
#         coords: List[Symbol],
#         domain: Domain = PredicateDomain(lambda *args: True),
#         embedding: Optional[Embedding] = None
#     ):
#         self.name = name
#         self.coords = list(coords)
#         self.dim = len(coords)
#         self.domain = domain
#         self.embedding = embedding
#         self.transitions: Dict[str, TransitionMap] = {}
#
#     def add_transition(
#         self,
#         other: Chart,
#         forward: Dict[Symbol, Expr],
#         inverse: Optional[Dict[Symbol, Expr]] = None
#     ) -> None:
#         tm = TransitionMap(self.coords, other.coords, forward, inverse)
#         self.transitions[other.name] = tm
#
#     def contains(self, point: Tuple[float, ...]) -> bool:
#         return self.domain.contains(point)
#
#     def to_chart(self, point: Tuple[float, ...], other: Chart) -> Tuple[float, ...]:
#         if other.name not in self.transitions:
#             raise KeyError(f"No transition from {self.name} to {other.name}.")
#         return self.transitions[other.name].to_target(point)
#
#     def sample_grid(
#         self,
#         bounds: Optional[List[Tuple[float, float]]] = None,
#         num: int = 20
#     ) -> List[Tuple[float, ...]]:
#         rng = bounds or getattr(self.domain, 'bounds', None)
#         if not rng:
#             raise ValueError("Bounds must be provided or domain must be BoxDomain.")
#         axes = [np.linspace(lo, hi, num) for lo, hi in rng]
#         mesh = np.meshgrid(*axes)
#         pts = [tuple(mesh[d].flat[i] for d in range(self.dim)) for i in range(mesh[0].size)]
#         return [pt for pt in pts if self.contains(pt)]
#
# # ----------------------- Manifold Definition -----------------------
# class SymbolicManifold:
#     """
#     Abstract n-dimensional manifold represented by an atlas of charts.
#     """
#     def __init__(self, name: str):
#         self.name = name
#         self.charts: Dict[str, Chart] = {}
#         self.default: Optional[Chart] = None
#
#     def add_chart(self, chart: Chart, default: bool = False) -> None:
#         if chart.name in self.charts:
#             raise KeyError(f"Chart '{chart.name}' already exists in manifold '{self.name}'.")
#         self.charts[chart.name] = chart
#         if default or self.default is None:
#             self.default = chart
#
#     def get_chart(self, name: str) -> Chart:
#         if name not in self.charts:
#             raise KeyError(f"Chart '{name}' not found in manifold '{self.name}'.")
#         return self.charts[name]
#
#     def find_chart(self, point: Tuple[float, ...]) -> Chart:
#         for chart in self.charts.values():
#             try:
#                 if chart.contains(point): return chart
#             except Exception:
#                 continue
#         if self.default and self.default.contains(point): return self.default
#         raise ValueError(f"No chart contains point {point} in manifold '{self.name}'.")
#
#     def transition(
#         self,
#         point: Tuple[float, ...],
#         source: Optional[str] = None,
#         target: Optional[str] = None
#     ) -> Tuple[float, ...]:
#         src = self.charts[source] if source else self.find_chart(point)
#         tgt = self.charts[target] if target else self.default
#         if not src or not tgt:
#             raise ValueError("Source or target chart undefined.")
#         return src.to_chart(point, tgt)
#
# # -------------------- Future Extension Hooks --------------------
# class DomainParser:
#     """
#     Parse domain specification strings or objects into Domain instances.
#     """
#     @staticmethod
#     def from_string(spec: str, coords: List[Symbol]) -> Domain:
#         expr = sympify(spec)
#         if isinstance(expr, sp.Relational):
#             return InequalityDomain(expr, coords)
#         if isinstance(expr, sp.And) or isinstance(expr, sp.Or):
#             sub = [InequalityDomain(e, coords) for e in expr.args]
#             return UnionDomain(sub)
#         raise ValueError(f"Cannot parse domain spec: {spec}")
#
# class ChartValidator:
#     """
#     Validate chart and transition consistency.
#     """
#     @staticmethod
#     def validate_transition(
#         chart_a: Chart,
#         chart_b: Chart,
#         tol: float = 1e-6,
#         samples: int = 5
#     ) -> bool:
#         tm = chart_a.transitions.get(chart_b.name)
#         if not tm or not tm._inv_func: return False
#         bd_a = getattr(chart_a.domain, 'bounds', None)
#         bd_b = getattr(chart_b.domain, 'bounds', None)
#         if not bd_a or not bd_b: return False
#         pts = []
#         for _ in range(samples):
#             pt = tuple(np.random.uniform(max(bd_a[d][0], bd_b[d][0]),
#                                         min(bd_a[d][1], bd_b[d][1]))
#                         for d in range(chart_a.dim))
#             pts.append(pt)
#         for pt in pts:
#             tgt = tm.to_target(pt)
#             src = tm.to_source(tgt)
#             if any(abs(x-y) > tol for x,y in zip(pt,src)):
#                 return False
#         return True
#
# # ---------------------- Visualization Utilities ----------------------
# class AtlasVisualizer:
#     """
#     Utilities for plotting chart domains and embedded manifolds.
#     """
#     @staticmethod
#     def plot_domains(
#         atlas: SymbolicManifold,
#         chart_names: Optional[List[str]] = None,
#         resolution: int = 100
#     ) -> None:
#         import matplotlib.pyplot as plt
#         names = chart_names or list(atlas.charts.keys())
#         plt.figure()
#         for name in names:
#             chart = atlas.get_chart(name)
#             dom = chart.domain
#             if isinstance(dom, BoxDomain):
#                 xs = np.linspace(dom.bounds[0][0], dom.bounds[0][1], resolution)
#                 ys = np.linspace(dom.bounds[1][0], dom.bounds[1][1], resolution) \
#                         if chart.dim > 1 else [0]
#                 X,Y = np.meshgrid(xs,ys)
#                 mask = np.array([[dom.contains((x,y)) for x in xs] for y in ys])
#                 plt.contourf(X,Y,mask, alpha=0.3, label=chart.name)
#             elif isinstance(dom, InequalityDomain):
#                 xs = np.linspace(-1,1,resolution)
#                 ys = np.linspace(-1,1,resolution)
#                 X,Y = np.meshgrid(xs,ys)
#                 pts = np.vstack([X.flatten(),Y.flatten()]).T
#                 mask = np.array([dom.contains(tuple(pt)) for pt in pts]).reshape(X.shape)
#                 plt.contourf(X,Y,mask, alpha=0.3, label=chart.name)
#         plt.title(f"Chart Domains: {atlas.name}")
#         plt.xlabel(str(atlas.default.coords[0]))
#         if atlas.default.dim>1: plt.ylabel(str(atlas.default.coords[1]))
#         plt.legend(); plt.show()
#
#     @staticmethod
#     def plot_embedding(
#         chart: Chart,
#         points: List[Tuple[float,...]],
#         vectors: Optional[List[Tuple[float,...]]] = None
#     ) -> None:
#         import matplotlib.pyplot as plt
#         from mpl_toolkits.mplot3d import Axes3D # noqa
#         if chart.embedding is None:
#             raise ValueError("Chart has no embedding for visualization.")
#         emb = chart.embedding
#         pts = np.array([emb.evaluate(pt) for pt in points])
#         fig = plt.figure(); ax = fig.add_subplot(111,
#             projection='3d' if pts.shape[1]==3 else None)
#         if pts.shape[1]==3:
#             ax.scatter(pts[:,0],pts[:,1],pts[:,2],s=20)
#         else:
#             plt.scatter(pts[:,0],pts[:,1],s=20)
#         if vectors:
#             jac = sp.Matrix(emb.map_exprs).jacobian(chart.coords)
#             jac_func = lambdify(chart.coords,jac,'numpy')
#             for pt,vec in zip(points,vectors):
#                 Jp = np.array(jac_func(*pt),dtype=float)
#                 v_emb = Jp.dot(np.array(vec,dtype=float))
#                 if pts.shape[1]==3:
#                     ax.quiver(*emb.evaluate(pt),*v_emb,length=0.2,color='r')
#                 else:
#                     plt.quiver(pts[:,0],pts[:,1],v_emb[0],v_emb[1])
#         plt.title(f"Embedding of Chart {chart.name}"); plt.show()
# from __future__ import annotations
# import sympy as sp
# from sympy import Expr, Symbol, sympify, lambdify
# from typing import Callable, Dict, List, Tuple, Optional, Union
# import numpy as np
#
# # -------------------------- Domain Definitions --------------------------
# class Domain:
#     """
#     Abstract base class for chart domains in R^n.
#     """
#     def contains(self, point: Tuple[float, ...]) -> bool:
#         """
#         Determine if a numeric point lies in this domain.
#         """
#         raise NotImplementedError("Domain.contains must be implemented by subclasses.")
#
# class PredicateDomain(Domain):
#     """
#     Domain defined by a Python predicate f: R^n -> bool.
#     """
#     def __init__(self, predicate: Callable[..., bool]):
#         self.predicate = predicate
#     def contains(self, point: Tuple[float, ...]) -> bool:
#         return bool(self.predicate(*point))
#
# class BoxDomain(Domain):
#     """
#     Axis-aligned box: each coordinate x_i in [lo_i, hi_i].
#     """
#     def __init__(self, bounds: List[Tuple[float, float]]):
#         # bounds: list of (min, max) pairs for each coordinate
#         self.bounds = bounds
#     def contains(self, point: Tuple[float, ...]) -> bool:
#         return all(lo <= x <= hi for x, (lo, hi) in zip(point, self.bounds))
#
# class InequalityDomain(Domain):
#     """
#     Domain defined by a Sympy relational expression, e.g. x**2 + y**2 < 1.
#     """
#     def __init__(self, expr: Union[Expr, str], coords: List[Symbol]):
#         self.expr = sympify(expr)
#         self.coords = coords
#         # lambdify to a numeric function
#         self._func = lambdify(coords, self.expr, "numpy")
#     def contains(self, point: Tuple[float, ...]) -> bool:
#         return bool(self._func(*point))
#
# class UnionDomain(Domain):
#     """
#     Union of multiple domains: point is in any one.
#     """
#     def __init__(self, domains: List[Domain]):
#         self.domains = domains
#     def contains(self, point: Tuple[float, ...]) -> bool:
#         return any(domain.contains(point) for domain in self.domains)
#
# # ------------------------ Embedding Definitions ------------------------
# class Embedding:
#     """
#         Base class for embedding chart coordinates into R^m.
#         By default behaves like ParametricEmbedding: store coords and map_exprs, compile with lambdify.
#         """
#
#     def __init__(self, coords: List[Symbol], map_exprs: List[Expr]):
#         # store symbolic definitions
#         self.coords = coords
#         # sympify all expressions
#         self.map_exprs = [sympify(expr) for expr in map_exprs]
#         # compile to numeric
#         self._func = lambdify(self.coords, self.map_exprs, "numpy")
#
#     def evaluate(self, point: Tuple[float, ...]) -> Tuple[float, ...]:
#         """
#         Map a chart coordinate tuple to an embedded point in R^m.
#         """
#         arr = np.array(self._func(*point), dtype=float)
#         return tuple(arr.flatten())
#
#         # ------------------------ Embedding Definitions ------------------------(self, point: Tuple[float, ...]) -> Tuple[float, ...]:
#         """
#         Map a chart coordinate tuple to an embedded point in R^m.
#         """
#         raise NotImplementedError("Embedding.evaluate must be implemented by subclasses.")
#
#
# class ParametricEmbedding(Embedding):
#     """
#     Embedding defined by Sympy expressions map_exprs in terms of coords.
#     """
#     def __init__(self, coords: List[Symbol], map_exprs: List[Expr]):
#         self.coords = coords
#         self.map_exprs = [sympify(expr) for expr in map_exprs]
#         # Precompile with lambdify for performance
#         self._func = lambdify(coords, self.map_exprs, "numpy")
#     def evaluate(self, point: Tuple[float, ...]) -> Tuple[float, ...]:
#         arr = np.array(self._func(*point), dtype=float)
#         return tuple(arr.flatten())
#
# # ---------------------- Coordinate Transition ----------------------
# class TransitionMap:
#     """
#     Represents a transition map between two chart coordinate systems.
#     """
#     def __init__(
#         self,
#         source_coords: List[Symbol],
#         target_coords: List[Symbol],
#         forward_map: Dict[Symbol, Expr],
#         inverse_map: Optional[Dict[Symbol, Expr]] = None
#     ):
#         # Validate keys
#         if set(forward_map.keys()) != set(source_coords):
#             raise ValueError("Forward map must define images of all source coords.")
#         self.source_coords = source_coords
#         self.target_coords = target_coords
#         # Sympify expressions
#         self.forward_map = {sym: sympify(expr) for sym, expr in forward_map.items()}
#         self.inverse_map = {sym: sympify(expr) for sym, expr in (inverse_map or {}).items()}
#         # Lambdify numeric functions
#         self._fwd_func = lambdify(source_coords, list(self.forward_map.values()), "numpy")
#         self._inv_func = (
#             lambdify(target_coords, list(self.inverse_map.values()), "numpy")
#             if inverse_map
#             else None
#         )
#
#     def to_target(self, point: Tuple[float, ...]) -> Tuple[float, ...]:
#         """
#         Map a numeric point in source chart to coordinates in target chart.
#         """
#         arr = np.array(self._fwd_func(*point), dtype=float)
#         return tuple(arr.flatten())
#
#     def to_source(self, point: Tuple[float, ...]) -> Tuple[float, ...]:
#         """
#         Map a numeric point in target chart back to source chart.
#         Requires inverse_map to be provided.
#         """
#         if not self._inv_func:
#             raise ValueError("Inverse transition is not defined.")
#         arr = np.array(self._inv_func(*point), dtype=float)
#         return tuple(arr.flatten())
#
# # -------------------------- Chart Definition --------------------------
# class Chart:
#     """
#     Coordinate chart on a manifold.
#
#     Attributes:
#       name: unique identifier
#       coords: symbols for local R^n coordinates
#       domain: Domain instance specifying valid region
#       embedding: optional Embedding into R^m
#       transitions: mapping other_chart_name -> TransitionMap
#     """
#     def __init__(
#         self,
#         name: str,
#         coords: List[Symbol],
#         domain: Domain = PredicateDomain(lambda *args: True),
#         embedding: Optional[Embedding] = None
#     ):
#         self.name = name
#         self.coords = list(coords)
#         self.dim = len(coords)
#         self.domain = domain
#         self.embedding = embedding
#         # Transition maps to other charts in the same manifold
#         self.transitions: Dict[str, TransitionMap] = {}
#
#     def add_transition(
#         self,
#         other: Chart,
#         forward: Dict[Symbol, Expr],
#         inverse: Optional[Dict[Symbol, Expr]] = None
#     ) -> None:
#         """
#         Define a transition between this chart and another.
#         """
#         tm = TransitionMap(self.coords, other.coords, forward, inverse)
#         self.transitions[other.name] = tm
#
#     def contains(self, point: Tuple[float, ...]) -> bool:
#         """
#         Check if a numeric point lies within the chart's domain.
#         """
#         return self.domain.contains(point)
#
#     def to_chart(self, point: Tuple[float, ...], other: Chart) -> Tuple[float, ...]:
#         """
#         Transform a numeric point from this chart to another chart.
#         """
#         if other.name not in self.transitions:
#             raise KeyError(f"No transition from {self.name} to {other.name}.")
#         return self.transitions[other.name].to_target(point)
#
#     def sample_grid(
#         self,
#         bounds: Optional[List[Tuple[float, float]]] = None,
#         num: int = 20
#     ) -> List[Tuple[float, ...]]:
#         """
#         Generate a grid of points within the chart domain or given bounds.
#         """
#         rng = bounds or getattr(self.domain, 'bounds', None)
#         if not rng:
#             raise ValueError("Bounds must be provided or domain must be BoxDomain.")
#         # Create grid axes
#         axes = [np.linspace(lo, hi, num) for lo, hi in rng]
#         mesh = np.meshgrid(*axes)
#         # Flatten mesh to list of tuples
#         pts = [tuple(mesh[d].flat[i] for d in range(self.dim)) for i in range(mesh[0].size)]
#         # Filter by domain
#         return [pt for pt in pts if self.contains(pt)]
#
# # End of Part 1: core definitions up to Chart.sample_grid
#
# # ----------------------- Manifold Definition -----------------------
# class SymbolicManifold:
#     """
#     Abstract n-dimensional manifold represented by an atlas of charts.
#     """
#     def __init__(self, name: str):
#         self.name = name
#         self.charts: Dict[str, Chart] = {}
#         self.default: Optional[Chart] = None
#
#     def add_chart(self, chart: Chart, default: bool = False) -> None:
#         """
#         Add a new Chart to the manifold.
#         If default=True or no default set, mark this as default chart.
#         """
#         if chart.name in self.charts:
#             raise KeyError(f"Chart '{chart.name}' already exists in manifold '{self.name}'.")
#         self.charts[chart.name] = chart
#         if default or self.default is None:
#             self.default = chart
#
#     def get_chart(self, name: str) -> Chart:
#         """
#         Get a chart by name.
#         """
#         if name not in self.charts:
#             raise KeyError(f"Chart '{name}' not found in manifold '{self.name}'.")
#         return self.charts[name]
#
#     def find_chart(self, point: Tuple[float, ...]) -> Chart:
#         """
#         Find a chart whose domain contains the numeric point.
#         Returns default chart if multiple match or fallback.
#         """
#         for chart in self.charts.values():
#             try:
#                 if chart.contains(point):
#                     return chart
#             except Exception:
#                 continue
#         if self.default and self.default.contains(point):
#             return self.default
#         raise ValueError(f"No chart contains point {point} in manifold '{self.name}'.")
#
#     def transition(
#         self,
#         point: Tuple[float, ...],
#         source: Optional[str] = None,
#         target: Optional[str] = None
#     ) -> Tuple[float, ...]:
#         """
#         Transition a numeric point from one chart to another.
#         If source omitted, auto-detect via find_chart.
#         If target omitted, use default chart.
#         """
#         src_chart = self.charts[source] if source else self.find_chart(point)
#         tgt_chart = self.charts[target] if target else self.default
#         if not src_chart or not tgt_chart:
#             raise ValueError("Source or target chart undefined.")
#         return src_chart.to_chart(point, tgt_chart)
#
# # -------------------- Future Extension Hooks --------------------
# class DomainParser:
#     """
#     Parse domain specification strings or objects into Domain instances.
#     """
#     @staticmethod
#     def from_string(spec: str, coords: List[Symbol]) -> Domain:
#         """
#         Convert string like 'x**2+y**2<1' to an InequalityDomain.
#         TODO: handle logical AND/OR, multiple relations.
#         """
#         expr = sympify(spec)
#         if isinstance(expr, sp.Relational):
#             return InequalityDomain(expr, coords)
#         if isinstance(expr, sp.And) or isinstance(expr, sp.Or):
#             # Decompose into subdomains
#             sub = [InequalityDomain(e, coords) for e in expr.args]
#             return UnionDomain(sub)
#         raise ValueError(f"Cannot parse domain spec: {spec}")
#
# class ChartValidator:
#     """
#     Validate chart and transition consistency.
#     """
#     @staticmethod
#     def validate_transition(
#         chart_a: Chart,
#         chart_b: Chart,
#         tol: float = 1e-6,
#         samples: int = 5
#     ) -> bool:
#         """
#         Numerically test forward+inverse mapping consistency.
#         Samples points in overlapping domain.
#         """
#         tm = chart_a.transitions.get(chart_b.name)
#         if not tm or not tm._inv_func:
#             return False
#         # sample grid in intersection of box domains if available
#         bd_a = getattr(chart_a.domain, 'bounds', None)
#         bd_b = getattr(chart_b.domain, 'bounds', None)
#         if not bd_a or not bd_b:
#             return False  # cannot sample
#         samples_pts = []
#         for i in range(samples):
#             pt = tuple(
#                 np.random.uniform(max(bd_a[d][0], bd_b[d][0]), min(bd_a[d][1], bd_b[d][1]))
#                 for d in range(chart_a.dim)
#             )
#             samples_pts.append(pt)
#         for pt in samples_pts:
#             try:
#                 tgt = tm.to_target(pt)
#                 src = tm.to_source(tgt)
#             except Exception:
#                 return False
#             if any(abs(x-y) > tol for x,y in zip(pt,src)):
#                 return False
#         return True
#
# # End of Part 2: SymbolicManifold, DomainParser, ChartValidator
#
# # ---------------------- Visualization Utilities ----------------------
# class AtlasVisualizer:
#     """
#     Utilities for plotting chart domains and embedded manifolds.
#     """
#     @staticmethod
#     def plot_domains(
#         atlas: SymbolicManifold,
#         chart_names: Optional[List[str]] = None,
#         resolution: int = 100
#     ) -> None:
#         """
#         Plot abstract chart domains in parameter space using Matplotlib.
#         Supports only BoxDomain or InequalityDomain for now.
#         """
#         import matplotlib.pyplot as plt
#         names = chart_names or list(atlas.charts.keys())
#         plt.figure()
#         for name in names:
#             chart = atlas.get_chart(name)
#             domain = chart.domain
#             if isinstance(domain, BoxDomain):
#                 xs = np.linspace(domain.bounds[0][0], domain.bounds[0][1], resolution)
#                 ys = np.linspace(domain.bounds[1][0], domain.bounds[1][1], resolution) if chart.dim > 1 else [0]
#                 X, Y = np.meshgrid(xs, ys)
#                 mask = np.array([[domain.contains((x, y)) for x in xs] for y in ys])
#                 plt.contourf(X, Y, mask, alpha=0.3, label=name)
#             elif isinstance(domain, InequalityDomain):
#                 xs = np.linspace(-1, 1, resolution)
#                 ys = np.linspace(-1, 1, resolution)
#                 X, Y = np.meshgrid(xs, ys)
#                 pts = np.vstack([X.flatten(), Y.flatten()]).T
#                 mask = np.array([domain.contains(tuple(pt)) for pt in pts]).reshape(X.shape)
#                 plt.contourf(X, Y, mask, alpha=0.3, label=name)
#             else:
#                 continue
#         plt.title(f"Chart Domains: {atlas.name}")
#         plt.xlabel(str(atlas.default.coords[0]))
#         if atlas.default.dim > 1:
#             plt.ylabel(str(atlas.default.coords[1]))
#         plt.legend()
#         plt.show()
#
#     @staticmethod
#     def plot_embedding(
#         chart: Chart,
#         points: List[Tuple[float, ...]],
#         vectors: Optional[List[Tuple[float, ...]]] = None
#     ) -> None:
#         """
#         Plot a set of points (and optional tangent vectors) on the embedded manifold.
#         """
#         import matplotlib.pyplot as plt
#         from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
#         if chart.embedding is None:
#             raise ValueError("Chart has no embedding for visualization.")
#         emb = chart.embedding
#         # Evaluate points
#         pts_emb = np.array([emb.evaluate(pt) for pt in points])
#         fig = plt.figure()
#         ax = fig.add_subplot(111, projection='3d' if pts_emb.shape[1] == 3 else None)
#         if pts_emb.shape[1] == 3:
#             ax.scatter(pts_emb[:, 0], pts_emb[:, 1], pts_emb[:, 2], s=20)
#         else:
#             plt.scatter(pts_emb[:, 0], pts_emb[:, 1], s=20)
#         # Plot vectors if provided
#         if vectors:
#             jac = sp.Matrix(emb.map_exprs).jacobian(chart.coords)
#             jac_func = lambdify(chart.coords, jac, 'numpy')
#             for pt, vec in zip(points, vectors):
#                 Jp = np.array(jac_func(*pt), dtype=float)
#                 vec_emb = Jp.dot(np.array(vec, dtype=float))
#                 if pts_emb.shape[1] == 3:
#                     ax.quiver(
#                         *emb.evaluate(pt), *vec_emb, length=0.2, color='r'
#                     )
#                 else:
#                     plt.quiver(
#                         pts_emb[:, 0], pts_emb[:, 1], vec_emb[0], vec_emb[1]
#                     )
#         plt.title(f"Embedding of Chart {chart.name}")
#         plt.show()



