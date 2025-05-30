import sympy as sp
from sympy import Expr, Function, lambdify, Symbol
from typing import List, Tuple, Union, Callable

class Connection:
    """
    Abstract base class for affine connections.
    """
    def __init__(self, chart):
        self.chart = chart
        self.coords = chart.coords
        self.dim = len(self.coords)

    def parallel_transport_equations(
        self,
        curve_funcs: List[Union[Function, sp.Lambda, Callable]],
        vec_funcs: List[Function],
        t: Symbol = None
    ) -> List[Expr]:
        """
        Compute the symbolic ODEs for parallel transport along a parameterized curve.
        curve_funcs may be sympy Function(u(t),v(t)), sympy Lambda, or Python callables.
        vec_funcs are sympy functions V_i(t).
        Returns list of Exprs eq_i = 0.
        """
        # ensure a symbol for parameter
        if t is None:
            t = sp.Symbol('t')
        Gamma = self.Gamma  # Christoffel symbols
        eqs: List[Expr] = []
        for i in range(self.dim):
            expr_i = sp.Integer(0)
            for j in range(self.dim):
                for k in range(self.dim):
                    # fetch connection coefficient
                    gamma_ijk = Gamma[i][j][k]

                    # derivative of curve function j
                    cf_j = curve_funcs[j]
                    if isinstance(cf_j, sp.Lambda):
                        # sympy Lambda: Lambda(t_prm, expr)
                        dcf = sp.diff(cf_j.expr, cf_j.variables[0]).subs(cf_j.variables[0], t)
                    elif isinstance(cf_j, sp.Function):
                        dcf = sp.diff(cf_j, t)
                    elif callable(cf_j):
                        # wrap python callable as sympy lambda
                        sym_lambda = sp.Lambda(t, cf_j(t))
                        dcf = sp.diff(sym_lambda.expr, t)
                    else:
                        # assume sympy expression in t
                        dcf = sp.diff(cf_j, t)

                    # vector function
                    vf_k = vec_funcs[k]
                    expr_i -= gamma_ijk * dcf * vf_k
            eqs.append(expr_i)
        return eqs

class LeviCivitaConnection(Connection):
    """
    Computes Christoffel symbols from a RiemannianMetric.
    """
    def __init__(self, metric, chart):
        super().__init__(chart)
        self.metric = metric
        self.Gamma = self._compute_christoffel()

    def _compute_christoffel(self) -> List[List[List[Expr]]]:
        g = self.metric.g
        inv_g = g.inv()
        coords = self.coords
        dim = self.dim
        Gamma = [[[sp.simplify(
            sum(inv_g[i, m] * (sp.diff(g[m, j], coords[k]) +
                             sp.diff(g[m, k], coords[j]) -
                             sp.diff(g[j, k], coords[m]))
                for m in range(dim)) / 2
        ) for k in range(dim)] for j in range(dim)] for i in range(dim)]
        return Gamma

    def riemann_tensor(self) -> List[List[List[List[Expr]]]]:
        """
        Compute the Riemann curvature tensor R^i_{jkl}.
        """
        coords = self.coords
        dim = self.dim
        R = [[[[sp.Integer(0) for l in range(dim)] 
               for k in range(dim)] 
               for j in range(dim)] 
               for i in range(dim)]
        
        for i in range(dim):
            for j in range(dim):
                for k in range(dim):
                    for l in range(dim):
                        # R^i_{jkl} = ∂_k Γ^i_{jl} - ∂_l Γ^i_{jk} + Γ^i_{mk}Γ^m_{jl} - Γ^i_{ml}Γ^m_{jk}
                        term1 = sp.diff(self.Gamma[i][j][l], coords[k])
                        term2 = -sp.diff(self.Gamma[i][j][k], coords[l])
                        term3 = sum(self.Gamma[i][m][k] * self.Gamma[m][j][l] 
                                  for m in range(dim))
                        term4 = -sum(self.Gamma[i][m][l] * self.Gamma[m][j][k] 
                                   for m in range(dim))
                        R[i][j][k][l] = sp.simplify(term1 + term2 + term3 + term4)
        return R

    def ricci_tensor(self) -> sp.Matrix:
        """
        Compute the Ricci tensor Ric_{jk} = R^i_{jik}.
        """
        dim = self.dim
        R = self.riemann_tensor()
        Ric = sp.zeros(dim, dim)
        
        for j in range(dim):
            for k in range(dim):
                Ric[j,k] = sp.simplify(sum(R[i][j][i][k] for i in range(dim)))
        return Ric

    def scalar_curvature(self) -> Expr:
        """
        Compute the scalar curvature R = g^{ij}Ric_{ij}.
        """
        Ric = self.ricci_tensor()
        g_inv = self.metric.g.inv()
        dim = self.dim
        R = sp.simplify(sum(g_inv[i,j] * Ric[i,j] 
                           for i in range(dim) 
                           for j in range(dim)))
        return R

    def geodesic_equations(
        self,
        curve_funcs: List[Function] = None,
        t: Symbol = None
    ) -> Tuple[List[Expr], List[Function]]:
        """
        Returns the geodesic ODEs and symbolic function list [u(t), v(t), ...].
        """
        if t is None:
            t = sp.Symbol('t')
        # create symbolic functions
        funcs = [sp.Function(f'X{i}')(t) for i in range(self.dim)]
        eqs: List[Expr] = []
        for i in range(self.dim):
            # second derivative
            d2 = sp.diff(funcs[i], t, 2)
            # subtract connection term
            term = sum(self.Gamma[i][j][k] * sp.diff(funcs[j], t) * sp.diff(funcs[k], t)
                       for j in range(self.dim) for k in range(self.dim))
            eqs.append(sp.simplify(d2 + term))
        return eqs, funcs

# End of connections.py

# import sympy as sp
# from sympy import Expr, Function, symbols, latex
# from typing import List, Tuple, Any, Callable, TYPE_CHECKING
#
# from charts import Chart
# from Riemannian_metric import RiemannianMetric
# if TYPE_CHECKING:
#     from vector_fields import VectorField
#
# # ======================= connections.py =======================
# # Modular, extensible connection implementations
#
# # ---------------- Part 1: Abstract Base ----------------
# class Connection:
#     """
#     Abstract base for affine connections on a Chart.
#
#     Must implement:
#       - covariant_derivative
#       - parallel_transport_equations
#       - geodesic_equations
#     """
#     def __init__(self, chart: Chart):
#         self.chart = chart
#         self.coords = chart.coords
#
#     def covariant_derivative(
#         self,
#         vec_components: List[Expr],
#         direction_index: int
#     ) -> List[Expr]:
#         """
#         ∇_{∂_{direction}} V components.
#         """
#         raise NotImplementedError
#
#     def parallel_transport_equations(
#         self,
#         curve_funcs: List[Function],
#         vec_funcs: List[Function]
#     ) -> List[sp.Eq]:
#         """
#         ODEs for parallel transport: dV^i/dt + connection^i_{jk} x'^j V^k = 0.
#         """
#         raise NotImplementedError
#
#     def geodesic_equations(self) -> Tuple[List[sp.Eq], List[Function]]:
#         """
#         Equations x''^i + connection^i_{jk} x'^j x'^k = 0.
#         """
#         raise NotImplementedError
#
# # ---------------- Part 2: Levi-Civita Connection ----------------
# class LeviCivitaConnection(Connection):
#     """
#     Metric-compatible, torsion-free connection derived from a RiemannianMetric.
#     """
#     def __init__(self, metric: RiemannianMetric, chart: Chart):
#         super().__init__(chart)
#         self.metric = metric
#         self._Gamma: List[List[List[Expr]]] = None
#
#     @property
#     def Gamma(self) -> List[List[List[Expr]]]:
#         """ Christoffel symbols Γ^i_{jk}. """
#         if self._Gamma is None:
#             self._Gamma = self.metric.christoffel_symbols()
#         return self._Gamma
#
#     def covariant_derivative(
#         self,
#         vec_components: List[Expr],
#         direction_index: int
#     ) -> List[Expr]:
#         n = len(self.coords)
#         result = [0]*n
#         for i in range(n):
#             term = sp.diff(vec_components[i], self.coords[direction_index])
#             for j in range(n):
#                 term += self.Gamma[i][direction_index][j] * vec_components[j]
#             result[i] = sp.simplify(term)
#         return result
#
#     def parallel_transport_equations(
#         self,
#         curve_funcs: List[Function],
#         vec_funcs: List[Function]
#     ) -> List[sp.Eq]:
#         t = symbols('t')
#         eqs = []
#         for i in range(len(vec_funcs)):
#             expr = sp.diff(vec_funcs[i], t)
#             for j in range(len(curve_funcs)):
#                 for k in range(len(vec_funcs)):
#                     expr += self.Gamma[i][j][k] * sp.diff(curve_funcs[j], t) * vec_funcs[k]
#             eqs.append(sp.Eq(expr, 0))
#         return eqs
#
#     def geodesic_equations(self) -> Tuple[List[sp.Eq], List[Function]]:
#         t = symbols('t')
#         funcs = [Function(str(c))(t) for c in self.coords]
#         eqs = []
#         for i in range(len(funcs)):
#             expr = sp.diff(funcs[i], (t,2))
#             for j in range(len(funcs)):
#                 for k in range(len(funcs)):
#                     expr += self.Gamma[i][j][k] * sp.diff(funcs[j], t) * sp.diff(funcs[k], t)
#             eqs.append(sp.Eq(expr, 0))
#         return eqs, funcs
#
#     def to_latex(self) -> str:
#         lines = []
#         n = len(self.coords)
#         for i in range(n):
#             for j in range(n):
#                 for k in range(n):
#                     if self.Gamma[i][j][k] != 0:
#                         lines.append(f"\\Gamma^{{{i}}}_{{{j}{k}}} = {latex(self.Gamma[i][j][k])}")
#         return "\\\\n".join(lines)
#
# # ---------------- Part 3: Metric Connection Alias ----------------
# MetricConnection = LeviCivitaConnection
#
# # ---------------- Part 4: Custom Connection Example ----------------
# class CustomConnection(Connection):
#     """
#     Example: flat connection (all Γ^i_{jk} = 0).
#     """
#     def __init__(self, chart: Chart):
#         super().__init__(chart)
#
#     def covariant_derivative(self, vec_components: List[Expr], direction_index: int) -> List[Expr]:
#         return [sp.diff(vec_components[i], self.coords[direction_index]) for i in range(len(self.coords))]
#
#     def parallel_transport_equations(self, curve_funcs: List[Function], vec_funcs: List[Function]) -> List[sp.Eq]:
#         t = symbols('t')
#         return [sp.Eq(sp.diff(Vi, t), 0) for Vi in vec_funcs]
#
#     def geodesic_equations(self) -> Tuple[List[sp.Eq], List[Function]]:
#         t = symbols('t')
#         funcs = [Function(str(c))(t) for c in self.coords]
#         return [sp.Eq(sp.diff(fi, (t,2)), 0) for fi in funcs], funcs
#
# # ---------------- Part 5: Utility and Validation ----------------
# def validate_connection(conn: Connection) -> bool:
#     """
#     New utilities to test connection properties:
#     - metric compatibility: ∇ g = 0
#     - torsion-free: Γ^i_{jk} = Γ^i_{kj}
#     """
#     # Torsion-free check
#     coords = conn.coords
#     Gamma = getattr(conn, 'Gamma', None)
#     if Gamma:
#         n = len(coords)
#         for i in range(n):
#             for j in range(n):
#                 for k in range(n):
#                     if Gamma[i][j][k] != Gamma[i][k][j]:
#                         return False
#     # Metric compatibility for Levi-Civita
#     if isinstance(conn, LeviCivitaConnection):
#         metric = conn.metric.g
#         inv = conn.metric.invg
#         for l in range(len(coords)):
#             for i in range(len(coords)):
#                 for j in range(len(coords)):
#                     deriv = sp.diff(metric[i,j], coords[l])
#                     for k in range(len(coords)):
#                         deriv -= metric[k,j]*conn.Gamma[k][l][i] + metric[i,k]*conn.Gamma[k][l][j]
#                     if sp.simplify(deriv) != 0:
#                         return False
#     return True

# End of connections.py
