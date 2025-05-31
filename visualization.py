import sympy as sp
import numpy as np
import matplotlib
import os
# Force matplotlib to not use any Xwindows backend
if os.environ.get('DISPLAY', '') == '':
    print('No display found. Using non-interactive Agg backend')
    matplotlib.use('Agg')
else:
    try:
        # Try Qt5Agg first as it's generally more robust
        matplotlib.use('Qt5Agg')
    except:
        try:
            matplotlib.use('TkAgg')
        except:
            print('Warning: Could not initialize interactive backend. Using non-interactive Agg backend')
            matplotlib.use('Agg')

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, art3d
from typing import List, Tuple, Callable, Optional, Union

from sympy import lambdify
from charts import Chart
from vector_fields import VectorField
from connections import LeviCivitaConnection

# New: Optional advanced visualization backends
try:
    import open3d as o3d
    _OPEN3D_AVAILABLE = True
except ImportError:
    _OPEN3D_AVAILABLE = False
    print("Warning: open3d not found. 3D visualization will be limited.")

try:
    import ipywidgets as widgets
    from IPython.display import display, clear_output
    _IPYWIDGETS_AVAILABLE = True
except ImportError:
    _IPYWIDGETS_AVAILABLE = False

# ---------------- Basic Matplotlib Plotting ----------------
class ChartPlotter:
    @staticmethod
    def plot_chart_domain(chart: Chart, resolution: int = 50) -> None:
        dom = chart.domain
        if hasattr(dom, 'bounds'):
            xs = np.linspace(dom.bounds[0][0], dom.bounds[0][1], resolution)
            ys = np.linspace(dom.bounds[1][0], dom.bounds[1][1], resolution)
            X, Y = np.meshgrid(xs, ys)
            mask = np.array([[dom.contains((x, y)) for x in xs] for y in ys])
            plt.contourf(X, Y, mask, alpha=0.3)
            plt.title(f"Domain of chart {chart.name}")
            plt.xlabel(str(chart.coords[0]))
            plt.ylabel(str(chart.coords[1]))
            plt.show()

    @staticmethod
    def plot_chart_grid(chart: Chart, grid_lines: int = 10) -> None:
        rng = getattr(chart.domain, 'bounds', None)
        if not rng:
            raise ValueError("BoxDomain required for grid plotting.")
        xs = np.linspace(rng[0][0], rng[0][1], grid_lines)
        ys = np.linspace(rng[1][0], rng[1][1], grid_lines)
        for x in xs:
            plt.plot([x]*len(ys), ys, 'k:', alpha=0.5)
        for y in ys:
            plt.plot(xs, [y]*len(xs), 'k:', alpha=0.5)
        plt.title(f"Grid on chart {chart.name}")
        plt.xlabel(str(chart.coords[0]))
        plt.ylabel(str(chart.coords[1]))
        plt.show()

# ---------------- Open3D Extensions ----------------
class Open3DPlotter:
    """
    Advanced 3D plotting using Open3D.
    """
    @staticmethod
    def plot_embedding_surface(chart: Chart, resolution: int = 50) -> None:
        if not _OPEN3D_AVAILABLE:
            raise ImportError("Open3D is not available. Please install open3d.")

        dom = chart.domain
        if not hasattr(dom, 'bounds'):
            raise ValueError("Chart domain must have bounds for sampling.")

        coords = chart.coords
        emb_func = lambdify(coords, chart.embedding.map_exprs, 'numpy')

        u = np.linspace(dom.bounds[0][0], dom.bounds[0][1], resolution)
        v = np.linspace(dom.bounds[1][0], dom.bounds[1][1], resolution)
        uu, vv = np.meshgrid(u, v)
        pts = np.array([emb_func(ux, vx) for ux, vx in zip(uu.flatten(), vv.flatten())])

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        # Estimate normals for the point cloud
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        # Orient normals consistently
        pcd.orient_normals_consistent_tangent_plane(k=20)
        
        mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8)
        mesh.compute_vertex_normals()
        o3d.visualization.draw_geometries([mesh])

    @staticmethod
    def plot_vector_field(chart: Chart, vf: VectorField, resolution: int = 20, scale: float = 0.1) -> None:
        if not _OPEN3D_AVAILABLE:
            raise ImportError("Open3D is not available. Please install open3d.")

        coords = chart.coords
        bounds = chart.domain.bounds
        emb_func = lambdify(coords, chart.embedding.map_exprs, 'numpy')
        
        # Create Jacobian for transforming vectors
        J_matrix = sp.Matrix(chart.embedding.map_exprs).jacobian(coords)
        J_func = lambdify(coords, J_matrix, 'numpy')

        u = np.linspace(bounds[0][0], bounds[0][1], resolution)
        v = np.linspace(bounds[1][0], bounds[1][1], resolution)
        uu, vv = np.meshgrid(u, v)
        pts, lines, colors = [], [], []

        for ux, vx in zip(uu.flatten(), vv.flatten()):
            if not chart.domain.contains((ux, vx)):
                continue
            p = np.array(emb_func(ux, vx))
            vvec = np.array(vf.evaluate((ux, vx)))
            # Transform vector using Jacobian
            vvec_embedded = J_func(ux, vx).dot(vvec)
            pts.append(p)
            pts.append(p + scale * vvec_embedded)
            lines.append([len(pts) - 2, len(pts) - 1])
            colors.append([1, 0, 0])

        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(np.array(pts)),
            lines=o3d.utility.Vector2iVector(np.array(lines))
        )
        line_set.colors = o3d.utility.Vector3dVector(colors)
        o3d.visualization.draw_geometries([line_set])

    @staticmethod
    def export_scene_as_mesh(chart: Chart, filename: str, resolution: int = 50) -> None:
        if not _OPEN3D_AVAILABLE:
            raise ImportError("Open3D is not available. Please install open3d.")

        dom = chart.domain
        coords = chart.coords
        emb_func = lambdify(coords, chart.embedding.map_exprs, 'numpy')

        u = np.linspace(dom.bounds[0][0], dom.bounds[0][1], resolution)
        v = np.linspace(dom.bounds[1][0], dom.bounds[1][1], resolution)
        uu, vv = np.meshgrid(u, v)
        pts = np.array([emb_func(ux, vx) for ux, vx in zip(uu.flatten(), vv.flatten())])

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8)
        mesh.compute_vertex_normals()
        o3d.io.write_triangle_mesh(filename, mesh)

    @staticmethod
    def visualize_geodesic(chart: Chart, connection: LeviCivitaConnection, init_point: Tuple[float, float], init_velocity: Tuple[float, float], steps: int = 100, dt: float = 0.05) -> None:
        if not _OPEN3D_AVAILABLE:
            raise ImportError("Open3D is not available. Please install open3d.")

        coords = chart.coords
        emb_func = lambdify(coords, chart.embedding.map_exprs, 'numpy')
        pos = np.array(init_point, dtype=float)
        vel = np.array(init_velocity, dtype=float)
        traj = [emb_func(*pos)]

        Gamma = connection.Gamma

        for _ in range(steps):
            acc = np.zeros_like(pos)
            for i in range(len(pos)):
                for j in range(len(pos)):
                    for k in range(len(pos)):
                        acc[i] -= Gamma[i][j][k].evalf(subs={coords[0]: pos[0], coords[1]: pos[1]}) * vel[j] * vel[k]
            acc = np.array([float(a) for a in acc])
            pos += vel * dt
            vel += acc * dt
            traj.append(emb_func(*pos))

        traj = np.array(traj)
        line = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(traj),
            lines=o3d.utility.Vector2iVector([[i, i + 1] for i in range(len(traj) - 1)])
        )
        line.paint_uniform_color([0, 1, 0])
        o3d.visualization.draw_geometries([line])

    @staticmethod
    def visualize_flow(chart: Chart, vf: VectorField, start_points: List[Tuple[float, float]], steps: int = 100, dt: float = 0.05) -> None:
        if not _OPEN3D_AVAILABLE:
            raise ImportError("Open3D is not available. Please install open3d.")

        emb_func = lambdify(chart.coords, chart.embedding.map_exprs, 'numpy')
        trajectories = []

        for pt in start_points:
            pos = np.array(pt, dtype=float)
            traj = [emb_func(*pos)]
            for _ in range(steps):
                vel = np.array(vf.evaluate(tuple(pos)))
                pos += dt * vel
                traj.append(emb_func(*pos))
            trajectories.append(np.array(traj))

        lines = []
        for traj in trajectories:
            lines.append(o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(traj),
                lines=o3d.utility.Vector2iVector([[i, i + 1] for i in range(len(traj) - 1)])
            ))
            lines[-1].paint_uniform_color([0.8, 0.3, 0.3])

        o3d.visualization.draw_geometries(lines)

# ---------------- Scalar Field Visualizer ----------------
class ScalarFieldVisualizer:
    @staticmethod
    def plot_scalar_field(chart: Chart, scalar_expr: sp.Expr, resolution: int = 50, cmap: str = 'viridis', ax=None) -> None:
        coords = chart.coords
        scalar_func = lambdify(coords, scalar_expr, 'numpy')
        u = np.linspace(chart.domain.bounds[0][0], chart.domain.bounds[0][1], resolution)
        v = np.linspace(chart.domain.bounds[1][0], chart.domain.bounds[1][1], resolution)
        U, V = np.meshgrid(u, v)
        Z = scalar_func(U, V)

        if ax is None:
            fig = plt.figure(figsize=(12, 12))
            ax = fig.add_subplot(111, projection='3d')
            
        # Remove all background elements
        ax.set_axis_off()
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.grid(False)
            
        emb_func = lambdify(coords, chart.embedding.map_exprs, 'numpy')
        xyz = np.array([emb_func(ux, vx) for ux, vx in zip(U.flatten(), V.flatten())])
        X = xyz[:, 0].reshape(U.shape)
        Y = xyz[:, 1].reshape(U.shape)
        Z3 = xyz[:, 2].reshape(U.shape)
        
        # Handle color normalization carefully
        Z_min, Z_max = np.nanmin(Z), np.nanmax(Z)
        if Z_min == Z_max:
            Z_norm = np.zeros_like(Z)
        else:
            Z_norm = (Z - Z_min) / (Z_max - Z_min)
            
        # Plot the surface using the scalar field values for coloring
        surf = ax.plot_surface(X, Y, Z3, 
                             cmap=plt.cm.get_cmap(cmap),
                             vmin=Z_min, vmax=Z_max,
                             facecolors=None,
                             alpha=0.8)
        
        # Add a color bar
        plt.colorbar(surf, ax=ax)
        
        if ax is None:
            plt.show()

# ---------------- Interactive Tools ----------------
class InteractiveTools:
    @staticmethod
    def interactive_flow_slider(chart: Chart, vf: VectorField):
        if not _IPYWIDGETS_AVAILABLE:
            raise ImportError("ipywidgets is not installed. Please install ipywidgets to use this feature.")

        def callback(u, v, dt, steps):
            clear_output(wait=True)
            display(ui)
            Open3DPlotter.visualize_flow(chart, vf, start_points=[(u, v)], dt=dt, steps=int(steps))

        ui = widgets.interactive(
            callback,
            u=widgets.FloatSlider(min=chart.domain.bounds[0][0], max=chart.domain.bounds[0][1], step=0.1, value=0),
            v=widgets.FloatSlider(min=chart.domain.bounds[1][0], max=chart.domain.bounds[1][1], step=0.1, value=0),
            dt=widgets.FloatSlider(min=0.01, max=0.2, step=0.01, value=0.05),
            steps=widgets.IntSlider(min=10, max=300, step=10, value=100)
        )
        display(ui)

    @staticmethod
    def interactive_scalar_field(chart: Chart, field_exprs: List[sp.Expr], names: List[str] = None):
        if not _IPYWIDGETS_AVAILABLE:
            raise ImportError("ipywidgets is not installed. Please install ipywidgets to use this feature.")

        def callback(idx):
            clear_output(wait=True)
            display(ui)
            expr = field_exprs[idx]
            ScalarFieldVisualizer.plot_scalar_field(chart, expr)

        if names is None:
            names = [f"Field {i}" for i in range(len(field_exprs))]
        ui = widgets.interactive(
            callback,
            idx=widgets.Dropdown(options=list(enumerate(names)), description="Field:")
        )
        display(ui)

class ManifoldPlotter:
    """
    Wrapper for accessing all visualization utilities for manifolds.
    """
    def __init__(self, chart: Chart):
        self.chart = chart
        self._backend = 'open3d' if _OPEN3D_AVAILABLE else 'matplotlib'
        self._interactive_backend_available = matplotlib.get_backend() not in ['Agg']

    def set_backend(self, backend: str):
        """Set visualization backend ('open3d' or 'matplotlib')"""
        if backend not in ['open3d', 'matplotlib']:
            raise ValueError("Backend must be 'open3d' or 'matplotlib'")
        if backend == 'open3d' and not _OPEN3D_AVAILABLE:
            print("Warning: Open3D not available, falling back to matplotlib")
            backend = 'matplotlib'
        self._backend = backend

    def plot_domain(self):
        ChartPlotter.plot_chart_domain(self.chart)

    def plot_grid(self, grid_lines=10):
        ChartPlotter.plot_chart_grid(self.chart, grid_lines)

    def plot_surface(self, resolution=50, color='lightblue', alpha=0.4, with_grid=False, grid_lines=15):
        """Plot the surface with optional coordinate grid."""
        ax = self._plot_surface_matplotlib(resolution, color, alpha)
        if with_grid:
            self.plot_coordinate_grid(grid_lines=grid_lines, ax=ax)
        return ax

    def _plot_surface_open3d(self, resolution=50, color=None, alpha=1.0):
        """Improved Open3D surface visualization."""
        if not _OPEN3D_AVAILABLE:
            raise ImportError("Open3D is not available. Please install open3d.")

        dom = self.chart.domain
        coords = self.chart.coords
        emb_func = lambdify(coords, self.chart.embedding.map_exprs, 'numpy')

        # Create a denser grid for better surface quality
        u = np.linspace(dom.bounds[0][0], dom.bounds[0][1], resolution)
        v = np.linspace(dom.bounds[1][0], dom.bounds[1][1], resolution)
        uu, vv = np.meshgrid(u, v)
        
        # Generate points
        points = []
        triangles = []
        
        # Create points and triangles for mesh
        for i in range(resolution):
            for j in range(resolution):
                pt = emb_func(uu[i,j], vv[i,j])
                points.append(pt)
                
                # Create triangles (two for each grid cell)
                if i < resolution-1 and j < resolution-1:
                    # First triangle
                    triangles.append([
                        i * resolution + j,
                        (i+1) * resolution + j,
                        i * resolution + j + 1
                    ])
                    # Second triangle
                    triangles.append([
                        (i+1) * resolution + j,
                        (i+1) * resolution + j + 1,
                        i * resolution + j + 1
                    ])

        # Convert to numpy arrays
        points = np.array(points)
        triangles = np.array(triangles)

        # Create mesh directly instead of using Poisson reconstruction
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(points)
        mesh.triangles = o3d.utility.Vector3iVector(triangles)
        
        # Compute vertex normals for proper lighting
        mesh.compute_vertex_normals()
        
        # Set color with alpha
        if color is None:
            color = [0.7, 0.7, 0.9]  # Default light blue
        if isinstance(color, str):
            color = matplotlib.colors.to_rgb(color)
        
        # Convert color to the format Open3D expects
        color = np.asarray(color[:3]).reshape(3, 1)  # Only use RGB, reshape to (3,1)
        mesh.paint_uniform_color(color)

        # Create visualizer with custom settings
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(mesh)
        
        # Improve rendering settings
        render_option = vis.get_render_option()
        render_option.background_color = np.asarray([1, 1, 1])  # White background
        render_option.light_on = True
        render_option.mesh_show_back_face = True
        render_option.point_size = 1.0
        render_option.line_width = 1.0
        
        # Set material properties for transparency
        if alpha < 1.0:
            render_option.mesh_show_back_face = True
            render_option.mesh_show_wireframe = False
        
        # Set camera position for better view
        ctr = vis.get_view_control()
        ctr.set_zoom(0.8)
        ctr.set_front([1, 0, 0])
        ctr.set_lookat([0, 0, 0])
        ctr.set_up([0, 0, 1])
        
        vis.run()
        vis.destroy_window()

    def _plot_surface_matplotlib(self, resolution=50, color='lightblue', alpha=0.4):
        """Plot the surface using matplotlib."""
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Remove all background elements
        ax.set_axis_off()
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.grid(False)
        
        coords = self.chart.coords
        emb_func = lambdify(coords, self.chart.embedding.map_exprs, 'numpy')
        
        u = np.linspace(self.chart.domain.bounds[0][0], self.chart.domain.bounds[0][1], resolution)
        v = np.linspace(self.chart.domain.bounds[1][0], self.chart.domain.bounds[1][1], resolution)
        U, V = np.meshgrid(u, v)
        
        xyz = np.array([emb_func(ux, vx) for ux, vx in zip(U.flatten(), V.flatten())])
        X = xyz[:, 0].reshape(U.shape)
        Y = xyz[:, 1].reshape(U.shape)
        if xyz.shape[1] == 3:
            Z = xyz[:, 2].reshape(U.shape)
        else:
            Z = np.zeros_like(X)
        
        ax.plot_surface(X, Y, Z, color=color, alpha=alpha)
        return ax

    def plot_coordinate_grid(self, grid_lines=15, color='gray', alpha=0.2, linewidth=0.5, ax=None):
        """Plot coordinate grid lines on the manifold surface with improved visibility."""
        dom = self.chart.domain
        coords = self.chart.coords
        emb_func = lambdify(coords, self.chart.embedding.map_exprs, 'numpy')

        # Get current axis or create new one
        if ax is None:
            ax = plt.gca()
            if not hasattr(ax, 'get_zlim'):  # Check if current axis is 3D
                fig = plt.figure(figsize=(10, 10))
                ax = fig.add_subplot(111, projection='3d')

        # Plot phi-lines (longitude)
        phi_vals = np.linspace(dom.bounds[0][0], dom.bounds[0][1], grid_lines)
        theta_dense = np.linspace(dom.bounds[1][0], dom.bounds[1][1], 100)
        for phi in phi_vals:
            points = np.array([emb_func(phi, theta) for theta in theta_dense])
            ax.plot(points[:, 0], points[:, 1], points[:, 2], 
                   color=color, alpha=alpha, linestyle='-', linewidth=linewidth)

        # Plot theta-lines (latitude)
        theta_vals = np.linspace(dom.bounds[1][0], dom.bounds[1][1], grid_lines)
        phi_dense = np.linspace(dom.bounds[0][0], dom.bounds[0][1], 100)
        for theta in theta_vals:
            points = np.array([emb_func(phi, theta) for phi in phi_dense])
            ax.plot(points[:, 0], points[:, 1], points[:, 2], 
                   color=color, alpha=alpha, linestyle='-', linewidth=linewidth)
        
        return ax

    def plot_vector_field(self, vf: VectorField, resolution=20, scale=0.1):
        Open3DPlotter.plot_vector_field(self.chart, vf, resolution, scale)

    def export_mesh(self, filename: str, resolution=50):
        Open3DPlotter.export_scene_as_mesh(self.chart, filename, resolution)

    def visualize_geodesic(self, connection: LeviCivitaConnection, init_point, init_velocity, steps=100, dt=0.05):
        Open3DPlotter.visualize_geodesic(self.chart, connection, init_point, init_velocity, steps, dt)

    def visualize_flow(self, vf: VectorField, start_points, steps=100, dt=0.05):
        Open3DPlotter.visualize_flow(self.chart, vf, start_points, steps, dt)

    def scalar_field(self, expr: sp.Expr, resolution=50, cmap='viridis'):
        ScalarFieldVisualizer.plot_scalar_field(self.chart, expr, resolution, cmap)

    def interactive_flow(self, vf: VectorField):
        InteractiveTools.interactive_flow_slider(self.chart, vf)

    def interactive_scalar_fields(self, field_exprs: List[sp.Expr], names: List[str] = None):
        InteractiveTools.interactive_scalar_field(self.chart, field_exprs, names)

    def interactive_geodesics(self):
        """Interactive mode where clicking points draws geodesics"""
        if not self._interactive_backend_available:
            print("Warning: Interactive visualization not available. Using static plot instead.")
            self._plot_surface_matplotlib(resolution=30)
            return
            
        try:
            self._interactive_geodesics_mpl()
        except Exception as e:
            print(f"Warning: Interactive visualization failed ({str(e)}). Using static plot instead.")
            self._plot_surface_matplotlib(resolution=30)

    def _interactive_geodesics_mpl(self):
        """Interactive mode for geodesic visualization with error handling."""
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot the surface first
        u = np.linspace(self.chart.domain.bounds[0][0], self.chart.domain.bounds[0][1], 30)
        v = np.linspace(self.chart.domain.bounds[1][0], self.chart.domain.bounds[1][1], 30)
        U, V = np.meshgrid(u, v)
        
        coords = self.chart.coords
        emb_func = lambdify(coords, self.chart.embedding.map_exprs, 'numpy')
        pts = np.array([emb_func(ux, vx) for ux, vx in zip(U.flatten(), V.flatten())])
        X = pts[:, 0].reshape(U.shape)
        Y = pts[:, 1].reshape(U.shape)
        Z = pts[:, 2].reshape(U.shape)
        
        surf = ax.plot_surface(X, Y, Z, alpha=0.3, color='lightblue')
        ax.set_title('Click points to draw geodesics')
        
        points = []
        def onclick(event):
            if event.inaxes != ax:
                return
            
            if len(points) < 2:
                # Get the mouse click position in display coordinates
                x2d, y2d = event.xdata, event.ydata
                
                # Convert 2D click into 3D point on surface
                # First get the view direction
                azim = np.deg2rad(ax.azim)
                elev = np.deg2rad(ax.elev)
                
                # View direction vector
                view_dir = np.array([
                    np.cos(elev) * np.sin(azim),
                    np.cos(elev) * np.cos(azim),
                    np.sin(elev)
                ])
                
                # Find closest point on surface to the clicked point
                # Project all surface points to view plane
                pts_2d = np.column_stack([X.flatten(), Y.flatten(), Z.flatten()])
                view_pts = pts_2d.dot(view_dir)
                click_pt = np.array([x2d, y2d, 0]).dot(view_dir)
                
                # Find closest point
                idx = np.argmin(np.abs(view_pts - click_pt))
                closest_pt = pts_2d[idx]
                
                # Find corresponding u,v coordinates
                idx_u, idx_v = np.unravel_index(idx, U.shape)
                u_val, v_val = U[idx_u, idx_v], V[idx_u, idx_v]
                
                points.append((u_val, v_val))
                ax.scatter([closest_pt[0]], [closest_pt[1]], [closest_pt[2]], 
                          color='red', s=100)
                
                if len(points) == 2:
                    # Draw geodesic between points
                    p1, p2 = points
                    
                    # Create initial velocity vector
                    v1 = np.array(p2) - np.array(p1)
                    v1 = v1 / np.linalg.norm(v1)  # normalize
                    
                    # Set up geodesic initial conditions
                    initial_conditions = [(p1, v1)]
                    
                    # Plot the geodesic
                    FlowPlotter.plot_geodesics(self.chart.connection, initial_conditions, 
                                             t_span=(0, 2), num=100)
                    points.clear()
                
                fig.canvas.draw_idle()
        
        fig.canvas.mpl_connect('button_press_event', onclick)
        
        # Use a non-blocking show if possible
        try:
            plt.show(block=False)
            plt.pause(0.1)  # Small pause to ensure window is shown
        except:
            plt.show()  # Fallback to blocking show

class VectorFieldPlotter:
    """
    Dedicated interface for plotting vector fields in both 2D and 3D contexts.
    """
    def __init__(self, chart: Chart, vector_field: VectorField):
        self.chart = chart
        self.vf = vector_field

    def plot_open3d(self, resolution: int = 20, scale: float = 0.1):
        Open3DPlotter.plot_vector_field(self.chart, self.vf, resolution, scale)

    def plot_matplotlib(self, ax=None, resolution: int = 15, scale: float = 0.2, color='red', alpha=0.6):
        """Plot vector field using matplotlib with proper arrow heads."""
        if ax is None:
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, projection='3d')

        coords = self.chart.coords
        bounds = self.chart.domain.bounds
        emb_func = lambdify(coords, self.chart.embedding.map_exprs, 'numpy')
        
        # Create Jacobian for transforming vectors
        J_matrix = sp.Matrix(self.chart.embedding.map_exprs).jacobian(coords)
        J_func = lambdify(coords, J_matrix, 'numpy')

        u = np.linspace(bounds[0][0], bounds[0][1], resolution)
        v = np.linspace(bounds[1][0], bounds[1][1], resolution)
        uu, vv = np.meshgrid(u, v)

        for ux, vx in zip(uu.flatten(), vv.flatten()):
            if not self.chart.domain.contains((ux, vx)):
                continue
            p = np.array(emb_func(ux, vx))
            vvec = np.array(self.vf.evaluate((ux, vx)))
            # Transform vector using Jacobian
            vvec_embedded = J_func(ux, vx).dot(vvec)
            # Normalize and scale the vector
            vvec_embedded = vvec_embedded / (np.linalg.norm(vvec_embedded) + 1e-10) * scale
            
            # Plot arrow
            ax.quiver(p[0], p[1], p[2],
                     vvec_embedded[0], vvec_embedded[1], vvec_embedded[2],
                     color=color, alpha=alpha, arrow_length_ratio=0.2)
        
        return ax

class FlowPlotter:
    """
    Visualizes flows (integral curves) of vector fields.
    """
    def __init__(self, chart: Chart, vector_field: VectorField):
        self.chart = chart
        self.vf = vector_field

    def simulate(self, start_point: Tuple[float, float], steps: int = 100, dt: float = 0.05) -> List[np.ndarray]:
        coords = self.chart.coords
        emb_func = lambdify(coords, self.chart.embedding.map_exprs, 'numpy')
        pos = np.array(start_point, dtype=float)
        traj = [emb_func(*pos)]

        # Calculate total time and adjust dt if needed
        total_time = steps * dt
        
        for _ in range(steps):
            vel = np.array(self.vf.evaluate(tuple(pos)))
            pos += dt * vel
            traj.append(emb_func(*pos))

        return traj

    def plot(self, start_points: List[Tuple[float, float]], steps: int = 100, dt: float = 0.05, 
            t_span: Tuple[float, float] = None, backend='matplotlib', ax=None, color='blue', alpha=0.8):
        """Plot flow lines using either matplotlib or open3d."""
        if t_span is not None:
            # Calculate steps and dt from t_span
            total_time = t_span[1] - t_span[0]
            dt = total_time / steps

        if backend == 'open3d':
            if not _OPEN3D_AVAILABLE:
                raise ImportError("Open3D is not available. Please install open3d.")
            self._plot_open3d(start_points, steps, dt)
        else:
            return self._plot_matplotlib(start_points, steps, dt, ax, color, alpha)

    def _plot_matplotlib(self, start_points, steps, dt, ax=None, color='blue', alpha=0.8):
        """Plot flow lines using matplotlib."""
        if ax is None:
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, projection='3d')

        for pt in start_points:
            traj = self.simulate(pt, steps=steps, dt=dt)
            traj = np.array(traj)
            ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], 
                   color=color, alpha=alpha, linewidth=2)
        
        return ax

    def _plot_open3d(self, start_points, steps, dt):
        """Plot flow lines using Open3D."""
        lines = []
        for pt in start_points:
            traj = self.simulate(pt, steps=steps, dt=dt)
            traj = np.array(traj)
            line = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(traj),
                lines=o3d.utility.Vector2iVector([[i, i + 1] for i in range(len(traj) - 1)])
            )
            line.paint_uniform_color([0.5, 0.2, 0.7])
            lines.append(line)
        o3d.visualization.draw_geometries(lines)

    @staticmethod
    def plot_geodesics(connection, initial_conditions, t_span=(0, 1), num=100, ax=None, surface_alpha=0.3):
        """Plot geodesics from initial conditions using matplotlib."""
        from scipy.integrate import solve_ivp
        
        chart = connection.chart
        coords = chart.coords
        emb_func = lambdify(coords, chart.embedding.map_exprs, 'numpy')
        
        def geodesic_eq(t, state):
            x, y, dx, dy = state
            # Compute Christoffel symbols at current point
            Gamma = [
                [
                    [float(connection.Gamma[i][j][k].subs({coords[0]: x, coords[1]: y}))
                     for k in range(2)]
                    for j in range(2)
                ]
                for i in range(2)
            ]
            
            # Compute accelerations using geodesic equation
            ddx = -sum(Gamma[0][j][k] * state[2+j] * state[2+k] 
                      for j in range(2) for k in range(2))
            ddy = -sum(Gamma[1][j][k] * state[2+j] * state[2+k] 
                      for j in range(2) for k in range(2))
            return [dx, dy, ddx, ddy]
        
        if ax is None:
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, projection='3d')
        
        # Plot the surface first if no axis was provided
        if len(ax.collections) == 0:  # Check if surface hasn't been plotted yet
            u = np.linspace(chart.domain.bounds[0][0], chart.domain.bounds[0][1], 20)
            v = np.linspace(chart.domain.bounds[1][0], chart.domain.bounds[1][1], 20)
            U, V = np.meshgrid(u, v)
            pts = np.array([emb_func(ux, vx) for ux, vx in zip(U.flatten(), V.flatten())])
            X = pts[:, 0].reshape(U.shape)
            Y = pts[:, 1].reshape(U.shape)
            Z = pts[:, 2].reshape(U.shape)
            ax.plot_surface(X, Y, Z, alpha=surface_alpha, color='lightblue')
        
        for (x0, y0), (v0x, v0y) in initial_conditions:
            # Normalize initial velocity
            v_norm = np.sqrt(v0x**2 + v0y**2)
            if v_norm > 0:
                v0x, v0y = v0x/v_norm, v0y/v_norm
                
            sol = solve_ivp(
                geodesic_eq,
                t_span,
                [x0, y0, v0x, v0y],
                t_eval=np.linspace(t_span[0], t_span[1], num),
                method='RK45'
            )
            
            points = np.array([emb_func(x, y) for x, y in zip(sol.y[0], sol.y[1])])
            ax.plot(points[:, 0], points[:, 1], points[:, 2], 'r-', linewidth=2)
            start_pt = emb_func(x0, y0)
            ax.scatter([start_pt[0]], [start_pt[1]], [start_pt[2]], color='blue', s=100)
        
        ax.set_title('Geodesics')
        return ax

class ConnectionPlotter:
    """
    Visualizes geodesics and connection data from a Levi-Civita connection.
    """
    def __init__(self, chart: Chart, connection: LeviCivitaConnection):
        self.chart = chart
        self.conn = connection

    @staticmethod
    def plot_parallel_transport(connection, curve_funcs, init_vecs, t_span=(0, 2*np.pi), num=100):
        """
        Plot parallel transport of vectors along a curve.
        
        Args:
            connection: LeviCivitaConnection instance
            curve_funcs: List of functions [t -> x(t), t -> y(t)] defining the curve
            init_vecs: List of initial vectors to transport
            t_span: Time interval for integration
            num: Number of points to use for plotting
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        from scipy.integrate import solve_ivp
        
        chart = connection.chart
        coords = chart.coords
        emb_func = lambdify(coords, chart.embedding.map_exprs, 'numpy')
        
        # Pre-compute Jacobian function
        J_matrix = sp.Matrix(chart.embedding.map_exprs).jacobian(coords)
        J_func = lambdify(coords, J_matrix, 'numpy')
        
        # Convert curve functions to numpy if they're symbolic
        if isinstance(curve_funcs[0], sp.Expr):
            t = sp.Symbol('t')
            # For symbolic functions, also compute their derivatives
            curve_funcs_prime = [sp.diff(f, t) for f in curve_funcs]
            curve_funcs = [lambdify(t, f, 'numpy') for f in curve_funcs]
            curve_funcs_prime = [lambdify(t, f, 'numpy') for f in curve_funcs_prime]
        else:
            # For numerical functions, use finite differences for derivatives
            def make_prime(f):
                def f_prime(t, h=1e-7):
                    return (f(t + h) - f(t - h)) / (2 * h)
                return f_prime
            curve_funcs_prime = [make_prime(f) for f in curve_funcs]
        
        # Set up the parallel transport equation
        def parallel_transport_eq(t, state):
            # state = [x, y, v1, v2] where (x,y) is position and (v1,v2) is vector
            x, y = state[:2]
            v = state[2:]
            
            # Compute curve velocity using the derivatives
            dx = curve_funcs_prime[0](t)
            dy = curve_funcs_prime[1](t)
            
            # Evaluate Christoffel symbols at current point
            Gamma = [
                [
                    [float(connection.Gamma[i][j][k].subs({coords[0]: x, coords[1]: y}))
                     for k in range(2)]
                    for j in range(2)
                ]
                for i in range(2)
            ]
            
            # Parallel transport equation: dv^i/dt + Γⁱⱼₖ v^j dx^k/dt = 0
            dv = [-sum(Gamma[i][j][k] * v[j] * [dx, dy][k]
                      for j in range(2) for k in range(2))
                  for i in range(2)]
            
            return [dx, dy] + dv
        
        # Create figure
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot the surface
        u = np.linspace(chart.domain.bounds[0][0], chart.domain.bounds[0][1], 20)
        v = np.linspace(chart.domain.bounds[1][0], chart.domain.bounds[1][1], 20)
        U, V = np.meshgrid(u, v)
        pts = np.array([emb_func(ux, vx) for ux, vx in zip(U.flatten(), V.flatten())])
        X = pts[:, 0].reshape(U.shape)
        Y = pts[:, 1].reshape(U.shape)
        Z = pts[:, 2].reshape(U.shape)
        ax.plot_surface(X, Y, Z, alpha=0.3, color='lightblue')
        
        # Plot the curve and parallel transported vectors
        t_vals = np.linspace(t_span[0], t_span[1], num)
        curve_pts = np.array([emb_func(curve_funcs[0](t), curve_funcs[1](t)) for t in t_vals])
        ax.plot(curve_pts[:, 0], curve_pts[:, 1], curve_pts[:, 2], 'r-', linewidth=2)
        
        # Plot parallel transported vectors
        for v0 in init_vecs:
            # Initial conditions: [x0, y0, v0x, v0y]
            y0 = [curve_funcs[0](t_span[0]), curve_funcs[1](t_span[0])] + list(v0)
            
            # Solve parallel transport equation
            sol = solve_ivp(
                parallel_transport_eq,
                t_span,
                y0,
                t_eval=t_vals,
                method='RK45'
            )
            
            # Plot vectors at regular intervals
            for i in range(0, len(t_vals), len(t_vals)//10):
                pt = emb_func(sol.y[0][i], sol.y[1][i])
                v = np.array([sol.y[2][i], sol.y[3][i]])
                # Scale vector for visualization
                scale = 0.2
                v = v / np.linalg.norm(v) * scale
                # Convert vector to embedding space using pre-computed Jacobian
                v_emb = J_func(sol.y[0][i], sol.y[1][i]).dot(v)
                ax.quiver(pt[0], pt[1], pt[2],
                         v_emb[0], v_emb[1], v_emb[2],
                         color='blue', length=0.2)
        
        plt.title('Parallel Transport')
        plt.show()

    def plot_christoffel_component(self, i: int, j: int, k: int, resolution: int = 50):
        """Plot a specific Christoffel symbol as a scalar field."""
        ScalarFieldVisualizer.plot_scalar_field(
            self.chart,
            self.conn.Gamma[i][j][k],
            resolution
        )

    def interactive_christoffel(self):
        """Interactive visualization of Christoffel symbols."""
        if not _IPYWIDGETS_AVAILABLE:
            print("Warning: ipywidgets not available. Using static visualization.")
            self.plot_christoffel_component(0, 0, 0)
            return
            
        import ipywidgets as widgets
        from IPython.display import display
        
        n = len(self.chart.coords)
        
        def update(i, j, k):
            self.plot_christoffel_component(i, j, k)
        
        i_widget = widgets.IntSlider(min=0, max=n-1, description='i')
        j_widget = widgets.IntSlider(min=0, max=n-1, description='j')
        k_widget = widgets.IntSlider(min=0, max=n-1, description='k')
        
        widgets.interactive(update, i=i_widget, j=j_widget, k=k_widget)


