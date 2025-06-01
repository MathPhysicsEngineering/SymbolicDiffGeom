"""
Visualization Module for Symbolic Differential Geometry

This module provides comprehensive visualization tools for differential geometry concepts using both
Matplotlib and Open3D backends. It includes functionality for visualizing:
- Charts and coordinate systems
- Vector fields and their flows
- Geodesics and parallel transport
- Scalar fields and curvature
- Interactive tools for exploration

Key Components:
- Base plotting classes for different geometric objects
- Multiple backend support (Matplotlib/Open3D)
- Interactive widgets for dynamic visualization
- LaTeX documentation generation
"""

import sympy as sp  # Symbolic mathematics library for geometric calculations
import numpy as np  # Numerical computations library
import matplotlib   # Base matplotlib import for backend configuration
import os

# Backend Configuration
# --------------------
# Handle matplotlib backend selection based on display availability
# This ensures the code works in both interactive and non-interactive environments
if os.environ.get('DISPLAY', '') == '':
    print('No display found. Using non-interactive Agg backend')
    matplotlib.use('Agg')
else:
    try:
        # Qt5Agg offers better performance and stability for 3D visualization
        matplotlib.use('Qt5Agg')
    except:
        try:
            # Fallback to TkAgg if Qt5 is not available
            matplotlib.use('TkAgg')
        except:
            print('Warning: Could not initialize interactive backend. Using non-interactive Agg backend')
            matplotlib.use('Agg')

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, art3d
from typing import List, Tuple, Callable, Optional, Union

# Core sympy function for converting symbolic expressions to numerical functions
from sympy import lambdify  # lambdify converts sympy expressions to fast numerical functions

# Local imports from the differential geometry package
from charts import Chart  # Represents coordinate charts on manifolds
from vector_fields import VectorField  # Represents vector fields on manifolds
from connections import LeviCivitaConnection  # Handles parallel transport and geodesics

# Optional Advanced Visualization Backends
# --------------------------------------
# Open3D provides high-quality 3D visualization capabilities
try:
    import open3d as o3d
    _OPEN3D_AVAILABLE = True
except ImportError:
    _OPEN3D_AVAILABLE = False
    print("Warning: open3d not found. 3D visualization will be limited.")

# IPython widgets for interactive visualizations
try:
    import ipywidgets as widgets
    from IPython.display import display, clear_output
    _IPYWIDGETS_AVAILABLE = True
except ImportError:
    _IPYWIDGETS_AVAILABLE = False

# ---------------- Base Visualization Classes ----------------

class ChartPlotter:
    """
    Base class for visualizing chart domains and coordinate grids.
    
    This class provides fundamental plotting capabilities for coordinate charts,
    focusing on the visualization of:
    1. Chart domains - The regions where coordinate systems are valid
    2. Coordinate grids - The grid lines showing constant coordinate values
    
    Key Methods:
    - plot_chart_domain: Visualizes the valid region of a coordinate chart
    - plot_chart_grid: Draws coordinate grid lines
    
    Usage:
        chart = Chart(...)
        ChartPlotter.plot_chart_domain(chart)
        ChartPlotter.plot_chart_grid(chart, grid_lines=10)
    """
    
    @staticmethod
    def plot_chart_domain(chart: Chart, resolution: int = 50) -> None:
        """
        Visualizes the domain of a coordinate chart.
        
        Args:
            chart: The coordinate chart to visualize
            resolution: Number of points to use in each dimension
            
        The method uses contour plotting to show where the chart is valid,
        particularly useful for charts with non-rectangular domains.
        """
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
        """
        Plots coordinate grid lines in the chart domain.
        
        Args:
            chart: The coordinate chart
            grid_lines: Number of grid lines in each direction
            
        This visualization helps understand how coordinates vary across the domain
        and is particularly useful for seeing coordinate singularities.
        """
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
        coords = chart.coords
        emb_func = lambdify(coords, chart.embedding.map_exprs, 'numpy')

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

        # Create mesh
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(np.array(points))
        mesh.triangles = o3d.utility.Vector3iVector(np.array(triangles))
        
        # Compute vertex normals for proper lighting
        mesh.compute_vertex_normals()
        
        # Set a light blue color
        mesh.paint_uniform_color([0.7, 0.7, 0.9])

        # Create coordinate grid lines
        grid_lines = []
        grid_resolution = 15  # Number of grid lines in each direction
        
        # Create latitude lines
        phi_vals = np.linspace(dom.bounds[0][0], dom.bounds[0][1], grid_resolution)
        theta_dense = np.linspace(dom.bounds[1][0], dom.bounds[1][1], 100)
        for phi in phi_vals:
            points = np.array([emb_func(phi, theta) for theta in theta_dense])
            line = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(points),
                lines=o3d.utility.Vector2iVector([[i, i + 1] for i in range(len(points) - 1)])
            )
            line.paint_uniform_color([0.5, 0.5, 0.5])  # Gray color for grid lines
            grid_lines.append(line)

        # Create longitude lines
        theta_vals = np.linspace(dom.bounds[1][0], dom.bounds[1][1], grid_resolution)
        phi_dense = np.linspace(dom.bounds[0][0], dom.bounds[0][1], 100)
        for theta in theta_vals:
            points = np.array([emb_func(phi, theta) for phi in phi_dense])
            line = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(points),
                lines=o3d.utility.Vector2iVector([[i, i + 1] for i in range(len(points) - 1)])
            )
            line.paint_uniform_color([0.5, 0.5, 0.5])  # Gray color for grid lines
            grid_lines.append(line)

        # Set up visualizer with custom settings
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        
        # Add geometries
        vis.add_geometry(mesh)
        for line in grid_lines:
            vis.add_geometry(line)
        
        # Improve rendering settings
        render_option = vis.get_render_option()
        render_option.background_color = np.asarray([1, 1, 1])  # White background
        render_option.light_on = True
        render_option.mesh_show_back_face = True
        render_option.point_size = 1.0
        render_option.line_width = 1.0
        
        # Set material properties for better appearance
        render_option.mesh_show_wireframe = False
        
        # Set camera position for better view
        ctr = vis.get_view_control()
        ctr.set_zoom(0.8)
        ctr.set_front([1, 0, 0])
        ctr.set_lookat([0, 0, 0])
        ctr.set_up([0, 0, 1])
        
        vis.run()
        vis.destroy_window()

    @staticmethod
    def plot_vector_field(chart: Chart, vf: VectorField, resolution: int = 30, scale: float = 0.1) -> None:
        """
        Creates a high-quality visualization of a vector field on a surface.
        
        Args:
            chart: The coordinate chart on which the vector field is defined
            vf: The vector field to visualize
            resolution: Number of sample points in each direction
            scale: Scale factor for vector arrows
        
        This method visualizes a vector field by:
        1. Creating a surface mesh as the base
        2. Computing vector field values at sample points
        3. Creating arrows to represent vectors
        4. Setting up proper 3D visualization
        
        Technical Details:
        - Uses lambdify for efficient numerical evaluation
        - Transforms vectors using the chart's Jacobian
        - Creates arrow geometry using lines and cones
        - Handles proper vector scaling and normalization
        
        The visualization process involves several key steps:
        1. Surface Creation:
           - Generates a triangulated mesh of the surface
           - Uses higher resolution than vector sampling for smoothness
        
        2. Vector Field Computation:
           - Evaluates vector field at regular grid points
           - Transforms vectors from chart coordinates to ambient space
           - Normalizes and scales vectors for visualization
        
        3. Arrow Generation:
           - Creates line segments for vector shafts
           - Adds cone meshes for arrow heads
           - Properly orients arrows in 3D space
        
        4. Visualization Setup:
           - Configures lighting and rendering options
           - Sets up camera position and controls
           - Enables interactive viewing
        """
        if not _OPEN3D_AVAILABLE:
            raise ImportError("Open3D is not available. Please install open3d.")

        # First create the surface mesh
        dom = chart.domain
        coords = chart.coords
        # Convert symbolic expressions to numerical functions for efficiency
        emb_func = lambdify(coords, chart.embedding.map_exprs, 'numpy')
        # Create Jacobian for transforming vectors from chart to ambient space
        J_matrix = sp.Matrix(chart.embedding.map_exprs).jacobian(coords)
        J_func = lambdify(coords, J_matrix, 'numpy')

        # Create surface mesh with higher resolution for smooth appearance
        # Double the resolution for the surface compared to vector sampling
        u = np.linspace(dom.bounds[0][0], dom.bounds[0][1], resolution * 2)
        v = np.linspace(dom.bounds[1][0], dom.bounds[1][1], resolution * 2)
        uu, vv = np.meshgrid(u, v)  # Create coordinate grid

        # Generate surface mesh
        points = []
        triangles = []
        
        # Create points and triangles for mesh
        for i in range(len(u)):
            for j in range(len(v)):
                pt = emb_func(uu[i,j], vv[i,j])
                points.append(pt)
                
                if i < len(u)-1 and j < len(v)-1:
                    # Create two triangles for each grid cell
                    triangles.append([
                        i * len(v) + j,
                        (i+1) * len(v) + j,
                        i * len(v) + j + 1
                    ])
                    triangles.append([
                        (i+1) * len(v) + j,
                        (i+1) * len(v) + j + 1,
                        i * len(v) + j + 1
                    ])

        # Create and configure surface mesh
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(np.array(points))
        mesh.triangles = o3d.utility.Vector3iVector(np.array(triangles))
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color([0.7, 0.7, 0.9])  # Light blue color

        # Create vector field visualization
        vector_lines = []  # For vector shafts
        arrow_heads = []   # For arrow heads
        
        # Create denser grid for vector field
        resolution = resolution * 2  # Double resolution for denser field
        scale = scale * 0.5  # Halve scale for better proportions
        
        # Sample points for vector field
        u = np.linspace(dom.bounds[0][0], dom.bounds[0][1], resolution)
        v = np.linspace(dom.bounds[1][0], dom.bounds[1][1], resolution)
        uu, vv = np.meshgrid(u, v)

        # Create vectors at each sample point
        for ux, vx in zip(uu.flatten(), vv.flatten()):
            if not chart.domain.contains((ux, vx)):
                continue
                
            # Get point in ambient space
            p = np.array(emb_func(ux, vx))
            # Get vector components in chart coordinates
            vvec = np.array(vf.evaluate((ux, vx)))
            # Transform vector to ambient space using Jacobian
            vvec_embedded = J_func(ux, vx).dot(vvec)
            # Normalize and scale vector
            vvec_embedded = vvec_embedded / (np.linalg.norm(vvec_embedded) + 1e-10) * scale
            
            # Create line for vector shaft
            end_point = p + 0.8 * vvec_embedded  # Shaft is 80% of total length
            line = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector([p, end_point]),
                lines=o3d.utility.Vector2iVector([[0, 1]])
            )
            line.paint_uniform_color([1, 0, 0])  # Red color
            vector_lines.append(line)
            
            # Create arrow head using cone
            arrow_head_length = 0.2 * scale
            arrow_head_radius = 0.03 * scale
            direction = vvec_embedded / np.linalg.norm(vvec_embedded)
            
            # Create cone for arrow head
            arrow_head = o3d.geometry.TriangleMesh.create_cone(
                radius=arrow_head_radius, 
                height=arrow_head_length
            )
            
            # Rotate arrow head to point in vector direction
            z_axis = np.array([0, 0, 1])
            rot_axis = np.cross(z_axis, direction)
            rot_angle = np.arccos(np.dot(z_axis, direction))
            if np.linalg.norm(rot_axis) > 1e-6:
                rot_axis = rot_axis / np.linalg.norm(rot_axis)
                R = o3d.geometry.get_rotation_matrix_from_axis_angle(rot_axis * rot_angle)
                arrow_head.rotate(R)
            
            # Position arrow head at end of shaft
            arrow_head.translate(end_point)
            arrow_head.paint_uniform_color([1, 0, 0])  # Red color
            arrow_heads.append(arrow_head)

        # Combine all geometries for visualization
        geometries = [mesh] + vector_lines + arrow_heads

        # Set up visualizer
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        
        # Add all geometries to scene
        for geom in geometries:
            vis.add_geometry(geom)
        
        # Configure rendering settings
        render_option = vis.get_render_option()
        render_option.background_color = np.asarray([1, 1, 1])  # White background
        render_option.light_on = True
        render_option.mesh_show_back_face = True
        render_option.point_size = 1.0
        render_option.line_width = 2.0  # Thicker lines for better visibility
        
        # Set camera position
        ctr = vis.get_view_control()
        ctr.set_zoom(0.8)
        ctr.set_front([1, 0, 0])
        ctr.set_lookat([0, 0, 0])
        ctr.set_up([0, 0, 1])
        
        # Run visualization
        vis.run()
        vis.destroy_window()

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
    def visualize_flow(chart: Chart, vf: VectorField, start_points: List[Tuple[float, float]], 
                      steps: int = 2000, dt: float = 0.1, vector_resolution: int = 40, 
                      vector_scale: float = 0.02) -> None:
        """
        Creates an integrated visualization of both vector field and its flow lines.
        
        This method combines vector field visualization with flow line integration to show
        how the vector field influences particle motion on the surface. It's particularly
        useful for understanding dynamical systems on manifolds.
        
        Args:
            chart: The coordinate chart defining the surface
            vf: Vector field to visualize
            start_points: List of initial points for flow lines
            steps: Number of integration steps for each flow line
            dt: Time step for numerical integration
            vector_resolution: Number of vectors to display in each direction
            vector_scale: Scale factor for vector arrows
        
        The visualization consists of several components:
        1. Base Surface:
           - Triangulated mesh showing the manifold
           - Light blue color for good contrast
        
        2. Vector Field:
           - Arrows showing vector direction and magnitude
           - Properly transformed from chart to ambient space
           - Scaled for clear visualization
        
        3. Flow Lines:
           - Integral curves starting from specified points
           - Multiple offset lines for thicker appearance
           - Color-coded differently from vectors
        
        Technical Implementation:
        - Uses lambdify for efficient numerical computation
        - Employs Jacobian for proper vector transformation
        - Numerical integration for flow lines
        - Multiple rendering passes for proper depth ordering
        
        Note on Parameters:
        - steps and dt control flow line length
        - vector_resolution affects vector field density
        - vector_scale controls arrow size
        
        Key Concepts:
        1. Lambdify:
           - Converts symbolic expressions to fast numerical functions
           - Essential for efficient computation of many points
        
        2. Vector Transformation:
           - Uses Jacobian to map vectors from chart to ambient space
           - Preserves geometric meaning of vectors
        
        3. Flow Integration:
           - Simple Euler method for trajectory computation
           - Checks domain bounds to handle chart boundaries
        
        4. Visualization Techniques:
           - Multiple offset lines for thicker flow lines
           - Overlapping cones for bold arrow heads
           - Proper depth ordering for clear visualization
        """
        if not _OPEN3D_AVAILABLE:
            raise ImportError("Open3D is not available. Please install open3d.")

        # Setup for numerical computations
        dom = chart.domain
        coords = chart.coords
        # Convert symbolic expressions to numerical functions
        emb_func = lambdify(coords, chart.embedding.map_exprs, 'numpy')
        # Create Jacobian for vector transformation
        J_matrix = sp.Matrix(chart.embedding.map_exprs).jacobian(coords)
        J_func = lambdify(coords, J_matrix, 'numpy')

        # Create base surface mesh
        resolution = 50  # Higher resolution for smooth surface
        u = np.linspace(dom.bounds[0][0], dom.bounds[0][1], resolution)
        v = np.linspace(dom.bounds[1][0], dom.bounds[1][1], resolution)
        uu, vv = np.meshgrid(u, v)
        points = []
        triangles = []
        
        # Generate surface triangulation
        for i in range(resolution):
            for j in range(resolution):
                pt = emb_func(uu[i,j], vv[i,j])
                points.append(pt)
                
                if i < resolution-1 and j < resolution-1:
                    # Create two triangles per grid cell
                    triangles.append([
                        i * resolution + j,
                        (i+1) * resolution + j,
                        i * resolution + j + 1
                    ])
                    triangles.append([
                        (i+1) * resolution + j,
                        (i+1) * resolution + j + 1,
                        i * resolution + j + 1
                    ])

        # Create and configure surface mesh
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(np.array(points))
        mesh.triangles = o3d.utility.Vector3iVector(np.array(triangles))
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color([0.8, 0.8, 0.95])  # Light blue color

        # Create vector field visualization
        vector_lines = []  # For vector shafts
        arrow_heads = []   # For arrow heads
        
        # Sample points for vector field
        u = np.linspace(dom.bounds[0][0], dom.bounds[0][1], vector_resolution)
        v = np.linspace(dom.bounds[1][0], dom.bounds[1][1], vector_resolution)
        uu, vv = np.meshgrid(u, v)

        # Generate vector field visualization
        for ux, vx in zip(uu.flatten(), vv.flatten()):
            if not chart.domain.contains((ux, vx)):
                continue
            
            # Get point and vector in ambient space
            p = np.array(emb_func(ux, vx))
            vvec = np.array(vf.evaluate((ux, vx)))
            # Transform vector using Jacobian
            vvec_embedded = J_func(ux, vx).dot(vvec)
            # Normalize and scale
            vvec_embedded = vvec_embedded / (np.linalg.norm(vvec_embedded) + 1e-10) * vector_scale
            
            # Create line for vector shaft (shorter)
            end_point = p + 0.6 * vvec_embedded  # Shaft is 60% of total length
            line = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector([p, end_point]),
                lines=o3d.utility.Vector2iVector([[0, 1]])
            )
            line.paint_uniform_color([1, 0, 0])  # Red color
            vector_lines.append(line)
            
            # Create arrow head (larger and more prominent)
            arrow_head_length = 0.4 * vector_scale
            arrow_head_radius = 0.1 * vector_scale
            direction = vvec_embedded / np.linalg.norm(vvec_embedded)
            
            # Create multiple cones for thicker appearance
            for scale_factor in [1.0, 0.9, 0.8]:  # Three overlapping cones
                arrow_head = o3d.geometry.TriangleMesh.create_cone(
                    radius=arrow_head_radius * scale_factor,
                    height=arrow_head_length * scale_factor
                )
                
                # Rotate arrow head to align with vector
                z_axis = np.array([0, 0, 1])
                rot_axis = np.cross(z_axis, direction)
                rot_angle = np.arccos(np.dot(z_axis, direction))
                if np.linalg.norm(rot_axis) > 1e-6:
                    rot_axis = rot_axis / np.linalg.norm(rot_axis)
                    R = o3d.geometry.get_rotation_matrix_from_axis_angle(rot_axis * rot_angle)
                    arrow_head.rotate(R)
                
                arrow_head.translate(end_point)
                arrow_head.paint_uniform_color([1, 0, 0])  # Red color
                arrow_heads.append(arrow_head)

        # Create flow lines with numerical integration
        flow_lines = []
        for pt in start_points:
            pos = np.array(pt, dtype=float)
            traj = [emb_func(*pos)]
            
            # Integrate flow lines
            for _ in range(steps):
                vel = np.array(vf.evaluate(tuple(pos)))
                pos += dt * vel  # Euler integration
                if not chart.domain.contains(tuple(pos)):
                    break
                traj.append(emb_func(*pos))
            
            traj = np.array(traj)
            
            # Create thicker lines using multiple offset lines
            offset_scale = 0.005  # Scale for line thickness
            offsets = [
                [0, 0, 0],
                [offset_scale, 0, 0],
                [-offset_scale, 0, 0],
                [0, offset_scale, 0],
                [0, -offset_scale, 0]
            ]
            
            # Create offset lines for thicker appearance
            for offset in offsets:
                offset_traj = traj + offset
                line = o3d.geometry.LineSet(
                    points=o3d.utility.Vector3dVector(offset_traj),
                    lines=o3d.utility.Vector2iVector([[i, i + 1] for i in range(len(offset_traj) - 1)])
                )
                line.paint_uniform_color([0.2, 0.6, 1.0])  # Blue color for flow lines
                flow_lines.append(line)

        # Set up visualizer with proper rendering order
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        
        # Add geometries in correct order for proper transparency
        vis.add_geometry(mesh)  # Surface first
        for line in flow_lines:  # Flow lines second
            vis.add_geometry(line)
        for line in vector_lines:  # Vector lines third
            vis.add_geometry(line)
        for arrow in arrow_heads:  # Arrow heads last
            vis.add_geometry(arrow)
        
        # Configure rendering settings
        render_option = vis.get_render_option()
        render_option.background_color = np.asarray([1, 1, 1])  # White background
        render_option.light_on = True
        render_option.mesh_show_back_face = True
        render_option.point_size = 1.0
        render_option.line_width = 3.0  # Thicker lines for better visibility
        
        # Set up camera for optimal viewing
        ctr = vis.get_view_control()
        ctr.set_zoom(0.8)
        ctr.set_front([1, 0, 0])
        ctr.set_lookat([0, 0, 0])
        ctr.set_up([0, 0, 1])
        
        # Run interactive visualizer
        vis.run()
        vis.destroy_window()

    @staticmethod
    def visualize_parallel_transport(chart: Chart, connection: LeviCivitaConnection, 
                                   start_point: Tuple[float, float], 
                                   initial_vector: Tuple[float, float],
                                   initial_velocity: Tuple[float, float],
                                   steps: int = 100, dt: float = 0.05) -> None:
        """
        Visualizes parallel transport of a vector along a geodesic using Open3D.
        """
        if not _OPEN3D_AVAILABLE:
            raise ImportError("Open3D is not available. Please install open3d.")

        # Create surface mesh first
        dom = chart.domain
        coords = chart.coords
        emb_func = lambdify(coords, chart.embedding.map_exprs, 'numpy')
        J_matrix = sp.Matrix(chart.embedding.map_exprs).jacobian(coords)
        J_func = lambdify(coords, J_matrix, 'numpy')

        # Create surface mesh with higher resolution
        resolution = 50
        u = np.linspace(dom.bounds[0][0], dom.bounds[0][1], resolution)
        v = np.linspace(dom.bounds[1][0], dom.bounds[1][1], resolution)
        uu, vv = np.meshgrid(u, v)
        points = []
        triangles = []
        
        for i in range(resolution):
            for j in range(resolution):
                pt = emb_func(uu[i,j], vv[i,j])
                points.append(pt)
                
                if i < resolution-1 and j < resolution-1:
                    triangles.append([
                        i * resolution + j,
                        (i+1) * resolution + j,
                        i * resolution + j + 1
                    ])
                    triangles.append([
                        (i+1) * resolution + j,
                        (i+1) * resolution + j + 1,
                        i * resolution + j + 1
                    ])

        # Create surface mesh
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(np.array(points))
        mesh.triangles = o3d.utility.Vector3iVector(np.array(triangles))
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color([0.7, 0.7, 0.9])  # Light blue color

        # Compute geodesic and parallel transport
        pos = np.array(start_point, dtype=float)
        vel = np.array(initial_velocity, dtype=float)
        vec = np.array(initial_vector, dtype=float)
        
        # Store trajectory and transported vectors
        traj = [emb_func(*pos)]
        vectors = []
        
        # Get Christoffel symbols
        Gamma = connection.Gamma
        
        for _ in range(steps):
            # Store current vector
            v_embedded = J_func(*pos).dot(vec)
            vectors.append((np.array(traj[-1]), v_embedded))
            
            # Update position and velocity (geodesic equation)
            acc = np.zeros_like(pos, dtype=float)
            for i in range(len(pos)):
                for j in range(len(pos)):
                    for k in range(len(pos)):
                        acc[i] -= float(Gamma[i][j][k].subs({coords[0]: pos[0], coords[1]: pos[1]})) * vel[j] * vel[k]
            
            # Update vector (parallel transport equation)
            dvec = np.zeros_like(vec, dtype=float)
            for i in range(len(vec)):
                for j in range(len(pos)):
                    for k in range(len(pos)):
                        dvec[i] -= float(Gamma[i][j][k].subs({coords[0]: pos[0], coords[1]: pos[1]})) * vec[j] * vel[k]
            
            # Euler integration
            pos += vel * dt
            vel += acc * dt
            vec += dvec * dt
            
            # Normalize vector to maintain length
            vec = vec / (np.linalg.norm(vec) + 1e-10)
            
            # Store new position
            traj.append(emb_func(*pos))

        # Create geodesic line
        traj = np.array(traj)
        geodesic = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(traj),
            lines=o3d.utility.Vector2iVector([[i, i + 1] for i in range(len(traj) - 1)])
        )
        geodesic.paint_uniform_color([1, 0, 0])  # Red color for geodesic

        # Create parallel transported vectors
        vector_lines = []
        scale = 0.2  # Scale for vector visualization
        for i in range(0, len(vectors), len(vectors)//10):  # Show vectors at regular intervals
            base_point, v = vectors[i]
            # Normalize and scale vector
            v = v / (np.linalg.norm(v) + 1e-10) * scale
            end_point = base_point + v
            
            # Create vector line
            line = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector([base_point, end_point]),
                lines=o3d.utility.Vector2iVector([[0, 1]])
            )
            line.paint_uniform_color([0, 0, 1])  # Blue color for transported vectors
            vector_lines.append(line)
            
            # Create arrow head
            arrow_head = o3d.geometry.TriangleMesh.create_cone(radius=0.02, height=0.05)
            
            # Rotate arrow head to point in vector direction
            direction = v / np.linalg.norm(v)
            z_axis = np.array([0, 0, 1])
            rot_axis = np.cross(z_axis, direction)
            rot_angle = np.arccos(np.dot(z_axis, direction))
            if np.linalg.norm(rot_axis) > 1e-6:
                rot_axis = rot_axis / np.linalg.norm(rot_axis)
                R = o3d.geometry.get_rotation_matrix_from_axis_angle(rot_axis * rot_angle)
                arrow_head.rotate(R)
            
            arrow_head.translate(end_point)
            arrow_head.paint_uniform_color([0, 0, 1])
            vector_lines.append(arrow_head)

        # Combine all geometries
        geometries = [mesh, geodesic] + vector_lines

        # Set up visualizer
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        
        # Add all geometries
        for geom in geometries:
            vis.add_geometry(geom)
        
        # Improve rendering settings
        render_option = vis.get_render_option()
        render_option.background_color = np.asarray([1, 1, 1])
        render_option.light_on = True
        render_option.mesh_show_back_face = True
        render_option.point_size = 1.0
        render_option.line_width = 2.0
        
        # Set camera position
        ctr = vis.get_view_control()
        ctr.set_zoom(0.8)
        ctr.set_front([1, 0, 0])
        ctr.set_lookat([0, 0, 0])
        ctr.set_up([0, 0, 1])
        
        vis.run()
        vis.destroy_window()

# ---------------- Scalar Field Visualizer ----------------
class ScalarFieldVisualizer:
    @staticmethod
    def plot_scalar_field(chart: Chart, scalar_expr: sp.Expr, ax=None, resolution: int = 50, cmap: str = 'viridis', backend='matplotlib') -> None:
        if backend == 'open3d' and not _OPEN3D_AVAILABLE:
            print("Warning: Open3D not available, falling back to matplotlib")
            backend = 'matplotlib'

        coords = chart.coords
        scalar_func = lambdify(coords, scalar_expr, 'numpy')
        emb_func = lambdify(coords, chart.embedding.map_exprs, 'numpy')

        # Generate coordinate grid
        u = np.linspace(chart.domain.bounds[0][0], chart.domain.bounds[0][1], resolution)
        v = np.linspace(chart.domain.bounds[1][0], chart.domain.bounds[1][1], resolution)
        U, V = np.meshgrid(u, v)
        
        # Compute scalar field values and embedding coordinates
        Z = scalar_func(U, V)
        xyz = np.array([emb_func(ux, vx) for ux, vx in zip(U.flatten(), V.flatten())])
        X = xyz[:, 0].reshape(U.shape)
        Y = xyz[:, 1].reshape(U.shape)
        Z3 = xyz[:, 2].reshape(U.shape)

        if backend == 'matplotlib':
            if ax is None:
                fig = plt.figure(figsize=(12, 12))
                ax = fig.add_subplot(111, projection='3d')
            
            # Remove all background elements
            ax.set_axis_off()
            ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax.grid(False)
            
            # Handle color normalization carefully
            Z_min, Z_max = np.nanmin(Z), np.nanmax(Z)
            if Z_min == Z_max:
                # For constant scalar fields, create a uniform color array
                colors = np.tile(plt.cm.get_cmap(cmap)(0.5), (resolution, resolution, 1))
            else:
                Z_norm = (Z - Z_min) / (Z_max - Z_min)
                # Create color array with correct shape
                colors = np.array(plt.cm.get_cmap(cmap)(Z_norm))
                colors = colors.reshape(resolution, resolution, 4)  # Reshape to match surface dimensions
            
            # Plot the surface using the scalar field values for coloring
            surf = ax.plot_surface(X, Y, Z3, 
                                 facecolors=colors,
                                 vmin=Z_min, vmax=Z_max,
                                 alpha=0.8)
            
            # Add a color bar
            m = plt.cm.ScalarMappable(cmap=plt.cm.get_cmap(cmap))
            m.set_array(Z)
            plt.colorbar(m, ax=ax)
            
            # Set the view angle
            ax.view_init(elev=30, azim=45)
            
            if ax is None:
                plt.show()
            return ax

        elif backend == 'open3d':
            # Create triangles for the mesh
            triangles = []
            for i in range(resolution-1):
                for j in range(resolution-1):
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

            # Create mesh
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(xyz)
            mesh.triangles = o3d.utility.Vector3iVector(triangles)
            mesh.compute_vertex_normals()

            # Normalize scalar field values for coloring
            Z_min, Z_max = np.nanmin(Z), np.nanmax(Z)
            if Z_min == Z_max:
                # For constant scalar fields, use a uniform color
                colors = np.tile(plt.cm.get_cmap(cmap)(0.5)[:3], (len(xyz), 1))
            else:
                Z_norm = (Z.flatten() - Z_min) / (Z_max - Z_min)
                # Get colors from colormap
                colors = plt.cm.get_cmap(cmap)(Z_norm)[:, :3]  # Only take RGB, not alpha

            mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

            # Set up visualizer with custom settings
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

            # Set camera position for better view
            ctr = vis.get_view_control()
            ctr.set_zoom(0.8)
            ctr.set_front([1, 0, 0])
            ctr.set_lookat([0, 0, 0])
            ctr.set_up([0, 0, 1])

            vis.run()
            vis.destroy_window()

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

    def _plot_surface_matplotlib(self, resolution=50, color='lightblue', alpha=0.4, ax=None):
        """Plot the surface using matplotlib."""
        if ax is None:
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
    def plot_parallel_transport(connection, curve_funcs, init_vecs, t_span=(0, 2*np.pi), num=100, ax=None):
        """
        Plot parallel transport of vectors along a curve.
        
        Args:
            connection: LeviCivitaConnection instance
            curve_funcs: List of functions [t -> x(t), t -> y(t)] defining the curve
            init_vecs: List of initial vectors to transport
            t_span: Time interval for integration
            num: Number of points to use for plotting
            ax: Optional matplotlib axis for plotting
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
        
        # Pre-compute Christoffel symbols as functions
        Gamma_funcs = [
            [
                [lambdify(coords, connection.Gamma[i][j][k], 'numpy')
                 for k in range(2)]
                for j in range(2)
            ]
            for i in range(2)
        ]
        
        # Set up the parallel transport equation
        def parallel_transport_eq(t, state):
            # state = [x, y, v1, v2] where (x,y) is position and (v1,v2) is vector
            x, y = state[:2]
            v = state[2:]
            
            # Compute curve velocity using the derivatives
            dx = curve_funcs_prime[0](t)
            dy = curve_funcs_prime[1](t)
            
            # Evaluate Christoffel symbols at current point using pre-computed functions
            Gamma = [
                [
                    [float(np.real(Gamma_funcs[i][j][k](x, y)))
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
        
        # Create figure if not provided
        if ax is None:
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')
        
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
                v = v / (np.linalg.norm(v) + 1e-10) * scale
                # Convert vector to embedding space using pre-computed Jacobian
                v_emb = J_func(sol.y[0][i], sol.y[1][i]).dot(v)
                ax.quiver(pt[0], pt[1], pt[2],
                         v_emb[0], v_emb[1], v_emb[2],
                         color='blue', length=0.2)
        
        return ax

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

class SymbolicFlowComputer:
    """Computes symbolic expressions for flows and exponential maps when possible."""
    
    @staticmethod
    def compute_flow(vf: VectorField, t: sp.Symbol = None) -> Optional[sp.Matrix]:
        """
        Attempts to compute the symbolic flow of a vector field.
        Returns None if the flow cannot be computed symbolically.
        """
        if t is None:
            t = sp.Symbol('t')
            
        chart = vf.chart
        coords = chart.coords
        
        try:
            # For rotational vector field, try direct solution
            if all(not expr.has(*coords) for expr in vf.components):
                # Constant vector field
                flow = sp.Matrix([
                    coords[i] + t * vf.components[i]
                    for i in range(len(coords))
                ])
                return flow
            
            # Set up the system of ODEs
            system = sp.Matrix(vf.components)
            x = sp.Matrix(coords)
            
            # Try to solve the system symbolically
            try:
                flow = sp.solve_ode_system(system, x, t)
                if flow:
                    return flow
            except:
                pass
            
            # Try component-wise integration for simple cases
            flow = []
            for i, comp in enumerate(vf.components):
                try:
                    # Try to integrate each component separately
                    integral = sp.integrate(comp, t)
                    flow.append(coords[i] + integral)
                except:
                    return None
            
            if flow:
                return sp.Matrix(flow)
                
        except:
            pass
        
        return None

    @staticmethod
    def compute_exponential_map(connection: LeviCivitaConnection, point: Tuple[sp.Expr, sp.Expr] = None) -> Optional[sp.Matrix]:
        """
        Attempts to compute the symbolic exponential map at a point.
        Returns None if the map cannot be computed symbolically.
        """
        chart = connection.chart
        coords = chart.coords
        t = sp.Symbol('t')
        
        if point is None:
            # Use symbolic point
            point = tuple(sp.Symbol(f'p_{i}') for i in range(len(coords)))
            
        try:
            # Special case: sphere at north pole
            if all(x == 0 for x in point):
                # At north pole, geodesics are great circles
                v = tuple(sp.Symbol(f'v_{i}') for i in range(len(coords)))
                v_norm = sp.sqrt(sum(vi**2 for vi in v))
                exp_map = sp.Matrix([
                    t * v[0],
                    t * v[1]
                ])
                return exp_map
            
            # Set up the geodesic equation
            Gamma = connection.Gamma
            v = tuple(sp.Symbol(f'v_{i}') for i in range(len(coords)))
            
            # Geodesic equation: d²x^i/dt² + Γⁱⱼₖ dx^j/dt dx^k/dt = 0
            system = []
            for i in range(len(coords)):
                eq = sum(Gamma[i][j][k].subs({coords[0]: point[0], coords[1]: point[1]}) * v[j] * v[k]
                        for j in range(len(coords)) for k in range(len(coords)))
                system.append(-eq)
            
            # Try to solve the system symbolically
            system = sp.Matrix(system)
            exp_map = sp.solve_ode_system(system, sp.Matrix(v), t)
            
            if exp_map:
                return exp_map
                
        except:
            pass
            
        return None

    @staticmethod
    def export_to_latex(expr: sp.Matrix, filename: str) -> None:
        """
        Exports a symbolic expression to a LaTeX file.
        """
        with open(filename, 'w') as f:
            f.write('\\documentclass{article}\n')
            f.write('\\usepackage{amsmath}\n')
            f.write('\\begin{document}\n\n')
            
            f.write('\\[\n')
            f.write(sp.latex(expr))
            f.write('\n\\]\n\n')
            
            f.write('\\end{document}\n')

class LatexDocumentGenerator:
    """Generates a comprehensive LaTeX document explaining the geometry of a manifold."""
    
    @staticmethod
    def generate_sphere_documentation(chart: Chart, connection: LeviCivitaConnection, vector_fields, filename: str) -> None:
        """
        Generates a detailed LaTeX document explaining the geometry of the sphere.
        Includes all computations, figures, and detailed explanations.
        """
        coords = chart.coords
        phi, theta = coords
        
        # Create figures directory if it doesn't exist
        import os
        figures_dir = "figures"  # Use relative path
        os.makedirs(figures_dir, exist_ok=True)
        
        with open(filename, 'w') as f:
            # Document preamble
            f.write('\\documentclass[12pt]{article}\n')
            f.write('\\usepackage{amsmath,amsthm,amssymb}\n')
            f.write('\\usepackage[margin=1in]{geometry}\n')
            f.write('\\usepackage{tikz}\n')
            f.write('\\usepackage{graphicx}\n')
            f.write('\\usepackage{hyperref}\n')
            f.write('\\usepackage{float}\n')
            f.write('\\usepackage{mathtools}\n')
            f.write('\\usepackage{enumitem}\n')
            f.write('\\title{Detailed Analysis of the 2-Sphere: \\\\ A Complete Geometric Study}\n')
            f.write('\\author{Symbolic Differential Geometry Package}\n')
            f.write('\\date{\\today}\n')
            f.write('\\begin{document}\n')
            f.write('\\maketitle\n\n')
            
            # Abstract
            f.write('\\begin{abstract}\n')
            f.write('This document presents a comprehensive study of the geometry of the 2-sphere, including its parametrization, ')
            f.write('metric structure, connection, curvature, and vector fields. We provide detailed computations, visualizations, ')
            f.write('and explanations for each aspect, making this a complete reference for understanding spherical geometry.\n')
            f.write('\\end{abstract}\n\n')
            
            # Table of contents
            f.write('\\tableofcontents\n\\newpage\n\n')
            
            # 1. Introduction
            f.write('\\section{Introduction}\n')
            f.write('The 2-sphere $S^2$ is one of the most fundamental curved spaces in differential geometry. ')
            f.write('It serves as an excellent example for understanding the key concepts of Riemannian geometry. ')
            f.write('In this document, we will systematically explore its geometric properties, starting from basic definitions ')
            f.write('and building up to more advanced concepts like parallel transport and curvature.\n\n')
            
            # 2. Parametrization and Charts
            f.write('\\section{Parametrization and Charts}\n')
            f.write('\\subsection{Standard Spherical Coordinates}\n')
            f.write('We begin with the standard spherical coordinate parametrization of the unit sphere. ')
            f.write('The embedding into $\\mathbb{R}^3$ is given by:\n\\[\n')
            f.write('\\begin{aligned}\n')
            f.write('x &= ' + sp.latex(chart.embedding.map_exprs[0]) + '\\\\\n')
            f.write('y &= ' + sp.latex(chart.embedding.map_exprs[1]) + '\\\\\n')
            f.write('z &= ' + sp.latex(chart.embedding.map_exprs[2]) + '\n')
            f.write('\\end{aligned}\n\\]\n')
            f.write('where $\\phi \\in [0,\\pi]$ is the polar angle measured from the positive $z$-axis, ')
            f.write('and $\\theta \\in [0,2\\pi]$ is the azimuthal angle in the $x$-$y$ plane.\n\n')
            
            # Save coordinate grid visualization
            f.write('\\subsection{Coordinate Grid}\n')
            f.write('The coordinate grid on the sphere helps visualize how the parameters $\\phi$ and $\\theta$ ')
            f.write('cover the surface. The lines of constant $\\phi$ form parallels (circles of latitude), ')
            f.write('while lines of constant $\\theta$ form meridians (circles of longitude).\n\n')
            
            # 3. Tangent Space and Metric
            f.write('\\section{Tangent Space and Metric Structure}\n')
            f.write('\\subsection{Tangent Vectors}\n')
            J = sp.Matrix(chart.embedding.map_exprs).jacobian(coords)
            f.write('The tangent vectors at a point $(\\phi,\\theta)$ are obtained by differentiating the ')
            f.write('embedding map with respect to the coordinates:\n\\[\n')
            f.write('\\begin{aligned}\n')
            f.write('\\frac{\\partial}{\\partial \\phi} &= ' + sp.latex(J.col(0)) + '\\\\\n')
            f.write('\\frac{\\partial}{\\partial \\theta} &= ' + sp.latex(J.col(1)) + '\n')
            f.write('\\end{aligned}\n\\]\n')
            f.write('These vectors form a basis for the tangent space at each point.\n\n')
            
            f.write('\\subsection{First Fundamental Form}\n')
            f.write('The metric tensor (first fundamental form) is computed by taking inner products of the tangent vectors:\n\\[\n')
            f.write('g_{ij} = ' + sp.latex(chart.metric.g) + '\n\\]\n')
            f.write('This gives the line element:\n\\[\n')
            f.write('ds^2 = d\\phi^2 + \\sin^2\\phi\\,d\\theta^2\n\\]\n')
            f.write('The metric encodes how distances and angles are measured on the sphere.\n\n')
            
            # 4. Connection and Parallel Transport
            f.write('\\section{Levi-Civita Connection}\n')
            f.write('\\subsection{Christoffel Symbols}\n')
            f.write('The Levi-Civita connection is the unique torsion-free connection compatible with the metric. ')
            f.write('Its components are given by the Christoffel symbols:\n\\[\n')
            f.write('\\begin{aligned}\n')
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        if connection.Gamma[i][j][k] != 0:
                            f.write(f'\\Gamma^{i}_{{\\,{j}{k}}} &= {sp.latex(connection.Gamma[i][j][k])} \\\\\n')
            f.write('\\end{aligned}\n\\]\n')
            f.write('These symbols determine how vectors are parallel transported along curves on the sphere.\n\n')
            
            # 5. Curvature
            f.write('\\section{Curvature}\n')
            f.write('\\subsection{Riemann Curvature Tensor}\n')
            R = connection.riemann_tensor()
            f.write('The Riemann curvature tensor measures how parallel transport around infinitesimal loops fails to return vectors to their original position. ')
            f.write('Its non-zero components are:\n\\[\n')
            f.write('\\begin{aligned}\n')
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        for l in range(2):
                            if R[i][j][k][l] != 0:
                                f.write(f'R^{i}_{{\\,{j}{k}{l}}} &= {sp.latex(R[i][j][k][l])} \\\\\n')
            f.write('\\end{aligned}\n\\]\n\n')
            
            f.write('\\subsection{Ricci Tensor and Scalar Curvature}\n')
            Ric = connection.ricci_tensor()
            K = connection.scalar_curvature()
            f.write('The Ricci tensor is obtained by contracting the Riemann tensor:\n\\[\n')
            f.write('\\text{Ric} = ' + sp.latex(Ric) + '\n\\]\n')
            f.write('The scalar curvature (Gaussian curvature) is:\n\\[\n')
            f.write('K = ' + sp.latex(K) + '\n\\]\n')
            f.write('The constant value of 2 reflects the fact that the sphere has constant positive curvature.\n\n')
            
            # 6. Vector Fields and Flows
            f.write('\\section{Vector Fields and Their Flows}\n')
            for vf, name in vector_fields:
                f.write(f'\\subsection{{{name.title()} Vector Field}}\n')
                f.write(f'The {name} vector field is defined by its components:\n\\[\n')
                f.write('V = ' + sp.latex(sp.Matrix(vf.components)) + '\n\\]\n')
                
                # Compute and explain the flow
                t = sp.Symbol('t')
                flow = SymbolicFlowComputer.compute_flow(vf, t)
                if flow is not None:
                    f.write('The flow of this vector field can be computed explicitly:\n\\[\n')
                    f.write('\\gamma(t) = ' + sp.latex(flow) + '\n\\]\n')
                    f.write('This represents the position at time $t$ of a particle following the flow lines ')
                    f.write('starting from initial coordinates $(\\phi_0, \\theta_0)$.\n\n')
                else:
                    f.write('The flow of this vector field cannot be expressed in closed form, ')
                    f.write('but can be studied numerically and visualized.\n\n')
            
            # 7. Geodesics
            f.write('\\section{Geodesics}\n')
            f.write('The geodesic equations on the sphere are:\n\\[\n')
            f.write('\\begin{aligned}\n')
            for i in range(2):
                eq = sum(connection.Gamma[i][j][k] * sp.Symbol(f'\\dot{{x}}^{j}') * sp.Symbol(f'\\dot{{x}}^{k}')
                        for j in range(2) for k in range(2))
                f.write(f'\\ddot{{x}}^{i} + {sp.latex(eq)} &= 0 \\\\\n')
            f.write('\\end{aligned}\n\\]\n')
            f.write('where $x^0 = \\phi$ and $x^1 = \\theta$. These equations show that geodesics are great circles ')
            f.write('on the sphere, which are the curves of shortest distance between two points.\n\n')
            
            # 8. Parallel Transport Examples
            f.write('\\section{Parallel Transport Examples}\n')
            f.write('We conclude with two important examples of parallel transport on the sphere:\n\n')
            
            f.write('\\subsection{Transport Along a Great Circle}\n')
            f.write('When we parallel transport a vector along a great circle, the vector maintains ')
            f.write('a constant angle with the great circle. This leads to the interesting phenomenon ')
            f.write('that when transported around a closed loop, the vector does not return to its ')
            f.write('original direction, but is rotated by an angle proportional to the solid angle ')
            f.write('enclosed by the loop.\n\n')
            
            f.write('\\subsection{Transport Along a Meridian}\n')
            f.write('Parallel transport along a meridian demonstrates how vectors change direction ')
            f.write('relative to the local coordinate basis. A vector initially pointing East will ')
            f.write('maintain its angle with the meridian but appear to rotate relative to the ')
            f.write('coordinate grid.\n\n')
            
            # End document
            f.write('\\end{document}\n')

# ---------------- Open3D Advanced Visualization ----------------

class Open3DPlotter:
    """
    Advanced 3D visualization class using Open3D backend.
    
    This class provides high-quality 3D visualization capabilities for differential geometry
    objects using the Open3D library. It offers superior performance and interactivity
    compared to matplotlib for 3D visualization.
    
    Key Features:
    - High-performance 3D rendering
    - Interactive camera controls
    - Proper lighting and shading
    - Support for large point clouds and meshes
    
    Main Components:
    1. Surface Visualization
    2. Vector Field Plotting
    3. Flow Visualization
    4. Geodesic Visualization
    5. Parallel Transport Visualization
    
    The class uses lambdify extensively to convert symbolic expressions to fast numerical
    functions for efficient visualization.
    """
    
    @staticmethod
    def plot_embedding_surface(chart: Chart, resolution: int = 50) -> None:
        """
        Creates a high-quality visualization of an embedded surface.
        
        Args:
            chart: The coordinate chart defining the surface embedding
            resolution: Number of points in each coordinate direction
        
        This method:
        1. Creates a triangulated mesh of the surface
        2. Adds coordinate grid lines
        3. Sets up proper lighting and rendering
        4. Provides interactive 3D viewing
        
        Implementation Details:
        - Uses lambdify to convert symbolic embedding to numerical function
        - Creates triangulated mesh for efficient rendering
        - Adds coordinate grid lines for better understanding
        - Sets up proper lighting and camera positioning
        """
        if not _OPEN3D_AVAILABLE:
            raise ImportError("Open3D is not available. Please install open3d.")

        # Convert symbolic expressions to numerical functions
        dom = chart.domain
        coords = chart.coords
        # lambdify creates a fast numerical function from symbolic expressions
        emb_func = lambdify(coords, chart.embedding.map_exprs, 'numpy')

        # Create a denser grid for better surface quality
        u = np.linspace(dom.bounds[0][0], dom.bounds[0][1], resolution)
        v = np.linspace(dom.bounds[1][0], dom.bounds[1][1], resolution)
        uu, vv = np.meshgrid(u, v)
        
        # Generate mesh vertices and faces
        points = []
        triangles = []
        
        # Create points and triangles for mesh
        # This creates a triangulated surface for efficient rendering
        for i in range(resolution):
            for j in range(resolution):
                pt = emb_func(uu[i,j], vv[i,j])
                points.append(pt)
                
                # Create triangles (two for each grid cell)
                # This ensures proper surface coverage without gaps
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

        # Create mesh object
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(np.array(points))
        mesh.triangles = o3d.utility.Vector3iVector(np.array(triangles))
        
        # Compute vertex normals for proper lighting
        mesh.compute_vertex_normals()
        
        # Set a light blue color for the surface
        mesh.paint_uniform_color([0.7, 0.7, 0.9])

        # Create coordinate grid lines for better understanding
        grid_lines = []
        grid_resolution = 15  # Number of grid lines in each direction
        
        # Create latitude lines (constant phi)
        phi_vals = np.linspace(dom.bounds[0][0], dom.bounds[0][1], grid_resolution)
        theta_dense = np.linspace(dom.bounds[1][0], dom.bounds[1][1], 100)
        for phi in phi_vals:
            points = np.array([emb_func(phi, theta) for theta in theta_dense])
            line = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(points),
                lines=o3d.utility.Vector2iVector([[i, i + 1] for i in range(len(points) - 1)])
            )
            line.paint_uniform_color([0.5, 0.5, 0.5])  # Gray color for grid lines
            grid_lines.append(line)

        # Create longitude lines (constant theta)
        theta_vals = np.linspace(dom.bounds[1][0], dom.bounds[1][1], grid_resolution)
        phi_dense = np.linspace(dom.bounds[0][0], dom.bounds[0][1], 100)
        for theta in theta_vals:
            points = np.array([emb_func(phi, theta) for phi in phi_dense])
            line = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(points),
                lines=o3d.utility.Vector2iVector([[i, i + 1] for i in range(len(points) - 1)])
            )
            line.paint_uniform_color([0.5, 0.5, 0.5])  # Gray color for grid lines
            grid_lines.append(line)

        # Set up visualizer with custom settings
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        
        # Add geometries to the scene
        vis.add_geometry(mesh)
        for line in grid_lines:
            vis.add_geometry(line)
        
        # Configure rendering settings for optimal visualization
        render_option = vis.get_render_option()
        render_option.background_color = np.asarray([1, 1, 1])  # White background
        render_option.light_on = True
        render_option.mesh_show_back_face = True
        render_option.point_size = 1.0
        render_option.line_width = 1.0
        
        # Set material properties for better appearance
        render_option.mesh_show_wireframe = False
        
        # Set camera position for optimal initial view
        ctr = vis.get_view_control()
        ctr.set_zoom(0.8)
        ctr.set_front([1, 0, 0])
        ctr.set_lookat([0, 0, 0])
        ctr.set_up([0, 0, 1])
        
        # Run the visualizer
        vis.run()
        vis.destroy_window()


