"""
Demo script for S² using symbolic_diff_geom.
Covers: chart definitions, induced metric, Levi-Civita connection,
vector fields, flows, geodesics, exponential map, distance,
geodesic grids, manifold transitions, parallel transport, and curvature.
"""
import sympy as sp
import numpy as np
import os
from charts import Chart, SymbolicManifold, Embedding, BoxDomain
from vector_fields import VectorField
from visualization import (
    ChartPlotter,
    ManifoldPlotter,
    VectorFieldPlotter,
    FlowPlotter,
    ConnectionPlotter,
    ScalarFieldVisualizer,
    _OPEN3D_AVAILABLE,
    Open3DPlotter,
    SymbolicFlowComputer,
    LatexDocumentGenerator
)
import matplotlib.pyplot as plt
from sympy import lambdify

def save_matplotlib_figure(fig, filename):
    """Helper function to save matplotlib figures with consistent style and show them."""
    # Show the figure
    plt.show()
    
    # Save as PNG with high DPI for quality
    png_filename = filename.replace('.pdf', '.png')
    fig.savefig(png_filename, bbox_inches='tight', dpi=300)
    plt.close(fig)

def main():
    # Create figures directory
    os.makedirs("figures", exist_ok=True)

    print("\n1. Chart Construction and Basic Properties")
    print("----------------------------------------")

    # Coordinate symbols for north and south charts
    phi, theta = sp.symbols('phi theta', real=True)
    phi2, theta2 = sp.symbols('phi2 theta2', real=True)

    # Embedding map expressions for S²
    map_exprs_N = [sp.sin(phi)*sp.cos(theta), sp.sin(phi)*sp.sin(theta), sp.cos(phi)]
    map_exprs_S = [sp.sin(sp.pi - phi2)*sp.cos(theta2), sp.sin(sp.pi - phi2)*sp.sin(theta2), -sp.cos(phi2)]

    # Instantiate charts with BoxDomain bounds [0,π]×[0,2π]
    chart_N = Chart(
        name="north",
        coords=[phi, theta],
        domain=BoxDomain([(0, np.pi), (0, 2*np.pi)]),
        embedding=Embedding([phi, theta], map_exprs_N)
    )
    chart_S = Chart(
        name="south",
        coords=[phi2, theta2],
        domain=BoxDomain([(0, np.pi), (0, 2*np.pi)]),
        embedding=Embedding([phi2, theta2], map_exprs_S)
    )

    # Define transition maps between charts
    chart_N.add_transition(
        chart_S,
        forward={phi: sp.pi - phi2, theta: theta2},
        inverse={phi2: sp.pi - phi, theta2: theta}
    )
    chart_S.add_transition(
        chart_N,
        forward={phi2: sp.pi - phi, theta2: theta},
        inverse={phi: sp.pi - phi2, theta: theta2}
    )

    # Build the manifold
    M = SymbolicManifold('S2')
    M.add_chart(chart_N, default=True)
    M.add_chart(chart_S)

    # Create visualizations for the documentation
    print("\nGenerating visualizations...")
    plotter = ManifoldPlotter(chart_N)

    # 1. Coordinate grid visualization
    # Matplotlib version
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    plotter._plot_surface_matplotlib(resolution=50, color='lightblue', alpha=0.6, ax=ax)
    plotter.plot_coordinate_grid(ax=ax, grid_lines=15)
    ax.view_init(elev=30, azim=45)
    save_matplotlib_figure(fig, "figures/coordinate_grid.png")

    # Open3D version
    if _OPEN3D_AVAILABLE:
        Open3DPlotter.plot_embedding_surface(chart_N, resolution=50)

    # Define vector fields
    print("\n2. Vector Fields and Their Flows")
    vf_rot = VectorField(chart_N, [0, 1])  # Rotation around z-axis
    vf_grad = VectorField(chart_N, [-sp.sin(theta), sp.cos(theta)])  # Gradient-like field
    vf_radial = VectorField(chart_N, [1, 0])  # Radial field
    vector_fields = [
        (vf_rot, "rotational"),
        (vf_grad, "gradient"),
        (vf_radial, "radial")
    ]

    # Generate vector field visualizations
    for vf, name in vector_fields:
        # Matplotlib version
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        plotter._plot_surface_matplotlib(resolution=50, color='lightblue', alpha=0.3, ax=ax)
        VectorFieldPlotter(chart_N, vf).plot_matplotlib(ax=ax, resolution=30, scale=0.1)
        ax.view_init(elev=30, azim=45)
        save_matplotlib_figure(fig, f"figures/vector_field_{name}.png")

        # Create start points for flow visualization
        phi_values = np.linspace(np.pi/12, 11*np.pi/12, 12)  # More phi values
        theta_values = np.linspace(0, 2*np.pi, 16, endpoint=False)  # More theta values
        start_points = []
        for phi in phi_values:
            for theta in theta_values:
                if abs(phi - np.pi/2) > np.pi/8:  # Reduced gap around equator
                    start_points.append((phi, theta))
                elif len(start_points) % 2 == 0:  # Add more points near equator
                    start_points.append((phi, theta))

        # Flow visualization - Matplotlib
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        plotter._plot_surface_matplotlib(resolution=50, color='lightblue', alpha=0.3, ax=ax)
        VectorFieldPlotter(chart_N, vf).plot_matplotlib(ax=ax, resolution=30, scale=0.1)
        FlowPlotter(chart_N, vf).plot(start_points, steps=100, dt=0.05, ax=ax)
        ax.view_init(elev=30, azim=45)
        save_matplotlib_figure(fig, f"figures/flow_{name}.png")

    # Generate parallel transport visualizations
    print("\n3. Parallel Transport Examples")
    
    # Example 1: Along great circle (avoiding poles)
    # Matplotlib version
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    plotter._plot_surface_matplotlib(resolution=50, color='lightblue', alpha=0.3, ax=ax)
    ConnectionPlotter.plot_parallel_transport(
        chart_N.connection,
        [lambda t: np.pi/3, lambda t: t],  # Circle at constant latitude pi/3
        [(1, 0)],
        t_span=(0, 2*np.pi),
        ax=ax
    )
    ax.view_init(elev=30, azim=45)
    save_matplotlib_figure(fig, "figures/parallel_transport_great_circle.png")

    # Skip Open3D version since it's causing issues

    # Example 2: Along meridian (avoiding poles)
    # Matplotlib version
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    plotter._plot_surface_matplotlib(resolution=50, color='lightblue', alpha=0.3, ax=ax)
    ConnectionPlotter.plot_parallel_transport(
        chart_N.connection,
        [lambda t: np.pi/6 + t, lambda t: np.pi/4],  # Meridian at constant longitude pi/4
        [(0, 1)],
        t_span=(0, 2*np.pi/3),  # Shorter path to avoid poles
        ax=ax
    )
    ax.view_init(elev=30, azim=45)
    save_matplotlib_figure(fig, "figures/parallel_transport_meridian.png")

    # Generate LaTeX documentation
    print("\nGenerating comprehensive LaTeX documentation...")
    LatexDocumentGenerator.generate_sphere_documentation(
        chart_N,
        chart_N.connection,
        vector_fields,
        "sphere_geometry.tex"
    )
    print("Documentation generated in 'sphere_geometry.tex'")

    print("\nSphere demonstration complete.")

if __name__ == "__main__":
    main()
