# sphere.py — Comprehensive S^2 demonstration
"""
Demo script for S² using symbolic_diff_geom.
Covers: chart definitions, induced metric, Levi-Civita connection,
vector fields, flows, geodesics, exponential map, distance,
geodesic grids, manifold transitions, parallel transport, and curvature.
"""
import sympy as sp
import numpy as np
from charts import Chart, SymbolicManifold, Embedding, BoxDomain
from vector_fields import VectorField
from visualization import (
    ChartPlotter,
    ManifoldPlotter,
    VectorFieldPlotter,
    FlowPlotter,
    ConnectionPlotter,
    ScalarFieldVisualizer,
    _OPEN3D_AVAILABLE
)
import matplotlib.pyplot as plt

print("1. Chart Construction and Basic Properties")
print("----------------------------------------")

# Coordinate symbols for north and south charts
phi, theta = sp.symbols('phi theta', positive=True)
phi2, theta2 = sp.symbols('phi2 theta2', positive=True)

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

print("\n2. Metric Structure")
print("------------------")
# Print intrinsic data for each chart
for chart in [chart_N, chart_S]:
    print(f"\nChart: {chart.name}")
    print("Metric g_ij:")
    sp.pprint(chart.metric.g)
    print("\nChristoffel symbols Γ^i_{jk}:")
    for i in range(chart.dim):
        for j in range(chart.dim):
            for k in range(chart.dim):
                if chart.Gamma[i][j][k] != 0:
                    print(f"Γ^{i}_{j}{k} =", chart.Gamma[i][j][k])

print("\n3. Vector Fields and Flows")
print("-------------------------")
# Define various vector fields
vf_rot = VectorField(chart_N, [0, 1])  # Rotation around z-axis
vf_grad = VectorField(chart_N, [-sp.sin(theta), sp.cos(theta)])  # Gradient-like field
print("Rotational vector field:", vf_rot)
print("Gradient-like vector field:", vf_grad)

# Test vector field evaluation
test_point = (np.pi/4, np.pi/2)
print(f"\nEvaluating vector fields at {test_point}:")
print("Rotation field:", vf_rot.evaluate(test_point))
print("Gradient field:", vf_grad.evaluate(test_point))

print("\n4. Visualization Tests")
print("--------------------")
plotter = ManifoldPlotter(chart_N)

print("\nTesting matplotlib backend:")
plotter.set_backend('matplotlib')
# Plot surface with coordinate grid
plotter.plot_surface(color='lightblue', alpha=0.6, with_grid=True, grid_lines=20)

if _OPEN3D_AVAILABLE:
    print("\nTesting Open3D backend:")
    plotter.set_backend('open3d')
    plotter.plot_surface(color=[0.7, 0.7, 0.9])

print("\n4a. Vector Fields and Their Flows")
print("--------------------------------")
# Define some interesting vector fields
vf_rot = VectorField(chart_N, [0, 1])  # Rotation around z-axis
vf_grad = VectorField(chart_N, [-sp.sin(theta), sp.cos(theta)])  # Gradient-like field
vf_radial = VectorField(chart_N, [1, 0])  # Radial field

# Create figure for rotational vector field
print("\nPlotting rotational vector field with flows:")
fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(111, projection='3d')

# Plot the manifold with coordinate grid
plotter.set_backend('matplotlib')
plotter._plot_surface_matplotlib(resolution=50, color='lightblue', alpha=0.3)
plotter.plot_coordinate_grid(grid_lines=10, color='gray', alpha=0.2)

# Add vector field with shorter, denser arrows
VectorFieldPlotter(chart_N, vf_rot).plot_matplotlib(ax=ax, resolution=25, scale=0.08, color='red', alpha=0.7)

# Add flow lines with more points for smoother curves
start_points = [
    (np.pi/4, 0),
    (np.pi/3, np.pi/2),
    (np.pi/2, np.pi),
    (2*np.pi/3, 3*np.pi/2)
]
FlowPlotter(chart_N, vf_rot).plot(start_points, t_span=(0, 4*np.pi), steps=200, ax=ax, color='blue', alpha=0.8)

# Improve the view
ax.view_init(elev=30, azim=45)
ax.set_title("Rotational Vector Field and Flow Lines")
plt.show()

# Create figure for gradient vector field
print("\nPlotting gradient vector field with flows:")
fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(111, projection='3d')

# Plot the manifold with coordinate grid
plotter._plot_surface_matplotlib(resolution=50, color='lightblue', alpha=0.3)
plotter.plot_coordinate_grid(grid_lines=10, color='gray', alpha=0.2)

# Add vector field with shorter, denser arrows
VectorFieldPlotter(chart_N, vf_grad).plot_matplotlib(ax=ax, resolution=25, scale=0.08, color='red', alpha=0.7)

# Add flow lines with more points for smoother curves
start_points = [
    (np.pi/6, 0),
    (np.pi/4, np.pi/2),
    (np.pi/3, np.pi),
    (np.pi/2, 3*np.pi/2)
]
FlowPlotter(chart_N, vf_grad).plot(start_points, t_span=(0, 2), steps=200, ax=ax, color='blue', alpha=0.8)

# Improve the view
ax.view_init(elev=30, azim=45)
ax.set_title("Gradient Vector Field and Flow Lines")
plt.show()

# Create figure for radial vector field
print("\nPlotting radial vector field with flows:")
fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(111, projection='3d')

# Plot the manifold with coordinate grid
plotter._plot_surface_matplotlib(resolution=50, color='lightblue', alpha=0.3)
plotter.plot_coordinate_grid(grid_lines=10, color='gray', alpha=0.2)

# Add vector field with shorter, denser arrows
VectorFieldPlotter(chart_N, vf_radial).plot_matplotlib(ax=ax, resolution=25, scale=0.08, color='red', alpha=0.7)

# Add flow lines with more points for smoother curves
start_points = [
    (np.pi/6, 0),
    (np.pi/6, np.pi/2),
    (np.pi/6, np.pi),
    (np.pi/6, 3*np.pi/2)
]
FlowPlotter(chart_N, vf_radial).plot(start_points, t_span=(0, 1), steps=200, ax=ax, color='blue', alpha=0.8)

# Improve the view
ax.view_init(elev=30, azim=45)
ax.set_title("Radial Vector Field and Flow Lines")
plt.show()

print("\n5. Geodesics and Parallel Transport")
print("---------------------------------")
# Test geodesics from various initial conditions
initial_conditions = [
    ((np.pi/4, 0), (0, 1)),      # Horizontal circle
    ((np.pi/2, 0), (1, 0)),      # Great circle
    ((np.pi/4, np.pi/4), (1, 1))  # Generic geodesic
]

# Create figure for geodesics
fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(111, projection='3d')

# Plot the manifold with coordinate grid first
plotter._plot_surface_matplotlib(resolution=30, color='lightblue', alpha=0.3)
plotter.plot_coordinate_grid(grid_lines=10, color='gray', alpha=0.2)

# Plot geodesics
FlowPlotter.plot_geodesics(chart_N.connection, initial_conditions, t_span=(0, 2*np.pi), ax=ax)

# Improve the view
ax.view_init(elev=30, azim=45)
plt.show()

print("\n6. Curvature Analysis")
print("--------------------")
# Get the Levi-Civita connection
connection = chart_N.connection

# Compute and display Riemann curvature tensor
R = connection.riemann_tensor()
print("Riemann curvature tensor components (non-zero):")
for i in range(2):
    for j in range(2):
        for k in range(2):
            for l in range(2):
                if R[i][j][k][l] != 0:
                    print(f"R^{i}_{j}{k}{l} =", R[i][j][k][l])

# Compute and display Ricci tensor
Ric = connection.ricci_tensor()
print("\nRicci tensor:")
sp.pprint(Ric)

# Compute scalar curvature
K = connection.scalar_curvature()
print("\nScalar curvature:", K)

print("\n7. Interactive Features")
print("---------------------")
print("Starting interactive geodesic visualization...")
plotter.set_backend('matplotlib')
plotter.interactive_geodesics()

print("\n8. Scalar Field Visualization")
print("---------------------------")
# Create some interesting scalar fields on the sphere
scalar_fields = [
    sp.sin(phi)*sp.cos(theta),  # x coordinate
    sp.cos(phi),                # z coordinate
    sp.sin(phi)**2              # height squared
]
field_names = ['x coordinate', 'z coordinate', 'height squared']

for field, name in zip(scalar_fields, field_names):
    print(f"\nVisualizing scalar field: {name}")
    ScalarFieldVisualizer.plot_scalar_field(chart_N, field)

print("\nSphere demonstration complete.")
