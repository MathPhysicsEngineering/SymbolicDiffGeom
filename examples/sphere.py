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
from sympy import lambdify

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

# Create a single figure for each vector field with its flow
def plot_vector_field_with_flow(vf, title, t_span):
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')

    # Remove all background elements
    ax.set_axis_off()
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.grid(False)

    # Plot the manifold first (most transparent)
    coords = chart_N.coords
    emb_func = lambdify(coords, chart_N.embedding.map_exprs, 'numpy')
    u = np.linspace(chart_N.domain.bounds[0][0], chart_N.domain.bounds[0][1], 50)
    v = np.linspace(chart_N.domain.bounds[1][0], chart_N.domain.bounds[1][1], 50)
    U, V = np.meshgrid(u, v)
    xyz = np.array([emb_func(ux, vx) for ux, vx in zip(U.flatten(), V.flatten())])
    X = xyz[:, 0].reshape(U.shape)
    Y = xyz[:, 1].reshape(U.shape)
    Z = xyz[:, 2].reshape(U.shape)
    ax.plot_surface(X, Y, Z, color='lightblue', alpha=0.15)

    # Generate flow start points
    phi_values = np.linspace(np.pi/12, 11*np.pi/12, 12)
    theta_values = np.linspace(0, 2*np.pi, 8, endpoint=False)

    start_points = []
    for phi in phi_values:
        for theta in theta_values:
            if abs(phi - np.pi/2) > np.pi/6:
                start_points.append((phi, theta))
            elif len(start_points) % 2 == 0:
                start_points.append((phi, theta))

    # Add near-polar curves
    for theta in np.linspace(0, 2*np.pi, 12, endpoint=False):
        start_points.append((np.pi/24, theta))
        start_points.append((23*np.pi/24, theta))

    # Plot flow lines next
    FlowPlotter(chart_N, vf).plot(start_points, t_span=t_span, steps=200, ax=ax, color='blue', alpha=0.8)

    # Plot vector field on top
    VectorFieldPlotter(chart_N, vf).plot_matplotlib(ax=ax, resolution=25, scale=0.08, color='red', alpha=0.8)

    ax.view_init(elev=30, azim=45)
    ax.set_title(title)
    plt.show()

# Plot each vector field with its flow
print("\nPlotting rotational vector field with flows:")
plot_vector_field_with_flow(vf_rot, "Rotational Vector Field and Flow Lines", (0, 4*np.pi))

print("\nPlotting gradient vector field with flows:")
plot_vector_field_with_flow(vf_grad, "Gradient Vector Field and Flow Lines", (0, 2))

print("\nPlotting radial vector field with flows:")
plot_vector_field_with_flow(vf_radial, "Radial Vector Field and Flow Lines", (0, 1))

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

# Remove all background elements
ax.set_axis_off()
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.grid(False)

# Plot the manifold first
coords = chart_N.coords
emb_func = lambdify(coords, chart_N.embedding.map_exprs, 'numpy')
u = np.linspace(chart_N.domain.bounds[0][0], chart_N.domain.bounds[0][1], 50)
v = np.linspace(chart_N.domain.bounds[1][0], chart_N.domain.bounds[1][1], 50)
U, V = np.meshgrid(u, v)
xyz = np.array([emb_func(ux, vx) for ux, vx in zip(U.flatten(), V.flatten())])
X = xyz[:, 0].reshape(U.shape)
Y = xyz[:, 1].reshape(U.shape)
Z = xyz[:, 2].reshape(U.shape)
ax.plot_surface(X, Y, Z, color='lightblue', alpha=0.15)

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

# Visualize scalar curvature as a scalar field
print("\nVisualizing scalar curvature:")
# Create scalar field for Gaussian curvature
K_field = sp.simplify(K)  # Simplify the expression
print("\nGaussian curvature expression:", K_field)

# Plot scalar field
fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(111, projection='3d')

# Remove all background elements
ax.set_axis_off()
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.grid(False)

# Plot scalar field
ScalarFieldVisualizer.plot_scalar_field(chart_N, K_field, ax=ax, resolution=50, cmap='coolwarm')
ax.set_title("Gaussian Curvature")
plt.show()

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
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Remove all background elements
    ax.set_axis_off()
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.grid(False)
    
    # Plot scalar field
    ScalarFieldVisualizer.plot_scalar_field(chart_N, field, ax=ax, resolution=50)
    ax.view_init(elev=30, azim=45)
    plt.show()

print("\nSphere demonstration complete.")
