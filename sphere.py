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
from sympy.printing.latex import latex

print("\n3. Vector Fields, Flows, and Exponential Maps")
print("-----------------------------------------")
# Define various vector fields
vf_rot = VectorField(chart_N, [0, 1])  # Rotation around z-axis
vf_grad = VectorField(chart_N, [-sp.sin(theta), sp.cos(theta)])  # Gradient-like field
vf_radial = VectorField(chart_N, [1, 0])  # Radial field

print("Computing symbolic flows where possible...")
# For rotational vector field (can be computed explicitly)
t = sp.Symbol('t')
# Flow of rotational field is just rotation by angle t
flow_rot = {
    phi: phi,
    theta: (theta + t) % (2*sp.pi)
}
print("\nRotational vector field flow:")
print("φ(t) =", latex(flow_rot[phi]))
print("θ(t) =", latex(flow_rot[theta]))

# For gradient field
print("\nGradient vector field:")
print("Components:", latex([-sp.sin(theta), sp.cos(theta)]))

# Compute exponential map for special cases
print("\nComputing exponential map for special cases...")
# At north pole (phi=0), the exponential map is particularly simple
exp_north = {
    phi: 't',
    theta: 'θ_0'  # Initial angle is preserved
}
print("\nExponential map at north pole:")
print("φ(t) =", latex(exp_north[phi]))
print("θ(t) =", latex(exp_north[theta]))

# Create figure for rotational vector field with manifold
print("\nPlotting rotational vector field with flows:")
fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(111, projection='3d')

# Plot the manifold with coordinate grid
plotter.set_backend('matplotlib')
plotter._plot_surface_matplotlib(resolution=50, color='lightblue', alpha=0.4)
plotter.plot_coordinate_grid(grid_lines=15, color='gray', alpha=0.2)

# Add vector field
VectorFieldPlotter(chart_N, vf_rot).plot_matplotlib(ax=ax, resolution=20, scale=0.15, color='red', alpha=0.7)

# Add flow lines
start_points = [
    (np.pi/4, 0),
    (np.pi/3, np.pi/2),
    (np.pi/2, np.pi),
    (2*np.pi/3, 3*np.pi/2)
]
FlowPlotter(chart_N, vf_rot).plot(start_points, t_span=(0, 4*np.pi), ax=ax, color='blue', alpha=0.8)

# Improve the view
ax.view_init(elev=30, azim=45)
ax.set_title("Rotational Vector Field and Flow Lines")
plt.show()

# Create figure for gradient vector field with manifold
print("\nPlotting gradient vector field with flows:")
fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(111, projection='3d')

# Plot the manifold with coordinate grid
plotter._plot_surface_matplotlib(resolution=50, color='lightblue', alpha=0.4)
plotter.plot_coordinate_grid(grid_lines=15, color='gray', alpha=0.2)

# Add vector field
VectorFieldPlotter(chart_N, vf_grad).plot_matplotlib(ax=ax, resolution=20, scale=0.15, color='red', alpha=0.7)

# Add flow lines
start_points = [
    (np.pi/6, 0),
    (np.pi/4, np.pi/2),
    (np.pi/3, np.pi),
    (np.pi/2, 3*np.pi/2)
]
FlowPlotter(chart_N, vf_grad).plot(start_points, t_span=(0, 2), ax=ax, color='blue', alpha=0.8)

# Improve the view
ax.view_init(elev=30, azim=45)
ax.set_title("Gradient Vector Field and Flow Lines")
plt.show()

# Create figure for radial vector field with manifold
print("\nPlotting radial vector field with flows:")
fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(111, projection='3d')

# Plot the manifold with coordinate grid
plotter._plot_surface_matplotlib(resolution=50, color='lightblue', alpha=0.4)
plotter.plot_coordinate_grid(grid_lines=15, color='gray', alpha=0.2)

# Add vector field
VectorFieldPlotter(chart_N, vf_radial).plot_matplotlib(ax=ax, resolution=20, scale=0.15, color='red', alpha=0.7)

# Add flow lines
start_points = [
    (np.pi/6, 0),
    (np.pi/6, np.pi/2),
    (np.pi/6, np.pi),
    (np.pi/6, 3*np.pi/2)
]
FlowPlotter(chart_N, vf_radial).plot(start_points, t_span=(0, 1), ax=ax, color='blue', alpha=0.8)

# Improve the view
ax.view_init(elev=30, azim=45)
ax.set_title("Radial Vector Field and Flow Lines")
plt.show() 