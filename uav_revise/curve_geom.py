import numpy as np
import trimesh
import os
from curve import generate_piecewise_bezier_curve
from matplotlib import pyplot as plt

def build_tube_mesh_simple(curve_points, radius=0.02, circle_resolution=16):
    """Build a tube mesh along curve points with circles in YZ-plane."""
    vertices = []
    normals = []
    faces = []

    num_curve = len(curve_points)

    # YZ unit circle
    angles = np.linspace(0, 2*np.pi, circle_resolution, endpoint=False)
    circle_offsets = np.stack([
        np.zeros_like(angles),
        np.cos(angles),
        np.sin(angles)
    ], axis=1) * radius

    # Build vertices
    for p in curve_points:
        vertices.append(p + circle_offsets)
        normals.append(circle_offsets)

    vertices = np.vstack(vertices)
    normals = np.vstack(normals)

    # Build faces
    for i in range(num_curve - 1):
        for j in range(circle_resolution):
            curr = i * circle_resolution + j
            next = i * circle_resolution + (j + 1) % circle_resolution
            curr_next_ring = (i + 1) * circle_resolution + j
            next_next_ring = (i + 1) * circle_resolution + (j + 1) % circle_resolution

            faces.append([curr, next, curr_next_ring])
            faces.append([next, next_next_ring, curr_next_ring])

    # Build mesh
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    return vertices,normals,mesh

def build_open_thick_tube_mesh(curve_points, outer_radius=0.05, thickness=0.005, circle_resolution=16):
    """Build an open, thick-walled tube mesh along the curve points."""
    vertices = []
    faces = []

    num_curve = len(curve_points)
    inner_radius = outer_radius - thickness

    # YZ unit circle
    angles = np.linspace(0, 2*np.pi, circle_resolution, endpoint=False)
    circle_outer = np.stack([
        np.zeros_like(angles),
        np.cos(angles),
        np.sin(angles)
    ], axis=1) * outer_radius

    circle_inner = np.stack([
        np.zeros_like(angles),
        np.cos(angles),
        np.sin(angles)
    ], axis=1) * inner_radius

    # Build vertices
    for p in curve_points:
        vertices.append(p + circle_outer)  # outer ring
    for p in curve_points:
        vertices.append(p + circle_inner)  # inner ring

    vertices = np.vstack(vertices)

    # Connect outer surface
    for i in range(num_curve - 1):
        for j in range(circle_resolution):
            curr = i * circle_resolution + j
            next = i * circle_resolution + (j + 1) % circle_resolution
            curr_next_ring = (i + 1) * circle_resolution + j
            next_next_ring = (i + 1) * circle_resolution + (j + 1) % circle_resolution

            faces.append([curr, next, curr_next_ring])
            faces.append([next, next_next_ring, curr_next_ring])

    # Connect inner surface (flip normals)
    offset = num_curve * circle_resolution
    for i in range(num_curve - 1):
        for j in range(circle_resolution):
            curr = offset + i * circle_resolution + j
            next = offset + i * circle_resolution + (j + 1) % circle_resolution
            curr_next_ring = offset + (i + 1) * circle_resolution + j
            next_next_ring = offset + (i + 1) * circle_resolution + (j + 1) % circle_resolution

            faces.append([curr_next_ring, next, curr])
            faces.append([curr_next_ring, next_next_ring, next])

    # Connect side walls (inner to outer circles)
    for i in [0, num_curve - 1]:  # only first and last cross-section
        for j in range(circle_resolution):
            outer_curr = i * circle_resolution + j
            outer_next = i * circle_resolution + (j + 1) % circle_resolution
            inner_curr = offset + i * circle_resolution + j
            inner_next = offset + i * circle_resolution + (j + 1) % circle_resolution

            if i == 0:
                # start side
                faces.append([outer_curr, inner_curr, inner_next])
                faces.append([outer_curr, inner_next, outer_next])
            else:
                # end side
                faces.append([inner_curr, outer_curr, outer_next])
                faces.append([inner_curr, outer_next, inner_next])

    # Build mesh
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    return mesh

def save_mesh_as_obj(mesh, filename):
    """Save the mesh as .obj."""
    mesh.export(filename)
    print(f"Mesh saved to {filename}")

def sample_half_sphere_x(num_points=1000, radius=1.0, side='left'):
    """
    Generate points on the surface of a sphere where x < 0 (left) or x > 0 (right).
    """
    total = 0
    samples = []

    while len(samples) < num_points:
        # Sample spherical coordinates
        theta = np.random.uniform(0, 2 * np.pi)
        phi = np.arccos(np.random.uniform(-1, 1))  # for uniform sampling

        # Convert to Cartesian
        x = radius * np.sin(phi) * np.cos(theta)
        y = radius * np.sin(phi) * np.sin(theta)
        z = radius * np.cos(phi)

        # Keep only based on x condition
        if (side == 'left' and x < 0) or (side == 'right' and x > 0):
            samples.append([x, y, z])
        
        total += 1

    return np.array(samples)

if __name__ == "__main__":
    start = (0, 0, 5)
    end = (15, 0, 10)

    start = np.array(start, dtype=float)
    end = np.array(end, dtype=float)
    straightness = 5.0

    control_points = [
        [0, 0, 5],   # P0
        # [1, 2, 0],   # P1
        [5, 8, 3],   # P2
        # [4, 1, 0],   # P3
        [11, -10, 5],   # P4
        # [7, 2, 0],   # P5
        # [9, 3, 0],   # P6
        [15, 0, 10]   # P7
    ]
    control_points = np.array(control_points, dtype=float)
    # Generate the curve with more turning points
    curve = generate_piecewise_bezier_curve(control_points, num_points_per_segment=100)
    np.save("../Data/uav_revise/curve_full.npy",curve)
    circle_points, normals, mesh = build_tube_mesh_simple(curve,radius = 2.5,circle_resolution=64)
    start_normals = sample_half_sphere_x(num_points=400, radius=1.0, side = "left")
    start_sphere = 2.5 * start_normals + np.array([0, 0, 5])
    end_normals = sample_half_sphere_x(num_points=400, radius=1.0, side = "right")
    end_sphere = 2.5 * end_normals + np.array([15, 0, 10])
    circle_points = np.concatenate((start_sphere,circle_points,end_sphere))
    normals = np.concatenate((start_normals,normals,end_normals))
    print(circle_points.shape)
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(circle_points[:,0], circle_points[:,1], circle_points[:,2], s=1)
    ax.set_title("Half-sphere (x < 0)")
    ax.set_box_aspect([1,1,1])
    plt.show()
    np.save("../Data/uav_revise/circle_full.npy",circle_points)
    np.save("../Data/uav_revise/normal_full.npy",normals)

    exit()
    save_mesh_as_obj(mesh, "../Data/uav_revise/tube_simple.obj")

    mesh = build_open_thick_tube_mesh(curve,outer_radius = 2.5,circle_resolution=16,thickness=0.2)
    save_mesh_as_obj(mesh, "../Data/uav_revise/tube_thick.obj")
