import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Load the OBJ file
mesh = trimesh.load('../Data/uav_revise/tube_thick.obj')

# Extract vertices and faces
vertices = mesh.vertices
faces = mesh.faces

# Setup 3D plot
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

# Create polygons for faces
mesh_collection = Poly3DCollection(vertices[faces], alpha=0.2)
mesh_collection.set_facecolor((0.5, 0.7, 0.9))

ax.add_collection3d(mesh_collection)

# Auto scale to the mesh size
scale = vertices.flatten()
ax.auto_scale_xyz(scale, scale, scale)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
# ax.set_box_aspect([1,1,1])  # Equal aspect ratio

plt.show()