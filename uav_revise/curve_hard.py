import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import xml.etree.ElementTree as ET
import os
from xml.dom import minidom

def cubic_bezier(t, P0, P1, P2, P3):
    """Generates a cubic Bézier curve given four control points."""
    return (1 - t)**3 * P0 + 3 * (1 - t)**2 * t * P1 + 3 * (1 - t) * t**2 * P2 + t**3 * P3

# Generate a piecewise Bézier curve (multiple segments)
def generate_piecewise_bezier_curve(control_points, num_points_per_segment=100):
    """
    Generate a piecewise Bézier curve with multiple turning points.
    
    Parameters:
        control_points (list of lists): List of control points, where each set of 4 points defines one cubic Bézier segment.
        num_points_per_segment (int): Number of points to generate per Bézier segment.
        
    Returns:
        np.ndarray: Array of points along the generated piecewise Bézier curve.
    """
    curve = []
    for i in range(len(control_points) - 3):  # Ensure there are 4 points per segment
        # Control points for this segment
        P0, P1, P2, P3 = control_points[i], control_points[i + 1], control_points[i + 2], control_points[i + 3]
        
        # Generate points for this segment
        segment_points = np.array([cubic_bezier(t / (num_points_per_segment - 1), P0, P1, P2, P3) for t in range(num_points_per_segment)])
        
        # Add the segment points to the curve
        curve.append(segment_points)
    
    return np.vstack(curve)

def compute_tangents(curve_points):
    """Compute the tangent (first derivative) of the curve at each point."""
    tangents = np.gradient(curve_points, axis=0)
    tangents = tangents / np.linalg.norm(tangents, axis=1, keepdims=True)
    return tangents

def get_local_frame(tangent):
    """Return an orthonormal basis (x, y, z), where z = tangent."""
    z = tangent / np.linalg.norm(tangent)
    # Pick a non-parallel vector to construct basis
    arbitrary = np.array([0, 0, 1]) if abs(z[2]) < 0.9 else np.array([0, 1, 0])
    x = np.cross(arbitrary, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return x, y, z

def add_spheres_along_curve(input_xml, output_xml, curve_points, num_spheres_per_circle=8, circle_radius=0.3, sphere_radius=0.1, tangents = None):
    """
    Add spheres around each point in a circle perpendicular to the curve tangent.
    """
    tree = ET.parse(input_xml)
    root = tree.getroot()
    worldbody = root.find(".//worldbody")
    if worldbody is None:
        raise ValueError("No <worldbody> found in the XML.")

    # tangents = compute_tangents(curve_points)
    circles = []
    for i, (p, t) in enumerate(zip(curve_points, tangents)):
        x_axis, y_axis, _ = get_local_frame(t)
        circle = []
        for j in range(num_spheres_per_circle):
            theta = 2 * np.pi * j / num_spheres_per_circle
            offset = (np.cos(theta) * x_axis + np.sin(theta) * y_axis) * circle_radius
            position = p + offset
            circle.append(position)

            body = ET.Element("body", name=f"sphere_{i}_{j}", pos="%.5f %.5f %.5f" % tuple(position))
            ET.SubElement(body, "geom", type="sphere", size=str(sphere_radius), rgba="0.3 0.5 0.8 1")
            worldbody.append(body)
        circles.append(circle)


    # Pretty print the XML
    pretty_xml = minidom.parseString(ET.tostring(root, encoding="unicode")).toprettyxml(indent="  ")
    with open(output_xml, "w") as f:
        f.write(pretty_xml)
    print(f"Saved modified XML to {output_xml}")

    return circles

# Example usage
if __name__ == "__main__":
    start = (0, 0, 1)
    end = (10, 0, 5)

    start = np.array(start, dtype=float)
    end = np.array(end, dtype=float)
    straightness = 5.0

    control_points = [
        [0, 0, 5],   # P0
        # [1, 2, 0],   # P1
        [5, 16, 5],   # P2
        # [4, 1, 0],   # P3
        [10, -16, 8],   # P4
        # [7, 2, 0],   # P5
        # [9, 3, 0],   # P6
        [15, 0, 10]   # P7
    ]
    control_points = np.array(control_points, dtype=float)
    # Generate the curve with more turning points
    curve = generate_piecewise_bezier_curve(control_points, num_points_per_segment=100)
    input_xml = os.path.join("../mujoco_uav/bitcraze_crazyflie_2/","scene_clean.xml")
    outputput_xml = os.path.join("../mujoco_uav/bitcraze_crazyflie_2/","scene_revise_hard.xml")
    tangents = compute_tangents(curve)
    circle_points = add_spheres_along_curve(input_xml=input_xml,output_xml=outputput_xml,curve_points=curve[::5],num_spheres_per_circle=10, circle_radius=2.5,tangents=tangents[::5])
    np.save("../Data/uav_revise/curve_points_hard.npy",curve[::5])
    np.save("../Data/uav_revise/circle_points_hard.npy",circle_points)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(curve[:, 0], curve[:, 1], curve[:, 2], label="Bezier Curve", linewidth=2)
    ax.scatter(*zip(start, end), color='red', label="Start/End", s=50)

    # Set aspect ratio equal
    def set_axes_equal(ax):
        """Set 3D plot axes to equal scale."""
        ranges = np.array([
            ax.get_xlim3d(),
            ax.get_ylim3d(),
            ax.get_zlim3d()
        ])
        spans = ranges[:, 1] - ranges[:, 0]
        centers = np.mean(ranges, axis=1)
        radius = 0.5 * max(spans)
        new_ranges = np.array([centers - radius, centers + radius]).T
        ax.set_xlim3d(new_ranges[0])
        ax.set_ylim3d(new_ranges[1])
        ax.set_zlim3d(new_ranges[2])

    set_axes_equal(ax)

    # Optional: turn off axis ticks for clean look
    # ax.set_xticks([])
    # ax.set_yticks([])
    # ax.set_zticks([])

    # Label axes
    ax.set_xlabel("X", labelpad=10)
    ax.set_ylabel("Y", labelpad=10)
    ax.set_zlabel("Z", labelpad=10)

    # Nice view angle
    ax.view_init(elev=30, azim=120)

    # Title and legend
    ax.set_title("3D Quadratic Bézier Curve with Controlled Curvature")
    ax.legend()
    plt.tight_layout()
    plt.show()
