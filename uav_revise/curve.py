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

def add_spheres_in_circle_to_existing_xml(input_xml, output_xml, curve_points, num_spheres_per_point=8, circle_radius=0.3, sphere_radius=0.1, distance=0.2):
    """
    Add multiple spheres in a circular formation around each curve point in the YZ-plane and add them to the Mujoco XML.
    
    Parameters:
        input_xml (str): Path to the input Mujoco XML file.
        output_xml (str): Path where the modified XML will be saved.
        curve_points (np.ndarray): Array of (x, y, z) points for the curve.
        num_spheres_per_point (int): Number of spheres to place in a circle around each curve point.
        circle_radius (float): Radius of the circle along which to distribute spheres.
        sphere_radius (float): Radius of each individual sphere.
        distance (float): Distance offset between consecutive spheres in the circle.
    """
    # Parse the existing XML file
    tree = ET.parse(input_xml)
    root = tree.getroot()

    # Find the <worldbody> element
    worldbody = root.find(".//worldbody")
    if worldbody is None:
        raise ValueError("No <worldbody> element found in the input XML.")
    
    circles = []
    # Generate spheres for each curve point and add them to the XML
    for i, point in enumerate(curve_points):
        x, y, z = point
        
        circle = []
        # For each point, create multiple spheres arranged in a circle
        for j in range(num_spheres_per_point):
            angle = 2 * np.pi * j / num_spheres_per_point  # Angle for placing spheres in a circle
            y_offset = y + circle_radius * np.cos(angle)  # Y position along the circle
            z_offset = z + circle_radius * np.sin(angle)  # Z position along the circle

            point_cir = np.array((x,y_offset,z_offset))
            circle.append(point_cir)
            # Create a unique sphere name for each point in the circle
            sphere_name = f"sphere_{i}_{j}"

            # Create the body and geometry elements for the sphere
            body = ET.Element("body", name=sphere_name, pos=f"{x} {y_offset} {z_offset}")
            geom = ET.SubElement(body, "geom", type="sphere", size=str(sphere_radius), rgba="0.8 0.2 0.2 1")
            
            # Add the body element to the worldbody
            worldbody.append(body)

        circles.append(circle)

    # Pretty-print the modified XML
    xml_str = minidom.parseString(ET.tostring(root, encoding="unicode"))
    with open(output_xml, "w") as f:
        f.write(xml_str.toprettyxml(indent="  "))
    
    print(f"Modified Mujoco XML file saved as {output_xml}")
    return np.array(circles)
# Example usage
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
    input_xml = os.path.join("../mujoco_uav/bitcraze_crazyflie_2/","scene_clean.xml")
    outputput_xml = os.path.join("../mujoco_uav/bitcraze_crazyflie_2/","scene_revise.xml")
    circle_points = add_spheres_in_circle_to_existing_xml(input_xml=input_xml,output_xml=outputput_xml,curve_points=curve[::5],num_spheres_per_point=12, circle_radius=2.5)
    np.save("../Data/uav_revise/curve_points.npy",curve[::5])
    np.save("../Data/uav_revise/circle_points.npy",circle_points)
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
