import mujoco
import numpy as np
import mujoco.viewer
import xml.etree.ElementTree as ET
import os

def add_trajectories_to_xml(trajectories, colors, base_xml_path, target_xml_path):
    """
    Adds multiple trajectories as line connectors to a MuJoCo XML file, with each trajectory grouped into a separate body.

    Args:
        trajectories (list of np.ndarray): A list of numpy arrays, each of shape (N, 3), representing the trajectory points.
        colors (list of str): A list of RGBA color strings (e.g., '1 0 0 1') for each trajectory.
        base_xml_path (str): Path to the base XML file.
        target_xml_path (str): Path to save the modified XML file.
    """
    import xml.etree.ElementTree as ET
    import numpy as np

    # Validate inputs
    if len(trajectories) != len(colors):
        raise ValueError("The number of trajectories must match the number of colors.")

    # Load the base XML file
    base_xml = ET.parse(base_xml_path)
    root = base_xml.getroot()

    # Find the worldbody element
    worldbody = root.find('worldbody')
    if worldbody is None:
        raise ValueError("The XML file does not contain a 'worldbody' element.")

    # Add a body for each trajectory
    for idx, (trajectory, color) in enumerate(zip(trajectories, colors)):
        # Create a new body element for the trajectory
        body = ET.Element('body', {
            'name': f'trajectory_{idx}',
            'pos': '0 0 0'
        })

        # Add line connectors (geoms) to the body
        for i in range(len(trajectory) - 1):
            start = trajectory[i]
            end = trajectory[i + 1]

            # Create a geom element for the line
            line = ET.Element('geom', {
                'type': 'capsule',
                'fromto': f"{start[0]} {start[1]} {start[2]} {end[0]} {end[1]} {end[2]}",
                'size': '0.1',  # Thickness of the line
                'rgba': color,  # Color for this trajectory
            })
            body.append(line)

        # Append the body to the worldbody
        worldbody.append(body)

    # Save the modified XML to the target path
    base_xml.write(target_xml_path)

# traj_ids = [45] #0,26,38,45
# colors = ['0 1 0.0 1']  # RGBA colors for each trajectory
# traj_files = [os.path.join("../Data/uav_revise/traj_poly", f"traj_{i}.npy") for i in traj_ids]
# trajs = [np.load(f) for f in traj_files]
# trajectories = [traj[:, :3].reshape(-1,3) for traj in trajs]  # Extract only the first three columns (x, y, z)
# base_xml_path = os.path.join("../mujoco_uav/bitcraze_crazyflie_2/", "scene_revise_geom.xml")
# target_path = os.path.join("../mujoco_uav/bitcraze_crazyflie_2/", "tmp_poly_45.xml")

# add_trajectories_to_xml(trajectories, colors, base_xml_path, target_path)

traj_ids = [11] #0,6,9,11
colors = ['0 1 0.0 1'] 
# colors = ['1 0 0 1', '0.5 0.0 1.0 1', '1 0.2 0.0 1', '0 1 0.0 1']  # RGBA colors for each trajectory
traj_files = [os.path.join("../Data/uav_revise/traj_neural", f"traj_{i}.npy") for i in traj_ids]
trajs = [np.load(f) for f in traj_files]
trajectories = [traj[:, :3].reshape(-1,3) for traj in trajs]  # Extract only the first three columns (x, y, z)
base_xml_path = os.path.join("../mujoco_uav/bitcraze_crazyflie_2/", "scene_revise_geom.xml")
target_path = os.path.join("../mujoco_uav/bitcraze_crazyflie_2/", "tmp_neural_11.xml")

add_trajectories_to_xml(trajectories, colors, base_xml_path, target_path)