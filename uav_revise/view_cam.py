import mujoco
import mujoco.viewer
import cv2
import time
for i in (6,9,11):
    mjmodel = mujoco.MjModel.from_xml_path("../mujoco_uav/bitcraze_crazyflie_2/tmp_neural_{}.xml".format(i))
    mjdata = mujoco.MjData(mjmodel)
    v = mujoco.viewer.launch_passive(mjmodel, mjdata)
    time.sleep(0.1)
    renderer=mujoco.Renderer(mjmodel, 480, 640)
    
    renderer.update_scene(mjdata, "cam_traj_1")
    cam_photo_1 = renderer.render()
    renderer.update_scene(mjdata, "cam_traj_2")
    cam_photo_2 = renderer.render()
    renderer.update_scene(mjdata, "cam_traj_3")
    cam_photo_3 = renderer.render()

    cam_photo_1 = cv2.cvtColor(cam_photo_1, cv2.COLOR_RGB2BGR)
    cam_photo_2 = cv2.cvtColor(cam_photo_2, cv2.COLOR_RGB2BGR)
    cam_photo_3 = cv2.cvtColor(cam_photo_3, cv2.COLOR_RGB2BGR)
    cv2.imwrite("nn_{}_1.png".format(i), cam_photo_1)
    cv2.imwrite("nn_{}_2.png".format(i), cam_photo_2)
    cv2.imwrite("nn_{}_3.png".format(i), cam_photo_3)

    renderer.close()

    v.close()

