import mujoco
import mujoco.viewer
import open3d as o3d
from pynput import keyboard

from vis_utils import *

# initialization

# load mujoco_model from XML file ; https://mujoco.readthedocs.io/en/stable/python.html
model = mujoco.MjModel.from_xml_path('/Users/lilith/Documents/Research/iprl/mujoco/realtime_depth/model/cube/cube_3x3x3.xml')
data = mujoco.MjData(model)

# initial camera pos (set to main which I used GUI for)
camera_name = 'main'
camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)

camera_pos = model.cam_pos[camera_id].copy()  # Make a copy!
print(f"Initial camera position: {camera_pos}")

# mujoco_renderer ; https://colab.research.google.com/github/google-deepmind/mujoco/blob/main/python/tutorial.ipynb#scrollTo=pvh47r97huS4
renderer = mujoco.Renderer(model)
mujoco.mj_forward(model, data)
rgb, depth = render_rgbd(renderer, data, camera_id=camera_id) # initial depth ; https://github.com/google-deepmind/mujoco/blob/main/python/mujoco/renderer.py#L182

# open3d (for depth visualization) ; https://www.open3d.org/docs/latest/python_api/open3d.visualization.Visualizer.html
vis = o3d.visualization.Visualizer()
vis.create_window(window_name='Point Cloud')
points, colors = get_point_cloud(depth, rgb, model, camera_id)
pcd = o3d.geometry.PointCloud()  # Empty point cloud
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(colors)
vis.add_geometry(pcd)

# background thread: keyboard listener ; https://pynput.readthedocs.io/en/latest/keyboard.html
# initial keyboard state
KEYS_PRESSED = {'w': False, 's': False, 'a': False, 'd': False}

def on_press(key):
    if key.char in KEYS_PRESSED:
        print('key {0} pressed'.format(
            key.char))
        KEYS_PRESSED[key.char] = True
    else:
        print('key {0} pressed not handled'.format(
            key))

def on_release(key):
    if key.char in KEYS_PRESSED:
        print('key {0} released'.format(
            key.char))
        KEYS_PRESSED[key.char] = False
    if key == keyboard.Key.esc:
        # Stop listener
        return False

listener = keyboard.Listener(
    on_press=on_press,
    on_release=on_release) # non-blocking
listener.start()
print("Listener Started!")


# main loop:
print("Instructions: W/A/S/D to move camera, ESC to quit")
running = True
while running and listener.is_alive():
    # update camdra based on keyboard
    update_camera_from_keyboard(KEYS_PRESSED, model, camera_id, camera_pos)
    
    # render point cloud 
    rgb, depth = render_rgbd(renderer, data, camera_id=camera_id)
    points, colors = get_point_cloud(depth, model, camera_id)
    
    # Open3D vis
    pcd.points = o3d.utility.Vector3dVector(points) # https://www.open3d.org/docs/latest/python_api/open3d.utility.Vector3dVector.html
    pcd.colors = o3d.utility.Vector3dVector(colors)
    vis.update_geometry(pcd)
    
    # refresh window
    vis.poll_events()
    vis.update_renderer()
    
    if not vis.poll_events():
        running = False

# clean
vis.destroy_window()
listener.stop()
print("Done...!")