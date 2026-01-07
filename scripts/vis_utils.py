import mujoco
import numpy as np

def get_point_cloud(depth, rgb, model, camera_id):

    '''
    // cameras https://mujoco.readthedocs.io/en/stable/APIreference/APItypes.html
    mjtNum*   cam_pos;              // position rel. to body frame              (ncam x 3)
    mjtNum*   cam_fovy;             // y field-of-view (ortho ? len : deg)      (ncam x 1)
    int*      cam_resolution;       // resolution: pixels [width, height]       (ncam x 2)
    float*    cam_intrinsic;        // [focal length; principal point]          (ncam x 4)
    '''
    height, width = depth.shape 

    fovy = model.cam_fovy[camera_id]  # vertical FoV in degrees
    
    # focal length (pixels)
    fy = height / (2.0 * np.tan(np.radians(fovy) / 2.0))
    fx = fy
    
    # @ image center (principal point)
    cx = width / 2.0
    cy = height / 2.0

    # initialize pixel coord grd
    u = np.arange(width)
    v = np.arange(height)
    u, v = np.meshgrid(u, v)

    # depth -> 3D points w/ cam coords
    z = depth
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    # necessary reshape
    points = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    colors = rgb.reshape(-1, 3) / 255.0

    return points, colors

def update_camera_from_keyboard(KEYS_PRESSED, camera_pos, speed=0.01):
    changed = False
    if KEYS_PRESSED['w']:  # forward
        camera_pos[2] += speed
        changed = True
    if KEYS_PRESSED['s']:  # backward
        camera_pos[2] -= speed
        changed = True
    if KEYS_PRESSED['a']:  # left
        camera_pos[0] -= speed
        changed = True
    if KEYS_PRESSED['d']:  # right
        camera_pos[0] += speed
        changed = True

    return camera_pos, changed

def render_rgbd(renderer, data, camera_id):
    # render using curr camera
    renderer.update_scene(data, camera=camera_id)
    rgb = renderer.render()
    
    renderer.enable_depth_rendering()
    renderer.update_scene(data, camera=camera_id)
    depth = renderer.render()
    renderer.disable_depth_rendering()
    
    return rgb, depth