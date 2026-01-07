import mujoco
import numpy as np

def get_point_cloud(depth, rgb, model, camera_id):

    '''
    // cameras https://mujoco.readthedocs.io/en/stable/APIreference/APItypes.html
    mjtNum*   cam_pos;              // position rel. to body frame              (ncam x 3)
    mjtNum*   cam_mat0;             // global orientation in qpos0              (ncam x 9)
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

def update_camera_from_keyboard(KEYS_PRESSED, model, camera_pos, camera_id, speed=0.01):
    changed = False

    move_camera_frame = np.array([0.0, 0.0, 0.0]) # global

    if KEYS_PRESSED['p']:  # zoom in (forward)
        move_camera_frame[2] -= speed
        changed = True
    if KEYS_PRESSED['o']:  # zoom out (backward)
        move_camera_frame[2] += speed
        changed = True
    if KEYS_PRESSED['a']:  # left
        move_camera_frame[0] -= speed
        changed = True
    if KEYS_PRESSED['d']:  # right
        move_camera_frame[0] += speed
        changed = True
    if KEYS_PRESSED['w']:  # up
        move_camera_frame[1] += speed
        changed = True
    if KEYS_PRESSED['s']:  # down
        move_camera_frame[1] -= speed
        changed = True

    if changed:
        cam_mat = model.cam_mat0[camera_id].reshape(3, 3) # camera orientation in world frame
        print(f'camera matrix: {cam_mat}')
        move_world_frame = cam_mat @ move_camera_frame
        camera_pos += move_world_frame
        model.cam_pos[camera_id][:] = camera_pos


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