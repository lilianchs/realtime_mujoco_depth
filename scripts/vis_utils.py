from numpy import np

def get_point_cloud(depth, rgb, model, camera_id):

    '''
    // cameras https://mujoco.readthedocs.io/en/stable/APIreference/APItypes.html
    mjtNum*   cam_pos;              // position rel. to body frame              (ncam x 3)
    mjtNum*   cam_quat;             // orientation rel. to body frame           (ncam x 4)
    mjtNum*   cam_poscom0;          // global position rel. to sub-com in qpos0 (ncam x 3)
    mjtNum*   cam_pos0;             // global position rel. to body in qpos0    (ncam x 3)
    mjtNum*   cam_mat0;             // global orientation in qpos0              (ncam x 9)
    int*      cam_orthographic;     // orthographic camera; 0: no, 1: yes       (ncam x 1)
    mjtNum*   cam_fovy;             // y field-of-view (ortho ? len : deg)      (ncam x 1)
    int*      cam_resolution;       // resolution: pixels [width, height]       (ncam x 2)
    float*    cam_intrinsic;        // [focal length; principal point]          (ncam x 4)
    mjtNum*   cam_user;             // user data                                (ncam x nuser_cam)
    '''
    resolution = model.cam_resolution[camera_id]
    width, height = resolution[0], resolution[1]

    intrinsics = model.cam_intrinsic[camera_id * 4:(camera_id + 1) * 4]
    fx, fy, cx, cy = intrinsics[0], intrinsics[1], intrinsics[2], intrinsics[3]

    # initialize pixel coord grd
    u = np.arange(width)
    v = np.arange(height)
    u, v = np.meshgrid(u, v)

    # depth -> 3D points w/ cam coords
    z = depth
    x = (u - cx) * z / f
    y = (v - cy) * z / f

    # necessary reshape
    points = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    colors = rgb.reshape(-1, 3) / 255.0

    return points, colors

def update_camera_from_keyboard(KEYS_PRESSED, model, camera_id, camera_pos, speed=0.05):

    if KEYS_PRESSED['w']:  # forward
        camera_pos[2] -= speed
    if KEYS_PRESSED['s']:  # backward
        camera_pos[2] += speed
    if KEYS_PRESSED['a']:  # left
        camera_pos[0] -= speed
    if KEYS_PRESSED['d']:  # right
        camera_pos[0] += speed
    
    model.cam_pos[camera_id] = camera_pos # update
    return camera_pos   

def render_rgbd(renderer, data, camera_id):
    # render using curr camera
    renderer.update_scene(data, camera=camera_id)
    rgb = renderer.render()
    
    renderer.enable_depth_rendering()
    renderer.update_scene(data, camera=camera_id)
    depth = renderer.render()
    renderer.disable_depth_rendering()
    
    return rgb, depth