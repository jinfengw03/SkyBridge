import numpy as np
from typing import Tuple, List
from skylink_v2x.core.autonomy_stack.perception.sensor_transformer import SensorTransformer

# Projection utils leveraging project SensorTransformer for consistent conventions.

def world_points_to_image(points_w: np.ndarray, camera_pose, K: np.ndarray) -> np.ndarray:
    """Project world points (N,3) into image pixels using camera pose and intrinsics K.
    camera_pose can be either:
      - list/tuple [x,y,z, roll, yaw, pitch]
      - a 4x4 numpy array representing the camera-to-world transform (T_cam_world)
    Returns (N,2) with subpixel floats; caller clips to image bounds.
    """
    assert points_w.shape[1] == 3
    # Build homogeneous and transform to camera sensor frame
    homo = np.hstack([points_w, np.ones((points_w.shape[0], 1))]).T  # (4,N)
    if isinstance(camera_pose, (list, tuple)):
        sensor_pts = SensorTransformer.world_to_sensor_coords(homo, camera_pose)  # (4,N)
    else:
        # Assume a 4x4 T_cam_world
        T_cam_world = np.array(camera_pose, dtype=np.float32)
        T_world_to_sensor = np.linalg.inv(T_cam_world)
        sensor_pts = T_world_to_sensor @ homo
    cam_coords = np.apply_along_axis(SensorTransformer.convert_ue4_to_opencv, 0, sensor_pts[:3, :])  # (3,N)
    # Pinhole projection
    proj = (K @ cam_coords).T  # (N,3)
    # keep points in front of camera
    z = proj[:, 2]
    valid = z > 1e-3
    proj = proj[valid]
    proj[:, 0] /= proj[:, 2]
    proj[:, 1] /= proj[:, 2]
    return proj[:, :2]


def bev_mask_to_drone_mask(bev_mask: np.ndarray, grid_origin_xy: Tuple[float, float], res: float, camera_pose, K: np.ndarray, img_w: int, img_h: int, ground_z: float = 0.0) -> np.ndarray:
    """Project a BEV mask (H,W) on world XY (z=ground_z) into a camera mask image (img_h, img_w).
    """
    H, W = bev_mask.shape
    rs, cs = np.where(bev_mask > 0)
    if rs.size == 0:
        return np.zeros((img_h, img_w), dtype=np.uint8)
    xs = grid_origin_xy[0] + (cs + 0.5) * res
    ys = grid_origin_xy[1] + (rs + 0.5) * res
    zs = np.full_like(xs, ground_z, dtype=np.float32)
    pts = np.stack([xs, ys, zs], axis=1)
    uv = world_points_to_image(pts, camera_pose, K)
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    if uv.shape[0] == 0:
        return mask
    u = np.round(uv[:,0]).astype(np.int32)
    v = np.round(uv[:,1]).astype(np.int32)
    keep = (u>=0)&(u<img_w)&(v>=0)&(v<img_h)
    mask[v[keep], u[keep]] = 255
    return mask


def bev_mask_to_drone_mask_filled(
    bev_mask: np.ndarray,
    grid_origin_xy: Tuple[float, float],
    res: float,
    camera_pose,
    K: np.ndarray,
    img_w: int,
    img_h: int,
    ground_z: float = 0.0,
) -> np.ndarray:
    """Project a BEV mask (H,W) onto the camera image and fill regions for a dense mask.

    Strategy:
    - Try to use OpenCV contours to fill projected polygons.
    - If OpenCV is unavailable or projection is degenerate, fall back to point splatting.
    Returns uint8 mask where 255 indicates VISIBLE area.
    """
    try:
        import cv2  # type: ignore
    except Exception:
        cv2 = None

    H, W = bev_mask.shape
    out = np.zeros((img_h, img_w), dtype=np.uint8)
    if bev_mask.max() == 0:
        return out

    if cv2 is None:
        # Fallback: point splatting
        return bev_mask_to_drone_mask(
            bev_mask, grid_origin_xy, res, camera_pose, K, img_w, img_h, ground_z
        )

    # Find contours on BEV grid
    bev_u8 = (bev_mask.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(bev_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return out

    polys_img: List[np.ndarray] = []
    for cnt in contours:
        if cnt.shape[0] < 3:
            continue
        # cnt is (N,1,2) with (col=x, row=y) in BEV grid coords
        cnt = cnt[:, 0, :]  # (N,2)
        cs = cnt[:, 0].astype(np.float32)
        rs = cnt[:, 1].astype(np.float32)
        xs = grid_origin_xy[0] + (cs + 0.5) * res
        ys = grid_origin_xy[1] + (rs + 0.5) * res
        zs = np.full_like(xs, ground_z, dtype=np.float32)
        pts = np.stack([xs, ys, zs], axis=1)
        uv = world_points_to_image(pts, camera_pose, K)
        if uv.shape[0] < 3:
            continue
        poly = np.stack([uv[:, 0], uv[:, 1]], axis=1)
        # Filter to image bounds after rounding
        poly = np.round(poly).astype(np.int32)
        # It is OK if some vertices fall outside; fillPoly clips automatically
        polys_img.append(poly)

    if polys_img:
        cv2.fillPoly(out, polys_img, 255)
    else:
        # Fallback if no valid projected polygons
        out = bev_mask_to_drone_mask(
            bev_mask, grid_origin_xy, res, camera_pose, K, img_w, img_h, ground_z
        )
    return out
