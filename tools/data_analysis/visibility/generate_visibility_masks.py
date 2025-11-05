#!/usr/bin/env python3
"""
Offline generator for per-vehicle visibility annotations.

What it does (two parallel outputs):
- Keeps current path: per-vehicle masks projected to the drone image (for visualization videos).
- New: exports world-frame per-vehicle visibility annotations (polygons and/or local rasters),
    derived from each vehicle's own LiDAR-based BEV visibility (occlusion-aware via LOS).

World-frame export rationale:
- Supervise BEVFormer visibility head in a canonical frame. At training time, warp world-frame
    polygons/rasters to the drone-ego BEV (102.4 m × 102.4 m @ 0.4 m) via T_world→drone.
    This script optionally copies world_to_drone.npy from the drone agent folder into the timestamp root
    for easier downstream consumption.

Notes:
- This remains a scaffold reliant on run folder conventions; adjust paths as needed.
"""
import os
import json
import pickle
import numpy as np
from typing import Dict
from PIL import Image
import open3d as o3d

from skylink_v2x.data_analysis.visibility.lidar_visibility import BEVGrid, build_occupancy_from_lidar, los_visibility_from_occupancy
from skylink_v2x.data_analysis.visibility.projection import bev_mask_to_drone_mask, bev_mask_to_drone_mask_filled, world_points_to_image
from skylink_v2x.core.autonomy_stack.perception.sensor_transformer import SensorTransformer


def load_metadata(meta_pkl_path: str) -> Dict:
    with open(meta_pkl_path, 'rb') as f:
        return pickle.load(f)


def save_png_mask(path: str, mask: np.ndarray):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(mask).save(path)


def save_overlay(path: str, rgb_path: str, mask: np.ndarray, alpha: float = 0.4):
    try:
        rgb = Image.open(rgb_path).convert('RGB')
        if (rgb.size[0], rgb.size[1]) != (mask.shape[1], mask.shape[0]):
            rgb = rgb.resize((mask.shape[1], mask.shape[0]))
        import numpy as np
        arr = np.array(rgb, dtype=np.uint8)
        overlay = arr.copy()
        # tint visible area in green
        g = overlay[:, :, 1]
        g[mask > 0] = 255
        blended = (alpha * overlay + (1 - alpha) * arr).astype(np.uint8)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        Image.fromarray(blended).save(path)
    except Exception:
        # best-effort; ignore overlay failures
        pass


def decode_carla_depth_png(depth_png_path: str) -> np.ndarray:
    """Decode CARLA depth PNG saved by DepthSensor into meters (float32).
    CARLA encodes depth as D_norm = (R + G*256 + B*65536) / (256^3 - 1), range [0, 1].
    Metric depth (meters) = D_norm * 1000 (CARLA depth camera default far plane ~1000m).
    Uses PIL to avoid extra dependencies.
    """
    try:
        img = Image.open(depth_png_path).convert('RGB')
        png = np.array(img)
        if png.ndim == 3 and png.shape[2] >= 3:
            # CARLA: depth encoding uses R + G*256 + B*256^2
            R = png[:, :, 0].astype(np.float32)
            G = png[:, :, 1].astype(np.float32)
            B = png[:, :, 2].astype(np.float32)
            Dn = (R + 256.0 * G + 65536.0 * B) / 16777215.0  # 256^3 - 1
            return Dn * 1000.0
        else:
            return np.zeros(png.shape[:2], dtype=np.float32)
    except Exception:
        return np.zeros((1, 1), dtype=np.float32)


def rasterize_line_world(grid: BEVGrid, p0_xy: tuple, p1_xy: tuple, arr: np.ndarray, value: int = 1, step_m: float = None):
    """Rasterize a line from p0 to p1 in world XY onto arr (H,W) using small steps.
    step_m defaults to grid.resolution. Safe for short segments; not optimized.
    """
    if step_m is None:
        step_m = float(grid.resolution)
    x0, y0 = float(p0_xy[0]), float(p0_xy[1])
    x1, y1 = float(p1_xy[0]), float(p1_xy[1])
    dx = x1 - x0
    dy = y1 - y0
    dist = float(np.hypot(dx, dy))
    if dist < 1e-6:
        r, c = grid.world_to_grid(np.array([x0], dtype=np.float32), np.array([y0], dtype=np.float32))
        r, c = int(r[0]), int(c[0])
        if 0 <= r < arr.shape[0] and 0 <= c < arr.shape[1]:
            arr[r, c] = value
        return
    n = max(1, int(np.ceil(dist / max(step_m, 1e-3))))
    ts = np.linspace(0.0, 1.0, n + 1, dtype=np.float32)
    xs = x0 + ts * dx
    ys = y0 + ts * dy
    rs, cs = grid.world_to_grid(xs, ys)
    H, W = arr.shape
    valid = (rs >= 0) & (rs < H) & (cs >= 0) & (cs < W)
    arr[rs[valid].astype(np.int32), cs[valid].astype(np.int32)] = value


def camera_depth_visibility_on_bev(agent_folder: str,
                                   meta: Dict,
                                   grid: BEVGrid,
                                   ground_z: float,
                                   ego_pos: list,
                                   sample_stride: int = 6,
                                   max_depth_m: float = 150.0,
                                   max_range_m: float = 120.0,
                                   debug_draw: bool = False) -> np.ndarray:
    """Approximate per-vehicle BEV visibility using its depth cameras.
    For each depth camera PNG in the agent folder, decode depth, backproject sparse rays,
    and draw ground-projected visible segments from camera ground position up to the first hit.
    Returns a uint8 mask (H,W) with 1 for visible ground cells, 0 otherwise.
    """
    H, W = grid.size
    vis = np.zeros((H, W), dtype=np.uint8)
    # Vehicle world pose
    if ego_pos is None or len(ego_pos) < 6:
        return vis
    T_vehicle = SensorTransformer.cords_to_transform_matrix(ego_pos)
    # Collect depth sensor entries by name heuristic
    depth_entries = []
    for k, v in meta.items():
        if not isinstance(v, dict):
            continue
        name = str(k).lower()
        if 'depth' not in name:
            continue
        png_path = os.path.join(agent_folder, f"{k}.png")
        if not os.path.exists(png_path):
            continue
        K = get_intrinsics_from_meta(v)
        cords = v.get('cords', None)
        if cords is None:
            continue
        depth_entries.append((k, png_path, K, cords))

    if not depth_entries:
        return vis

    debug_info = {
        'num_depth_sensors': 0,
        'sensors': [],
        'ray_stride': int(sample_stride),
        'max_depth_m': float(max_depth_m),
        'max_range_m': float(max_range_m),
        'rays_sampled': 0,
        'rays_with_ground': 0,
        'rays_drawn': 0,
    }
    # Iterate depth sensors
    for name, png_path, K, cords in depth_entries:
        entry_dbg = {'name': name, 'png': os.path.basename(png_path)}
        D = decode_carla_depth_png(png_path)
        if D.size <= 1:
            entry_dbg['decoded_shape'] = list(D.shape)
            entry_dbg['note'] = 'decode_failed_or_empty'
            debug_info['sensors'].append(entry_dbg)
            continue
        Himg, Wimg = D.shape
        entry_dbg['decoded_shape'] = [int(Himg), int(Wimg)]
        try:
            entry_dbg['depth_stats'] = {
                'min': float(np.min(D)),
                'max': float(np.max(D)),
                'mean': float(np.mean(D)),
            }
        except Exception:
            pass
        fx = float(K[0, 0]); fy = float(K[1, 1]); cx = float(K[0, 2]); cy = float(K[1, 2])
        T_cam_rel = SensorTransformer.cords_to_transform_matrix(cords)
        T_cam_world = T_vehicle @ T_cam_rel
    # Camera origin in world and ground-projected
        Cw = (T_cam_world @ np.array([0, 0, 0, 1], dtype=np.float32))[:3]
        Cg = np.array([Cw[0], Cw[1], ground_z], dtype=np.float32)
        # Sample pixels
        ss = max(1, int(sample_stride))
        us = np.arange(0, Wimg, ss, dtype=np.int32)
        vs = np.arange(0, Himg, ss, dtype=np.int32)
        # Clamp unreasonable depths
        D = np.nan_to_num(D, nan=0.0, posinf=max_depth_m, neginf=0.0)
        D = np.clip(D, 0.0, max_depth_m)
        for v in vs:
            ycv = (v - cy) / max(fy, 1e-6)
            for u in us:
                d = float(D[v, u])
                if d <= 0.1:  # skip invalid/too-near
                    continue
                xcv = (u - cx) / max(fx, 1e-6)
                # Build ray in SENSOR frame consistent with SensorTransformer's projection:
                # x_cv = Y_s, y_cv = -Z_s, z_cv = X_s -> dir_s = [X_s, Y_s, Z_s] = [1, x_cv, -y_cv]
                dir_s = np.array([1.0, xcv, -ycv], dtype=np.float32)
                dir_s = dir_s / max(np.linalg.norm(dir_s), 1e-6)
                # Transform ray to world frame
                Rcw = T_cam_world[:3, :3]
                dir_w = (Rcw @ dir_s)
                dir_w = dir_w / max(np.linalg.norm(dir_w), 1e-6)
                # Intersection with ground plane z = ground_z
                denom = float(dir_w[2])
                t_ground = np.inf
                if abs(denom) > 1e-6:
                    t_ground = (ground_z - float(Cw[2])) / denom
                debug_info['rays_sampled'] += 1
                if np.isfinite(t_ground) and t_ground > 0:
                    debug_info['rays_with_ground'] += 1
                # Only contribute if the ray intersects ground in front of the camera
                if not np.isfinite(t_ground) or t_ground <= 0:
                    continue
                # Choose segment end: up to first hit or ground intersection, whichever comes first
                t_end = min(d, t_ground)
                # Clip to max range
                t_end = min(t_end, max_depth_m)
                if t_end <= 0:
                    continue
                Pend = Cw + t_end * dir_w
                # Ground-projected segment end XY
                Pg = np.array([Pend[0], Pend[1], ground_z], dtype=np.float32)
                # Enforce a max ground range constraint
                if max_range_m is not None and max_range_m > 0:
                    # distance on ground from camera ground point
                    dx = float(Pg[0] - Cg[0]); dy = float(Pg[1] - Cg[1])
                    gdist = float(np.hypot(dx, dy))
                    if gdist > max_range_m:
                        scale = max_range_m / max(gdist, 1e-6)
                        Pg[0] = Cg[0] + dx * scale
                        Pg[1] = Cg[1] + dy * scale
                # Draw segment on grid from camera ground point to Pg
                rasterize_line_world(grid, (Cg[0], Cg[1]), (Pg[0], Pg[1]), vis, value=1, step_m=grid.resolution)
                debug_info['rays_drawn'] += 1
        debug_info['sensors'].append(entry_dbg)
        debug_info['num_depth_sensors'] += 1
    # Save debug json if requested
    if debug_draw:
        try:
            with open(os.path.join(agent_folder, 'camera_depth_debug.json'), 'w') as jf:
                json.dump({
                    **debug_info,
                    'vis_nonzero': int(np.count_nonzero(vis)),
                }, jf, indent=2)
        except Exception:
            pass
    return vis


def bev_mask_to_world_polygons(vis_bev: np.ndarray, grid: BEVGrid, ground_z: float = 0.0):
    """Convert a BEV visibility mask into one or more world-frame polygons.
    Returns: list of polygons, each as an (N,2) float32 array [[x,y],...].
    Uses OpenCV contours if available; falls back to sampling mask cells.
    """
    try:
        import cv2  # type: ignore
    except Exception:
        cv2 = None

    H, W = vis_bev.shape
    polys = []
    if vis_bev.max() == 0:
        return polys
    if cv2 is None:
        # Fallback: sample on a coarse grid to build small convex-like polygons per connected chunk
        rs, cs = np.where(vis_bev > 0)
        if rs.size == 0:
            return polys
        xs = grid.origin_xy[0] + (cs + 0.5) * grid.resolution
        ys = grid.origin_xy[1] + (rs + 0.5) * grid.resolution
        pts = np.stack([xs, ys], axis=1).astype(np.float32)
        polys.append(pts)
        return polys

    bev_u8 = (vis_bev.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(bev_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cnt.shape[0] < 3:
            continue
        cnt = cnt[:, 0, :]  # (N,2) in (col=x, row=y)
        cs = cnt[:, 0].astype(np.float32)
        rs = cnt[:, 1].astype(np.float32)
        xs = grid.origin_xy[0] + (cs + 0.5) * grid.resolution
        ys = grid.origin_xy[1] + (rs + 0.5) * grid.resolution
        poly = np.stack([xs, ys], axis=1)
        # Optional: simplify polygon using approxPolyDP for compactness
        try:
            import cv2  # type: ignore
            eps = max(1.0, 0.5 * grid.resolution)
            poly_int = np.round(np.stack([cs, rs], axis=1)).astype(np.int32).reshape(-1, 1, 2)
            simp = cv2.approxPolyDP(poly_int, epsilon=eps, closed=True)
            simp = simp.reshape(-1, 2)
            xs_s = grid.origin_xy[0] + (simp[:, 0].astype(np.float32) + 0.5) * grid.resolution
            ys_s = grid.origin_xy[1] + (simp[:, 1].astype(np.float32) + 0.5) * grid.resolution
            poly = np.stack([xs_s, ys_s], axis=1)
        except Exception:
            pass
        polys.append(poly.astype(np.float32))
    return polys


def save_world_visibility(agent_folder: str,
                          actor_id: int,
                          vis_bev: np.ndarray,
                          grid: BEVGrid,
                          ground_z: float,
                          blocker_ids: list,
                          blocker_hit_counts: dict,
                          meta_extra: Dict):
    """Persist per-vehicle world-frame visibility in two formats:
    - Polygons: JSON with list of world-frame polygons.
    - Local raster: NPZ containing vis_bev, origin_xy, res, and size.
    Also include first-blocker info if provided.
    """
    os.makedirs(agent_folder, exist_ok=True)
    # Polygons
    polys = bev_mask_to_world_polygons(vis_bev, grid, ground_z)
    out_json = {
        'actor_id': int(actor_id),
        'format': 'world_polygons',
        'polygons': [p.tolist() for p in polys],
        'bev_meta_vehicle': {
            'origin_xy': [float(grid.origin_xy[0]), float(grid.origin_xy[1])],
            'resolution_m': float(grid.resolution),
            'size_hw': [int(grid.size[0]), int(grid.size[1])],
            'ground_z': float(ground_z),
        },
        'first_blocker_actor_ids': [int(b) for b in (blocker_ids or [])],
        'first_blocker_hit_counts': {int(k): int(v) for k, v in (blocker_hit_counts.items() if isinstance(blocker_hit_counts, dict) else [])},
        'meta': meta_extra or {},
    }
    with open(os.path.join(agent_folder, f"visibility_world_actor_{actor_id:06d}.json"), 'w') as f:
        json.dump(out_json, f, indent=2)

    # Local raster (for debugging or alternative consumption)
    np.savez_compressed(
        os.path.join(agent_folder, f"visibility_world_actor_{actor_id:06d}.npz"),
        vis_bev=vis_bev.astype(np.uint8),
        origin_xy=np.array(grid.origin_xy, dtype=np.float32),
        resolution=np.array([grid.resolution], dtype=np.float32),
        size=np.array(list(grid.size), dtype=np.int32),
        ground_z=np.array([ground_z], dtype=np.float32),
    )


def save_lidar_projection_overlay(out_path: str, rgb_path: str, points_world: np.ndarray, T_cam_world: np.ndarray, K: np.ndarray, max_points: int = 5000):
    """Project a subset of LiDAR world points to the drone image and save an overlay for debugging.
    Returns a small dict with counts for diagnostics.
    Always saves an image (even if no points project in front of the camera).
    """
    stats = {"lidar_points": int(points_world.shape[0]), "proj_points": 0, "in_image_points": 0}
    try:
        base = Image.open(rgb_path).convert('RGB')
        W, H = base.size
        # sample points
        pts = points_world
        if pts.shape[0] > max_points:
            idx = np.random.choice(pts.shape[0], size=max_points, replace=False)
            pts = pts[idx]
        if pts.shape[0] > 0:
            uv = world_points_to_image(pts, T_cam_world, K)
            stats["proj_points"] = int(uv.shape[0])
            import numpy as np
            import PIL.ImageDraw as ImageDraw
            draw = ImageDraw.Draw(base)
            if uv.shape[0] > 0:
                u = np.round(uv[:, 0]).astype(np.int32)
                v = np.round(uv[:, 1]).astype(np.int32)
                keep = (u >= 0) & (u < W) & (v >= 0) & (v < H)
                stats["in_image_points"] = int(np.count_nonzero(keep))
                u = u[keep]
                v = v[keep]
                for x, y in zip(u.tolist(), v.tolist()):
                    # small green dot
                    draw.ellipse([(x-1, y-1), (x+1, y+1)], fill=(0, 255, 0))
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        base.save(out_path)
    except Exception:
        # best-effort; ignore failures
        return stats
    return stats


def save_sensor_axes_overlay(out_path: str, rgb_path: str, T_sensor_world: np.ndarray, T_cam_world: np.ndarray, K: np.ndarray):
    """Project sensor origin and +X (forward) axis into the drone image and draw markers.
    Red: origin, Blue: forward endpoint (1m).
    """
    try:
        base = Image.open(rgb_path).convert('RGB')
        W, H = base.size
        import numpy as np
        import PIL.ImageDraw as ImageDraw
        draw = ImageDraw.Draw(base)
        origin_w = (T_sensor_world @ np.array([0, 0, 0, 1], dtype=np.float32))[:3]
        forward_w = (T_sensor_world @ np.array([1, 0, 0, 1], dtype=np.float32))[:3]
        uv = world_points_to_image(np.stack([origin_w, forward_w], axis=0), T_cam_world, K)
        if uv.shape[0] >= 1:
            u = np.round(uv[:, 0]).astype(np.int32)
            v = np.round(uv[:, 1]).astype(np.int32)
            # origin
            if 0 <= u[0] < W and 0 <= v[0] < H:
                draw.ellipse([(u[0]-3, v[0]-3), (u[0]+3, v[0]+3)], fill=(255, 0, 0))
            # forward
            if uv.shape[0] > 1 and 0 <= u[1] < W and 0 <= v[1] < H:
                draw.ellipse([(u[1]-3, v[1]-3), (u[1]+3, v[1]+3)], fill=(0, 0, 255))
                # draw a line
                if 0 <= u[0] < W and 0 <= v[0] < H:
                    draw.line([(u[0], v[0]), (u[1], v[1])], fill=(0, 0, 255), width=2)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        base.save(out_path)
    except Exception:
        pass


def save_actor_centers_overlay(out_path: str, rgb_path: str, objects: dict, T_cam_world: np.ndarray, K: np.ndarray):
    """Project actor centers (their world locations) to the drone image for a quick sanity check."""
    try:
        base = Image.open(rgb_path).convert('RGB')
        W, H = base.size
        import numpy as np
        import PIL.ImageDraw as ImageDraw
        draw = ImageDraw.Draw(base)
        centers = []
        ids = []
        for aid, entry in objects.items():
            loc = entry.get('location', None)
            if loc is None or len(loc) < 3:
                continue
            centers.append([loc[0], loc[1], loc[2]])
            ids.append(str(aid))
        if centers:
            centers = np.array(centers, dtype=np.float32)
            uv = world_points_to_image(centers, T_cam_world, K)
            u = np.round(uv[:, 0]).astype(np.int32)
            v = np.round(uv[:, 1]).astype(np.int32)
            for i in range(uv.shape[0]):
                if 0 <= u[i] < W and 0 <= v[i] < H:
                    draw.ellipse([(u[i]-2, v[i]-2), (u[i]+2, v[i]+2)], fill=(0, 255, 0))
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        base.save(out_path)
    except Exception:
        pass


def get_intrinsics_from_meta(cam_meta: dict) -> np.ndarray:
    """Return 3x3 intrinsics from metadata, with a robust fallback if missing.
    Prefers a saved 'intrinsic' matrix; else computes from fov and resolution.
    """
    K = cam_meta.get('intrinsic', None)
    if K is not None:
        return np.array(K, dtype=np.float32)
    try:
        width = int(cam_meta.get('image_size_x'))
        height = int(cam_meta.get('image_size_y'))
        fov = float(cam_meta.get('fov'))
        tmp = {'image_size_x': width, 'image_size_y': height, 'fov': fov}
        return SensorTransformer.compute_camera_intrinsics(tmp).astype(np.float32)
    except Exception:
        # Last resort: identity with a warning-ish behavior
        return np.eye(3, dtype=np.float32)


def estimate_ground_z_from_lidar(points_world: np.ndarray, percentile: float = 5.0) -> float:
    """Estimate ground plane height using a low percentile of LiDAR z.
    Robust to sparse outliers; assumes ground has many low-z points.
    """
    if points_world.shape[0] == 0:
        return 0.0
    zs = points_world[:, 2]
    try:
        return float(np.percentile(zs, percentile))
    except Exception:
        return float(zs.min())


def matrix_to_cords(T: np.ndarray):
    """Convert a 4x4 transform matrix back to cords [x,y,z, roll, yaw, pitch] (deg)
    consistent with SensorTransformer.cords_to_transform_matrix.
    """
    x, y, z = T[0,3], T[1,3], T[2,3]
    # Using the same convention as cords_to_transform_matrix
    # T[2,0] = sp
    sp = T[2,0]
    pitch = np.arcsin(sp)
    cp = np.cos(pitch)
    # T[2,1] = -cp*sr, T[2,2] = cp*cr
    sr = -T[2,1] / max(cp, 1e-8)
    cr =  T[2,2] / max(cp, 1e-8)
    roll = np.arctan2(sr, cr)
    # T[1,0] = sy*cp, T[0,0] = cp*cy
    sy = T[1,0] / max(cp, 1e-8)
    cy = T[0,0] / max(cp, 1e-8)
    yaw = np.arctan2(sy, cy)
    return [float(x), float(y), float(z), np.degrees(roll), np.degrees(yaw), np.degrees(pitch)]


def compute_bbox2d_from_object(obj_entry: dict, camera_pose, K: np.ndarray, img_w: int, img_h: int):
    """Project an object's 3D bounding box to 2D bbox in the given camera.
    obj_entry has keys: 'location' (actor cords), 'center' (bbox cords relative to actor), 'extent' ([ex,ey,ez])
    Returns (x1,y1,x2,y2) clipped to image, or None if behind camera/empty.
    """
    try:
        actor_cords = obj_entry['location']  # [x,y,z,roll,yaw,pitch]
        bb_cords = obj_entry['center']       # [x,y,z,roll,yaw,pitch] relative to actor
        ex, ey, ez = obj_entry['extent']     # [ex,ey,ez]
    except Exception:
        return None

    verts = np.array([
        [ ex,  ey, -ez, 1],
        [-ex,  ey, -ez, 1],
        [-ex, -ey, -ez, 1],
        [ ex, -ey, -ez, 1],
        [ ex,  ey,  ez, 1],
        [-ex,  ey,  ez, 1],
        [-ex, -ey,  ez, 1],
        [ ex, -ey,  ez, 1],
    ], dtype=np.float32)  # (8,4)

    T_actor = SensorTransformer.cords_to_transform_matrix(actor_cords)
    T_bb_rel = SensorTransformer.cords_to_transform_matrix(bb_cords)
    T_bb_world = T_actor @ T_bb_rel
    world_pts = (T_bb_world @ verts.T)  # (4,8)
    # Project to image using robust projection util (supports 4x4 pose)
    uv = world_points_to_image(world_pts[:3, :].T, camera_pose, K)
    if uv.shape[0] == 0:
        return None
    u = uv[:, 0]
    v = uv[:, 1]
    # compute bbox and clip
    x1 = max(0, int(np.floor(u.min())))
    y1 = max(0, int(np.floor(v.min())))
    x2 = min(img_w-1, int(np.ceil(u.max())))
    y2 = min(img_h-1, int(np.ceil(v.max())))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def project_bbox_polygon_from_object(obj_entry: dict, camera_pose, K: np.ndarray, img_w: int, img_h: int):
    """Project object's 3D bbox and return a 2D polygon approximating its footprint in the image.
    Returns an Nx2 int32 array of pixel coordinates (convex hull of projected 8 corners) or None.
    """
    try:
        actor_cords = obj_entry['location']
        bb_cords = obj_entry['center']
        ex, ey, ez = obj_entry['extent']
    except Exception:
        return None
    # 8 bbox vertices in local bbox frame
    verts = np.array([
        [ ex,  ey, -ez, 1],
        [-ex,  ey, -ez, 1],
        [-ex, -ey, -ez, 1],
        [ ex, -ey, -ez, 1],
        [ ex,  ey,  ez, 1],
        [-ex,  ey,  ez, 1],
        [-ex, -ey,  ez, 1],
        [ ex, -ey,  ez, 1],
    ], dtype=np.float32)  # (8,4)

    T_actor = SensorTransformer.cords_to_transform_matrix(actor_cords)
    T_bb_rel = SensorTransformer.cords_to_transform_matrix(bb_cords)
    T_bb_world = T_actor @ T_bb_rel
    world_pts = (T_bb_world @ verts.T).T[:, :3]  # (8,3)
    uv = world_points_to_image(world_pts, camera_pose, K)
    if uv.shape[0] < 3:
        return None
    # Round to integer pixel coords and filter to image bounds (keep slightly outside; fill will clip)
    poly = np.round(uv).astype(np.int32)
    # Compute convex hull to ensure a proper polygon
    try:
        import cv2  # type: ignore
        hull = cv2.convexHull(poly.reshape(-1,1,2))  # (M,1,2)
        hull = hull.reshape(-1,2)
        return hull
    except Exception:
        # Fallback: return the raw projected corners
        return poly


def build_vehicle_mask_on_grid(objects: dict, grid: BEVGrid):
    """Rasterize vehicle bounding boxes onto the BEV grid to form a vehicle mask.
    objects entries are expected to include 'location', 'center', and 'extent'.
    """
    try:
        import cv2  # type: ignore
    except Exception:
        cv2 = None
    H, W = grid.size
    veh_mask = np.zeros((H, W), dtype=np.uint8)
    veh_label_map = np.zeros((H, W), dtype=np.int32)
    relaxed_used = False
    if objects is None or len(objects) == 0:
        return veh_mask
    def rasterize(entries, label_with_actor_id: bool = False):
        for act_id, entry in entries:
            if not isinstance(entry, dict):
                continue
            try:
                loc = entry['location']
                center = entry['center']
                ex, ey, _ = entry['extent']
            except Exception:
                continue
            corners = np.array([
                [ ex,  ey, 1],
                [-ex,  ey, 1],
                [-ex, -ey, 1],
                [ ex, -ey, 1],
            ], dtype=np.float32)
            T_actor = SensorTransformer.cords_to_transform_matrix(loc)
            T_bb_rel = SensorTransformer.cords_to_transform_matrix(center)
            T_bb_world = T_actor @ T_bb_rel
            world_pts = (T_bb_world @ np.hstack([corners[:, :2], np.zeros((4,1), dtype=np.float32), np.ones((4,1), dtype=np.float32)]).T).T
            xs = world_pts[:, 0]
            ys = world_pts[:, 1]
            rs, cs = grid.world_to_grid(xs, ys)
            poly = np.stack([cs, rs], axis=1).astype(np.int32)
            if cv2 is not None:
                cv2.fillPoly(veh_mask, [poly.reshape(-1,1,2)], 1)
                if label_with_actor_id and act_id is not None:
                    cv2.fillPoly(veh_label_map, [poly.reshape(-1,1,2)], int(act_id))
            else:
                valid = (rs>=0)&(rs<H)&(cs>=0)&(cs<W)
                veh_mask[rs[valid], cs[valid]] = 1
                if label_with_actor_id and act_id is not None:
                    veh_label_map[rs[valid], cs[valid]] = int(act_id)

    # First pass: heuristic vehicles only
    veh_like = []
    for k, entry in objects.items():
        if not isinstance(entry, dict):
            continue
        t = entry.get('type', entry.get('category', '')).lower()
        if ('vehicle' in t) or ('car' in t) or ('truck' in t) or ('bus' in t):
            try:
                veh_like.append((int(k), entry))
            except Exception:
                veh_like.append((None, entry))
    rasterize(veh_like, label_with_actor_id=True)

    # If empty, relax filter: rasterize all entries that have extent
    if np.count_nonzero(veh_mask) == 0:
        relaxed_used = True
        any_entries = []
        for k, entry in objects.items():
            if isinstance(entry, dict) and 'extent' in entry and 'location' in entry and 'center' in entry:
                try:
                    any_entries.append((int(k), entry))
                except Exception:
                    any_entries.append((None, entry))
        rasterize(any_entries, label_with_actor_id=True)

    return veh_mask, veh_label_map, relaxed_used


def build_instance_to_actor_map(objects: dict, id_map: np.ndarray, camera_pose, K: np.ndarray) -> dict:
    """Map instance IDs to actor IDs by projecting 3D bboxes and taking dominant instance ID in bbox.
    Returns dict: inst_id -> (actor_id, pixel_count)
    """
    h, w = id_map.shape
    inst2actor = {}
    for actor_id, entry in objects.items():
        bbox = compute_bbox2d_from_object(entry, camera_pose, K, w, h)
        if bbox is None:
            continue
        x1, y1, x2, y2 = bbox
        crop = id_map[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        vals, counts = np.unique(crop, return_counts=True)
        mask = vals > 0
        vals = vals[mask]
        counts = counts[mask]
        if vals.size == 0:
            continue
        inst = int(vals[np.argmax(counts)])
        pix = int(counts.max())
        if inst not in inst2actor or pix > inst2actor[inst][1]:
            try:
                aid = int(actor_id)
            except Exception:
                # keys may be strings
                aid = int(actor_id)
            inst2actor[inst] = (aid, pix)
    return inst2actor


def is_first_blocker_on_ray(grid: BEVGrid, occ: np.ndarray, veh_label_map: np.ndarray, ego_xy, target_xy, actor_id: int, step: float) -> bool:
    """Cast a single ray from ego_xy to target_xy. If the first occupied cell belongs to actor_id, return True."""
    import math
    # Compute direction and distance
    dx = target_xy[0] - ego_xy[0]
    dy = target_xy[1] - ego_xy[1]
    dist_total = math.hypot(dx, dy)
    if dist_total < 1e-3:
        return False
    dirx = dx / dist_total
    diry = dy / dist_total
    H, W = grid.size
    d = step
    while d < dist_total + step:
        x = ego_xy[0] + d * dirx
        y = ego_xy[1] + d * diry
        r, c = grid.world_to_grid(np.array([x]), np.array([y]))
        r, c = int(r[0]), int(c[0])
        if r < 0 or r >= H or c < 0 or c >= W:
            return False
        if occ[r, c] == 1:
            # First blocker encountered; check label map
            return veh_label_map is not None and veh_label_map[r, c] == int(actor_id)
        d += step
    return False


def generate_for_scene(scene_folder: str, config: Dict):
    # Iterate timestamps
    only_ts = config.get('only_timestamp', None)
    all_ts = sorted([d for d in os.listdir(scene_folder) if d.startswith('timestamp_')])
    only_ts = config.get('only_timestamp', None)
    for ts_name in all_ts:
        if only_ts is not None and ts_name != only_ts:
            continue
        if not ts_name.startswith('timestamp_'):
            continue
        if only_ts is not None and ts_name != only_ts:
            continue
        ts_path = os.path.join(scene_folder, ts_name)
        print(f"[Visibility] Processing {ts_name} ...")
        # find drone agent folder (heuristic: agent type in metadata)
        agent_dirs = [d for d in os.listdir(ts_path) if d.startswith('agent_')]
        drone_agent = None
        for ad in agent_dirs:
            meta = os.path.join(ts_path, ad, 'metadata.pkl')
            if not os.path.exists(meta):
                continue
            data = load_metadata(meta)
            if data.get('agent_type','') == 'drone':
                drone_agent = ad
                break
        if drone_agent is None:
            print(f"[Visibility]   Skip: no drone agent found in {ts_name}")
            continue
        drone_path = os.path.join(ts_path, drone_agent)
        # Load drone inst ID map; fallback scan if named sensor missing
        id_map_file = os.path.join(drone_path, f"{config['drone_seg_name']}_id_map.npy")
        if not os.path.exists(id_map_file):
            # fallback: find any *_id_map.npy
            candidates = [f for f in os.listdir(drone_path) if f.endswith('_id_map.npy')]
            if candidates:
                id_map_file = os.path.join(drone_path, candidates[0])
                print(f"[Visibility]   Using fallback id_map: {os.path.basename(id_map_file)}")
            else:
                print(f"[Visibility]   Skip: no id_map file in {drone_path}")
                continue
        id_map = np.load(id_map_file)
        h_ids, w_ids = id_map.shape
    # Load drone intrinsics and camera cords
        drone_meta = load_metadata(os.path.join(drone_path, 'metadata.pkl'))
        cam_meta = drone_meta.get(config['drone_seg_name'], drone_meta.get(config['drone_rgb_name'], {}))
        K = get_intrinsics_from_meta(cam_meta)
        camera_rel_cords = cam_meta.get('cords', None)
        if camera_rel_cords is None:
            # try rgb camera if seg not present
            cam_meta = drone_meta.get(config['drone_rgb_name'], {})
            K = get_intrinsics_from_meta(cam_meta)
            camera_rel_cords = cam_meta.get('cords', None)
        # Compose to world cords using drone odometry (ego_pos)
        odom = drone_meta.get('odometry', {})
        drone_world_cords = odom.get('ego_pos', None)
        if camera_rel_cords is None or (drone_world_cords is not None and len(drone_world_cords) < 6):
            print(f"[Visibility]   Skip: missing camera cords or drone odometry")
            continue
        if drone_world_cords is None:
            print(f"[Visibility]   Skip: missing drone odometry")
            print(f"[Visibility]   Skip: missing camera cords in drone metadata")
            continue
        T_actor = SensorTransformer.cords_to_transform_matrix(drone_world_cords)
        T_cam_rel = SensorTransformer.cords_to_transform_matrix(camera_rel_cords)
        T_cam_world = T_actor @ T_cam_rel

        # Try copying world_to_drone.npy to timestamp root for convenience (if available)
        try:
            w2d_src = os.path.join(drone_path, 'world_to_drone.npy')
            if os.path.exists(w2d_src):
                w2d_dst = os.path.join(ts_path, 'world_to_drone.npy')
                if not os.path.exists(w2d_dst):
                    import shutil as _shutil
                    _shutil.copy2(w2d_src, w2d_dst)
        except Exception:
            pass

    # Get vehicle list present in this frame from timestamp-level objects.pkl
        obj_pkl = os.path.join(ts_path, 'objects.pkl')
        if not os.path.exists(obj_pkl):
            # No objects info for this timestamp; proceed without it
            objects = {}
        else:
            with open(obj_pkl, 'rb') as f:
                objects = pickle.load(f)

        # Progress: how many actors in id_map
        # print(f"[Visibility] {ts_name}: id_map present, unique actors: {len(np.unique(id_map))}")

        # Map instance IDs to actor IDs using objects and projection (for diagnostics only)
        inst2actor = build_instance_to_actor_map(objects, id_map, T_cam_world, K)
        mapped_actor_ids = np.array([v[0] for v in inst2actor.values()], dtype=np.int64)

        saved_count = 0
        # List present agent ids and read their metadata to determine agent_type
        present_agents = sorted([int(d.split('_')[-1]) for d in agent_dirs])
        present_agent_types = {}
        for aid in present_agents:
            try:
                m = load_metadata(os.path.join(ts_path, f"agent_{aid:06d}", 'metadata.pkl'))
                present_agent_types[int(aid)] = m.get('agent_type', '')
            except Exception:
                present_agent_types[int(aid)] = ''
        # If objects.pkl is missing some present VEHICLE agents, synthesize minimal entries
        # so we can build a vehicle label map and attribute first blockers. Do not synthesize for drone/rsu.
        try:
            synth_added = 0
            for aid in present_agents:
                if str(aid) in objects or int(aid) in objects:
                    continue
                # Only synthesize for vehicles
                if present_agent_types.get(int(aid), '').lower() != 'vehicle':
                    continue
                agent_folder = os.path.join(ts_path, f"agent_{aid:06d}")
                meta_path = os.path.join(agent_folder, 'metadata.pkl')
                if not os.path.exists(meta_path):
                    continue
                m = load_metadata(meta_path)
                odo = m.get('odometry', {})
                ego_pos = odo.get('ego_pos', None)
                if ego_pos is None or len(ego_pos) < 6:
                    continue
                # Synthesize a coarse vehicle entry with default extent; good enough for BEV rasterization
                obj_entry = {
                    'type': 'vehicle',
                    'category': 'vehicle',
                    'location': ego_pos,
                    'center': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    'extent': [1.0, 0.5, 1.0],  # approx half-length/half-width/half-height
                }
                objects[int(aid)] = obj_entry
                synth_added += 1
            if synth_added > 0:
                print(f"[Visibility]   Synthesized {synth_added} missing vehicle entries into objects for label map attribution")
        except Exception:
            pass
        present_set = set(present_agents)
        mapped_present = [aid for aid in mapped_actor_ids if int(aid) in present_set]
        print(f"[Visibility]   id_map actors (mapped): {len(inst2actor)}, present agent folders: {len(present_agents)}")
        if len(mapped_actor_ids) > 0:
            print(f"[Visibility]   sample mapped actor_ids: {mapped_actor_ids[:10]}")
            print(f"[Visibility]   mapped & present count: {len(mapped_present)}")
        else:
            print(f"[Visibility]   mapped set empty; proceeding to generate masks for all present agents with LiDAR")

    # Optionally restrict masks to vehicles only (based on objects.pkl types)
        vehicles_only = bool(config.get('vehicles_only', True))
        vehicle_ids = set()
        if vehicles_only and isinstance(objects, dict) and len(objects) > 0:
            for k, v in objects.items():
                # Heuristic: object entry may contain 'type' or 'category'
                t = v.get('type', v.get('category', '')).lower() if isinstance(v, dict) else ''
                if 'vehicle' in t or 'car' in t or 'truck' in t or 'bus' in t:
                    try:
                        vehicle_ids.add(int(k))
                    except Exception:
                        pass
            # Also incorporate present agents whose metadata marks them as vehicles
            for aid, atype in present_agent_types.items():
                if str(atype).lower() == 'vehicle':
                    vehicle_ids.add(int(aid))

        # Save actor centers overlay once (debug)
        if config.get('debug', False):
            rgb_png = os.path.join(drone_path, f"{config['drone_rgb_name']}.png")
            out_centers = os.path.join(drone_path, f"actor_centers_overlay.png")
            save_actor_centers_overlay(out_centers, rgb_png, objects, T_cam_world, K)

        for aid in present_agents:
            actor_id = int(aid)
            # Locate this actor's per-agent folder
            agent_folder = os.path.join(ts_path, f"agent_{actor_id:06d}")
            if not os.path.exists(agent_folder):
                # skip actors without sensors/logs (background not instrumented)
                continue
            # Skip non-vehicle agents when vehicles_only is enabled, using combined knowledge
            if vehicles_only:
                if vehicle_ids and (actor_id not in vehicle_ids):
                    continue
                # Extra guard: if metadata exists and says not a vehicle, skip
                atype = present_agent_types.get(actor_id, '')
                if atype and atype.lower() != 'vehicle':
                    continue
            # Load LiDAR points
            pcd_path = os.path.join(agent_folder, 'lidar.pcd')  # adjust if name differs with your logger
            exists_folder = os.path.isdir(agent_folder)
            if not exists_folder:
                # print(f"[Visibility]   Actor {actor_id}: no agent folder")
                continue
            if not os.path.exists(pcd_path):
                # No lidar recorded for this actor (not instrumented or filtered)
                # print(f"[Visibility]   Actor {actor_id}: has folder but no lidar.pcd, skip")
                continue
            pcd = o3d.io.read_point_cloud(pcd_path)
            points = np.asarray(pcd.points).astype(np.float32)

            # Transform LiDAR points from sensor frame to world.
            # NOTE: metadata['lidar']['lidar_pose'] is the sensor pose RELATIVE to the parent actor (vehicle/drone),
            # so we must compose with the agent world pose from odometry.
            meta = load_metadata(os.path.join(agent_folder, 'metadata.pkl'))
            # Try to locate lidar pose metadata saved by AgentLogger
            lidar_meta = meta.get('lidar', {})
            if isinstance(lidar_meta, dict) and 'lidar_pose' in lidar_meta:
                lidar_cords = lidar_meta['lidar_pose']
            else:
                # fallback: no transform; treat as already in world (unlikely)
                lidar_cords = None

            if lidar_cords is not None and points.shape[0] > 0:
                pts = np.hstack([points, np.ones((points.shape[0],1), dtype=np.float32)])  # (N,4)
                # Compose sensor RELATIVE pose with agent WORLD pose
                odo = meta.get('odometry', {})
                ego_pos = odo.get('ego_pos', None)
                if ego_pos is not None and len(ego_pos) >= 6:
                    T_agent_world = SensorTransformer.cords_to_transform_matrix(ego_pos)
                    T_sensor_rel = SensorTransformer.cords_to_transform_matrix(lidar_cords)
                    T_sensor_world = T_agent_world @ T_sensor_rel
                    # print(T_sensor_world)
                    world = (T_sensor_world @ pts.T).T
                    points_world = world[:, :3]
                else:
                    # If odometry missing, best-effort assume lidar_cords was world cords
                    world = SensorTransformer.sensor_to_world_coords(pts.T, lidar_cords).T
                    points_world = world[:, :3]
            else:
                points_world = points

            # Build BEV grid around the vehicle
            # Read vehicle pose from metadata (prefer LiDAR pose x,y)
            ego_cords = None
            # Use odometry for ego_xy (world), not sensor-relative cords
            odo = meta.get('odometry', {})
            ego_pos = odo.get('ego_pos', None)
            if ego_pos is not None and len(ego_pos) >= 2:
                ego_xy = (ego_pos[0], ego_pos[1])
            else:
                # Fallback to scene objects
                obj = objects.get(actor_id, None)
                if obj is None:
                    continue
                ego_pos = obj['location']
                ego_xy = (ego_pos[0], ego_pos[1])

            res = config.get('bev_resolution', 0.5)
            size = tuple(config.get('bev_size', [512, 512]))
            origin_xy = (ego_xy[0] - size[1]*res/2.0, ego_xy[1] - size[0]*res/2.0)
            grid = BEVGrid(origin_xy, size, res)

            # Determine ground height per agent if requested (used for both occupancy and projection)
            ground_z = config.get('ground_z', None)
            if ground_z is None and config.get('estimate_ground_z', True):
                ground_z = estimate_ground_z_from_lidar(points_world)
            if ground_z is None:
                ground_z = 0.0

            # Build occupancy with ground-relative z and optional ego clear radius and ego footprint clearing
            clear_polys_world = None
            if config.get('ego_clear_polygon', False):
                # Build an approximate ego rectangle around the LiDAR/vehicle footprint
                # If we have the object's extent, use that; else default to a 2m x 1m box
                try:
                    obj = objects.get(actor_id, None)
                    if obj is not None and 'extent' in obj:
                        ex, ey, _ = obj['extent']
                    else:
                        ex, ey = 1.0, 0.5
                except Exception:
                    ex, ey = 1.0, 0.5
                # Vehicle pose in world
                try:
                    if ego_pos is None or len(ego_pos) < 6:
                        ego_pos = [ego_xy[0], ego_xy[1], ground_z, 0, 0, 0]
                    T_vehicle = SensorTransformer.cords_to_transform_matrix(ego_pos)
                except Exception:
                    T_vehicle = np.eye(4, dtype=np.float32)
                rect_vehicle = np.array([
                    [ ex,  ey, 0, 1],
                    [-ex,  ey, 0, 1],
                    [-ex, -ey, 0, 1],
                    [ ex, -ey, 0, 1],
                ], dtype=np.float32)
                rect_world = (T_vehicle @ rect_vehicle.T).T[:, :2]
                clear_polys_world = [rect_world]

            # Vehicle mask for optional inclusion of first blocker
            veh_mask = None
            veh_label_map = None
            include_first_blocker = bool(config.get('include_first_vehicle_blocker', False))
            if include_first_blocker:
                veh_mask, veh_label_map, veh_mask_relaxed = build_vehicle_mask_on_grid(objects, grid)
                # If labels are relaxed and strict behavior is desired, disable inclusion for safety
                if veh_mask_relaxed and not bool(config.get('allow_relaxed_vehicle_blocker', False)):
                    veh_label_map = None

            # Compute label(s) per requested type
            label_type = str(config.get('label_type', 'lidar_los')).lower()
            vis_bev_lidar = None
            occ = None
            blocker_ids = []
            blocker_hit_counts = {}
            if label_type in ('lidar_los', 'fused_union', 'fused_intersection'):
                occ = build_occupancy_from_lidar(
                    points_world,
                    grid,
                    z_thresh=config.get('z_thresh', 0.5),
                    ground_z=ground_z,
                    ego_xy=ego_xy,
                    clear_radius=config.get('ego_clear_radius', None),
                    clear_polygons_world=clear_polys_world,
                    clear_margin_m=config.get('ego_clear_margin', None),
                )
                vis_bev_lidar, blocker_ids, blocker_hit_counts = los_visibility_from_occupancy(
                    grid,
                    occ,
                    ego_xy,
                    max_range=config.get('max_range', 120.0),
                    step=res,
                    vehicle_label_map=veh_label_map,
                    include_first_vehicle_blocker=include_first_blocker,
                    blocker_min_hits=int(config.get('first_vehicle_blocker_min_hits', 0)),
                )
            vis_bev_cam = None
            if label_type in ('camera_depth', 'fused_union', 'fused_intersection'):
                # camera-based visibility from vehicle depth sensors
                vis_bev_cam = camera_depth_visibility_on_bev(
                    agent_folder=agent_folder,
                    meta=meta,
                    grid=grid,
                    ground_z=ground_z,
                    ego_pos=ego_pos,
                    sample_stride=int(config.get('depth_sample_stride', 6)),
                    max_depth_m=float(config.get('depth_max_m', 150.0)),
                    max_range_m=float(config.get('max_range', 120.0)),
                    debug_draw=bool(config.get('debug', False)),
                )
            # Fuse according to label_type
            if label_type == 'lidar_los' and vis_bev_lidar is not None:
                vis_bev = vis_bev_lidar
            elif label_type == 'camera_depth' and vis_bev_cam is not None:
                vis_bev = vis_bev_cam
            elif label_type == 'fused_union':
                if vis_bev_lidar is None and vis_bev_cam is None:
                    vis_bev = np.zeros(grid.size, dtype=np.uint8)
                elif vis_bev_lidar is None:
                    vis_bev = vis_bev_cam
                elif vis_bev_cam is None:
                    vis_bev = vis_bev_lidar
                else:
                    vis_bev = ((vis_bev_lidar > 0) | (vis_bev_cam > 0)).astype(np.uint8)
            elif label_type == 'fused_intersection':
                if vis_bev_lidar is None or vis_bev_cam is None:
                    # if one missing, fall back to available one
                    vis_bev = vis_bev_lidar if vis_bev_lidar is not None else (
                        vis_bev_cam if vis_bev_cam is not None else np.zeros(grid.size, dtype=np.uint8))
                else:
                    vis_bev = ((vis_bev_lidar > 0) & (vis_bev_cam > 0)).astype(np.uint8)
            else:
                # default safety
                vis_bev = vis_bev_lidar if vis_bev_lidar is not None else (
                    vis_bev_cam if vis_bev_cam is not None else np.zeros(grid.size, dtype=np.uint8))

            # Optional: save BEV occupancy/visibility debug images
            if config.get('save_bev_debug', False):
                try:
                    from PIL import Image as _Image
                    os.makedirs(agent_folder, exist_ok=True)
                    _Image.fromarray((occ*255).astype(np.uint8)).save(os.path.join(agent_folder, f"bev_occupancy_actor_{actor_id:06d}.png"))
                    _Image.fromarray((vis_bev*255).astype(np.uint8)).save(os.path.join(agent_folder, f"bev_visibility_actor_{actor_id:06d}.png"))
                    if veh_mask is not None:
                        _Image.fromarray((veh_mask*255).astype(np.uint8)).save(os.path.join(agent_folder, f"bev_vehicle_mask_{actor_id:06d}.png"))
                    if veh_label_map is not None:
                        # Normalize labels for visualization
                        vis_lab = veh_label_map.copy().astype(np.uint16)
                        vis_lab = (255.0 * (vis_lab % 997) / 997.0).astype(np.uint8)
                        _Image.fromarray(vis_lab).save(os.path.join(agent_folder, f"bev_vehicle_label_{actor_id:06d}.png"))
                except Exception:
                    pass

            # If we need first-blocker verification but haven't built occupancy (e.g., camera_depth), do it now
            if include_first_blocker and veh_label_map is not None and occ is None:
                try:
                    occ = build_occupancy_from_lidar(
                        points_world,
                        grid,
                        z_thresh=config.get('z_thresh', 0.5),
                        ground_z=ground_z,
                        ego_xy=ego_xy,
                        clear_radius=config.get('ego_clear_radius', None),
                        clear_polygons_world=clear_polys_world,
                        clear_margin_m=config.get('ego_clear_margin', None),
                    )
                except Exception:
                    occ = None

            # 1) New: save world-frame visibility (polygons + local raster)
            meta_extra = {
                'bev_resolution_used': float(res),
                'bev_size_hw_used': list(size),
                'label_type': label_type,
            }
            if config.get('export_world', True):
                try:
                    save_world_visibility(
                        agent_folder=agent_folder,
                        actor_id=actor_id,
                        vis_bev=vis_bev,
                        grid=grid,
                        ground_z=ground_z,
                        blocker_ids=blocker_ids if config.get('include_first_vehicle_blocker', False) else [],
                        blocker_hit_counts=blocker_hit_counts if config.get('include_first_vehicle_blocker', False) else {},
                        meta_extra=meta_extra,
                    )
                except Exception as _e:
                    print(f"[Visibility]   Warning: failed to save world-frame visibility for actor {actor_id}: {_e}")

            # 2) Existing: project to drone image (for visualization videos)
            if config.get('filled_projection', True):
                mask = bev_mask_to_drone_mask_filled(
                    vis_bev, origin_xy, res, T_cam_world, K, img_w=w_ids, img_h=h_ids, ground_z=ground_z
                )
            else:
                mask = bev_mask_to_drone_mask(
                    vis_bev, origin_xy, res, T_cam_world, K, img_w=w_ids, img_h=h_ids, ground_z=ground_z
                )

            # If requested, also mark first-blocker vehicles as visible by projecting their GT 3D boxes into the drone image
            # Two paths:
            #  - Strict: blocker_ids returned from LOS where first occupied cell matches a labeled vehicle in veh_label_map.
            #  - Relaxed (or no labels): for each projected vehicle bbox that faces the ego, cast a single ray to its center and verify it’s the first blocker.
            if include_first_blocker:
                img_h, img_w = mask.shape
                step = res
                filled_bboxes = []  # list of (x1,y1,x2,y2)
                def fill_actor_bbox(aid: int):
                    entry = objects.get(aid, None)
                    if not isinstance(entry, dict):
                        return False
                    # Prefer oriented polygon fill from projected 3D bbox to preserve direction
                    poly = project_bbox_polygon_from_object(entry, T_cam_world, K, img_w, img_h)
                    if poly is not None and poly.shape[0] >= 3:
                        try:
                            import cv2  # type: ignore
                            cv2.fillPoly(mask, [poly.reshape(-1,1,2)], 255)
                            filled_bboxes.append((int(poly[:,0].min()), int(poly[:,1].min()), int(poly[:,0].max()), int(poly[:,1].max())))
                            return True
                        except Exception:
                            pass
                    # Fallback to axis-aligned bbox if polygon path fails
                    bbox = compute_bbox2d_from_object(entry, T_cam_world, K, img_w, img_h)
                    if bbox is None:
                        return False
                    x1, y1, x2, y2 = bbox
                    mask[y1:y2, x1:x2] = 255
                    filled_bboxes.append((x1, y1, x2, y2))
                    return True
                used_ids = []
                # Preferred strict list
                if blocker_ids:
                    for bid in blocker_ids:
                        if fill_actor_bbox(bid):
                            used_ids.append(int(bid))
                else:
                    # Relaxed: iterate GT vehicles, check first-blocker along ray to their center
                    for k, entry in objects.items():
                        if not isinstance(entry, dict):
                            continue
                        t = entry.get('type', entry.get('category', '')).lower()
                        if ('vehicle' not in t) and ('car' not in t) and ('truck' not in t) and ('bus' not in t):
                            continue
                        # Compute vehicle center XY from 3D bbox center in world
                        try:
                            loc = entry['location']
                            center = entry['center']
                        except Exception:
                            continue
                        T_actor = SensorTransformer.cords_to_transform_matrix(loc)
                        T_bb_rel = SensorTransformer.cords_to_transform_matrix(center)
                        bb_world = T_actor @ T_bb_rel
                        cx, cy = float(bb_world[0,3]), float(bb_world[1,3])
                        # Verify first blocker on ray
                        try:
                            aid = int(k)
                        except Exception:
                            continue
                        if veh_label_map is not None:
                            # If we do have labels but blocker_ids empty (e.g., strict gating), verify per-ray
                            if not is_first_blocker_on_ray(grid, occ, veh_label_map, ego_xy, (cx, cy), aid, step):
                                continue
                        else:
                            # No labels: cannot safely verify
                            continue
                        if fill_actor_bbox(aid):
                            used_ids.append(int(aid))
                # Optionally save an overlay with drawn blocker bboxes for visual confirmation
                if config.get('debug', False) and filled_bboxes:
                    try:
                        from PIL import Image as _Image
                        import PIL.ImageDraw as _ImageDraw
                        rgb_png = os.path.join(drone_path, f"{config['drone_rgb_name']}.png")
                        base = _Image.open(rgb_png).convert('RGB')
                        draw = _ImageDraw.Draw(base)
                        for (x1,y1,x2,y2) in filled_bboxes:
                            draw.rectangle([(x1,y1),(x2,y2)], outline=(255,0,0), width=2)
                        base.save(os.path.join(agent_folder, f"first_blockers_overlay.png"))
                    except Exception:
                        pass

            # Mask semantics: 255 => VISIBLE. Allow invert via config if needed.
            if config.get('invert_mask', False):
                mask = 255 - mask

            # Save
            out_mask = os.path.join(agent_folder, f"mask_actor_{actor_id:06d}.png")
            save_png_mask(out_mask, mask)
            if config.get('save_overlay', False):
                rgb_png = os.path.join(drone_path, f"{config['drone_rgb_name']}.png")
                out_overlay = os.path.join(agent_folder, f"mask_actor_{actor_id:06d}_overlay.png")
                save_overlay(out_overlay, rgb_png, mask)
            # Optional debug: project LiDAR into drone image
            if config.get('debug', False):
                rgb_png = os.path.join(drone_path, f"{config['drone_rgb_name']}.png")
                out_proj = os.path.join(agent_folder, f"lidar_proj_actor_{actor_id:06d}.png")
                proj_stats = save_lidar_projection_overlay(out_proj, rgb_png, points_world, T_cam_world, K)
                # sensor axes overlay
                try:
                    T_sensor_world_mat = T_sensor_world if lidar_cords is not None and ego_pos is not None and len(ego_pos) >= 6 else None
                except NameError:
                    T_sensor_world_mat = None
                if T_sensor_world_mat is not None:
                    out_axes = os.path.join(agent_folder, f"lidar_axes_actor_{actor_id:06d}.png")
                    save_sensor_axes_overlay(out_axes, rgb_png, T_sensor_world_mat, T_cam_world, K)
                # Save a tiny JSON with poses
                dbg = {
                    'drone_world_cords': drone_world_cords,
                    'camera_rel_cords': camera_rel_cords,
                    'T_cam_world': np.asarray(T_cam_world).tolist(),
                    'agent_odometry_ego_pos': ego_pos if ego_pos is not None else None,
                    'lidar_pose_relative': lidar_cords if lidar_cords is not None else None,
                    'camera_image_size': [int(cam_meta.get('image_size_x', 0) or 0), int(cam_meta.get('image_size_y', 0) or 0)],
                    'ego_xy': [float(ego_xy[0]), float(ego_xy[1])],
                    'ground_z': float(ground_z),
                    'lidar_proj_stats': proj_stats,
                    'bev_occ_nonzero': int(np.count_nonzero(occ)),
                    'bev_vis_nonzero': int(np.count_nonzero(vis_bev)),
                    'veh_mask_nonzero': int(np.count_nonzero(veh_mask)) if veh_mask is not None else 0,
                    'veh_label_nonzero': int(np.count_nonzero(veh_label_map)) if veh_label_map is not None else 0,
                    'veh_mask_relaxed_used': bool(veh_mask is not None and 'veh_mask_relaxed' in locals() and veh_mask_relaxed),
                    'blocker_actor_ids': [int(b) for b in blocker_ids] if include_first_blocker else [],
                    'blocker_actor_ids_filled': used_ids if include_first_blocker else [],
                    'blocker_hit_counts': {int(k): int(v) for k, v in (blocker_hit_counts.items() if isinstance(blocker_hit_counts, dict) else [])},
                    'first_vehicle_blocker_min_hits': int(config.get('first_vehicle_blocker_min_hits', 0)),
                }
                with open(os.path.join(agent_folder, f"projection_debug.json"), 'w') as f:
                    json.dump(dbg, f, indent=2)
            saved_count += 1

        print(f"[Visibility]   Saved masks: {saved_count}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-folder', required=True, help='Path to a run folder (contains timestamp_*)')
    parser.add_argument('--drone-rgb-name', default='bev_camera')
    parser.add_argument('--drone-seg-name', default='bev_instseg')
    parser.add_argument('--label-type', type=str, default='lidar_los', choices=['lidar_los','camera_depth','fused_union','fused_intersection'],
                        help='Which label to generate: LiDAR LOS, camera depth, or fused variants')
    parser.add_argument('--bev-resolution', type=float, default=0.5)
    parser.add_argument('--bev-size', type=int, nargs=2, default=[512,512])
    parser.add_argument('--max-range', type=float, default=120.0)
    parser.add_argument('--z-thresh', type=float, default=0.5)
    parser.add_argument('--depth-sample-stride', type=int, default=6, help='Pixel stride when sampling depth rays for camera-based visibility')
    parser.add_argument('--depth-max-m', type=float, default=150.0, help='Max depth in meters to consider from camera depth when generating visibility')
    parser.add_argument('--ego-clear-radius', type=float, default=None, help='Meters to clear around ego in occupancy to avoid self-blocking')
    parser.add_argument('--ego-clear-polygon', action='store_true', help='Additionally clear an ego footprint polygon from occupancy')
    parser.add_argument('--ego-clear-margin', type=float, default=None, help='Optional dilation margin in meters for ego footprint clearing')
    parser.add_argument('--invert-mask', action='store_true', help='Invert semantics so white=occluded (default white=visible)')
    parser.add_argument('--no-filled-proj', action='store_true', help='Disable filled polygon projection and use point splatting')
    parser.add_argument('--save-overlay', action='store_true', help='Save an RGB overlay to visually verify mask alignment')
    parser.add_argument('--save-bev-debug', action='store_true', help='Also save BEV occupancy/visibility debug PNGs per agent')
    parser.add_argument('--no-vehicles-only', action='store_true', help='Do not filter to vehicles; generate for all present agents with LiDAR')
    parser.add_argument('--no-estimate-ground-z', action='store_true', help='Disable ground_z estimation from LiDAR')
    parser.add_argument('--only-timestamp', type=str, default=None, help='Process only this timestamp (e.g., timestamp_000010) for debugging')
    parser.add_argument('--debug', action='store_true', help='Save per-agent LiDAR projections and pose JSON for debugging transforms')
    parser.add_argument('--include-first-vehicle-blocker', action='store_true', help='If a ray hits an occupied cell that belongs to a vehicle, mark that first blocker cell visible too')
    parser.add_argument('--allow-relaxed-vehicle-blocker', action='store_true', help='Allow inclusion when vehicle labels had to be relaxed (not strictly typed vehicles). Default is strict only.')
    parser.add_argument('--first-vehicle-blocker-min-hits', type=int, default=0, help='Minimum number of LiDAR LOS ray intersections required to mark a first-hit vehicle as visible (0 disables threshold).')
    parser.add_argument('--no-export-world', action='store_true', help='Disable exporting world-frame visibility annotations (polygons and local raster)')
    args = parser.parse_args()

    cfg = {
        'drone_rgb_name': args.drone_rgb_name,
        'drone_seg_name': args.drone_seg_name,
        'label_type': args.label_type,
        'bev_resolution': args.bev_resolution,
        'bev_size': args.bev_size,
        'max_range': args.max_range,
        'z_thresh': args.z_thresh,
        'depth_sample_stride': args.depth_sample_stride,
        'depth_max_m': args.depth_max_m,
        'ego_clear_radius': args.ego_clear_radius,
        'ego_clear_polygon': args.ego_clear_polygon,
        'ego_clear_margin': args.ego_clear_margin,
        'invert_mask': args.invert_mask,
        'filled_projection': not args.no_filled_proj,
        'save_overlay': args.save_overlay,
        'save_bev_debug': args.save_bev_debug,
        'vehicles_only': not args.no_vehicles_only,
        'estimate_ground_z': not args.no_estimate_ground_z,
        'debug': args.debug,
        'only_timestamp': args.only_timestamp,
        'include_first_vehicle_blocker': args.include_first_vehicle_blocker,
        'allow_relaxed_vehicle_blocker': args.allow_relaxed_vehicle_blocker,
        'first_vehicle_blocker_min_hits': args.first_vehicle_blocker_min_hits,
    }

    # Always generate image-space masks; world export can be toggled via config flag
    cfg['export_world'] = not args.no_export_world
    generate_for_scene(args.run_folder, cfg)

if __name__ == '__main__':
    main()
