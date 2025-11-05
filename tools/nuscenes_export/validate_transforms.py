#!/usr/bin/env python3
"""
Quick nuScenes-style transform validator for SkyLink exports.

Checks for a scene:
- Builds T_world_from_{ego, sensor} using ego_pose and calibrated_sensor tables.
- Projects LiDAR TOP points into each camera and reports in-image ratio and basic stats.
- Optional image overlay output for spot-checking.

Usage:
  python validate_transforms.py --root data/nuscenes --scene-idx 0 --max-samples 5 --save-overlays

Notes:
- Assumes calibrated_sensor translation/rotation encode sensor->ego transform.
- Assumes ego_pose translation/rotation encode ego->world transform.
- Works with images in samples/CAM_*/ and LiDAR PCD in samples/LIDAR_TOP/.
"""
import os
import json
import argparse
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import open3d as o3d
import cv2

NUS_VERSION_DEFAULTS = ["v1.0-trainval", "v1.0-test", "v1.0-custom"]


def load_table(path: Path):
    with open(path, 'r') as f:
        return json.load(f)


def quat_wxyz_to_rot(q: List[float]) -> np.ndarray:
    # q as [w,x,y,z]
    w, x, y, z = q
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    R = np.array([
        [1-2*(yy+zz), 2*(xy - wz), 2*(xz + wy)],
        [2*(xy + wz), 1-2*(xx+zz), 2*(yz - wx)],
        [2*(xz - wy), 2*(yz + wx), 1-2*(xx+yy)]
    ], dtype=np.float64)
    return R


def make_SE3(t: List[float], q: List[float]) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = quat_wxyz_to_rot(q)
    T[:3, 3] = np.array(t, dtype=np.float64)
    return T


def pick_version(root: Path, preferred: str = None) -> str:
    if preferred:
        return preferred
    for v in NUS_VERSION_DEFAULTS:
        if (root / v / 'sample.json').exists():
            return v
    raise FileNotFoundError("Could not find a nuScenes version folder under root; tried: " + ", ".join(NUS_VERSION_DEFAULTS))


def build_indices(root: Path, nus_version: str) -> Dict[str, Dict[str, dict]]:
    ver = root / nus_version
    idx = {
        'sensor': {r['token']: r for r in load_table(ver / 'sensor.json')},
        'calibrated_sensor': {r['token']: r for r in load_table(ver / 'calibrated_sensor.json')},
        'ego_pose': {r['token']: r for r in load_table(ver / 'ego_pose.json')},
        'sample': {r['token']: r for r in load_table(ver / 'sample.json')},
        'sample_data': load_table(ver / 'sample_data.json'),
        'scene': load_table(ver / 'scene.json'),
        'sample_annotation': {r['token']: r for r in load_table(ver / 'sample_annotation.json')} if (ver / 'sample_annotation.json').exists() else {},
    }
    # Build helper maps: calibrated_sensor_token -> (modality, channel)
    sensor_by_token = idx['sensor']
    cs_map: Dict[str, Tuple[str, str]] = {}
    for cs_tok, cs in idx['calibrated_sensor'].items():
        st = cs['sensor_token']
        srow = sensor_by_token.get(st, {})
        cs_map[cs_tok] = (srow.get('modality', ''), srow.get('channel', ''))
    idx['cs_mod_ch'] = cs_map
    return idx


def gather_sample_data_for_sample(idx: Dict, sample_token: str) -> List[dict]:
    return [r for r in idx['sample_data'] if r['sample_token'] == sample_token]


def pick_pair(idx: Dict, sd_list: List[dict], lidar_channel: str = None, cam_regex: str = None) -> Tuple[dict, List[dict]]:
    import re as _re
    cams: List[dict] = []
    lidar: dict = None
    cs_mod_ch = idx['cs_mod_ch']
    # First, annotate each sd with modality/channel
    for r in sd_list:
        mod, ch = cs_mod_ch.get(r['calibrated_sensor_token'], ('', ''))
        r['__modality'] = mod
        r['__channel'] = ch
    # Pick lidar by channel preference or first lidar
    lidar_candidates = [r for r in sd_list if r.get('__modality') == 'lidar']
    if lidar_channel:
        for r in lidar_candidates:
            if r.get('__channel') == lidar_channel:
                lidar = r; break
    lidar = lidar or (lidar_candidates[0] if lidar_candidates else None)
    # Pick cameras filtered by regex if provided
    for r in sd_list:
        if r.get('__modality') != 'camera':
            continue
        ch = r.get('__channel') or ''
        if cam_regex and not _re.search(cam_regex, ch):
            continue
        cams.append(r)
    return lidar, cams


def read_pcd_ascii(path: Path) -> np.ndarray:
    # Use open3d for robustness
    pcd = o3d.io.read_point_cloud(str(path))
    pts = np.asarray(pcd.points, dtype=np.float64)
    return pts

def read_lidar_points(path: Path) -> np.ndarray:
    sfx = path.suffix.lower()
    if sfx == '.bin':  # nuScenes .pcd.bin
        raw = np.fromfile(str(path), dtype=np.float32)
        # 尝试按 5 或 4 列解析（不同导出可能带 ring/intensity）
        if raw.size % 5 == 0:
            pts = raw.reshape(-1, 5)[:, :3]
        elif raw.size % 4 == 0:
            pts = raw.reshape(-1, 4)[:, :3]
        else:
            raise ValueError(f"Unexpected lidar binary size: {raw.size}")
        return pts.astype(np.float64)
    else:
        # fallback: Open3D for .pcd/.ply 等
        pcd = o3d.io.read_point_cloud(str(path))
        return np.asarray(pcd.points, dtype=np.float64)

def project_points(K: np.ndarray, T_cam_from_world: np.ndarray, Pw: np.ndarray, w: int, h: int) -> Tuple[int, int, float]:
    # Pw: (N,3)
    N = Pw.shape[0]
    Pwh = np.concatenate([Pw, np.ones((N,1))], axis=1).T  # (4,N)
    Pc = (T_cam_from_world @ Pwh)[:3]  # (3,N)
    z = Pc[2]
    valid = z > 0.1
    Pc = Pc[:, valid]
    if Pc.shape[1] == 0:
        return 0, 0, 0.0
    uv = (K @ (Pc / Pc[2]))
    u, v = uv[0], uv[1]
    in_im = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    return int(valid.sum()), int(in_im.sum()), float(in_im.sum() / max(1, valid.sum()))


def box_corners_wlh(w: float, l: float, h: float) -> np.ndarray:
    # nuScenes box size order is [w, l, h]; corners in box frame (X forward, Y left, Z up)
    x = l / 2.0; y = w / 2.0; z = h / 2.0
    corners = np.array([
        [ x,  y,  z], [ x, -y,  z], [-x, -y,  z], [-x,  y,  z],
        [ x,  y, -z], [ x, -y, -z], [-x, -y, -z], [-x,  y, -z]
    ], dtype=np.float64)
    return corners


def draw_box(img: np.ndarray, uv: np.ndarray, color=(0, 255, 255)) -> None:
    # uv: (8,2) image coords
    uv = uv.astype(int)
    edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    for a,b in edges:
        pa = tuple(uv[a]); pb = tuple(uv[b])
        cv2.line(img, pa, pb, color, 2)


def depth_colormap(z: np.ndarray, vmin: float = 0.0, vmax: float = 80.0) -> np.ndarray:
    zc = np.clip((z - vmin) / max(1e-6, (vmax - vmin)), 0.0, 1.0)
    cm = (255 * np.stack([zc, 1 - np.abs(2*zc - 1), 1 - zc], axis=-1)).astype(np.uint8)
    return cm


def run(root: Path, scene_idx: int, max_samples: int, save_overlays: bool, nus_version: str = None, out_dir: Path = None, lidar_channel: str = None, cam_regex: str = None, max_points: int = 200000, pcd_frame: str = 'sensor'):
    nus_version = pick_version(root, nus_version)
    idx = build_indices(root, nus_version)
    scenes = idx['scene']
    if scene_idx >= len(scenes):
        print(f"Scene index {scene_idx} out of range. Available: {len(scenes)}")
        return
    scene = scenes[scene_idx]
    first_token = scene['first_sample_token']
    # Reconstruct chain of samples starting at first
    samples = []
    tok = first_token
    while tok:
        s = idx['sample'].get(tok)
        if not s:
            break
        samples.append(s)
        tok = s.get('next', '')
        if max_samples and len(samples) >= max_samples:
            break

    ver = root / nus_version
    samples_dir = root  # filenames in sample_data already include 'samples/<CHANNEL>/...'
    vis_dir = out_dir or (root / 'vis')
    vis_dir.mkdir(parents=True, exist_ok=True)

    # Build quick lookup: calib token -> (K, T_ego_from_sensor)
    cs_rows = {k: v for k, v in idx['calibrated_sensor'].items()}

    ok = True
    for s in samples:
        sd_list = gather_sample_data_for_sample(idx, s['token'])
        lidar_sd, cam_sds = pick_pair(idx, sd_list, lidar_channel=lidar_channel, cam_regex=cam_regex)
        if lidar_sd is None:
            print('No lidar sample_data for sample', s['token'])
            ok = False
            continue
        ego_pose = idx['ego_pose'][lidar_sd['ego_pose_token']]
        T_world_from_ego = make_SE3(ego_pose['translation'], ego_pose['rotation'])
        # LiDAR
        cs_li = cs_rows[lidar_sd['calibrated_sensor_token']]
        T_ego_from_lidar = make_SE3(cs_li['translation'], cs_li['rotation'])
        T_world_from_lidar = T_world_from_ego @ T_ego_from_lidar
        # read points
        lidar_path = samples_dir / lidar_sd['filename']
        Pw = read_lidar_points(lidar_path)
        if Pw.shape[0] > max_points:
            sel = np.random.choice(Pw.shape[0], size=max_points, replace=False)
            Pw = Pw[sel]
        # Iterate cameras
        for cam_sd in cam_sds:
            if 'CAM' not in cam_sd['filename']:
                continue
            cs_ca = cs_rows[cam_sd['calibrated_sensor_token']]
            K = np.array(cs_ca.get('camera_intrinsic', [[0,0,0],[0,0,0],[0,0,1]]), dtype=np.float64)
            if K.shape != (3,3):
                print('Bad K shape for', cam_sd['filename'])
                ok = False
                continue
            T_ego_from_cam = make_SE3(cs_ca['translation'], cs_ca['rotation'])
            ego_pose_cam = idx['ego_pose'][cam_sd['ego_pose_token']]
            T_world_from_ego_cam = make_SE3(ego_pose_cam['translation'], ego_pose_cam['rotation'])
            T_world_from_cam = T_world_from_ego @ T_ego_from_cam
            T_cam_from_world = np.linalg.inv(T_world_from_cam)
            # Build camera-from-<frame> transform depending on point frame
            if pcd_frame == 'sensor':
                # points are in LiDAR sensor frame
                T_cam_from_lidar = np.linalg.inv(T_world_from_cam) @ T_world_from_lidar
                # transform all points once per camera
                Psh = np.concatenate([Pw, np.ones((Pw.shape[0],1))], axis=1).T
                Pc_all = (T_cam_from_lidar @ Psh)[:3]
            elif pcd_frame == 'world':
                # points are already in world
                Pwh = np.concatenate([Pw, np.ones((Pw.shape[0],1))], axis=1).T
                Pc_all = (T_cam_from_world @ Pwh)[:3]
            elif pcd_frame == 'ego':
                # points are in ego frame at that timestamp
                T_cam_from_ego = np.linalg.inv(T_ego_from_cam)
                Peh = np.concatenate([Pw, np.ones((Pw.shape[0],1))], axis=1).T
                Pc_all = (T_cam_from_ego @ Peh)[:3]
            else:
                raise ValueError("pcd_frame must be one of {'sensor','world','ego'}")

            # image size: if not stored in sample_data, derive from file
            img_path = samples_dir / cam_sd['filename']
            img = cv2.imread(str(img_path))
            if img is None:
                print('Missing image', img_path)
                ok = False
                continue
            h, w = img.shape[:2]
            # stats from Pc_all
            z = Pc_all[2]
            valid_mask = z > 0.1
            inside_mask = np.zeros_like(valid_mask, dtype=bool)
            if np.any(valid_mask):
                uv = (K @ (Pc_all[:, valid_mask] / Pc_all[2, valid_mask]))
                u, v = uv[0], uv[1]
                inside_mask_idx = (u >= 0) & (u < w) & (v >= 0) & (v < h)
                # scatter back into full-size mask for counts
                inside_mask[np.where(valid_mask)[0][inside_mask_idx]] = True
            valid = int(valid_mask.sum())
            inside = int(inside_mask.sum())
            ratio = float(inside / max(1, valid))
            print(f"{cam_sd['filename']}: points_front={valid}, in_image={inside}, ratio={ratio:.3f}")
            if save_overlays:
                # dense overlay with depth coloring
                Pc = Pc_all[:, valid_mask]
                if Pc.shape[1] > 0:
                    uv = (K @ (Pc / Pc[2]))
                    u, v = uv[0], uv[1]
                    m = (u >= 0) & (u < w) & (v >= 0) & (v < h)
                    overlay = img.copy()
                    # color by depth
                    zc = z[valid_mask][m]
                    colors = depth_colormap(zc, 0.0, 80.0)
                    uu = u[m].astype(int)
                    vv = v[m].astype(int)
                    for (uu_i, vv_i), col in zip(zip(uu, vv), colors):
                        cv2.circle(overlay, (int(uu_i), int(vv_i)), 1, tuple(int(c) for c in col.tolist()), -1)
                    # draw 3D boxes if available for this sample
                    anns = [r for r in idx.get('sample_annotation', {}).values() if r.get('sample_token') == s['token']]
                    for ann in anns:
                        t = np.array(ann['translation'], dtype=np.float64)
                        R = quat_wxyz_to_rot(ann['rotation'])
                        wlh = ann['size']  # [w, l, h]
                        corners = box_corners_wlh(wlh[0], wlh[1], wlh[2])  # (8,3)
                        Pw_box = (R @ corners.T).T + t[None, :]
                        Pwh_box = np.concatenate([Pw_box, np.ones((8,1))], axis=1).T
                        Pc_box = (T_cam_from_world @ Pwh_box)[:3]
                        if np.all(Pc_box[2] > 0.1):
                            uv_box = (K @ (Pc_box / Pc_box[2]))
                            uv_box = np.stack([uv_box[0], uv_box[1]], axis=1).T  # (2,8)
                            uv_box = uv_box.T  # (8,2)
                            draw_box(overlay, uv_box, color=(0,255,255))
                    # stats text
                    txt = f"{cam_sd.get('__channel','')}: front={valid} in={inside} ratio={ratio:.3f} (pcd:{pcd_frame})"
                    cv2.putText(overlay, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                    out_rel = Path(cam_sd['filename']).with_suffix('')
                    out_path = (vis_dir / (str(out_rel).replace('/', '_') + '_overlay.png'))
                    cv2.imwrite(str(out_path), overlay)
    if ok:
        print('TF validation finished: no blocking issues found (see ratios above).')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True, help='Export root, e.g., data/nuscenes')
    ap.add_argument('--scene-idx', type=int, default=0)
    ap.add_argument('--max-samples', type=int, default=3)
    ap.add_argument('--save-overlays', action='store_true')
    ap.add_argument('--nus-version', type=str, default=None, help='Version folder under root; auto-detect if omitted')
    ap.add_argument('--out-dir', type=str, default=None, help='Directory to write visualization overlays; default root/vis')
    ap.add_argument('--lidar-channel', type=str, default=None, help='Exact lidar channel to use (e.g., LIDAR_TOP)')
    ap.add_argument('--cam-regex', type=str, default=None, help='Regex to select camera channels (e.g., ^CAM_.*FRONT.*$)')
    ap.add_argument('--max-points', type=int, default=200000, help='Max LiDAR points to project for speed')
    ap.add_argument('--pcd-frame', type=str, default='sensor', choices=['sensor','world','ego'], help='Frame of PCD coordinates')
    args = ap.parse_args()
    out_dir = Path(args.out_dir) if args.out_dir else None
    run(Path(args.root), args.scene_idx, args.max_samples, args.save_overlays, nus_version=args.nus_version, out_dir=out_dir, lidar_channel=args.lidar_channel, cam_regex=args.cam_regex, max_points=args.max_points, pcd_frame=args.pcd_frame)
