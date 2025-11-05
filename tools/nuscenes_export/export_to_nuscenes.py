#!/usr/bin/env python3
"""
Export SkyLink-V2X runs to a nuScenes-style dataset (v1.0-custom).

MVP: RGB cameras + LiDAR for a chosen ego (vehicle or drone), with GT 3D boxes from objects.pkl.

Tables written: sensor, calibrated_sensor, ego_pose, sample, sample_data, scene, log, instance, sample_annotation, category, attribute, visibility, map.
"""
import os
import re
import json
import uuid
import shutil
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from collections import defaultdict


# -------------------------------
# Helpers
# -------------------------------


def uuid5_str(namespace: uuid.UUID, name: str) -> str:
    return str(uuid.uuid5(namespace, name))


NS = uuid.UUID("12345678-1234-5678-1234-567812345678")  # deterministic namespace


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def list_timestamps(run_folder: Path) -> List[Path]:
    return sorted([p for p in run_folder.iterdir() if p.is_dir() and p.name.startswith("timestamp_")])


def extract_ts_index(name: str) -> Optional[int]:
    m = re.match(r"timestamp_(\d+)$", name)
    return int(m.group(1)) if m else None


# removed: legacy cords_to_T; use cords_to_T_with_order instead


def cords_to_T_with_order(cords: List[float], angle_order: str = 'roll-yaw-pitch', cords_field_order: str = 'r-y-p') -> np.ndarray:
    """
    Build SE3 from cords with configurable Euler order.
    cords expected shape: [x, y, z, roll, yaw, pitch] in degrees.

    angle_order options:
    - 'roll-yaw-pitch': R = Rz(yaw) @ Ry(pitch) @ Rx(roll)  (default, matches legacy cords_to_T)
    - 'roll-pitch-yaw': R = Ry(pitch) @ Rz(yaw) @ Rx(roll)
    """
    x, y, z = cords[0], cords[1], cords[2]
    # Determine which indices correspond to yaw/pitch in metadata
    # cords_field_order: 'r-y-p' means cords[3]=roll, cords[4]=yaw, cords[5]=pitch
    #                    'r-p-y' means cords[3]=roll, cords[4]=pitch, cords[5]=yaw
    roll = cords[3]
    if cords_field_order == 'r-p-y':
        pitch = cords[4]; yaw = cords[5]
    else:  # 'r-y-p'
        yaw = cords[4]; pitch = cords[5]

    r = np.deg2rad(roll); yv = np.deg2rad(yaw); p = np.deg2rad(pitch)
    cr, sr = np.cos(r), np.sin(r)
    cy, sy = np.cos(yv), np.sin(yv)
    cp, sp = np.cos(p), np.sin(p)

    # Rx, Ry, Rz components
    Rx = np.array([[1, 0, 0],
                   [0, cr, -sr],
                   [0, sr, cr]], dtype=np.float64)
    Ry = np.array([[cp, 0, sp],
                   [0, 1, 0],
                   [-sp, 0, cp]], dtype=np.float64)
    Rz = np.array([[cy, -sy, 0],
                   [sy,  cy, 0],
                   [0,    0, 1]], dtype=np.float64)

    if angle_order == 'roll-pitch-yaw':
        R = Ry @ Rz @ Rx
    else:  # 'roll-yaw-pitch' (default)
        R = Rz @ Ry @ Rx

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[0, 3] = x; T[1, 3] = y; T[2, 3] = z
    return T


def rotmat_to_quat(R: np.ndarray) -> List[float]:
    # Returns [w,x,y,z] per nuScenes convention
    q = np.empty(4, dtype=np.float64)
    t = np.trace(R)
    if t > 0.0:
        s = np.sqrt(t + 1.0) * 2.0
        q[0] = 0.25 * s
        q[1] = (R[2, 1] - R[1, 2]) / s
        q[2] = (R[0, 2] - R[2, 0]) / s
        q[3] = (R[1, 0] - R[0, 1]) / s
    else:
        i = np.argmax([R[0, 0], R[1, 1], R[2, 2]])
        if i == 0:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            q[0] = (R[2, 1] - R[1, 2]) / s
            q[1] = 0.25 * s
            q[2] = (R[0, 1] + R[1, 0]) / s
            q[3] = (R[0, 2] + R[2, 0]) / s
        elif i == 1:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            q[0] = (R[0, 2] - R[2, 0]) / s
            q[1] = (R[0, 1] + R[1, 0]) / s
            q[2] = 0.25 * s
            q[3] = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            q[0] = (R[1, 0] - R[0, 1]) / s
            q[1] = (R[0, 2] + R[2, 0]) / s
            q[2] = (R[1, 2] + R[2, 1]) / s
            q[3] = 0.25 * s
    # Normalize to avoid numerical scaling issues
    norm = np.linalg.norm(q)
    if norm > 1e-12:
        q = q / norm
    return [float(q[0]), float(q[1]), float(q[2]), float(q[3])]


# -------------------------------
# Frame conversions (CARLA -> nuScenes)
# -------------------------------

# Handedness change-of-basis: flip Y axis (left-handed Unreal/CARLA -> right-handed nuScenes/CV)
S_FLIP_Y = np.diag([1.0, -1.0, 1.0])

# Unreal camera axes to CV camera axes: X_cv=+Y_unreal, Y_cv=-Z_unreal, Z_cv=+X_unreal
R_FIX_CAM = np.array([[0.0, 1.0, 0.0],
                      [0.0, 0.0, -1.0],
                      [1.0, 0.0, 0.0]], dtype=float)


def flip_R_carla_to_nus(R_carla: np.ndarray) -> np.ndarray:
    return S_FLIP_Y @ R_carla @ S_FLIP_Y


def flip_t_carla_to_nus(t_carla: np.ndarray) -> np.ndarray:
    return (S_FLIP_Y @ t_carla.reshape(3, 1)).reshape(3)


def flip_T_carla_to_nus(T_carla: np.ndarray) -> np.ndarray:
    Rn = flip_R_carla_to_nus(T_carla[:3, :3])
    tn = flip_t_carla_to_nus(T_carla[:3, 3])
    Tn = np.eye(4, dtype=float)
    Tn[:3, :3] = Rn
    Tn[:3, 3] = tn
    return Tn


def load_pickle(path: Path):
    import pickle
    with open(path, 'rb') as f:
        return pickle.load(f)


def load_json(path: Path):
    with open(path, 'r') as f:
        return json.load(f)


def try_symlink_or_copy(src: Path, dst: Path, mode: str = "symlink"):
    ensure_dir(dst.parent)
    if mode == "symlink":
        try:
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            dst.symlink_to(src.resolve())
            return
        except Exception:
            pass
    shutil.copy2(src, dst)


# removed: unused helpers is_camera_sensor_entry / is_lidar_sensor_entry


# -------------------------------
# Export logic
# -------------------------------

def export_run(run_folder: Path,
               out_root: Path,
               ego_mode: str,
               fps: float,
               copy_mode: str,
               channel_map: Dict,
               category_map: Dict,
               max_frames: Optional[int] = None,
               nus_version: str = "v1.0-trainval",
               meta_root: Optional[Path] = None):
    timestamps = list_timestamps(run_folder)
    if not timestamps:
        print(f"No timestamp_* folders in {run_folder}")
        return

    # Gather present agents per frame to find candidate egos
    agent_ids = set()
    for ts in timestamps:
        for p in ts.iterdir():
            if p.is_dir() and p.name.startswith('agent_'):
                try:
                    agent_ids.add(int(p.name.split('_')[-1]))
                except Exception:
                    pass

    # Determine ego candidates
    def agent_type_for(ts_path: Path, aid: int) -> str:
        meta = ts_path / f"agent_{aid:06d}" / "metadata.pkl"
        if not meta.exists():
            return ''
        try:
            m = load_pickle(meta)
            return str(m.get('agent_type', '')).lower()
        except Exception:
            return ''

    first_ts = timestamps[0]
    candidates = []
    for aid in sorted(agent_ids):
        t = agent_type_for(first_ts, aid)
        if ego_mode == 'vehicle' and t == 'vehicle':
            candidates.append((aid, t))
        elif ego_mode == 'drone' and t == 'drone':
            candidates.append((aid, t))
        elif ego_mode == 'both' and t in ('vehicle', 'drone'):
            candidates.append((aid, t))

    if not candidates:
        print(f"No ego candidates found for mode {ego_mode}")
        return

    # Prepare output folders
    ver_dir = out_root / nus_version
    samples_dir = out_root / "samples"
    ensure_dir(ver_dir)
    ensure_dir(samples_dir)

    # Load existing tables for incremental export
    def read_table_or_empty(name: str) -> List[Dict]:
        p = ver_dir / f"{name}.json"
        if p.exists():
            try:
                with open(p, 'r') as f:
                    return json.load(f)
            except Exception:
                return []
        return []

    existing_sensor_rows = read_table_or_empty("sensor")
    existing_calib_rows = read_table_or_empty("calibrated_sensor")
    existing_ego_pose_rows = read_table_or_empty("ego_pose")
    existing_sample_rows = read_table_or_empty("sample")
    existing_sample_data_rows = read_table_or_empty("sample_data")
    existing_scene_rows = read_table_or_empty("scene")
    existing_log_rows = read_table_or_empty("log")
    existing_instance_rows = read_table_or_empty("instance")
    existing_ann_rows = read_table_or_empty("sample_annotation")
    existing_category_rows = read_table_or_empty("category")
    existing_attribute_rows = read_table_or_empty("attribute")
    existing_visibility_rows = read_table_or_empty("visibility")
    existing_map_rows = read_table_or_empty("map")

    existing_scene_names = {r.get('name','') for r in existing_scene_rows}
    # Determine next scene index in nuScenes style (scene-0001, scene-0002, ...)
    scene_index_re = re.compile(r"^scene-(\d+)$")
    existing_scene_indices = []
    for r in existing_scene_rows:
        name = r.get('name', '')
        m = scene_index_re.match(name)
        if m:
            try:
                existing_scene_indices.append(int(m.group(1)))
            except Exception:
                pass
    next_scene_index = (max(existing_scene_indices) + 1) if existing_scene_indices else 1
    existing_scene_tokens = {r.get('token') for r in existing_scene_rows}
    existing_tokens = {
        'sensor': {r.get('token') for r in existing_sensor_rows},
        'calib': {r.get('token') for r in existing_calib_rows},
        'ego_pose': {r.get('token') for r in existing_ego_pose_rows},
        'sample': {r.get('token') for r in existing_sample_rows},
        'sample_data': {r.get('token') for r in existing_sample_data_rows},
        'scene': existing_scene_tokens,
        'log': {r.get('token') for r in existing_log_rows},
        'instance': {r.get('token') for r in existing_instance_rows},
        'ann': {r.get('token') for r in existing_ann_rows},
        'category': {r.get('token') for r in existing_category_rows},
        'attribute': {r.get('token') for r in existing_attribute_rows},
        'visibility': {r.get('token') for r in existing_visibility_rows},
        'map': {r.get('token') for r in existing_map_rows},
    }

    # Tables (new rows produced in this run)
    sensor_rows = []
    calib_rows = []
    ego_pose_rows = []
    sample_rows = []
    sample_data_rows = []
    scene_rows = []
    log_rows = []
    instance_rows = []
    ann_rows = []
    category_rows = []
    attribute_rows = []
    visibility_rows = []
    map_rows = []
    # Deferred scene stats updates for incremental merge
    scene_stats_updates: Dict[str, Tuple[str, str, int]] = {}
    # Mapping stats for debugging category coverage
    mapping_stats = {
        'total_annotations': 0,
        'mapped': 0,
        'dropped': 0,
        'by_source': defaultdict(lambda: {'count': 0, 'last_mapped_to': None})
    }

    # Helpers for linking (will be reset per ego/scene)
    sd_token_to_index: Dict[str, int] = {}
    last_sd_token_by_channel: Dict[str, str] = {}
    sample_token_to_index: Dict[str, int] = {}
    inst_token_to_ann_list: Dict[str, List[Tuple[int, str]]] = {}
    ann_token_to_index: Dict[str, int] = {}

    # Seed category table from mapping file (or meta for test)
    seen_categories = set()
    if nus_version.endswith('test') and (meta_root is not None) and not existing_category_rows:
        try:
            with open((Path(meta_root) / nus_version / 'category.json'), 'r') as f:
                category_rows = json.load(f)
                seen_categories = {r.get('name') for r in category_rows}
        except Exception:
            category_rows = []
            seen_categories = set()
    for group, m in category_map.items():
        for k, v in m.items():
            if v not in seen_categories:
                category_rows.append({
                    "token": uuid5_str(NS, f"category:{v}"),
                    "name": v,
                    "description": f"mapped from {group}:{k}"
                })
                seen_categories.add(v)

    # Export per ego
    for (ego_aid, ego_type) in candidates:
        # Reset per-ego/scene link state to avoid cross-scene chaining
        sd_token_to_index = {}
        last_sd_token_by_channel = {}
        sample_token_to_index = {}
        inst_token_to_ann_list = {}
        ann_token_to_index = {}
        print(f"Exporting ego {ego_aid} type={ego_type}")
        log_token = uuid5_str(NS, f"log:{run_folder}:{ego_aid}")
        if log_token not in existing_tokens['log']:
            log_rows.append({
                "token": log_token,
                "logfile": str(run_folder),
                "vehicle": ego_type,
                "date_captured": "",
                "location": "carla"
            })

        # Single scene per run for now
        scene_token = uuid5_str(NS, f"scene:{run_folder}:{ego_aid}")
        # scene name in nuScenes format: scene-XXXX (zero-padded), incrementing across exports
        # Ensure uniqueness even when exporting multiple egos in one call
        while True:
            scene_name = f"scene-{next_scene_index:04d}"
            next_scene_index += 1
            if scene_name not in existing_scene_names:
                break
        # Skip creating a duplicate scene if token already exists
        if scene_token not in existing_tokens['scene']:
            scene_rows.append({
                "token": scene_token,
                "name": scene_name,
                "description": f"SkyLink export (ego {ego_aid}, mode {ego_mode})",
                "log_token": log_token,
                "nbr_samples": 0,  # filled later
                "first_sample_token": "",
                "last_sample_token": ""
            })
            existing_scene_names.add(scene_name)

        # Build sensor/calibrated_sensor from first frame metadata
        first_meta_path = first_ts / f"agent_{ego_aid:06d}" / "metadata.pkl"
        if not first_meta_path.exists():
            print(f"  Skip ego {ego_aid}: missing metadata at first frame")
            continue
        first_meta = load_pickle(first_meta_path)

        # Scan sensors present on ego (RGB cameras + LiDAR). Metadata does not store sensor_type,
        # so infer modality from available fields, filename presence, and naming.
        sensor_name_to_tokens = {}
        agent_dir_first = first_ts / f"agent_{ego_aid:06d}"
        for sname, sentry in first_meta.items():
            if not isinstance(sentry, dict):
                continue
            if sname in ("agent_type", "odometry"):
                continue

            sname_l = sname.lower()
            img_first = agent_dir_first / f"{sname}.png"
            pcd_first = agent_dir_first / f"{sname}.pcd"

            modality = None
            cords = None

            # Prefer explicit LiDAR metadata when a matching file exists
            if 'lidar_pose' in sentry and pcd_first.exists():
                modality = 'lidar'
                cords = sentry.get('lidar_pose', None)
            # Camera metadata typically has 'cords' and 'intrinsic'; filter out depth/segmentation by name
            elif ('cords' in sentry and 'intrinsic' in sentry and img_first.exists()
                  and ('depth' not in sname_l) and ('seg' not in sname_l)):
                modality = 'camera'
                cords = sentry.get('cords', None)

            if modality is None or cords is None or len(cords) < 6:
                continue

            # Map to nuScenes channel
            if modality == 'camera':
                channel = channel_map.get('camera', {}).get(sname, f"CAM_CUSTOM_{sname}")
            else:
                channel = channel_map.get('lidar', {}).get(sname, f"LIDAR_{sname.upper()}")

            sensor_token = uuid5_str(NS, f"sensor:{ego_aid}:{sname}:{channel}")
            if sensor_token not in existing_tokens['sensor']:
                sensor_rows.append({
                "token": sensor_token,
                "channel": channel,
                "modality": modality
                })

            # Calibrated sensor (sensor->ego extrinsics) and intrinsics for cameras
            # Build sensor->ego from metadata cords using fixed conventions
            # Conventions baked-in: cords are [x,y,z, roll, yaw, pitch] (r-y-p),
            # angle order roll-yaw-pitch, and the cords encode ego->sensor (need to invert rotation only).
            T_es_raw = cords_to_T_with_order(cords, 'roll-yaw-pitch', 'r-y-p')
            R_es = T_es_raw[:3, :3]
            t_es = T_es_raw[:3, 3]
            # Invert ONLY the rotation (rotation needs to be inverted for sensor->ego)
            # Translation represents sensor POSITION in ego frame, use directly
            R_carla = R_es.T
            t_carla = t_es  # Use position directly, not inverted!
            # Convert to nuScenes right-handed frame; apply camera axis fix for cameras
            if modality == 'camera':
                # Basis-change for target (ego): Unreal -> nuScenes uses a handedness flip on Y.
                # For the source (camera) frame: Unreal camera axes -> CV camera axes via R_FIX_CAM.
                # General rule: R_new = C_target * R_old * C_source^{-1}
                # Here: C_target = S_FLIP_Y, C_source = R_FIX_CAM (CV_from_Unreal), so C_source^{-1} = R_FIX_CAM.T
                
                # Per-camera correction: yaw angle affects how handedness flip interacts with rotation
                # Extract yaw from original metadata to determine correction
                yaw_deg = cords[4]  # cords are [x, y, z, roll, yaw, pitch]
                
                # Base transform works for yaw=0° (CAM_FRONT, CAM_DOWN)
                R_nus = (S_FLIP_Y @ R_carla) @ R_FIX_CAM.T
                
                # Apply yaw-specific correction for non-zero yaw cameras
                if abs(yaw_deg - 90.0) < 1.0:  # CAM_LEFT (yaw=90°)
                    # Need to negate yaw effect after handedness flip
                    R_yaw_correction = np.array([
                        [0, -1, 0],
                        [1, 0, 0],
                        [0, 0, 1]
                    ], dtype=np.float64)
                    R_nus = R_nus @ R_yaw_correction
                elif abs(yaw_deg + 90.0) < 1.0:  # CAM_RIGHT (yaw=-90°)
                    # Need to negate yaw effect after handedness flip
                    R_yaw_correction = np.array([
                        [0, 1, 0],
                        [-1, 0, 0],
                        [0, 0, 1]
                    ], dtype=np.float64)
                    R_nus = R_nus @ R_yaw_correction
                elif abs(abs(yaw_deg) - 180.0) < 1.0:  # CAM_BACK (yaw=±180°)
                    # Need to negate yaw effect after handedness flip
                    R_yaw_correction = np.array([
                        [-1, 0, 0],
                        [0, -1, 0],
                        [0, 0, 1]
                    ], dtype=np.float64)
                    R_nus = R_nus @ R_yaw_correction
                
                t_nus = flip_t_carla_to_nus(t_carla)
            else:
                R_nus = flip_R_carla_to_nus(R_carla)
                t_nus = flip_t_carla_to_nus(t_carla)
            q = rotmat_to_quat(R_nus)
            t = [float(t_nus[0]), float(t_nus[1]), float(t_nus[2])]
            calib_token = uuid5_str(NS, f"calib:{ego_aid}:{sname}")
            row = {
                "token": calib_token,
                "sensor_token": sensor_token,
                "translation": t,
                "rotation": q,
                "camera_intrinsic": []
            }
            if modality == 'camera':
                K = sentry.get('intrinsic', None)
                if K is None:
                    # try compute from fov/res if present in metadata
                    try:
                        w = int(sentry.get('image_size_x'))
                        h = int(sentry.get('image_size_y'))
                        fov = float(sentry.get('fov'))
                        fx = w / (2.0 * np.tan(np.deg2rad(fov) / 2.0))
                        cx, cy = w / 2.0, h / 2.0
                        row["camera_intrinsic"] = [[fx, 0, cx], [0, fx, cy], [0, 0, 1]]
                    except Exception:
                        row["camera_intrinsic"] = [[0,0,0],[0,0,0],[0,0,1]]
                else:
                    k = np.array(K, dtype=float).reshape(3, 3).tolist()
                    row["camera_intrinsic"] = k
            if calib_token not in existing_tokens['calib']:
                calib_rows.append(row)
            sensor_name_to_tokens[sname] = (sensor_token, calib_token, channel, modality)

    # Iterate frames → samples (prev/next like official), ego_poses, sample_data (with prev/next), and annotations
        prev_sample_token = ""
        start_ts_us = 0
        dt_us = int(1e6 / max(1e-6, fps))
        sample_tokens = []
        frame_count = 0
        for ts_idx, ts in enumerate(timestamps):
            if max_frames is not None and frame_count >= max_frames:
                break
            ego_meta_path = ts / f"agent_{ego_aid:06d}" / "metadata.pkl"
            if not ego_meta_path.exists():
                continue
            m = load_pickle(ego_meta_path)
            odom = (m.get('odometry') or {})
            ego_pos = odom.get('ego_pos', None)
            if ego_pos is None or len(ego_pos) < 6:
                continue

            # Timestamp (µs)
            ts_us = start_ts_us + ts_idx * dt_us

            # ego_pose
            # Ego pose: fixed conventions as above
            T_we = cords_to_T_with_order(ego_pos, 'roll-yaw-pitch', 'r-y-p')
            R_we_nus = flip_R_carla_to_nus(T_we[:3, :3])
            t_we_nus = flip_t_carla_to_nus(T_we[:3, 3])
            q = rotmat_to_quat(R_we_nus)
            t = [float(t_we_nus[0]), float(t_we_nus[1]), float(t_we_nus[2])]
            ego_pose_token = uuid5_str(NS, f"ego_pose:{ego_aid}:{ts.name}:{t}")
            if ego_pose_token not in existing_tokens['ego_pose']:
                ego_pose_rows.append({
                "token": ego_pose_token,
                "timestamp": int(ts_us),
                "rotation": q,
                "translation": t
                })

            # sample (official format: no embedded data/anns; only prev/next and scene_token)
            sample_token = uuid5_str(NS, f"sample:{ego_aid}:{ts.name}")
            sample_row = {
                "token": sample_token,
                "timestamp": int(ts_us),
                "scene_token": scene_token,
                "prev": prev_sample_token,
                "next": ""
            }
            sample_exists = sample_token in existing_tokens['sample']
            if not sample_exists:
                sample_rows.append(sample_row)
                cur_sample_index = len(sample_rows) - 1
                sample_token_to_index[sample_token] = cur_sample_index
                if prev_sample_token:
                    # patch prev sample's next
                    prev_idx = sample_token_to_index[prev_sample_token]
                    sample_rows[prev_idx]["next"] = sample_token
            sample_tokens.append(sample_token)

            # sample_data for ego sensors
            for sname, (sensor_token, calib_token, channel, modality) in sensor_name_to_tokens.items():
                agent_dir = ts / f"agent_{ego_aid:06d}"
                if modality == 'camera':
                    fpath = agent_dir / f"{sname}.png"
                else:
                    # lidar saved as <name>.pcd in our logger
                    fpath = agent_dir / f"{sname}.pcd"
                if not fpath.exists():
                    continue

                # Ensure camera channels start with CAM_ for downstream loaders
                ch = channel
                if modality == 'camera' and not ch.upper().startswith('CAM_'):
                    ch = f"CAM_{ch}"

                rel_dst = Path("samples") / ch / f"{ts.name}_{ego_aid:06d}_{fpath.name}"
                try_symlink_or_copy(fpath, out_root / rel_dst, copy_mode)
                sd_token = uuid5_str(NS, f"sd:{ego_aid}:{ts.name}:{sname}")
                # prev/next chaining per channel
                prev_sd = last_sd_token_by_channel.get(ch, "")
                sd_row = {
                    "token": sd_token,
                    "sample_token": sample_token,
                    "ego_pose_token": ego_pose_token,
                    "calibrated_sensor_token": calib_token,
                    "filename": str(rel_dst),
                    "fileformat": "png" if modality == 'camera' else "pcd",
                    "width": int(first_meta.get(sname, {}).get('image_size_x', 0)) if modality == 'camera' else 0,
                    "height": int(first_meta.get(sname, {}).get('image_size_y', 0)) if modality == 'camera' else 0,
                    "timestamp": int(ts_us),
                    "is_key_frame": True,
                    "prev": prev_sd,
                    "next": ""
                }
                if sd_token not in existing_tokens['sample_data']:
                    sample_data_rows.append(sd_row)
                    sd_token_to_index[sd_token] = len(sample_data_rows) - 1
                if prev_sd:
                    # patch previous sd next
                    # Only patch if previous sd is in this batch; cross-batch chaining is not supported to avoid cross-scene links
                    if prev_sd in sd_token_to_index:
                        prev_idx = sd_token_to_index[prev_sd]
                        sample_data_rows[prev_idx]["next"] = sd_token
                last_sd_token_by_channel[ch] = sd_token

                # (nuScenes stores sample_data relation via sample_token; do not embed into sample row)

            # sample_annotation from timestamp objects.pkl
            obj_pkl = ts / "objects.pkl"
            create_anns = not nus_version.endswith('test')
            if create_anns and obj_pkl.exists():
                try:
                    objects = load_pickle(obj_pkl)
                except Exception:
                    objects = {}
                # Track tokens per sample if needed for debugging; nuScenes does not embed in sample.json
                ann_tokens_this_sample: List[str] = []
                for k, entry in objects.items():
                    if not isinstance(entry, dict):
                        continue
                    try:
                        aid = int(k)
                    except Exception:
                        continue
                    # 3D box pose
                    loc = entry.get('location', None)
                    center = entry.get('center', [0, 0, 0, 0, 0, 0])
                    ext = entry.get('extent', [1, 0.5, 1])
                    if loc is None or len(ext) < 3:
                        continue
                    # Compose world T of bbox center
                    T_actor = cords_to_T_with_order(loc, 'roll-yaw-pitch', 'r-y-p')
                    T_bb = cords_to_T_with_order(center, 'roll-yaw-pitch', 'r-y-p')
                    # Use actor_plus_center by default
                    T = T_actor @ T_bb
                    R = T[:3, :3]
                    t_vec = T[:3, 3]
                    # Convert annotation pose to nuScenes right-handed global frame
                    R_nus = flip_R_carla_to_nus(R)
                    t_nus = flip_t_carla_to_nus(t_vec)
                    q = rotmat_to_quat(R_nus)
                    t = [float(t_nus[0]), float(t_nus[1]), float(t_nus[2])]
                    # nuScenes size: width, length, height
                    # Fixed mapping: size_map='yx'
                    l = float(2.0 * ext[0]); w = float(2.0 * ext[1]); h = float(2.0 * ext[2])
                    # Map category; prefer explicit type/category, then CARLA blueprint id (bp_id), else fallback
                    raw_type_val = entry.get('type') or entry.get('category') or entry.get('bp_id')
                    if raw_type_val is None:
                        # As a last resort, try numeric class id (dataset-specific); encode as string for mapping rules
                        cls_id = entry.get('class')
                        raw_type_val = f"class:{cls_id}" if cls_id is not None else 'object.unknown'
                    raw_type = str(raw_type_val)
                    tname = raw_type.lower()
                    mapping_stats['total_annotations'] += 1
                    category_name = map_category_name(tname, category_map)
                    if not category_name:
                        mapping_stats['dropped'] += 1
                        s = mapping_stats['by_source'][tname]
                        s['count'] += 1
                        s['last_mapped_to'] = None
                        continue
                    mapping_stats['mapped'] += 1
                    s = mapping_stats['by_source'][tname]
                    s['count'] += 1
                    s['last_mapped_to'] = category_name
                    # Ensure category exists in category table (incremental-safe)
                    if category_name not in {r.get('name') for r in (category_rows or [])}:
                        category_rows.append({
                            "token": uuid5_str(NS, f"category:{category_name}"),
                            "name": category_name,
                            "description": f"auto-added from object type '{tname}'"
                        })
                    # Instance token per actor id within this scene (avoid cross-scene merges)
                    inst_token = uuid5_str(NS, f"instance:{scene_token}:{aid}")
                    if not any(r.get('token') == inst_token for r in instance_rows):
                        instance_rows.append({
                            "token": inst_token,
                            "category_token": uuid5_str(NS, f"category:{category_name}"),
                            "nbr_annotations": 0,
                            "first_annotation_token": "",
                            "last_annotation_token": ""
                        })
                    ann_token = uuid5_str(NS, f"ann:{sample_token}:{aid}")
                    ann_row = {
                        "token": ann_token,
                        "sample_token": sample_token,
                        "instance_token": inst_token,
                        "translation": t,
                        "size": [w, l, h],
                        "rotation": q,
                        "num_lidar_pts": 1,
                        "num_radar_pts": 0,
                        "visibility_token": "4",
                        "attribute_tokens": [],
                        "category_name": category_name,
                        "prev": "",
                        "next": ""
                    }
                    ann_rows.append(ann_row)
                    ann_token_to_index[ann_token] = len(ann_rows) - 1
                    # Track per-instance temporal order
                    inst_token_to_ann_list.setdefault(inst_token, []).append((int(ts_us), ann_token))
                    ann_tokens_this_sample.append(ann_token)

                # Do not attach anns into sample row (official schema)

            prev_sample_token = sample_token
            frame_count += 1

        # Update scene first/last sample tokens and nbr_samples (defer to merge stage)
        if sample_tokens:
            scene_stats_updates[scene_token] = (sample_tokens[0], sample_tokens[-1], len(sample_tokens))

        # Link annotations prev/next per instance and update instance first/last/nbr_annotations
        if not nus_version.endswith('test'):
            for inst_token, items in inst_token_to_ann_list.items():
                items_sorted = sorted(items, key=lambda x: x[0])  # by timestamp
                if not items_sorted:
                    continue
                first_ann_tok = items_sorted[0][1]
                last_ann_tok = items_sorted[-1][1]
                # Update chain
                for i in range(len(items_sorted)):
                    tok = items_sorted[i][1]
                    idx = ann_token_to_index.get(tok)
                    if idx is None:
                        continue
                    prev_tok = items_sorted[i-1][1] if i > 0 else ""
                    next_tok = items_sorted[i+1][1] if i < len(items_sorted)-1 else ""
                    ann_rows[idx]['prev'] = prev_tok
                    ann_rows[idx]['next'] = next_tok
                # Update instance row
                for r in instance_rows:
                    if r['token'] == inst_token:
                        r['first_annotation_token'] = first_ann_tok
                        r['last_annotation_token'] = last_ann_tok
                        r['nbr_annotations'] = len(items_sorted)
                        break

    # Seed attribute.json (official list) if empty; prefer meta for test if provided
    if not attribute_rows and not existing_attribute_rows:
        if nus_version.endswith('test') and (meta_root is not None):
            meta_ver_dir = Path(meta_root) / nus_version
            try:
                with open(meta_ver_dir / 'attribute.json', 'r') as f:
                    attribute_rows = json.load(f)
            except Exception:
                attribute_rows = []
        if not attribute_rows:
            attribute_rows = [
            {"token": "cb5118da1ab342aa947717dc53544259", "name": "vehicle.moving", "description": "Vehicle is moving."},
            {"token": "c3246a1e22a14fcb878aa61e69ae3329", "name": "vehicle.stopped", "description": "Vehicle, with a driver/rider in/on it, is currently stationary but has an intent to move."},
            {"token": "58aa28b1c2a54dc88e169808c07331e3", "name": "vehicle.parked", "description": "Vehicle is stationary (usually for longer duration) with no immediate intent to move."},
            {"token": "a14936d865eb4216b396adae8cb3939c", "name": "cycle.with_rider", "description": "There is a rider on the bicycle or motorcycle."},
            {"token": "5a655f9751944309a277276b8f473452", "name": "cycle.without_rider", "description": "There is NO rider on the bicycle or motorcycle."},
            {"token": "03aa62109bf043afafdea7d875dd4f43", "name": "pedestrian.sitting_lying_down", "description": "The human is sitting or lying down."},
            {"token": "4d8821270b4a47e3a8a300cbec48188e", "name": "pedestrian.standing", "description": "The human is standing."},
            {"token": "ab83627ff28b465b85c427162dec722f", "name": "pedestrian.moving", "description": "The human is moving."}
            ]

    # Seed visibility.json (official 4 levels)
    if not visibility_rows and not existing_visibility_rows:
        if nus_version.endswith('test') and (meta_root is not None):
            meta_ver_dir = Path(meta_root) / nus_version
            try:
                with open(meta_ver_dir / 'visibility.json', 'r') as f:
                    visibility_rows = json.load(f)
            except Exception:
                visibility_rows = []
        if not visibility_rows:
            visibility_rows = [
            {"description": "visibility of whole object is between 0 and 40%", "token": "1", "level": "v0-40"},
            {"description": "visibility of whole object is between 40 and 60%", "token": "2", "level": "v40-60"},
            {"description": "visibility of whole object is between 60 and 80%", "token": "3", "level": "v60-80"},
            {"description": "visibility of whole object is between 80 and 100%", "token": "4", "level": "v80-100"}
            ]

    # Seed map.json
    if nus_version.endswith('test') and (meta_root is not None):
        # Load map.json from meta and copy map images
        meta_ver_dir = Path(meta_root) / nus_version
        try:
            with open(meta_ver_dir / 'map.json', 'r') as f:
                map_rows = json.load(f)
            # Ensure map images exist in out_root/maps
            ensure_dir(out_root / 'maps')
            meta_maps_dir = Path(meta_root) / 'maps'
            for row in map_rows:
                src = meta_maps_dir / Path(row['filename']).name
                dst = out_root / 'maps' / Path(row['filename']).name
                if src.exists() and not dst.exists():
                    shutil.copy2(src, dst)
        except Exception:
            map_rows = []
    if not map_rows and not existing_map_rows:
        carla_map_token = uuid5_str(NS, "map:carla")
        map_rows = [{
            "category": "semantic_prior",
            "token": carla_map_token,
            "filename": f"maps/{carla_map_token}.png",
            "log_tokens": []
        }]
        # Create a tiny placeholder map image so consumers don't fail on missing file
        try:
            ensure_dir(out_root / 'maps')
            placeholder_path = out_root / 'maps' / f"{carla_map_token}.png"
            if not placeholder_path.exists():
                # Minimal 1x1 PNG bytes
                png_bytes = bytes([
                    0x89,0x50,0x4E,0x47,0x0D,0x0A,0x1A,0x0A,0x00,0x00,0x00,0x0D,0x49,0x48,0x44,0x52,
                    0x00,0x00,0x00,0x01,0x00,0x00,0x00,0x01,0x08,0x02,0x00,0x00,0x00,0x90,0x77,0x53,0xDE,
                    0x00,0x00,0x00,0x0A,0x49,0x44,0x41,0x54,0x78,0x9C,0x63,0x60,0x00,0x00,0x00,0x02,0x00,0x01,
                    0xE2,0x21,0xBC,0x33,0x00,0x00,0x00,0x00,0x49,0x45,0x4E,0x44,0xAE,0x42,0x60,0x82
                ])
                with open(placeholder_path, 'wb') as f:
                    f.write(png_bytes)
        except Exception:
            pass
    # Append new log tokens to first map entry (simple approach)
    all_new_log_tokens = [r["token"] for r in log_rows]
    if all_new_log_tokens:
        # if existing map rows exist, we'll merge later; for now ensure local map_rows gets logs
        if map_rows:
            lt = set(map_rows[0].get('log_tokens', []))
            for tkn in all_new_log_tokens:
                if tkn not in lt:
                    map_rows[0].setdefault('log_tokens', []).append(tkn)

    # Merge with existing tables (incremental)
    def merge_by_token(existing: List[Dict], new: List[Dict]) -> List[Dict]:
        seen = {r.get('token') for r in existing}
        merged = list(existing)
        for r in new:
            tok = r.get('token')
            if tok not in seen:
                merged.append(r)
                seen.add(tok)
        return merged

    sensor_rows = merge_by_token(existing_sensor_rows, sensor_rows)
    calib_rows = merge_by_token(existing_calib_rows, calib_rows)
    ego_pose_rows = merge_by_token(existing_ego_pose_rows, ego_pose_rows)
    sample_rows = merge_by_token(existing_sample_rows, sample_rows)
    sample_data_rows = merge_by_token(existing_sample_data_rows, sample_data_rows)
    # For scenes/logs we also merge by token
    scene_rows = merge_by_token(existing_scene_rows, scene_rows)
    # Apply deferred scene stats updates
    if scene_stats_updates:
        idx_by_token = {r.get('token'): i for i, r in enumerate(scene_rows)}
        for stoken, (fst, lst, cnt) in scene_stats_updates.items():
            i = idx_by_token.get(stoken)
            if i is not None:
                scene_rows[i]['first_sample_token'] = fst
                scene_rows[i]['last_sample_token'] = lst
                scene_rows[i]['nbr_samples'] = cnt
    log_rows = merge_by_token(existing_log_rows, log_rows)
    # For test split, annotations/instances may be empty; merge safely
    instance_rows = merge_by_token(existing_instance_rows, instance_rows)
    ann_rows = merge_by_token(existing_ann_rows, ann_rows)
    # Categories: keep existing if present, else merge
    category_rows = merge_by_token(existing_category_rows, category_rows)
    # Attributes/visibility: prefer existing if present, else use generated/default
    attribute_rows = existing_attribute_rows or attribute_rows
    visibility_rows = existing_visibility_rows or visibility_rows
    # Map: if existing present, prefer existing and extend log_tokens of first entry with any new logs
    if existing_map_rows:
        map_rows = existing_map_rows
        if map_rows and all_new_log_tokens:
            lt = set(map_rows[0].get('log_tokens', []))
            for tkn in all_new_log_tokens:
                if tkn not in lt:
                    map_rows[0].setdefault('log_tokens', []).append(tkn)
    else:
        map_rows = map_rows

    # Write tables
    write_table(ver_dir / "sensor.json", sensor_rows)
    write_table(ver_dir / "calibrated_sensor.json", calib_rows)
    write_table(ver_dir / "ego_pose.json", ego_pose_rows)
    write_table(ver_dir / "sample.json", sample_rows)
    write_table(ver_dir / "sample_data.json", sample_data_rows)
    write_table(ver_dir / "scene.json", scene_rows)
    write_table(ver_dir / "log.json", log_rows)
    write_table(ver_dir / "instance.json", instance_rows)
    write_table(ver_dir / "sample_annotation.json", ann_rows)
    write_table(ver_dir / "category.json", category_rows)
    write_table(ver_dir / "attribute.json", attribute_rows)
    write_table(ver_dir / "visibility.json", visibility_rows)
    write_table(ver_dir / "map.json", map_rows)

    # Write mapping stats for visibility if any annotations were processed
    try:
        if mapping_stats['total_annotations'] > 0:
            # Convert defaultdict to normal dict for JSON serialization
            by_src = {k: v for k, v in mapping_stats['by_source'].items()}
            stats_out = dict(mapping_stats)
            stats_out['by_source'] = by_src
            with open(ver_dir / 'mapping_stats.json', 'w') as f:
                json.dump(stats_out, f, indent=2)
    except Exception:
        pass

    print(f"Export complete → {out_root}")


def write_table(path: Path, rows: List[Dict]):
    ensure_dir(path.parent)
    with open(path, 'w') as f:
        json.dump(rows, f, indent=2)


def map_category_name(tname: str, cmap: Dict) -> Optional[str]:
    """
    Map source category/type string to a valid nuScenes category_name that the SDK
    can map into its 10-class detection taxonomy via category_to_detection_name.

    Returns None if unmappable, in which case the caller should drop the box.
    """
    t = (tname or "").lower()

    # 1) Explicit user-provided mappings (substring match)
    try:
        direct_map: Dict[str, str] = cmap.get('mapping', {})
        for patt, target in direct_map.items():
            if patt and patt.lower() in t:
                return target
    except Exception:
        pass

    # Common vehicle classes
    if "trailer" in t:
        return "vehicle.trailer"
    if any(k in t for k in ("cons", "construction")):
        return "vehicle.construction"
    if "truck" in t:
        return "vehicle.truck"
    if "bus" in t:
        return "vehicle.bus"
    # Motorcycle first to avoid matching 'bike' from 'motorbike'
    if any(k in t for k in ("motorcycle", "motorbike", "mcycle", "scooter")):
        return "vehicle.motorcycle"
    # Bicycle next (avoid false positives from 'motorbike' handled above)
    if any(k in t for k in ("bicycle", "bike", "cycle")):
        return "vehicle.bicycle"
    if any(k in t for k in ("car", "sedan", "vehicle")):
        return "vehicle.car"

    # Pedestrians
    if any(k in t for k in ("ped", "person", "walker", "human")):
        # Use the generic adult subcategory
        return "human.pedestrian.adult"

    # Movable objects
    if "cone" in t:
        return "movable_object.trafficcone"
    if "barrier" in t:
        return "movable_object.barrier"

    # Unmappable
    return None


def load_mapping_json(path: Optional[str], fallback: Path) -> Dict:
    if path:
        with open(path, 'r') as f:
            return json.load(f)
    with open(fallback, 'r') as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser(description="Export SkyLink run to nuScenes-style dataset")
    ap.add_argument('--run-folder', required=True)
    ap.add_argument('--out-root', required=True)
    ap.add_argument('--ego-mode', choices=['vehicle','drone','both'], default='vehicle')
    ap.add_argument('--fps', type=float, default=20.0)
    ap.add_argument('--copy-mode', choices=['copy','symlink'], default='symlink')
    ap.add_argument('--channel-map', type=str, default=None)
    ap.add_argument('--category-map', type=str, default=None)
    ap.add_argument('--max-frames', type=int, default=None)
    ap.add_argument('--nus-version', choices=['v1.0-trainval','v1.0-test'], default='v1.0-trainval', help='Target nuScenes version folder name')
    ap.add_argument('--meta-root', type=str, default=None, help='Path to meta root containing maps/ and version folder (used for test)')
    args = ap.parse_args()

    run_folder = Path(args.run_folder)
    out_root = Path(args.out_root)
    channel_map = load_mapping_json(args.channel_map, Path(__file__).parent / 'mappings' / 'channel_map_default.json')
    category_map = load_mapping_json(args.category_map, Path(__file__).parent / 'mappings' / 'category_map_default.json')

    export_run(run_folder, out_root, args.ego_mode, args.fps, args.copy_mode, channel_map, category_map, args.max_frames,
               nus_version=args.nus_version,
               meta_root=Path(args.meta_root) if args.meta_root else None)


if __name__ == '__main__':
    main()
