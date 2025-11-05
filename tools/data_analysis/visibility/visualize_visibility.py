#!/usr/bin/env python3
"""
Visualize world-frame per-vehicle visibility results.

Supports two modes:
1) Single file(s): pass one or more NPZ and/or JSON files to visualize.
2) Timestamp directory: pass a timestamp folder containing agent_*/ files, and it will
   overlay all agents' visibility masks (NPZ) and optionally polygons (JSON).

Usage examples:
  # Single actor NPZ
  python visualize_visibility.py --npz skylink_data/.../agent_000454/visibility_world_actor_000454.npz

  # Overlay its polygons (if generated)
  python visualize_visibility.py \
    --npz skylink_data/.../agent_000454/visibility_world_actor_000454.npz \
    --json skylink_data/.../agent_000454/visibility_world_actor_000454.json \
    --show-json

  # Visualize all actors under one timestamp
  python visualize_visibility.py --timestamp-dir skylink_data/.../timestamp_000123 --show-json

Notes:
 - The NPZ contains a raster mask in world coordinates along with origin/resolution/size.
 - The JSON contains world-frame polygons; showing them is optional via --show-json.
"""
import os
import glob
import json
import argparse
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon


def load_npz(npz_path: str):
    data = np.load(npz_path)
    vis = data["vis_bev"].astype(np.uint8)
    origin_xy = data["origin_xy"].astype(np.float32)
    res = float(data["resolution"][0]) if data["resolution"].shape else float(data["resolution"]) 
    size = tuple(int(x) for x in data["size"])
    ground_z = float(data["ground_z"][0]) if data["ground_z"].shape else float(data["ground_z"]) 
    return vis, origin_xy, res, size, ground_z


def extent_from_meta(origin_xy: np.ndarray, res: float, size: Tuple[int, int]) -> Tuple[float, float, float, float]:
    H, W = size
    x0, y0 = float(origin_xy[0]), float(origin_xy[1])
    x1 = x0 + W * res
    y1 = y0 + H * res
    return (x0, x1, y0, y1)


def draw_npz(ax, npz_path: str, alpha: float = 0.5, cmap: str = "Greens", label: str = None):
    vis, origin_xy, res, size, _ = load_npz(npz_path)
    x0, x1, y0, y1 = extent_from_meta(origin_xy, res, size)
    im = ax.imshow(vis, cmap=cmap, origin="lower", extent=[x0, x1, y0, y1], interpolation="nearest", alpha=alpha)
    if label:
        im.set_label(label)
    return im


def draw_json_polygons(ax, json_path: str, face_alpha: float = 0.15, edge_width: float = 1.0, color: str = None, label: str = None):
    try:
        with open(json_path, "r") as f:
            jd = json.load(f)
    except Exception as e:
        print(f"Failed to read JSON {json_path}: {e}")
        return
    polys = jd.get("polygons", [])
    used_label = False
    for poly in polys:
        if not poly or len(poly) < 3:
            continue
        pts = np.array(poly, dtype=np.float32)
        patch = MplPolygon(pts, closed=True, fill=True, facecolor=color or "lime", edgecolor=color or "green", alpha=face_alpha, linewidth=edge_width)
        # Apply label to the first polygon only to avoid duplicate legend entries
        if label and (not used_label):
            patch.set_label(label)
            used_label = True
        ax.add_patch(patch)


def find_timestamp_artifacts(ts_dir: str) -> Tuple[List[str], List[str]]:
    npzs = sorted(glob.glob(os.path.join(ts_dir, "agent_*", "visibility_world_actor_*.npz")))
    jsons = sorted(glob.glob(os.path.join(ts_dir, "agent_*", "visibility_world_actor_*.json")))
    return npzs, jsons


def main():
    ap = argparse.ArgumentParser(description="Visualize world-frame per-vehicle visibility (NPZ/JSON)")
    ap.add_argument("--npz", nargs="*", default=[], help="One or more NPZ files to render")
    ap.add_argument("--json", nargs="*", default=[], help="Optional JSON polygon files to render")
    ap.add_argument("--timestamp-dir", type=str, default=None, help="Path to a timestamp directory (contains agent_*/)")
    ap.add_argument("--show-json", action="store_true", help="Overlay polygons from JSON files if available")
    ap.add_argument("--alpha", type=float, default=0.35, help="Alpha for raster masks")
    ap.add_argument("--edge-width", type=float, default=1.0, help="Edge width for polygon outlines")
    ap.add_argument("--cmap", type=str, default="Greens", help="Matplotlib colormap for raster visibility")
    ap.add_argument("--figsize", type=float, nargs=2, default=[8.0,8.0], help="Figure size in inches")
    ap.add_argument("--save", type=str, default=None, help="If set, save the figure to this path instead of showing")
    args = ap.parse_args()

    fig, ax = plt.subplots(1, 1, figsize=tuple(args.figsize))
    ax.set_aspect("equal", adjustable="box")

    npz_list = list(args.npz)
    json_list = list(args.json)

    if args.timestamp_dir:
        ts_npzs, ts_jsons = find_timestamp_artifacts(args.timestamp_dir)
        if ts_npzs:
            print(f"Found {len(ts_npzs)} NPZ files under {args.timestamp_dir}")
            npz_list.extend(ts_npzs)
        if args.show_json and ts_jsons:
            print(f"Found {len(ts_jsons)} JSON files under {args.timestamp_dir}")
            json_list.extend(ts_jsons)

    # Draw NPZ rasters first (background)
    for i, npz_path in enumerate(sorted(set(npz_list))):
        if not os.path.exists(npz_path):
            print(f"Skip missing NPZ: {npz_path}")
            continue
        label = os.path.basename(os.path.dirname(npz_path))  # agent_xxx folder
        draw_npz(ax, npz_path, alpha=args.alpha, cmap=args.cmap, label=label)

    # Draw polygons on top (optional)
    if args.show_json:
        for jpath in sorted(set(json_list)):
            if not os.path.exists(jpath):
                print(f"Skip missing JSON: {jpath}")
                continue
            label = os.path.basename(os.path.dirname(jpath))
            draw_json_polygons(ax, jpath, face_alpha=max(0.05, args.alpha*0.6), edge_width=args.edge_width, color=None, label=label)

    ax.set_title("World-frame visibility")
    ax.set_xlabel("world X (m)")
    ax.set_ylabel("world Y (m)")
    try:
        ax.legend(loc="best", fontsize=8)
    except Exception:
        pass

    if args.save:
        os.makedirs(os.path.dirname(args.save), exist_ok=True)
        plt.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"Saved visualization to {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
