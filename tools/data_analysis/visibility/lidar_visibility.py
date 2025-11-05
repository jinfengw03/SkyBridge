import numpy as np
from typing import Tuple, List, Optional
try:
    import cv2  # type: ignore
except Exception:
    cv2 = None

# Simple LiDAR BEV visibility computation using LOS over an occupancy grid.
# Grid convention: world XY -> array (H, W), with origin and resolution provided by config.

class BEVGrid:
    def __init__(self, origin_xy: Tuple[float, float], size: Tuple[int, int], resolution: float):
        self.origin_xy = origin_xy  # bottom-left world (x,y)
        self.size = size            # (H, W)
        self.resolution = resolution

    def world_to_grid(self, x: np.ndarray, y: np.ndarray):
        gx = ((x - self.origin_xy[0]) / self.resolution)
        gy = ((y - self.origin_xy[1]) / self.resolution)
        return gy.astype(np.int32), gx.astype(np.int32)  # row, col

    def grid_to_world_center(self, r: np.ndarray, c: np.ndarray):
        x = self.origin_xy[0] + (c + 0.5) * self.resolution
        y = self.origin_xy[1] + (r + 0.5) * self.resolution
        return x, y


def build_occupancy_from_lidar(
    points_xyz: np.ndarray,
    grid: BEVGrid,
    z_thresh: float = 0.5,
    ground_z: float = None,
    ego_xy: Tuple[float, float] = None,
    clear_radius: float = None,
    clear_polygons_world: Optional[List[np.ndarray]] = None,
    clear_margin_m: Optional[float] = None,
):
    """Build a 2D occupancy grid from LiDAR points.
    By default, marks cells occupied where height above ground exceeds z_thresh.
    Also optionally clears a small radius around ego to avoid self-blocking.

    Args:
        points_xyz: (N,3) LiDAR points in world frame
        grid: BEVGrid
        z_thresh: threshold height above ground for occupancy
        ground_z: ground reference height (if None, use absolute z)
        ego_xy: ego world (x,y) for optional clear radius
        clear_radius: meters to clear around ego
    """
    occ = np.zeros(grid.size, dtype=np.uint8)
    if points_xyz.shape[0] == 0:
        return occ
    xs, ys, zs = points_xyz[:, 0], points_xyz[:, 1], points_xyz[:, 2]
    if ground_z is not None:
        zrel = zs - ground_z
        mask = zrel > z_thresh
    else:
        mask = zs > z_thresh
    if not mask.any():
        return occ
    r, c = grid.world_to_grid(xs[mask], ys[mask])
    valid = (r >= 0) & (r < grid.size[0]) & (c >= 0) & (c < grid.size[1])
    occ[r[valid], c[valid]] = 1

    # Optionally clear a small disk around ego to prevent blocking at the origin
    if ego_xy is not None and clear_radius is not None and clear_radius > 0:
        # Compute grid coords of a disk around ego
        # Create a small window around the ego and clear cells within radius
        H, W = grid.size
        # Center cell of ego
        er, ec = grid.world_to_grid(np.array([ego_xy[0]]), np.array([ego_xy[1]]))
        er, ec = int(er[0]), int(ec[0])
        rmin = max(0, er - int(clear_radius / grid.resolution) - 2)
        rmax = min(H - 1, er + int(clear_radius / grid.resolution) + 2)
        cmin = max(0, ec - int(clear_radius / grid.resolution) - 2)
        cmax = min(W - 1, ec + int(clear_radius / grid.resolution) + 2)
        rr, cc = np.ogrid[rmin:rmax+1, cmin:cmax+1]
        # Compute world center coords of these cells and clear within radius
        xs_cell, ys_cell = grid.grid_to_world_center(rr - 0, cc - 0)
        dist = np.sqrt((xs_cell - ego_xy[0])**2 + (ys_cell - ego_xy[1])**2)
        occ[rmin:rmax+1, cmin:cmax+1][dist <= clear_radius] = 0

    # Optionally clear occupancy inside provided polygons (e.g., ego footprint) with margin
    if clear_polygons_world and cv2 is not None:
        H, W = grid.size
        mask_grid = np.zeros((H, W), dtype=np.uint8)
        for poly in clear_polygons_world:
            if poly is None or len(poly) == 0:
                continue
            # Convert world XY polygon to grid pixel coordinates (col=x, row=y)
            rs, cs = grid.world_to_grid(poly[:, 0], poly[:, 1])
            pts = np.stack([cs, rs], axis=1).astype(np.int32)
            pts = pts.reshape(-1, 1, 2)
            cv2.fillPoly(mask_grid, [pts], 1)
        if clear_margin_m is not None and clear_margin_m > 0:
            k = max(1, int(round(clear_margin_m / grid.resolution)))
            kernel = np.ones((2*k+1, 2*k+1), dtype=np.uint8)
            mask_grid = cv2.dilate(mask_grid, kernel, iterations=1)
        occ[mask_grid > 0] = 0

    return occ


def los_visibility_from_occupancy(
    grid: BEVGrid,
    occ: np.ndarray,
    ego_xy: Tuple[float, float],
    max_range: float = 120.0,
    step: float = 0.5,
    vehicle_label_map: Optional[np.ndarray] = None,
    include_first_vehicle_blocker: bool = False,
    blocker_min_hits: int = 0,
):
    """Compute visible free space using simple LOS ray casting over occupancy grid.
    Args:
        grid: BEV grid definition
        occ: occupancy grid (1 = occupied)
        ego_xy: origin of rays
        max_range: maximum ray length
        step: distance increment along each ray (meters)
        vehicle_label_map: per-cell actor id labels for vehicles; required for strict first-blocker attribution
        include_first_vehicle_blocker: if True, return IDs of vehicles that were first blockers on any ray
        blocker_min_hits: minimum number of distinct ray intersections required to include a vehicle as a blocker (0 disables threshold)

    Returns:
        - vis: binary mask (H, W) where 1 indicates visible.
        - blocker_actor_ids: sorted list of actor IDs that were first blockers along any ray (filtered by blocker_min_hits if > 0).
        - blocker_hit_counts: dict mapping actor_id -> number of rays that first-hit that actor (empty if attribution disabled/missing labels)
    """
    H, W = grid.size
    vis = np.zeros((H, W), dtype=np.uint8)
    blocker_ids = set()
    blocker_hit_counts = {}

    # Precompute polar rays over 360 degrees
    angles = np.deg2rad(np.arange(0, 360, 1))
    # Mark the starting cell as visible to avoid a blank mask when origin cell is occupied
    er, ec = grid.world_to_grid(np.array([ego_xy[0]]), np.array([ego_xy[1]]))
    er, ec = int(er[0]), int(ec[0])
    if 0 <= er < H and 0 <= ec < W:
        vis[er, ec] = 1
    for theta in angles:
        # march along the ray
        dist = step  # start one step away to avoid self-occupancy blocking immediately
        blocked = False
        while dist < max_range and not blocked:
            x = ego_xy[0] + dist * np.cos(theta)
            y = ego_xy[1] + dist * np.sin(theta)
            r, c = grid.world_to_grid(np.array([x]), np.array([y]))
            r, c = int(r[0]), int(c[0])
            if r < 0 or r >= H or c < 0 or c >= W:
                break
            # If this cell is occupied, the ray stops. Optionally include the blocker if it's a vehicle.
            if occ[r, c] == 1:
                if include_first_vehicle_blocker and vehicle_label_map is not None:
                    if 0 <= r < H and 0 <= c < W and vehicle_label_map[r, c] > 0:
                        vis[r, c] = 1
                        aid = int(vehicle_label_map[r, c])
                        blocker_ids.add(aid)
                        # count ray intersections per actor id
                        blocker_hit_counts[aid] = blocker_hit_counts.get(aid, 0) + 1
                blocked = True
                break
            # Free cell -> mark visible and continue
            vis[r, c] = 1
            dist += step
    # Apply hit-count threshold if requested
    if include_first_vehicle_blocker and blocker_min_hits > 0 and blocker_hit_counts:
        blocker_ids = {aid for aid in blocker_ids if blocker_hit_counts.get(aid, 0) >= int(blocker_min_hits)}
    return vis, sorted(list(blocker_ids)), blocker_hit_counts
