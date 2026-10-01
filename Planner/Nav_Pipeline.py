"""
Glasses -> Depth Anything 3 -> Open3D point cloud -> Lagrangian/Hamiltonian planner -> arrow overlay

Camera frame convention (same as DA3 / OpenCV): x right, y down, z forward.
Ground frame (derived from the fitted floor plane): lateral (+ = right), forward.

Usage:
    python nav_pipeline.py                      # live Pupil Labs glasses
    python nav_pipeline.py --video scene.mp4    # recording (processed frame by frame)
    python nav_pipeline.py --goal gaze          # steer toward where the wearer is looking
"""
import argparse
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np
import open3d as o3d

cv2.imshow("cv/av bug", np.zeros(1))
cv2.destroyAllWindows()

UP_CAM = np.array([0.0, -1.0, 0.0])  # "up" in camera coordinates (y points down)


@dataclass
class Cfg:
    # ground grid (metres)
    x_half: float = 3.0      # grid spans lateral +-x_half
    z_max: float = 6.0       # and forward 0..z_max
    res: float = 0.10        # cell size
    # obstacle slab above the floor (metres)
    h_min: float = 0.20
    h_max: float = 2.00
    # point cloud
    stride: int = 2
    voxel: float = 0.05
    conf_pct: float = 30.0   # drop the least confident N% of depth pixels
    cam_height: float = 0.0  # >0: rescale the cloud so the floor sits this far below the camera
    # potential field U = k_att * |q-goal| + repulsion from obstacles
    k_att: float = 1.0
    k_rep: float = 2.0
    d0: float = 1.2          # obstacle influence radius
    k_blur: float = 15.0     # smooth "mass" repulsion: gives a sideways push even for a flat wall dead ahead
    blur_sigma: float = 0.7  # metres
    goal_dist: float = 4.0   # default goal: straight ahead
    goal_mode: str = "ahead"  # "ahead" | "gaze"
    # particle dynamics
    mass: float = 1.0
    v0: float = 1.0          # initial speed along current heading
    dt: float = 0.05
    steps: int = 120
    gamma: float = 0.3       # small damping so the particle settles instead of orbiting
    look: float = 1.5        # arrow points along the path at this arc length
    smooth: float = 0.7      # EMA on the heading vector (0 = no smoothing)


# --------------------------------------------------------------------------- depth model
class DepthModel:
    def __init__(self, name: str, res: int):
        import torch
        from depth_anything_3.api import DepthAnything3

        self.torch, self.res = torch, res
        dev = (
            "cuda" if torch.cuda.is_available()
            else "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
            else "cpu"
        )
        print(f"Loading {name} on {dev}...")
        self.model = DepthAnything3.from_pretrained(name).to(dev)

    def infer(self, rgb):
        """Returns depth (H, W), intrinsics K (3, 3) at that depth resolution, conf (H, W) or None."""
        with self.torch.inference_mode():
            pred = self.model.inference([rgb], process_res=self.res)
        depth = np.asarray(pred.depth[0], dtype=np.float32)
        conf = None if pred.conf is None else np.asarray(pred.conf[0], dtype=np.float32)
        if pred.intrinsics is not None:
            K = np.asarray(pred.intrinsics[0], dtype=np.float64)
        else:  # fallback: assume ~100 deg horizontal FOV
            h, w = depth.shape
            f = (w / 2) / np.tan(np.radians(50))
            K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])
        return depth, K, conf


# --------------------------------------------------------------------------- Open3D point cloud
def depth_to_cloud(depth, K, rgb, cfg):
    H, W = depth.shape
    s = cfg.stride
    v, u = np.mgrid[0:H:s, 0:W:s]
    z = depth[::s, ::s]
    ok = np.isfinite(z) & (z > 0.1) & (z < 30.0)
    pts = np.stack([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z], -1)
    return pts, ok, (v, u)


def make_pcd(pts, colors, voxel):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    return pcd.voxel_down_sample(voxel)


def fit_floor(pcd, prev):
    """RANSAC floor plane. Returns (n, d) with n pointing up and n.p + d = height above floor."""
    pts = np.asarray(pcd.points)
    cand = pts[pts[:, 1] > 0.5]  # only points clearly below the camera can be floor
    if len(cand) < 200:
        return prev
    c = o3d.geometry.PointCloud()
    c.points = o3d.utility.Vector3dVector(cand)
    (a, b, cc, d), _ = c.segment_plane(distance_threshold=0.05, ransac_n=3, num_iterations=300)
    n = np.array([a, b, cc])
    norm = np.linalg.norm(n)
    n, d = n / norm, d / norm
    if n @ UP_CAM < 0:
        n, d = -n, -d
    if n @ UP_CAM < np.cos(np.radians(35)) or d <= 0.3:  # not floor-like -> keep previous
        return prev
    return n, d


# --------------------------------------------------------------------------- ground map + potential
def ground_basis(n):
    z = np.array([0.0, 0.0, 1.0])
    f = z - (z @ n) * n
    if np.linalg.norm(f) < 1e-3:
        f = np.array([0.0, 0.0, 1.0])
    f /= np.linalg.norm(f)
    return np.cross(f, n), f  # right, forward


def obstacle_grid(pts, n, d, r, f, cfg):
    h = pts @ n + d
    p = pts[(h > cfg.h_min) & (h < cfg.h_max)]
    lat, fwd = p @ r, p @ f
    NX, NZ = int(2 * cfg.x_half / cfg.res), int(cfg.z_max / cfg.res)
    ok = (fwd >= 0) & (np.abs(lat) < cfg.x_half) & (fwd < cfg.z_max)
    ix = np.floor((lat[ok] + cfg.x_half) / cfg.res).astype(int)
    iz = np.floor(fwd[ok] / cfg.res).astype(int)
    cnt = np.zeros((NZ, NX), np.int32)
    np.add.at(cnt, (iz, ix), 1)
    return cnt >= 2


def build_potential(occ, goal, cfg):
    NZ, NX = occ.shape
    dist = cv2.distanceTransform((~occ).astype(np.uint8) * 255, cv2.DIST_L2, 3) * cfg.res
    dist = np.minimum(dist, 10.0)
    xs = (np.arange(NX) + 0.5) * cfg.res - cfg.x_half
    zs = (np.arange(NZ) + 0.5) * cfg.res
    X, Z = np.meshgrid(xs, zs)
    d_c = np.maximum(dist, 0.1)
    U_rep = np.where(dist < cfg.d0, 0.5 * cfg.k_rep * (1 / d_c - 1 / cfg.d0) ** 2, 0.0)
    U_att = cfg.k_att * np.sqrt((X - goal[0]) ** 2 + (Z - goal[1]) ** 2 + 0.25)
    sig = cfg.blur_sigma / cfg.res
    U_blur = cfg.k_blur * cv2.GaussianBlur(occ.astype(np.float32), (0, 0), sig)
    U_wall = 5.0 * np.clip(np.abs(X) - (cfg.x_half - 0.4), 0, None) ** 2
    return U_att + np.minimum(U_rep, 100.0) + U_blur + U_wall, dist


def sample(F, q, cfg):
    NZ, NX = F.shape
    fx = np.clip((q[0] + cfg.x_half) / cfg.res - 0.5, 0, NX - 1 - 1e-6)
    fz = np.clip(q[1] / cfg.res - 0.5, 0, NZ - 1 - 1e-6)
    x0, z0 = int(fx), int(fz)
    ax, az = fx - x0, fz - z0
    return (F[z0, x0] * (1 - ax) * (1 - az) + F[z0, x0 + 1] * ax * (1 - az)
            + F[z0 + 1, x0] * (1 - ax) * az + F[z0 + 1, x0 + 1] * ax * az)


# --------------------------------------------------------------------------- Lagrangian / Hamiltonian
def hamiltonian_rollout(U, goal, cfg, alpha0=0.0):
    """
    Wearer = point mass m moving on the ground plane in potential U(q).
        Lagrangian   L(q, v) = 1/2 m |v|^2 - U(q)          (used for the action S = sum L dt)
        Hamiltonian  H(q, p) = |p|^2 / 2m + U(q),  p = m v
        Hamilton:    dq/dt = p/m,   dp/dt = -grad U(q)   (-gamma p for light damping)
    Integrated with velocity-Verlet (symplectic leapfrog). Starts at the wearer with heading alpha0 (0 = straight ahead).
    """
    gU_z, gU_x = np.gradient(U, cfg.res)

    def force(q):
        return -np.array([sample(gU_x, q, cfg), sample(gU_z, q, cfg)])

    q = np.zeros(2)
    p = cfg.mass * cfg.v0 * np.array([np.sin(alpha0), np.cos(alpha0)])
    F = force(q)
    traj, H, S = [q.copy()], [], 0.0
    for _ in range(cfg.steps):
        p = p + 0.5 * cfg.dt * F
        q = q + cfg.dt * p / cfg.mass
        F = force(q)
        p = p + 0.5 * cfg.dt * F
        p = p * np.exp(-cfg.gamma * cfg.dt)
        T, V = p @ p / (2 * cfg.mass), sample(U, q, cfg)
        S += (T - V) * cfg.dt
        H.append(T + V)
        traj.append(q.copy())
        if (np.linalg.norm(q - goal) < 0.3 or abs(q[0]) >= cfg.x_half
                or q[1] >= cfg.z_max or q[1] < 0):
            break
    return np.array(traj), np.array(H), S


def heading_from_path(traj, look):
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(traj, axis=0), axis=1))])
    t = traj[min(np.searchsorted(s, look), len(traj) - 1)]
    return float(np.arctan2(t[0], t[1]))  # radians, + = turn right


# --------------------------------------------------------------------------- full pipeline
class NavPipeline:
    def __init__(self, depth_model, cfg: Cfg):
        self.depth, self.cfg = depth_model, cfg
        self.plane = (UP_CAM.copy(), cfg.cam_height or 1.6)
        self.vec = np.array([0.0, 1.0])  # smoothed (sin theta, cos theta)

    def step(self, bgr, gaze=None):
        cfg = self.cfg
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        t0 = time.time()
        depth, K, conf = self.depth.infer(rgb)
        H, W = depth.shape
        small = cv2.resize(rgb, (W, H))

        pts, ok, (v, u) = depth_to_cloud(depth, K, small, cfg)
        if conf is not None:
            c = conf[:: cfg.stride, :: cfg.stride]
            ok &= c >= np.percentile(c, cfg.conf_pct)
        pcd = make_pcd(pts[ok], small[v, u][ok] / 255.0, cfg.voxel)

        n, d = self.plane = fit_floor(pcd, self.plane)
        scale = cfg.cam_height / d if cfg.cam_height > 0 else 1.0
        P = np.asarray(pcd.points) * scale
        d *= scale
        r, f = ground_basis(n)

        goal = np.array([0.0, cfg.goal_dist])
        if cfg.goal_mode == "gaze" and gaze is not None:
            gu = np.clip(gaze[0] * W / bgr.shape[1], 0, W - 1)
            gv = np.clip(gaze[1] * H / bgr.shape[0], 0, H - 1)
            z = depth[int(gv), int(gu)] * scale
            p = np.array([(gu - K[0, 2]) * z / K[0, 0], (gv - K[1, 2]) * z / K[1, 1], z])
            goal = np.array([p @ r, p @ f])
        goal = np.clip(goal, [-cfg.x_half + 0.3, 0.5], [cfg.x_half - 0.3, cfg.z_max - 0.3])

        occ = obstacle_grid(P, n, d, r, f, cfg)
        U, dist = build_potential(occ, goal, cfg)
        # initial heading leans toward the freer side; also breaks the symmetric "wall dead ahead" trap
        near = occ[: int(4.0 / cfg.res)]
        left, right = near[:, : near.shape[1] // 2].sum(), near[:, near.shape[1] // 2 :].sum()
        alpha0 = float(np.clip(0.6 * (left - right) / (left + right + 1e-6), -0.4, 0.4))
        traj, Hs, S = hamiltonian_rollout(U, goal, cfg, alpha0)

        theta = heading_from_path(traj, cfg.look)
        new = np.array([np.sin(theta), np.cos(theta)])
        self.vec = cfg.smooth * self.vec + (1 - cfg.smooth) * new
        theta_s = float(np.arctan2(self.vec[0], self.vec[1]))
        clearance = min(sample(dist, q, cfg) for q in traj)

        return dict(theta=theta_s, blocked=clearance < 0.4, H=float(Hs[-1]) if len(Hs) else 0.0,
                    H_drift=float(np.ptp(Hs)) if len(Hs) else 0.0, S=S, ms=(time.time() - t0) * 1e3,
                    debug=self._debug_map(U, traj, goal))

    def _debug_map(self, U, traj, goal):
        cfg = self.cfg
        img = cv2.normalize(np.clip(U, 0, np.percentile(U, 98)), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        img = cv2.applyColorMap(img, cv2.COLORMAP_VIRIDIS)
        NZ = img.shape[0]
        px = lambda q: (int((q[0] + cfg.x_half) / cfg.res), int(NZ - 1 - q[1] / cfg.res))
        cv2.polylines(img, [np.array([px(q) for q in traj])], False, (255, 255, 255), 1)
        cv2.circle(img, px(goal), 3, (0, 0, 255), -1)
        return cv2.resize(img, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)


# --------------------------------------------------------------------------- overlay
def draw_overlay(img, res, debug=False):
    h, w = img.shape[:2]
    c = (w // 2, h - 90)
    tip = (int(c[0] + 110 * np.sin(res["theta"])), int(c[1] - 110 * np.cos(res["theta"])))
    color = (0, 0, 255) if res["blocked"] else (0, 255, 0)
    cv2.arrowedLine(img, c, tip, (0, 0, 0), 18, tipLength=0.4)
    cv2.arrowedLine(img, c, tip, color, 10, tipLength=0.4)
    txt = f"{np.degrees(res['theta']):+.0f} deg  H={res['H']:.2f}  S={res['S']:.1f}  {res['ms']:.0f} ms"
    cv2.putText(img, txt, (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    if debug:
        dm = res["debug"]
        img[10 : 10 + dm.shape[0], w - 10 - dm.shape[1] : w - 10] = dm


# --------------------------------------------------------------------------- frame sources
def live_source():
    from pupil_labs.realtime_api.simple import discover_one_device, Device  # noqa: E402

    print("Looking for the next best device...")
    # device = discover_one_device(max_search_duration_seconds=10)
    device = Device(address="145.137.153.144", port=8080)
    if device is None:
        print("No device found.")
        raise SystemExit(-1)
    print(f"Connecting to {device}...")
    try:
        while True:
            m = device.receive_matched_scene_and_eyes_video_frames_and_gaze()
            if not m:
                print("Not able to find a match!")
                continue
            yield m.scene.bgr_pixels.copy(), (m.gaze.x, m.gaze.y)
    finally:
        print("Stopping...")
        device.close()


def video_source(path):
    cap = cv2.VideoCapture(path)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                return
            yield frame, None
    finally:
        cap.release()


class Worker(threading.Thread):
    """Runs the (slow) depth + planning loop on the newest frame so the video display stays smooth."""

    def __init__(self, pipe):
        super().__init__(daemon=True)
        self.pipe, self.lock, self.job, self.result, self.alive = pipe, threading.Lock(), None, None, True

    def submit(self, frame, gaze):
        with self.lock:
            self.job = (frame, gaze)

    def run(self):
        while self.alive:
            with self.lock:
                job, self.job = self.job, None
            if job is None:
                time.sleep(0.005)
                continue
            self.result = self.pipe.step(*job)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", help="recording to process instead of the live stream")
    ap.add_argument("--model", default="depth-anything/DA3NESTED-GIANT-LARGE")
    ap.add_argument("--res", type=int, default=504, help="DA3 process_res; lower = faster")
    ap.add_argument("--goal", choices=["ahead", "gaze"], default="ahead")
    ap.add_argument("--cam-height", type=float, default=0.0, help="metres; rescales depth using the floor plane")
    ap.add_argument("--debug", action="store_true", help="show potential map + simulated path inset")
    a = ap.parse_args()

    pipe = NavPipeline(DepthModel(a.model, a.res), Cfg(goal_mode=a.goal, cam_height=a.cam_height))
    src = live_source()
    worker = Worker(pipe)
    print("Starting worker thread...")
    worker.start()
    try:
        for frame, gaze in src:
            if worker:
                worker.submit(frame, gaze)
                res = worker.result
            else:
                res = pipe.step(frame, gaze)
            vis = frame.copy()
            if res:
                draw_overlay(vis, res, a.debug)
            cv2.imshow("Navigation", vis)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    except KeyboardInterrupt:
        pass
    finally:
        if worker:
            worker.alive = False
        src.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()