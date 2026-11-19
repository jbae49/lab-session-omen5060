#!/usr/bin/env python3
"""Render a tracking motion .npz to mp4 (headless, no trained policy needed).

Example:
  MUJOCO_GL=egl python scripts/render_motion_npz.py \
      --motion-file src/assets/motions/g1/test_motion1_t084_g1_bicycle_mjlab_fixed.npz \
      --bike-motion-npz src/assets/motions/g1/test_motion1_t084_g1_bicycle_motion.npz \
      --output src/assets/motions/g1/test_motion1_t084_g1_bicycle_preview.mp4

For accurate robot+bike mesh visualization, use the GMR XML + combined_qpos instead:
  MUJOCO_GL=egl python scripts/render_combined_motion.py \\
      --xml src/assets/gmr/g1_mocap_29dof_collision.xml \\
      --motion-npz src/assets/motions/g1/test_motion1_t084_g1_bicycle_motion.npz \\
      --output src/assets/motions/g1/test_motion1_t084_g1_bicycle_preview.mp4
"""
from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch
from tqdm import tqdm

import mjlab
from mjlab.entity import Entity
from mjlab.scene import Scene
from mjlab.sim.sim import Simulation, SimulationCfg
from mjlab.utils.lab_api.math import matrix_from_quat, quat_slerp
from mjlab.viewer.native.visualizer import MujocoNativeDebugVisualizer
from mjlab.viewer.offscreen_renderer import OffscreenRenderer
from mjlab.viewer.viewer_config import ViewerConfig
from src.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg

# Approximate bike part visuals (proxy geometry, not mesh-accurate).
BIKE_PART_STYLE: dict[str, dict] = {
    "frame": {"kind": "ellipsoid", "size": (0.45, 0.04, 0.10), "color": (0.15, 0.35, 0.85, 0.95)},
    "fork": {"kind": "ellipsoid", "size": (0.03, 0.03, 0.18), "color": (0.20, 0.45, 0.90, 0.95)},
    "rear_wheel": {"kind": "wheel", "radius": 0.34, "width": 0.03, "color": (0.10, 0.10, 0.10, 0.95)},
    "front_wheel": {"kind": "wheel", "radius": 0.34, "width": 0.03, "color": (0.10, 0.10, 0.10, 0.95)},
    "crank": {"kind": "sphere", "radius": 0.05, "color": (0.75, 0.75, 0.20, 0.95)},
    "pedal_left": {"kind": "sphere", "radius": 0.04, "color": (0.90, 0.35, 0.15, 0.95)},
    "pedal_right": {"kind": "sphere", "radius": 0.04, "color": (0.90, 0.35, 0.15, 0.95)},
}


def _compute_frame_blend(times: np.ndarray, input_frames: int, duration: float):
    phase = np.clip(times / duration, 0.0, 1.0)
    index_0 = np.floor(phase * (input_frames - 1)).astype(np.int64)
    index_1 = np.minimum(index_0 + 1, input_frames - 1)
    blend = phase * (input_frames - 1) - index_0
    return index_0, index_1, blend.astype(np.float32)


def _lerp(a: np.ndarray, b: np.ndarray, blend: np.ndarray) -> np.ndarray:
    while blend.ndim < a.ndim:
        blend = blend[..., None]
    return a * (1.0 - blend) + b * blend


def _slerp_quat_sequence(q0: np.ndarray, q1: np.ndarray, blend: float) -> np.ndarray:
    out = quat_slerp(torch.from_numpy(q0), torch.from_numpy(q1), blend)
    return out.numpy()


def _slerp_body_quats(
    quats: np.ndarray, index_0: np.ndarray, index_1: np.ndarray, blend: np.ndarray
) -> np.ndarray:
    out_len = len(blend)
    nbody = quats.shape[1]
    out = np.zeros((out_len, nbody, 4), dtype=np.float32)
    for t in range(out_len):
        b = float(blend[t])
        for j in range(nbody):
            out[t, j] = _slerp_quat_sequence(quats[index_0[t], j], quats[index_1[t], j], b)
    return out


def resample_bike_poses(
    bike_pos: np.ndarray,
    bike_quat: np.ndarray,
    input_fps: float,
    output_fps: float,
    output_frames: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Resample bike rigid-body poses to match a robot motion timeline."""
    input_frames = bike_pos.shape[0]
    duration = (input_frames - 1) / input_fps
    output_dt = 1.0 / output_fps
    times = np.arange(0.0, duration + 0.5 * output_dt, output_dt, dtype=np.float32)
    times = times[times <= duration + 1e-9][:output_frames]
    if len(times) < output_frames:
        pad = np.full(output_frames - len(times), duration, dtype=np.float32)
        times = np.concatenate([times, pad])

    i0, i1, blend = _compute_frame_blend(times, input_frames, duration)
    pos_out = _lerp(bike_pos[i0], bike_pos[i1], blend)
    quat_out = _slerp_body_quats(bike_quat, i0, i1, blend)
    return pos_out.astype(np.float32), quat_out.astype(np.float32)


def load_motion(path: Path):
    d = np.load(path)
    fps = float(np.asarray(d["fps"]).reshape(-1)[0])
    return {
        "fps": fps,
        "joint_pos": torch.tensor(d["joint_pos"], dtype=torch.float32),
        "body_pos_w": torch.tensor(d["body_pos_w"], dtype=torch.float32),
        "body_quat_w": torch.tensor(d["body_quat_w"], dtype=torch.float32),
    }


def load_bike_motion(path: Path) -> dict:
    d = np.load(path, allow_pickle=True)
    return {
        "fps": float(np.asarray(d["fps"]).reshape(-1)[0]),
        "names": [str(n) for n in d["bike_body_names"]],
        "pos_w": d["bike_body_pos_w"].astype(np.float32),
        "quat_w": d["bike_body_quat_w"].astype(np.float32),
    }


def _quat_to_mat_np(quat_wxyz: np.ndarray) -> np.ndarray:
    mat = matrix_from_quat(torch.from_numpy(quat_wxyz).float()).numpy()
    return mat.reshape(3, 3)


def _draw_bike_part(vis: MujocoNativeDebugVisualizer, name: str, pos: np.ndarray, quat: np.ndarray) -> None:
    style = BIKE_PART_STYLE.get(name)
    if style is None:
        vis.add_sphere(pos, 0.05, (0.8, 0.2, 0.8, 0.9))
        return

    mat = _quat_to_mat_np(quat)
    kind = style["kind"]
    color = style["color"]
    if kind == "sphere":
        vis.add_sphere(pos, style["radius"], color)
    elif kind == "ellipsoid":
        vis.add_ellipsoid(pos, np.asarray(style["size"], dtype=np.float64), mat, color)
    elif kind == "wheel":
        # Wheel axle is local Y in the mocap bike frames.
        axis = mat[:, 1]
        half_width = style["width"] * 0.5
        vis.add_cylinder(pos - axis * half_width, pos + axis * half_width, style["radius"], color)


def draw_bike(
    vis: MujocoNativeDebugVisualizer,
    names: list[str],
    pos_w: np.ndarray,
    quat_w: np.ndarray,
) -> None:
    name_to_idx = {n: i for i, n in enumerate(names)}
    for name in names:
        idx = name_to_idx[name]
        _draw_bike_part(vis, name, pos_w[idx], quat_w[idx])

    if "rear_wheel" in name_to_idx and "front_wheel" in name_to_idx:
        rw = pos_w[name_to_idx["rear_wheel"]]
        fw = pos_w[name_to_idx["front_wheel"]]
        vis.add_cylinder(rw, fw, 0.015, (0.15, 0.35, 0.85, 0.7))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion-file", type=Path, required=True)
    parser.add_argument(
        "--bike-motion-npz",
        type=Path,
        default=None,
        help="Optional GMR motion.npz with bike_body_* keys (rough proxy only; use render_combined_motion.py for real mesh)",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--stride", type=int, default=1, help="Render every N frames")
    args = parser.parse_args()

    motion = load_motion(args.motion_file)
    n_frames = motion["joint_pos"].shape[0]
    device = "cpu"

    bike = None
    if args.bike_motion_npz is not None:
        bike_raw = load_bike_motion(args.bike_motion_npz)
        bike_pos, bike_quat = resample_bike_poses(
            bike_raw["pos_w"],
            bike_raw["quat_w"],
            input_fps=bike_raw["fps"],
            output_fps=motion["fps"],
            output_frames=n_frames,
        )
        bike = {"names": bike_raw["names"], "pos_w": bike_pos, "quat_w": bike_quat}
        print(
            f"Bike overlay: {args.bike_motion_npz.name} "
            f"({bike_raw['pos_w'].shape[0]}@{bike_raw['fps']:.0f}fps -> {n_frames}@{motion['fps']:.0f}fps)"
        )

    env_cfg = unitree_g1_flat_tracking_env_cfg(has_state_estimation=True, play=True)
    sim_cfg = env_cfg.sim
    sim_cfg.mujoco.timestep = 1.0 / motion["fps"]

    scene = Scene(env_cfg.scene, device=device)
    model = scene.compile()
    sim = Simulation(num_envs=1, cfg=sim_cfg, model=model, device=device)
    scene.initialize(sim.mj_model, sim.model, sim.data)

    robot: Entity = scene["robot"]
    robot_joint_indexes = robot.find_joints(
        [
            "left_hip_pitch_joint",
            "left_hip_roll_joint",
            "left_hip_yaw_joint",
            "left_knee_joint",
            "left_ankle_pitch_joint",
            "left_ankle_roll_joint",
            "right_hip_pitch_joint",
            "right_hip_roll_joint",
            "right_hip_yaw_joint",
            "right_knee_joint",
            "right_ankle_pitch_joint",
            "right_ankle_roll_joint",
            "waist_yaw_joint",
            "waist_roll_joint",
            "waist_pitch_joint",
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_joint",
            "left_wrist_roll_joint",
            "left_wrist_pitch_joint",
            "left_wrist_yaw_joint",
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
            "right_wrist_roll_joint",
            "right_wrist_pitch_joint",
            "right_wrist_yaw_joint",
        ],
        preserve_order=True,
    )[0]

    viewer_cfg = ViewerConfig(
        width=args.width,
        height=args.height,
        origin_type=ViewerConfig.OriginType.ASSET_ROOT,
        entity_name="robot",
        distance=4.5 if bike is not None else 3.0,
        elevation=-10.0,
        azimuth=120.0,
    )
    renderer = OffscreenRenderer(model=sim.mj_model, cfg=viewer_cfg, scene=scene)
    renderer.initialize()

    frames_out: list[np.ndarray] = []
    indices = range(0, n_frames, args.stride)

    print(f"Rendering {len(list(indices))}/{n_frames} frames @ {motion['fps']:.1f} fps")
    scene.reset()

    for t in tqdm(indices):
        root_pos = motion["body_pos_w"][t, 0:1].to(device)
        root_quat = motion["body_quat_w"][t, 0:1].to(device)
        dof_pos = motion["joint_pos"][t : t + 1].to(device)

        root_states = robot.data.default_root_state.clone()
        root_states[:, 0:3] = root_pos
        root_states[:, 3:7] = root_quat
        robot.write_root_state_to_sim(root_states)

        joint_pos = robot.data.default_joint_pos.clone()
        joint_vel = robot.data.default_joint_vel.clone()
        joint_pos[:, robot_joint_indexes] = dof_pos
        robot.write_joint_state_to_sim(joint_pos, joint_vel)

        sim.forward()
        scene.update(sim.mj_model.opt.timestep)

        def _debug_vis(vis: MujocoNativeDebugVisualizer) -> None:
            if bike is not None:
                draw_bike(vis, bike["names"], bike["pos_w"][t], bike["quat_w"][t])

        renderer.update(sim.data, debug_vis_callback=_debug_vis if bike is not None else None)
        frames_out.append(renderer.render())

    args.output.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(
        args.output,
        np.stack(frames_out),
        fps=motion["fps"] / args.stride,
        codec="libx264",
        pixelformat="yuv420p",
    )
    print(f"Saved {args.output} ({len(frames_out)} frames)")


if __name__ == "__main__":
    main()
