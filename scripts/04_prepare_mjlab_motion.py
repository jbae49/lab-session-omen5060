#!/usr/bin/env python3
"""Convert free-base G1 qpos to MJLab's full 50 Hz reference archive."""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp


LAB_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_XML = (
    LAB_ROOT
    / "repos/unitree_rl_mjlab/src/assets/robots/unitree_g1/xmls/g1.xml"
)


def resample(qpos: np.ndarray, source_fps: float, target_fps: float) -> np.ndarray:
    source_t = np.arange(len(qpos), dtype=np.float64) / source_fps
    target_t = np.arange(round(source_t[-1] * target_fps) + 1) / target_fps
    target_t[-1] = min(target_t[-1], source_t[-1])
    out = np.empty((len(target_t), qpos.shape[1]), dtype=np.float64)
    for column in [*range(3), *range(7, qpos.shape[1])]:
        out[:, column] = np.interp(target_t, source_t, qpos[:, column])
    xyzw = qpos[:, [4, 5, 6, 3]]
    interp = Slerp(source_t, Rotation.from_quat(xyzw))(target_t).as_quat()
    out[:, 3:7] = interp[:, [3, 0, 1, 2]]
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML)
    parser.add_argument("--fps", type=float, default=50.0)
    args = parser.parse_args()

    source = np.load(args.input, allow_pickle=True)
    source_qpos = np.asarray(source["qpos"], dtype=np.float64)
    source_fps = float(np.asarray(source["fps"]).reshape(-1)[0])
    model = mujoco.MjModel.from_xml_path(str(args.xml))
    if source_qpos.shape[1] != model.nq:
        raise ValueError(f"qpos width={source_qpos.shape[1]}, MuJoCo nq={model.nq}")
    qpos = resample(source_qpos, source_fps, args.fps)

    qvel = np.zeros((len(qpos), model.nv), dtype=np.float64)
    dt = 1.0 / args.fps
    for i in range(len(qpos)):
        a, b, span = (i, i + 1, dt) if i == 0 else (
            (i - 1, i, dt) if i == len(qpos) - 1 else (i - 1, i + 1, 2 * dt)
        )
        mujoco.mj_differentiatePos(model, qvel[i], span, qpos[a], qpos[b])

    data = mujoco.MjData(model)
    nframe, nbody = len(qpos), model.nbody - 1
    pos = np.empty((nframe, nbody, 3), np.float32)
    quat = np.empty((nframe, nbody, 4), np.float32)
    lin = np.empty_like(pos)
    ang = np.empty_like(pos)
    velocity = np.empty(6, dtype=np.float64)
    for i, (q, v) in enumerate(zip(qpos, qvel, strict=True)):
        data.qpos[:] = q
        data.qvel[:] = v
        mujoco.mj_forward(model, data)
        pos[i], quat[i] = data.xpos[1:], data.xquat[1:]
        for body_id in range(1, model.nbody):
            mujoco.mj_objectVelocity(
                model, data, mujoco.mjtObj.mjOBJ_BODY, body_id, velocity, 0
            )
            ang[i, body_id - 1], lin[i, body_id - 1] = velocity[:3], velocity[3:]

    joint_names = np.asarray([
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        for i in range(1, model.njnt)
    ])
    body_names = np.asarray([
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        for i in range(1, model.nbody)
    ])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        joint_pos=qpos[:, 7:].astype(np.float32),
        joint_vel=qvel[:, 6:].astype(np.float32),
        body_pos_w=pos,
        body_quat_w=quat,
        body_lin_vel_w=lin,
        body_ang_vel_w=ang,
        fps=np.asarray([args.fps], np.float64),
        qpos=qpos.astype(np.float32),
        qvel=qvel.astype(np.float32),
        joint_names=joint_names,
        body_names=body_names,
        source_file=np.asarray(str(args.input.resolve())),
        source_fps=np.asarray(source_fps),
    )
    print(f"Saved {args.output}: frames={nframe}, fps={args.fps:g}, joints=29")


if __name__ == "__main__":
    main()
