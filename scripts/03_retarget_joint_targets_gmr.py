#!/usr/bin/env python3
"""Retarget pre-baked SMPL-X joint targets to Unitree G1 29-DoF."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch


LAB_ROOT = Path(__file__).resolve().parents[1]
GMR_ROOT = LAB_ROOT / "repos" / "GMR"
sys.path.insert(0, str(GMR_ROOT))

from general_motion_retargeting import GeneralMotionRetargeting  # noqa: E402
from general_motion_retargeting.kinematics_model import KinematicsModel  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    data = np.load(args.input, allow_pickle=False)
    if str(np.asarray(data["format"]).item()) != "gmr_joint_targets_v1":
        raise ValueError("Expected format='gmr_joint_targets_v1'")
    names = [str(name) for name in data["joint_names"]]
    positions = np.asarray(data["joint_positions"], np.float64)
    quaternions = np.asarray(data["joint_quaternions_wxyz"], np.float64)
    if positions.shape[:2] != quaternions.shape[:2] or quaternions.shape[-1] != 4:
        raise ValueError(f"Invalid target shapes: {positions.shape}, {quaternions.shape}")

    frames = [
        {name: (positions[t, j], quaternions[t, j]) for j, name in enumerate(names)}
        for t in range(len(positions))
    ]
    retargeter = GeneralMotionRetargeting(
        src_human="smplx",
        tgt_robot="unitree_g1",
        actual_human_height=float(np.asarray(data["human_height"]).item()),
    )
    qpos = np.asarray([retargeter.retarget(frame).copy() for frame in frames])

    # One constant vertical translation preserves real jumps while putting the
    # globally lowest robot point on the floor.
    fk = KinematicsModel(retargeter.xml_file, device=args.device)
    body_pos, _ = fk.forward_kinematics(
        torch.as_tensor(qpos[:, :3], dtype=torch.float32, device=args.device),
        torch.as_tensor(qpos[:, 3:7][:, [1, 2, 3, 0]], dtype=torch.float32, device=args.device),
        torch.as_tensor(qpos[:, 7:], dtype=torch.float32, device=args.device),
    )
    qpos[:, 2] -= float(body_pos[..., 2].min().item())
    qpos[:, :2] -= qpos[0, :2]
    if qpos.shape[1] != 36 or not np.isfinite(qpos).all():
        raise ValueError(f"Invalid G1 qpos: {qpos.shape}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        qpos=qpos.astype(np.float32),
        fps=np.asarray(float(np.asarray(data["fps"]).item()), np.float64),
        robot=np.asarray("unitree_g1"),
        dof=np.int32(29),
        source_file=np.asarray(str(args.input.resolve())),
    )
    print(f"Saved {args.output}: qpos={qpos.shape}")


if __name__ == "__main__":
    main()
