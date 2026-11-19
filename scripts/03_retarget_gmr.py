#!/usr/bin/env python3
"""Headless SMPL-X -> Unitree G1 29-DoF retargeting with GMR."""

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
from general_motion_retargeting.utils.smpl import (  # noqa: E402
    get_smplx_data_offline_fast,
    load_smplx_file,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    model_dir = GMR_ROOT / "assets" / "body_models"
    smplx_data, body_model, smplx_output, human_height = load_smplx_file(
        args.input, model_dir
    )
    frames, fps = get_smplx_data_offline_fast(
        smplx_data, body_model, smplx_output, tgt_fps=args.fps
    )
    retargeter = GeneralMotionRetargeting(
        src_human="smplx",
        tgt_robot="unitree_g1",
        actual_human_height=human_height,
    )
    qpos = np.asarray([retargeter.retarget(frame).copy() for frame in frames])
    # GMR returns floating-base quaternion in wxyz; MJLab qpos uses wxyz too.
    # Apply one constant translation based on the lowest point over the entire
    # clip.  Do not ground every frame: that would erase real jump height.
    fk = KinematicsModel(retargeter.xml_file, device=args.device)
    root_pos = qpos[:, :3].copy()
    root_xyzw = qpos[:, 3:7][:, [1, 2, 3, 0]].copy()
    dof_pos = qpos[:, 7:].copy()
    body_pos, _ = fk.forward_kinematics(
        torch.as_tensor(root_pos, dtype=torch.float32, device=args.device),
        torch.as_tensor(root_xyzw, dtype=torch.float32, device=args.device),
        torch.as_tensor(dof_pos, dtype=torch.float32, device=args.device),
    )
    qpos[:, 2] -= float(body_pos[..., 2].min().item())
    qpos[:, :2] -= qpos[0, :2]
    if qpos.shape[1] != 36:
        raise ValueError(f"Expected free base 7 + 29 joints, got {qpos.shape}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        qpos=qpos.astype(np.float32),
        fps=np.asarray(float(fps), dtype=np.float64),
        robot=np.asarray("unitree_g1"),
        dof=np.int32(29),
        source_file=np.asarray(str(args.input.resolve())),
        human_height=np.float32(human_height),
    )
    print(f"Saved {args.output}: {len(qpos)} frames, {fps:.6f} fps, qpos={qpos.shape}")


if __name__ == "__main__":
    main()
