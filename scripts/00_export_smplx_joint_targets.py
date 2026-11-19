#!/usr/bin/env python3
"""Instructor utility: bake licensed SMPL-X parameters into GMR joint targets.

The output contains only animated joint positions/orientations.  Students can
retarget it without PromptHMR checkpoints, Google Drive downloads, or SMPL-X
body-model files.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import smplx
import torch
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation, Slerp
from smplx.joint_names import JOINT_NAMES


GMR_JOINTS = (
    "pelvis", "spine3",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_foot", "right_foot",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()

    source = np.load(args.input, allow_pickle=True)
    betas = np.asarray(source["betas"], dtype=np.float32).reshape(-1)
    gender = str(np.asarray(source["gender"]).item())
    nframe = int(source["pose_body"].shape[0])
    body_model = smplx.create(
        args.model_dir, "smplx", gender=gender, use_pca=False, num_betas=len(betas)
    )
    zeros3 = torch.zeros(nframe, 3)
    with torch.inference_mode():
        body = body_model(
            betas=torch.from_numpy(betas).view(1, -1),
            global_orient=torch.from_numpy(source["root_orient"]).float(),
            body_pose=torch.from_numpy(source["pose_body"]).float(),
            transl=torch.from_numpy(source["trans"]).float(),
            left_hand_pose=torch.zeros(nframe, 45),
            right_hand_pose=torch.zeros(nframe, 45),
            jaw_pose=zeros3,
            leye_pose=zeros3,
            reye_pose=zeros3,
            expression=torch.zeros(nframe, body_model.num_expression_coeffs),
            return_full_pose=True,
        )

    parents = body_model.parents.cpu().numpy()
    all_names = list(JOINT_NAMES[: len(parents)])
    local_pose = body.full_pose.reshape(nframe, -1, 3).cpu().numpy()
    positions = body.joints[:, : len(parents)].cpu().numpy()
    source_fps = float(np.asarray(source["mocap_frame_rate"]).item())
    source_t = np.arange(nframe, dtype=np.float64) / source_fps
    output_frames = max(2, int(np.floor(nframe * args.fps / source_fps)))
    target_t = np.linspace(source_t[0], source_t[-1], output_frames)

    local_interp = np.empty((output_frames, len(parents), 3), np.float64)
    for joint_id in range(len(parents)):
        local_interp[:, joint_id] = Slerp(
            source_t, Rotation.from_rotvec(local_pose[:, joint_id])
        )(target_t).as_rotvec()
    pos_interp = interp1d(source_t, positions, axis=0)(target_t)

    selected = [all_names.index(name) for name in GMR_JOINTS]
    quat_wxyz = np.empty((output_frames, len(selected), 4), np.float32)
    for frame in range(output_frames):
        global_rotations: list[Rotation] = []
        for joint_id, parent_id in enumerate(parents):
            local = Rotation.from_rotvec(local_interp[frame, joint_id])
            global_rotations.append(
                local if parent_id < 0 else global_rotations[parent_id] * local
            )
        for out_id, joint_id in enumerate(selected):
            xyzw = global_rotations[joint_id].as_quat()
            quat_wxyz[frame, out_id] = xyzw[[3, 0, 1, 2]]

    human_height = float(1.66 + 0.1 * betas[0])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        format=np.asarray("gmr_joint_targets_v1"),
        joint_names=np.asarray(GMR_JOINTS),
        joint_positions=pos_interp[:, selected].astype(np.float32),
        joint_quaternions_wxyz=quat_wxyz,
        fps=np.asarray(args.fps, np.float32),
        human_height=np.asarray(human_height, np.float32),
        source_description=np.asarray("PromptHMR SMPL-X output, baked for class retargeting"),
    )
    print(f"Saved {args.output}: {output_frames} frames, {len(selected)} GMR joints")


if __name__ == "__main__":
    main()
