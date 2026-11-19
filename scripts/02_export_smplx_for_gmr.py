#!/usr/bin/env python3
"""Convert one PromptHMR world track to the SMPL-X NPZ expected by GMR."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
from scipy.spatial.transform import Rotation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="PromptHMR results.pkl")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--track-id", type=int, default=None)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument(
        "--video",
        type=Path,
        default=None,
        help="Source video used to infer fps when --fps is omitted.",
    )
    args = parser.parse_args()

    results = joblib.load(args.input)
    people = results["people"]
    track_id = args.track_id
    if track_id is None:
        if len(people) != 1:
            raise ValueError(
                f"Found tracks {sorted(people)}. Pass --track-id after inspecting them."
            )
        track_id = next(iter(people))
    track = people[track_id]
    world = track["smplx_world"]
    pose = np.asarray(world["pose"], dtype=np.float32)
    shape = np.asarray(world["shape"], dtype=np.float32)
    trans = np.asarray(world["trans"], dtype=np.float32)

    # PromptHMR world is Y-up; GMR's SMPL-X loader assumes Z-up.
    world_to_z_up = Rotation.from_rotvec(np.array([np.pi / 2.0, 0.0, 0.0]))
    root_orient = (
        world_to_z_up * Rotation.from_rotvec(pose[:, :3])
    ).as_rotvec().astype(np.float32)
    trans_z_up = np.stack((trans[:, 0], -trans[:, 2], trans[:, 1]), axis=-1)
    trans_z_up[:, :2] -= trans_z_up[0, :2]

    fps = args.fps
    if fps is None and args.video is not None:
        import cv2

        capture = cv2.VideoCapture(str(args.video))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        capture.release()
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError(f"Could not read fps from {args.video}")
    if fps is None:
        raise ValueError("Pass --video to infer fps or provide --fps explicitly.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        pose_body=pose[:, 3:66],
        root_orient=root_orient,
        trans=trans_z_up.astype(np.float32),
        betas=shape.mean(axis=0).astype(np.float32),
        gender=np.asarray("neutral"),
        mocap_frame_rate=np.float32(fps),
        source_track_id=np.int32(track_id),
        source_results=np.asarray(str(args.input.resolve())),
    )
    print(f"Saved {args.output}: track={track_id}, frames={len(pose)}, fps={fps:g}")


if __name__ == "__main__":
    main()
