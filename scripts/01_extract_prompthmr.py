#!/usr/bin/env python3
"""Run PromptHMR on one video with deterministic subject selection."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


LAB_ROOT = Path(__file__).resolve().parents[1]
PROMPTHMR_ROOT = LAB_ROOT / "repos" / "PromptHMR"
CALL_CWD = Path.cwd()
sys.path.insert(0, str(PROMPTHMR_ROOT))
os.chdir(PROMPTHMR_ROOT)

from pipeline import Pipeline  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Video -> selected, temporally tracked SMPL-X body."
    )
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--static-camera", action="store_true")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument(
        "--person-rank",
        type=int,
        default=0,
        help="0 is the largest/highest-confidence person at the initialization frame.",
    )
    parser.add_argument("--num-people", type=int, default=1)
    parser.add_argument("--target-x", type=float, default=None)
    parser.add_argument("--target-y", type=float, default=None)
    parser.add_argument("--det-thresh", type=float, default=0.8)
    parser.add_argument("--debug-masks", action="store_true")
    parser.add_argument("--max-height", type=int, default=640)
    parser.add_argument("--max-fps", type=int, default=30)
    parser.add_argument("--seg-batch-size", type=int, default=2)
    parser.add_argument("--phmr-batch-size", type=int, default=1)
    args = parser.parse_args()

    video = args.video.expanduser()
    output = args.output.expanduser()
    video = (CALL_CWD / video).resolve() if not video.is_absolute() else video.resolve()
    output = (CALL_CWD / output).resolve() if not output.is_absolute() else output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    pipeline = Pipeline(static_cam=args.static_camera)
    pipeline.cfg.init_start_frame = args.start_frame
    pipeline.cfg.init_person_rank = args.person_rank
    if (args.target_x is None) != (args.target_y is None):
        raise ValueError("Pass both --target-x and --target-y, or neither.")
    pipeline.cfg.init_target_point = (
        None if args.target_x is None else [args.target_x, args.target_y]
    )
    pipeline.cfg.init_num_people = args.num_people
    pipeline.cfg.num_max_people = args.num_people
    pipeline.cfg.det_thresh = args.det_thresh
    pipeline.cfg.debug_masks = args.debug_masks
    pipeline.cfg.max_height = args.max_height
    pipeline.cfg.max_fps = args.max_fps
    pipeline.cfg.seg_batch_size = args.seg_batch_size
    pipeline.cfg.phmr_batch_size = args.phmr_batch_size
    pipeline(str(video), str(output), save_only_essential=True)

    print(f"PromptHMR result: {output / 'results.pkl'}")
    print(f"Bodies requested: {args.num_people}; initialization frame: {args.start_frame}")


if __name__ == "__main__":
    main()
