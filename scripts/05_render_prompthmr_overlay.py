#!/usr/bin/env python3
"""Render a selected PromptHMR camera-space SMPL-X track over the video."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import cv2
import joblib
import numpy as np
import pyrender
import torch
import trimesh


LAB_ROOT = Path(__file__).resolve().parents[1]
PROMPTHMR_ROOT = LAB_ROOT / "repos" / "PromptHMR"
CALL_CWD = Path.cwd()
sys.path.insert(0, str(PROMPTHMR_ROOT))
os.chdir(PROMPTHMR_ROOT)

from data_config import SMPLX_PATH  # noqa: E402
from prompt_hmr.smpl_family import SMPLX  # noqa: E402


def render_mesh(
    renderer: pyrender.OffscreenRenderer,
    vertices: np.ndarray,
    faces: np.ndarray,
    background: np.ndarray,
    focal: float,
    center: np.ndarray,
) -> np.ndarray:
    """Render OpenCV-camera vertices with EGL/pyrender (no PyTorch3D needed)."""
    scene = pyrender.Scene(
        bg_color=np.zeros(4, dtype=np.float32),
        ambient_light=np.full(3, 0.35, dtype=np.float32),
    )
    camera = pyrender.IntrinsicsCamera(
        fx=focal, fy=focal, cx=float(center[0]), cy=float(center[1])
    )
    scene.add(camera, pose=np.eye(4))
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
    scene.add(light, pose=np.eye(4))
    mesh = trimesh.Trimesh(vertices, faces, process=False)
    mesh.apply_transform(
        trimesh.transformations.rotation_matrix(np.pi, [1.0, 0.0, 0.0])
    )
    material = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.1,
        roughnessFactor=0.7,
        baseColorFactor=(0.2, 0.72, 1.0, 1.0),
    )
    scene.add(pyrender.Mesh.from_trimesh(mesh, material=material))
    rgba, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    output = background.copy()
    mask = depth > 0
    output[mask] = rgba[mask, :3]
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--track-id", type=int, default=None)
    args = parser.parse_args()

    for name in ("video", "results", "output"):
        path = getattr(args, name).expanduser()
        path = (CALL_CWD / path).resolve() if not path.is_absolute() else path.resolve()
        setattr(args, name, path)

    results = joblib.load(args.results)
    people = results["people"]
    track_id = args.track_id
    if track_id is None:
        if len(people) != 1:
            raise ValueError(f"Found tracks {sorted(people)}; pass --track-id.")
        track_id = next(iter(people))
    if track_id not in people:
        raise KeyError(f"Track {track_id} is not in {args.results}")
    track = people[track_id]

    capture = cv2.VideoCapture(str(args.video))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if not capture.isOpened() or width <= 0 or height <= 0:
        raise RuntimeError(f"Could not open {args.video}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create {args.output}")

    camera = results["camera"]
    center = np.asarray(camera["img_center"], dtype=np.float32)
    body = SMPLX(SMPLX_PATH).cuda().eval()
    faces = body.faces.astype(np.int64)
    renderer = pyrender.OffscreenRenderer(width, height)
    local_by_frame = {int(frame): i for i, frame in enumerate(track["frames"])}
    smplx_cam = track["smplx_cam"]

    with torch.inference_mode():
        for frame_index in range(frame_count):
            ok, frame_bgr = capture.read()
            if not ok:
                break
            local_index = local_by_frame.get(frame_index)
            if local_index is not None:
                rotations = torch.as_tensor(
                    smplx_cam["rotmat"][local_index : local_index + 1],
                    dtype=torch.float32,
                    device="cuda",
                )
                betas = torch.as_tensor(
                    smplx_cam["shape"][local_index : local_index + 1],
                    dtype=torch.float32,
                    device="cuda",
                )
                translation = torch.as_tensor(
                    smplx_cam["trans"][local_index : local_index + 1],
                    dtype=torch.float32,
                    device="cuda",
                )
                vertices = body(
                    global_orient=rotations[:, :1],
                    body_pose=rotations[:, 1:22],
                    betas=betas,
                    transl=translation,
                    pose2rot=False,
                ).vertices
                background = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                rendered = render_mesh(
                    renderer,
                    vertices[0].detach().cpu().numpy(),
                    faces,
                    background,
                    float(camera["img_focal"]),
                    center,
                )
                frame_bgr = cv2.cvtColor(
                    rendered.astype(np.uint8), cv2.COLOR_RGB2BGR
                )
            writer.write(frame_bgr)
            if (frame_index + 1) % 60 == 0 or frame_index + 1 == frame_count:
                print(f"Rendered {frame_index + 1}/{frame_count}", flush=True)

    capture.release()
    writer.release()
    renderer.delete()
    print(f"Saved {args.output.resolve()} (track {track_id})")


if __name__ == "__main__":
    main()
