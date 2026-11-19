#!/usr/bin/env python3
"""Fast, non-destructive validation of the lab bundle and licensed models."""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def check(label: str, ok: bool, detail: str) -> bool:
    print(f"[{'OK' if ok else 'MISSING'}] {label}: {detail}")
    return ok


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=("files", "retarget", "prompthmr", "gmr", "mjlab"),
        default="files",
    )
    args = parser.parse_args()
    ok = True
    required = [
        ROOT / "repos/PromptHMR",
        ROOT / "repos/GMR",
        ROOT / "repos/unitree_rl_mjlab",
        ROOT / "sample_data/input/jump.mov",
    ]
    for path in required:
        ok &= check(path.name, path.exists(), str(path))
    if args.stage == "retarget":
        import numpy as np

        target_file = ROOT / "sample_data/input/jump_smplx_joint_targets_30fps.npz"
        ok &= check("prepared joint motion", target_file.is_file(), str(target_file))
        if target_file.is_file():
            motion = np.load(target_file, allow_pickle=False)
            fmt = str(np.asarray(motion["format"]).item())
            positions = motion["joint_positions"]
            quaternions = motion["joint_quaternions_wxyz"]
            ok &= check("joint target format", fmt == "gmr_joint_targets_v1", fmt)
            ok &= check(
                "joint target shapes",
                positions.ndim == 3
                and positions.shape[1:] == (14, 3)
                and quaternions.shape == (*positions.shape[:2], 4),
                f"position={positions.shape}, quaternion={quaternions.shape}",
            )
        for module in ("torch", "mujoco", "mink", "qpsolvers", "general_motion_retargeting"):
            ok &= check(module, importlib.util.find_spec(module) is not None, "Python import")
    if args.stage == "prompthmr":
        import torch

        ok &= check("PyTorch >= 2.7", tuple(map(int, torch.__version__.split("+")[0].split(".")[:2])) >= (2, 7), torch.__version__)
        ok &= check("CUDA available", torch.cuda.is_available(), str(torch.version.cuda))
        if torch.cuda.is_available():
            capability = torch.cuda.get_device_capability()
            ok &= check("RTX 5060 kernel support", capability >= (12, 0), f"{torch.cuda.get_device_name(0)}, sm_{capability[0]}{capability[1]}")
        for module in ("detectron2", "sam2", "xformers"):
            ok &= check(module, importlib.util.find_spec(module) is not None, "Python import")
        body_root = ROOT / "repos/PromptHMR/data/body_models"
        body_files = [
            body_root / "smplx" / f"SMPLX_{gender}.npz"
            for gender in ("NEUTRAL", "MALE", "FEMALE")
        ] + [
            body_root / "smpl" / f"SMPL_{gender}.pkl"
            for gender in ("NEUTRAL", "MALE", "FEMALE")
        ] + [
            body_root / "smplx2smpl.pkl",
            body_root / "smplx2smpl_joints.npy",
        ]
        for path in body_files:
            ok &= check(
                path.name,
                path.is_file() and not path.is_symlink(),
                f"{path} (must be a real file)",
            )
        for path in [
            ROOT / "repos/PromptHMR/data/pretrain/phmr/checkpoint.ckpt",
            ROOT / "repos/PromptHMR/data/pretrain/phmr_vid/prhmr_release_002.ckpt",
            ROOT / "repos/PromptHMR/data/pretrain/sam2_ckpts/sam2_hiera_tiny.pt",
            ROOT / "repos/PromptHMR/data/pretrain/vitpose-h-coco_25.pth",
        ]:
            ok &= check(path.name, path.exists(), str(path))
    if args.stage == "gmr":
        for module in ("mujoco", "smplx", "general_motion_retargeting"):
            ok &= check(module, importlib.util.find_spec(module) is not None, "Python import")
        for name in ("SMPLX_NEUTRAL.npz", "SMPLX_MALE.npz", "SMPLX_FEMALE.npz"):
            path = ROOT / "repos/GMR/assets/body_models/smplx" / name
            ok &= check(
                name,
                path.is_file() and not path.is_symlink(),
                f"{path} (must be a real file)",
            )
    if args.stage == "mjlab":
        for module in ("torch", "mujoco", "scipy", "mjlab", "rsl_rl"):
            ok &= check(module, importlib.util.find_spec(module) is not None, "Python import")
        import mujoco
        import warp

        version = tuple(map(int, mujoco.__version__.split(".")[:2]))
        ok &= check("MuJoCo >= 3.5", version >= (3, 5), mujoco.__version__)
        ok &= check(
            "MuJoCo MULTICCD enum",
            hasattr(mujoco.mjtEnableBit, "mjENBL_MULTICCD"),
            "mujoco.mjtEnableBit.mjENBL_MULTICCD",
        )
        ok &= check(
            "Warp version",
            warp.__version__ == "1.12.0",
            f"{warp.__version__} (required: 1.12.0)",
        )
        try:
            from mujoco_warp._src import sensor  # noqa: F401
        except Exception as exc:  # pragma: no cover - diagnostic path
            ok &= check("MuJoCo-Warp sensor kernels", False, repr(exc))
        else:
            ok &= check("MuJoCo-Warp sensor kernels", True, "import and parse")
        if subprocess.run(["nvidia-smi"], stdout=subprocess.DEVNULL).returncode == 0:
            check("NVIDIA driver", True, "nvidia-smi works")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
