"""Script to play RL agent with RSL-RL."""

import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from src.tasks.tracking.mdp import MotionCommandCfg
from mjlab.utils.os import get_wandb_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer


@dataclass(frozen=True)
class PlayConfig:
  agent: Literal["zero", "random", "trained"] = "trained"
  checkpoint_file: str | None = None
  motion_file: str | None = None
  num_envs: int | None = None
  device: str | None = None
  video: bool = False
  video_length: int = 200
  video_height: int | None = None
  video_width: int | None = None
  camera: int | str | None = None
  viewer: Literal["auto", "native", "viser"] = "auto"
  viser_port: int | None = None
  """Use this TCP port for the Viser server instead of its default."""
  assistance_scale: float | None = None
  """Freeze bicycle curriculum assistance at this value in [0, 1] during play."""
  no_terminations: bool = False
  """Disable all termination conditions (useful for viewing motions with dummy agents)."""
  no_debug_vis: bool = False
  """Disable reference ghost / debug visualization (motion command debug_vis)."""
  show_collision: bool = False
  """Show collision geoms (group 3) together with visual meshes."""
  collision_only: bool = False
  """Hide visual meshes and show only collision geoms (group 3). Viser also gets Mesh/Collision toggles."""
  start_frame: int = 0
  """Tracking only: spawn robot+bike at this motion frame and run policy inference from there."""
  resmimic_base_policy: str | None = None
  """If set, policy actions are residuals added to frozen ResMimic base."""
  resmimic_residual_scale: float = 1.0

  # Internal flag used by demo script.
  _demo_mode: tyro.conf.Suppress[bool] = False


def run_play(task_id: str, cfg: PlayConfig):
  configure_torch_backends()

  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  env_cfg = load_env_cfg(task_id, play=True)
  agent_cfg = load_rl_cfg(task_id)

  if cfg.assistance_scale is not None:
    scale = float(cfg.assistance_scale)
    if not 0.0 <= scale <= 1.0:
      raise ValueError("--assistance-scale must be in [0, 1]")
    init_cfg = env_cfg.events.get("bicycle_curriculum_init")
    if init_cfg is None:
      raise ValueError(f"Task {task_id!r} has no bicycle curriculum to override")
    mode = init_cfg.params.get("mode")
    init_cfg.params["start_scale"] = scale
    # Effectively freeze the selected scale for an interactive play session.
    init_cfg.params["decay_until_iter"] = 1.0e12
    if mode == "PD":
      # A single play env would otherwise receive the first cohort level (0).
      init_cfg.params["pd_assist_levels"] = (1.0,)
    print(f"[INFO]: Frozen bicycle assistance: mode={mode}, scale={scale:.3f}")

  DUMMY_MODE = cfg.agent in {"zero", "random"}
  TRAINED_MODE = not DUMMY_MODE

  # Disable terminations if requested (useful for viewing motions).
  if cfg.no_terminations:
    env_cfg.terminations = {}
    print("[INFO]: Terminations disabled")

  # Check if this is a tracking task by checking for motion command.
  is_tracking_task = "motion" in env_cfg.commands and isinstance(
    env_cfg.commands["motion"], MotionCommandCfg
  )

  if is_tracking_task and cfg._demo_mode:
    # Demo mode: use uniform sampling to see more diversity with num_envs > 1.
    motion_cmd = env_cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)
    motion_cmd.sampling_mode = "uniform"

  if is_tracking_task:
    motion_cmd = env_cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)

    # Check for local motion file first (works for both dummy and trained modes).
    if cfg.motion_file is not None and Path(cfg.motion_file).exists():
      print(f"[INFO]: Using local motion file: {cfg.motion_file}")
      motion_cmd.motion_file = cfg.motion_file
    elif DUMMY_MODE:
      if not cfg.registry_name:
        raise ValueError(
          "Tracking tasks require either:\n"
          "  --motion-file /path/to/motion.npz (local file)\n"
          "  --registry-name your-org/motions/motion-name (download from WandB)"
        )

    if cfg.no_debug_vis:
      motion_cmd.debug_vis = False
      print("[INFO]: Reference ghost / motion debug_vis disabled")

    if cfg.start_frame > 0:
      # Play cfgs already use sampling_mode='start'; override the reset index
      # so robot + bike teleport to mid-clip and the policy tracks from there.
      motion_cmd.sampling_mode = "start"
      motion_cmd.start_frame = int(cfg.start_frame)
      print(
        f"[INFO]: Motion start_frame={motion_cmd.start_frame} "
        "(policy inference begins at this reference frame on each reset)"
      )
  elif cfg.start_frame > 0:
    print("[WARN]: --start-frame ignored (task has no motion command)")
  log_dir: Path | None = None
  resume_path: Path | None = None
  if TRAINED_MODE:
    log_root_path = (Path("logs") / "rsl_rl" / agent_cfg.experiment_name).resolve()
    if cfg.checkpoint_file is not None:
      resume_path = Path(cfg.checkpoint_file)
      if not resume_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {resume_path}")
      print(f"[INFO]: Loading checkpoint: {resume_path.name}")
    else:
      if cfg.wandb_run_path is None:
        raise ValueError(
          "`wandb_run_path` is required when `checkpoint_file` is not provided."
        )
      resume_path, was_cached = get_wandb_checkpoint_path(
        log_root_path, Path(cfg.wandb_run_path)
      )
      # Extract run_id and checkpoint name from path for display.
      run_id = resume_path.parent.name
      checkpoint_name = resume_path.name
      cached_str = "cached" if was_cached else "downloaded"
      print(
        f"[INFO]: Loading checkpoint: {checkpoint_name} (run: {run_id}, {cached_str})"
      )
    log_dir = resume_path.parent

  if cfg.num_envs is not None:
    env_cfg.scene.num_envs = cfg.num_envs
  if cfg.video_height is not None:
    env_cfg.viewer.height = cfg.video_height
  if cfg.video_width is not None:
    env_cfg.viewer.width = cfg.video_width

  render_mode = "rgb_array" if (TRAINED_MODE and cfg.video) else None
  if cfg.video and DUMMY_MODE:
    print(
      "[WARN] Video recording with dummy agents is disabled (no checkpoint/log_dir)."
    )
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)

  if TRAINED_MODE and cfg.video:
    print("[INFO] Recording videos during play")
    assert log_dir is not None  # log_dir is set in TRAINED_MODE block
    env = VideoRecorder(
      env,
      video_folder=log_dir / "videos" / "play",
      step_trigger=lambda step: step == 0,
      video_length=cfg.video_length,
      disable_logger=True,
    )

  if cfg.resmimic_base_policy:
    from src.tasks.tracking.rl import ResMimicResidualVecEnvWrapper

    env = ResMimicResidualVecEnvWrapper(
      env,
      base_policy_path=cfg.resmimic_base_policy,
      clip_actions=agent_cfg.clip_actions,
      residual_scale=cfg.resmimic_residual_scale,
    )
    print(f"[INFO]: ResMimic residual wrapper enabled ({cfg.resmimic_base_policy})")
  else:
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  if DUMMY_MODE:
    action_shape: tuple[int, ...] = env.unwrapped.action_space.shape
    if cfg.agent == "zero":

      class PolicyZero:
        def __call__(self, obs) -> torch.Tensor:
          del obs
          return torch.zeros(action_shape, device=env.unwrapped.device)

      policy = PolicyZero()
    else:

      class PolicyRandom:
        def __call__(self, obs) -> torch.Tensor:
          del obs
          return 2 * torch.rand(action_shape, device=env.unwrapped.device) - 1

      policy = PolicyRandom()
  else:
    runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    runner.load(
      str(resume_path), load_cfg={"actor": True}, strict=True, map_location=device
    )
    policy = runner.get_inference_policy(device=device)

  # Handle "auto" viewer selection.
  if cfg.viewer == "auto":
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    resolved_viewer = "native" if has_display else "viser"
    del has_display
  else:
    resolved_viewer = cfg.viewer

  if resolved_viewer == "native":
    viewer = NativeMujocoViewer(env, policy)
    _patch_play_viz(viewer, cfg, backend="native")
    viewer.run()
  elif resolved_viewer == "viser":
    viser_server = None
    if cfg.viser_port is not None:
      import viser

      viser_server = viser.ViserServer(
        host="0.0.0.0", port=int(cfg.viser_port), label="mjlab play"
      )
      print(f"[INFO]: Viser URL: http://localhost:{cfg.viser_port}")
    viewer = ViserPlayViewer(env, policy, viser_server=viser_server)
    _patch_play_viz(viewer, cfg, backend="viser")
    viewer.run()
  else:
    raise RuntimeError(f"Unsupported viewer backend: {resolved_viewer}")

  env.close()


def _apply_viser_geom_preset(scene, *, collision_only: bool) -> None:
  """Mesh view = G0–G2 (+G4/G5); collision-only = G3 only."""
  if collision_only:
    scene.geom_groups_visible = [False, False, False, True, False, False]
  else:
    scene.geom_groups_visible = [True, True, True, False, True, True]
  scene._sync_visibilities()
  scene._request_update()


def _patch_play_viz(
  viewer, cfg: PlayConfig, backend: Literal["native", "viser"]
) -> None:
  """Hook viewer.setup so collision / ghost overrides apply after scene exists."""
  want_collision_ui = cfg.show_collision or cfg.collision_only
  if not (want_collision_ui or cfg.no_debug_vis):
    return

  orig_setup = viewer.setup

  def setup_with_viz() -> None:
    orig_setup()
    if backend == "viser":
      scene = viewer._scene
      if cfg.no_debug_vis:
        scene.debug_visualization_enabled = False
        print("[INFO]: Viser debug viz / reference ghost disabled")

      if want_collision_ui:
        viewer._play_collision_only = bool(cfg.collision_only)
        with viewer._server.gui.add_folder("Display"):
          status = viewer._server.gui.add_markdown(
            "**View:** collision-only" if cfg.collision_only else "**View:** mesh"
          )
          collision_btn = viewer._server.gui.add_button(
            "Collision only (G3)",
            hint="Hide visual meshes; show collision geoms only.",
          )
          mesh_btn = viewer._server.gui.add_button(
            "Mesh view",
            hint="Show visual meshes; hide collision geoms.",
          )
          both_btn = viewer._server.gui.add_button(
            "Mesh + collision",
            hint="Show visual meshes and collision geoms together.",
          )

          def _set_status(label: str) -> None:
            status.content = f"**View:** {label}"

          @collision_btn.on_click
          def _(_) -> None:
            viewer._play_collision_only = True
            _apply_viser_geom_preset(scene, collision_only=True)
            _set_status("collision-only")

          @mesh_btn.on_click
          def _(_) -> None:
            viewer._play_collision_only = False
            _apply_viser_geom_preset(scene, collision_only=False)
            _set_status("mesh")

          @both_btn.on_click
          def _(_) -> None:
            viewer._play_collision_only = False
            scene.geom_groups_visible = [True, True, True, True, True, True]
            scene._sync_visibilities()
            scene._request_update()
            _set_status("mesh + collision")

        if cfg.collision_only:
          _apply_viser_geom_preset(scene, collision_only=True)
          print("[INFO]: Viser collision-only mode (Display tab to toggle)")
        elif cfg.show_collision:
          scene.geom_groups_visible = [True, True, True, True, True, True]
          scene._sync_visibilities()
          scene._request_update()
          print("[INFO]: Viser mesh + collision (Display tab to toggle)")
      return

    if cfg.no_debug_vis:
      viewer._show_debug_vis = False
      print("[INFO]: Native debug viz / reference ghost disabled")
    v = getattr(viewer, "viewer", None)
    if v is not None and want_collision_ui:
      if cfg.collision_only:
        v.opt.geomgroup[:] = [0, 0, 0, 1, 0, 0]
        print("[INFO]: Native collision-only mode")
      elif cfg.show_collision:
        v.opt.geomgroup[3] = 1
        print("[INFO]: Collision geoms (group 3) enabled in native viewer")

  viewer.setup = setup_with_viz  # type: ignore[method-assign]


def main():
  # Parse first argument to choose the task.
  # Import tasks to populate the registry.
  import mjlab.tasks  # noqa: F401
  import src.tasks

  all_tasks = list_tasks()
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )

  # Parse the rest of the arguments + allow overriding env_cfg and agent_cfg.
  agent_cfg = load_rl_cfg(chosen_task)

  args = tyro.cli(
    PlayConfig,
    args=remaining_args,
    default=PlayConfig(),
    prog=sys.argv[0] + f" {chosen_task}",
    config=mjlab.TYRO_FLAGS,
  )
  del remaining_args, agent_cfg

  run_play(chosen_task, args)


if __name__ == "__main__":
  main()
