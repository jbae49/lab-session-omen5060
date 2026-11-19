"""RMA-style dual privileged encoders for bicycle policies."""

from __future__ import annotations

import copy
import os

import torch
from torch import nn

from rsl_rl.models import MLPModel
from rsl_rl.modules import MLP


class BicycleDualPrivilegedEncoderModel(MLPModel):
  """Encode robot and bicycle privilege separately before the policy MLP.

  The actor observation set must resolve to exactly three manager groups in
  this order: regular, robot_privileged, bicycle_privileged.  The two
  privileged vectors are independently compressed to ``latent_dim`` while the
  regular observation remains uncompressed.
  """

  def __init__(
    self,
    *args,
    encoder_hidden_dims: tuple[int, ...] | list[int] = (256, 128),
    latent_dim: int = 8,
    **kwargs,
  ) -> None:
    self.encoder_hidden_dims = tuple(encoder_hidden_dims)
    self.latent_dim = int(latent_dim)
    self._nan_diagnostics = os.environ.get(
      "MJLAB_DUAL_LATENT_NAN_DIAGNOSTICS", "0"
    ) == "1"
    if self.latent_dim <= 0:
      raise ValueError("latent_dim must be positive")
    super().__init__(*args, **kwargs)
    if len(self.obs_groups) != 3:
      raise ValueError(
        "BicycleDualPrivilegedEncoderModel needs exactly three actor groups: "
        "regular, robot_privileged, bicycle_privileged"
      )
    activation = kwargs.get("activation", "elu")
    self.robot_encoder = MLP(
      self._group_dims[1],
      self.latent_dim,
      self.encoder_hidden_dims,
      activation,
      last_activation=activation,
    )
    self.bicycle_encoder = MLP(
      self._group_dims[2],
      self.latent_dim,
      self.encoder_hidden_dims,
      activation,
      last_activation=activation,
    )

  def _get_obs_dim(self, obs, obs_groups, obs_set):
    groups, total_dim = super()._get_obs_dim(obs, obs_groups, obs_set)
    self._group_dims = tuple(int(obs[name].shape[-1]) for name in groups)
    return groups, total_dim

  def _get_latent_dim(self) -> int:
    if len(self._group_dims) != 3:
      return self.obs_dim
    return self._group_dims[0] + 2 * self.latent_dim

  def get_latent(self, obs, masks=None, hidden_state=None):
    del masks, hidden_state
    raw = torch.cat([obs[name] for name in self.obs_groups], dim=-1)
    self._check_finite("raw actor observation", raw)
    normalized = self.obs_normalizer(raw)
    self._check_finite("normalized actor observation", normalized)
    regular, robot_privileged, bicycle_privileged = torch.split(
      normalized, self._group_dims, dim=-1
    )
    robot_latent = self.robot_encoder(robot_privileged)
    self._check_finite("robot privileged latent", robot_latent)
    bicycle_latent = self.bicycle_encoder(bicycle_privileged)
    self._check_finite("bicycle privileged latent", bicycle_latent)
    return torch.cat((regular, robot_latent, bicycle_latent), dim=-1)

  def _check_finite(self, name: str, tensor: torch.Tensor) -> None:
    if not self._nan_diagnostics or torch.isfinite(tensor).all():
      return
    flat = tensor.reshape(tensor.shape[0], -1)
    invalid = ~torch.isfinite(flat)
    env_ids = invalid.any(dim=-1).nonzero(as_tuple=True)[0]
    channels = invalid[env_ids].nonzero(as_tuple=False)[:, 1].unique()
    bad_parameters = [
      param_name
      for param_name, parameter in self.named_parameters()
      if not torch.isfinite(parameter).all()
    ]
    print("\n[Dual-latent NaN diagnostics]", flush=True)
    print(f"  first bad stage: {name}", flush=True)
    print(f"  env IDs: {env_ids.detach().cpu().tolist()}", flush=True)
    print(f"  channels: {channels.detach().cpu().tolist()}", flush=True)
    print(f"  non-finite model parameters: {bad_parameters}", flush=True)
    raise FloatingPointError(
      f"Non-finite values at {name}; envs={env_ids[:10].detach().cpu().tolist()}, "
      f"channels={channels.detach().cpu().tolist()}"
    )

  def as_jit(self) -> nn.Module:
    return _ExportedBicycleDualLatentModel(self)

  def as_onnx(self, verbose: bool = False) -> nn.Module:
    del verbose
    return _ExportedBicycleDualLatentModel(self)


class _ExportedBicycleDualLatentModel(nn.Module):
  """Inference wrapper accepting the pre-concatenated three raw groups."""

  is_recurrent: bool = False

  def __init__(self, model: BicycleDualPrivilegedEncoderModel) -> None:
    super().__init__()
    self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
    self.robot_encoder = copy.deepcopy(model.robot_encoder)
    self.bicycle_encoder = copy.deepcopy(model.bicycle_encoder)
    self.mlp = copy.deepcopy(model.mlp)
    self.regular_dim = model._group_dims[0]
    self.robot_privileged_dim = model._group_dims[1]
    self.bicycle_privileged_dim = model._group_dims[2]
    self.input_size = model.obs_dim
    if model.distribution is not None:
      self.deterministic_output = model.distribution.as_deterministic_output_module()
    else:
      self.deterministic_output = nn.Identity()

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.obs_normalizer(x)
    regular, robot_privileged, bicycle_privileged = torch.split(
      x,
      (
        self.regular_dim,
        self.robot_privileged_dim,
        self.bicycle_privileged_dim,
      ),
      dim=-1,
    )
    latent = torch.cat(
      (
        regular,
        self.robot_encoder(robot_privileged),
        self.bicycle_encoder(bicycle_privileged),
      ),
      dim=-1,
    )
    return self.deterministic_output(self.mlp(latent))

  def get_dummy_inputs(self) -> tuple[torch.Tensor]:
    return (torch.zeros(1, self.input_size),)

  @property
  def input_names(self) -> list[str]:
    return ["obs"]

  @property
  def output_names(self) -> list[str]:
    return ["actions"]

  @torch.jit.export
  def reset(self) -> None:
    pass
