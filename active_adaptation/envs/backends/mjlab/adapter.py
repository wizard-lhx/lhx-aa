from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import torch

from typing_extensions import override

from active_adaptation.envs.adapters import SimAdapter, SceneAdapter
from mjlab.entity.data import EntityData
from mjlab.sensor.contact_sensor import ContactData

if TYPE_CHECKING:
    from active_adaptation.envs.backends.mjlab.viewer import MjLabViewer
    from mjlab.scene import Scene
    from mjlab.sim import Simulation
    from mjlab.viewer.offscreen_renderer import OffscreenRenderer
    from mjlab.viewer.viewer_config import ViewerConfig


def _install_entity_data_aliases() -> None:
    if not hasattr(EntityData, "root_link_state_w"):
        EntityData.root_link_state_w = property(
            lambda self: torch.cat([self.root_link_pose_w, self.root_link_vel_w], dim=-1)
        )
    if not hasattr(EntityData, "body_lin_vel_w"):
        EntityData.body_lin_vel_w = property(lambda self: self.body_link_lin_vel_w)
    if not hasattr(EntityData, "body_ang_vel_w"):
        EntityData.body_ang_vel_w = property(lambda self: self.body_link_ang_vel_w)
    if not hasattr(EntityData, "joint_limits"):
        EntityData.joint_limits = property(lambda self: self.joint_pos_limits)
    if not hasattr(EntityData, "applied_torque"):
        EntityData.applied_torque = property(lambda self: self.actuator_force)


_install_entity_data_aliases()


def _install_contact_data_aliases() -> None:
    if not hasattr(ContactData, "net_forces_w"):
        ContactData.net_forces_w = property(lambda self: self.force)


_install_contact_data_aliases()


try:
    from isaaclab.utils.string import resolve_matching_names
except ModuleNotFoundError:
    from mjlab.utils.lab_api.string import resolve_matching_names


class MjlabEntityIndexingAdapter:
    _JOINT_FIELDS = {"joint_ids", "joint_q_adr", "joint_v_adr", "ctrl_ids"}

    def __init__(self, indexing, joint_ids: torch.Tensor, body_ids: torch.Tensor):
        self._indexing = indexing
        self._joint_ids = joint_ids
        self._body_ids = body_ids

    def __getattr__(self, name):
        value = getattr(self._indexing, name)
        if isinstance(value, torch.Tensor):
            if name in self._JOINT_FIELDS:
                return value[self._joint_ids.to(value.device)]
            if name == "body_ids":
                return value[self._body_ids.to(value.device)]
        return value


class MjlabEntityDataAdapter:
    _JOINT_FIELDS = {
        "default_joint_pos",
        "default_joint_vel",
        "default_joint_pos_limits",
        "joint_limits",
        "joint_pos_limits",
        "soft_joint_pos_limits",
        "joint_pos",
        "joint_pos_biased",
        "joint_vel",
        "joint_acc",
        "joint_pos_target",
        "joint_vel_target",
        "encoder_bias",
    }
    _ACTUATOR_FIELDS = {
        "actuator_force",
        "applied_torque",
        "joint_effort_target",
    }

    def __init__(
        self,
        data,
        joint_ids: torch.Tensor,
        body_ids: torch.Tensor,
        actuator_ids: torch.Tensor,
    ):
        self._data = data
        self._joint_ids = joint_ids
        self._body_ids = body_ids
        self._actuator_ids = actuator_ids
        self._indexing = MjlabEntityIndexingAdapter(
            data.indexing,
            joint_ids,
            body_ids,
        )

    def __getattr__(self, name):
        if name == "indexing":
            return self._indexing

        value = getattr(self._data, name)
        if not isinstance(value, torch.Tensor):
            return value

        if name in self._JOINT_FIELDS:
            return value[:, self._joint_ids.to(value.device)]
        if name in self._ACTUATOR_FIELDS:
            return value[:, self._actuator_ids.to(value.device)]
        if name.startswith("body_"):
            return value[:, self._body_ids.to(value.device)]
        return value


class MjlabEntityAdapter:
    def __init__(self, entity, sim):
        self._entity = entity
        self.cfg = entity.cfg

        joint_names = list(entity.joint_names)
        body_names = list(entity.body_names)
        ctrl_ids = entity.data.indexing.ctrl_ids
        actuator_names = [
            sim.mj_model.actuator(int(ctrl_id)).name.split("/")[-1]
            for ctrl_id in ctrl_ids.cpu()
        ]

        self._joint_names = list(entity.cfg.joint_names_simulation)
        self._body_names = list(entity.cfg.body_names_simulation)
        device = entity.data.joint_pos.device
        self._joint_ids = torch.tensor(
            [joint_names.index(name) for name in self._joint_names],
            dtype=torch.long,
            device=device,
        )
        self._body_ids = torch.tensor(
            [body_names.index(name) for name in self._body_names],
            dtype=torch.long,
            device=device,
        )
        self._actuator_ids = torch.tensor(
            [actuator_names.index(name) for name in self._joint_names],
            dtype=torch.long,
            device=device,
        )

    @property
    def data(self):
        return MjlabEntityDataAdapter(
            self._entity.data,
            self._joint_ids,
            self._body_ids,
            self._actuator_ids,
        )

    @property
    def joint_names(self):
        return tuple(self._joint_names)

    @property
    def body_names(self):
        return tuple(self._body_names)

    @property
    def num_joints(self):
        return len(self._joint_names)

    @property
    def num_bodies(self):
        return len(self._body_names)

    def find_joints(self, name_keys, preserve_order: bool = False):
        return resolve_matching_names(name_keys, self._joint_names, preserve_order)

    def find_bodies(self, name_keys, preserve_order: bool = False):
        return resolve_matching_names(name_keys, self._body_names, preserve_order)

    def _map_joint_ids(self, joint_ids):
        if joint_ids is None:
            return self._joint_ids
        if isinstance(joint_ids, slice):
            external_ids = torch.arange(
                len(self._joint_ids),
                device=self._joint_ids.device,
            )[joint_ids]
        else:
            external_ids = torch.as_tensor(
                joint_ids,
                dtype=torch.long,
                device=self._joint_ids.device,
            )
        return self._joint_ids[external_ids].reshape(-1)

    def _set_joint_tensor(self, tensor, value, joint_ids, env_ids):
        joint_ids = self._map_joint_ids(joint_ids).to(tensor.device)
        if env_ids is None:
            env_ids = slice(None)
        elif not isinstance(env_ids, slice):
            env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=tensor.device)

        if isinstance(env_ids, slice):
            tensor[env_ids, joint_ids] = value
        else:
            tensor[env_ids[:, None], joint_ids[None, :]] = value

    def set_joint_position_target(self, position, joint_ids=None, env_ids=None):
        self._set_joint_tensor(
            self._entity.data.joint_pos_target,
            position,
            joint_ids,
            env_ids,
        )

    def set_joint_velocity_target(self, velocity, joint_ids=None, env_ids=None):
        self._set_joint_tensor(
            self._entity.data.joint_vel_target,
            velocity,
            joint_ids,
            env_ids,
        )

    def set_joint_effort_target(self, effort, joint_ids=None, env_ids=None):
        self._set_joint_tensor(
            self._entity.data.joint_effort_target,
            effort,
            joint_ids,
            env_ids,
        )

    def write_joint_state_to_sim(
        self,
        position,
        velocity,
        joint_ids=None,
        env_ids=None,
    ):
        self._entity.write_joint_state_to_sim(
            position,
            velocity,
            joint_ids=self._map_joint_ids(joint_ids),
            env_ids=env_ids,
        )

    def write_joint_position_to_sim(self, position, joint_ids=None, env_ids=None):
        self._entity.write_joint_position_to_sim(
            position,
            joint_ids=self._map_joint_ids(joint_ids),
            env_ids=env_ids,
        )

    def write_joint_velocity_to_sim(self, velocity, joint_ids=None, env_ids=None):
        self._entity.write_joint_velocity_to_sim(
            velocity,
            joint_ids=self._map_joint_ids(joint_ids),
            env_ids=env_ids,
        )

    def __getattr__(self, name):
        return getattr(self._entity, name)


class MjlabSimAdapter(SimAdapter):
    def __init__(
        self,
        sim: "Simulation",
        viewer: "MjLabViewer" = None,
        viewer_cfg: "ViewerConfig" = None,
        scene: "Scene" = None,
    ):
        self._sim = sim
        self.viewer = viewer
        self._viewer_cfg = viewer_cfg
        self._scene = scene
        self._offscreen_renderer: "OffscreenRenderer | None" = None

    def get_physics_dt(self) -> float:
        return self._sim.cfg.mujoco.timestep

    def has_gui(self) -> bool:
        return self.viewer is not None

    def step(self, render: bool = False) -> None:
        self._sim.step()

    def render(self) -> None:
        if self.viewer is not None:
            self.viewer.update()

    def render_rgb_array(self) -> np.ndarray:
        renderer = self._get_offscreen_renderer()
        renderer.update(self._sim.data)
        return renderer.render()

    def set_camera_view(self, eye=None, target=None, **kwargs) -> None:
        if eye is None or target is None or self._viewer_cfg is None:
            return

        eye = np.asarray(eye, dtype=float)
        target = np.asarray(target, dtype=float)
        delta = eye - target
        distance = float(np.linalg.norm(delta))
        if distance <= 1e-8:
            return

        planar = math.hypot(float(delta[0]), float(delta[1]))
        self._viewer_cfg.lookat = tuple(float(v) for v in target.tolist())
        self._viewer_cfg.distance = distance
        self._viewer_cfg.azimuth = math.degrees(math.atan2(delta[1], delta[0]))
        self._viewer_cfg.elevation = -math.degrees(math.atan2(delta[2], planar))

        if self._offscreen_renderer is not None:
            self._offscreen_renderer.close()
            self._offscreen_renderer = None

    def close(self) -> None:
        if self._offscreen_renderer is not None:
            self._offscreen_renderer.close()
            self._offscreen_renderer = None

    def _get_offscreen_renderer(self) -> "OffscreenRenderer":
        if self._offscreen_renderer is None:
            if self._viewer_cfg is None or self._scene is None:
                raise ValueError("MjLab offscreen renderer is not configured.")

            from mjlab.viewer.offscreen_renderer import OffscreenRenderer

            renderer = OffscreenRenderer(
                model=self._sim.mj_model,
                cfg=self._viewer_cfg,
                scene=self._scene,
            )
            renderer.initialize()
            self._offscreen_renderer = renderer
        return self._offscreen_renderer

    def __getattr__(self, name):
        return getattr(self._sim, name)


class MjlabSceneAdapter(SceneAdapter):
    def __init__(self, scene: Scene, sim: Simulation):
        self._scene = scene
        self._sim = sim
        self._articulations = {
            name: MjlabEntityAdapter(entity, sim)
            for name, entity in scene.entities.items()
            if hasattr(entity.cfg, "joint_names_simulation")
        }

    @override
    def zero_external_wrenches(self) -> None:
        for asset in self._scene.entities.values():
            asset.data.data.xfrc_applied.zero_()

    @property
    def articulations(self):
        return self._articulations

    def __getattr__(self, name):
        return getattr(self._scene, name)

    @property
    def ground_mesh(self):
        """Warp mesh for the mjlab terrain body (name ``terrain``), for ray height queries."""
        if hasattr(self, "_ground_mesh"):
            return self._ground_mesh

        if self._scene.terrain is None:
            self._ground_mesh = None
            return self._ground_mesh

        import mujoco
        import numpy as np
        import trimesh
        import warp as wp
        from mujoco import mjtGeom
        from mjviser.conversions import create_primitive_mesh, mujoco_mesh_to_trimesh

        mj_model = self._sim.mj_model
        terrain_bid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "terrain")
        if terrain_bid < 0:
            self._ground_mesh = None
            return self._ground_mesh

        mj_data = mujoco.MjData(mj_model)
        mujoco.mj_forward(mj_model, mj_data)

        parts: list[trimesh.Trimesh] = []
        for gid in range(mj_model.ngeom):
            if mj_model.geom_bodyid[gid] != terrain_bid:
                continue
            gt = int(mj_model.geom_type[gid])
            if gt == int(mjtGeom.mjGEOM_MESH):
                mesh = mujoco_mesh_to_trimesh(mj_model, gid)
            else:
                mesh = create_primitive_mesh(mj_model, gid)
            if mesh is None and gt == int(mjtGeom.mjGEOM_PLANE):
                mesh = trimesh.creation.box(extents=[400.0, 400.0, 0.1])
                mesh.apply_translation([0.0, 0.0, -0.05])
            if mesh is None:
                continue
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = mj_data.geom_xmat[gid].reshape(3, 3)
            T[:3, 3] = mj_data.geom_xpos[gid]
            mesh.apply_transform(T)
            parts.append(mesh)

        if not parts:
            self._ground_mesh = None
            return self._ground_mesh

        combined = parts[0] if len(parts) == 1 else trimesh.util.concatenate(parts)
        device = wp.get_device(str(self._scene.device))
        self._ground_mesh = wp.Mesh(
            points=wp.array(
                np.asarray(combined.vertices, dtype=np.float32),
                dtype=wp.vec3,
                device=device,
            ),
            indices=wp.array(
                np.asarray(combined.faces, dtype=np.int32).flatten(),
                dtype=wp.int32,
                device=device,
            ),
        )
        return self._ground_mesh


__all__ = [
    "MjlabSimAdapter",
    "MjlabSceneAdapter",
]
