# Copyright (c) Meta Platforms, Inc. and affiliates.
import os

import numpy as np
from PIL import Image
import torch
import math
import utils3d
from PIL import Image
import numpy as np
from copy import deepcopy
from pytorch3d.transforms import quaternion_multiply, quaternion_invert
from sam3d_objects.model.backbone.tdfy_dit.utils import render_utils

from sam3d_objects.utils.visualization import SceneVisualizer

def _yaw_pitch_r_fov_to_extrinsics_intrinsics(yaws, pitchs, rs, fovs):
    is_list = isinstance(yaws, list)
    if not is_list:
        yaws = [yaws]
        pitchs = [pitchs]
    if not isinstance(rs, list):
        rs = [rs] * len(yaws)
    if not isinstance(fovs, list):
        fovs = [fovs] * len(yaws)
    extrinsics = []
    intrinsics = []
    for yaw, pitch, r, fov in zip(yaws, pitchs, rs, fovs):
        fov = torch.deg2rad(torch.tensor(float(fov))).cuda()
        yaw = torch.tensor(float(yaw)).cuda()
        pitch = torch.tensor(float(pitch)).cuda()
        orig = (
            torch.tensor(
                [
                    torch.sin(yaw) * torch.cos(pitch),
                    torch.sin(pitch),
                    torch.cos(yaw) * torch.cos(pitch),
                ]
            ).cuda()
            * r
        )
        extr = utils3d.torch.extrinsics_look_at(
            orig,
            torch.tensor([0, 0, 0]).float().cuda(),
            torch.tensor([0, 1, 0]).float().cuda(),
        )
        intr = utils3d.torch.intrinsics_from_fov_xy(fov, fov)
        extrinsics.append(extr)
        intrinsics.append(intr)
    if not is_list:
        extrinsics = extrinsics[0]
        intrinsics = intrinsics[0]
    return extrinsics, intrinsics


def render_video(
    sample,
    resolution=512,
    bg_color=(0, 0, 0),
    num_frames=300,
    r=2.0,
    fov=40,
    pitch_deg=0,
    yaw_start_deg=-90,
    **kwargs,
):

    yaws = (
        torch.linspace(0, 2 * torch.pi, num_frames) + math.radians(yaw_start_deg)
    ).tolist()
    pitch = [math.radians(pitch_deg)] * num_frames

    extr, intr = _yaw_pitch_r_fov_to_extrinsics_intrinsics(yaws, pitch, r, fov)

    return render_utils.render_frames(
        sample,
        extr,
        intr,
        {"resolution": resolution, "bg_color": bg_color, "backend": "gsplat"},
        **kwargs,
    )


def ready_gaussian_for_video_rendering(scene_gs, in_place=False, fix_alignment=False):
    if fix_alignment:
        scene_gs = _fix_gaussian_alignment(scene_gs, in_place=in_place)
    scene_gs = normalized_gaussian(scene_gs, in_place=fix_alignment)
    return scene_gs


def _fix_gaussian_alignment(scene_gs, in_place=False):
    if not in_place:
        scene_gs = deepcopy(scene_gs)

    device = scene_gs._xyz.device
    dtype = scene_gs._xyz.dtype
    scene_gs._xyz = (
        scene_gs._xyz
        @ torch.tensor(
            [
                [-1, 0, 0],
                [0, 0, 1],
                [0, 1, 0],
            ],
            device=device,
            dtype=dtype,
        ).T
    )
    return scene_gs


def normalized_gaussian(scene_gs, in_place=False, outlier_percentile=None):
    if not in_place:
        scene_gs = deepcopy(scene_gs)

    orig_xyz = scene_gs.get_xyz
    orig_scale = scene_gs.get_scaling

    active_mask = (scene_gs.get_opacity > 0.9).squeeze()
    inv_scale = (
        orig_xyz[active_mask].max(dim=0)[0] - orig_xyz[active_mask].min(dim=0)[0]
    ).max()
    norm_scale = orig_scale / inv_scale
    norm_xyz = orig_xyz / inv_scale

    if outlier_percentile is None:
        lower_bound_xyz = torch.min(norm_xyz[active_mask], dim=0)[0]
        upper_bound_xyz = torch.max(norm_xyz[active_mask], dim=0)[0]
    else:
        lower_bound_xyz = torch.quantile(
            norm_xyz[active_mask],
            outlier_percentile,
            dim=0,
        )
        upper_bound_xyz = torch.quantile(
            norm_xyz[active_mask],
            1.0 - outlier_percentile,
            dim=0,
        )

    center = (lower_bound_xyz + upper_bound_xyz) / 2
    norm_xyz = norm_xyz - center
    scene_gs.from_xyz(norm_xyz)
    scene_gs.mininum_kernel_size /= inv_scale.item()
    scene_gs.from_scaling(norm_scale)
    return scene_gs


def make_scene(*outputs, in_place=False):
    if not in_place:
        outputs = [deepcopy(output) for output in outputs]

    all_outs = []
    minimum_kernel_size = float("inf")
    for output in outputs:
        # move gaussians to scene frame of reference
        PC = SceneVisualizer.object_pointcloud(
            points_local=output["gaussian"][0].get_xyz.unsqueeze(0),
            quat_l2c=output["rotation"],
            trans_l2c=output["translation"],
            scale_l2c=output["scale"],
        )
        output["gaussian"][0].from_xyz(PC.points_list()[0])
        # must ... ROTATE
        output["gaussian"][0].from_rotation(
            quaternion_multiply(
                quaternion_invert(output["rotation"]),
                output["gaussian"][0].get_rotation,
            )
        )
        scale = output["gaussian"][0].get_scaling
        adjusted_scale = scale * output["scale"]
        assert (
            output["scale"][0, 0].item()
            == output["scale"][0, 1].item()
            == output["scale"][0, 2].item()
        )
        output["gaussian"][0].mininum_kernel_size *= output["scale"][0, 0].item()
        adjusted_scale = torch.maximum(
            adjusted_scale,
            torch.tensor(
                output["gaussian"][0].mininum_kernel_size * 1.1,
                device=adjusted_scale.device,
            ),
        )
        output["gaussian"][0].from_scaling(adjusted_scale)
        minimum_kernel_size = min(
            minimum_kernel_size,
            output["gaussian"][0].mininum_kernel_size,
        )
        all_outs.append(output)

    # merge gaussians
    scene_gs = all_outs[0]["gaussian"][0]
    scene_gs.mininum_kernel_size = minimum_kernel_size
    for out in all_outs[1:]:
        out_gs = out["gaussian"][0]
        scene_gs._xyz = torch.cat([scene_gs._xyz, out_gs._xyz], dim=0)
        scene_gs._features_dc = torch.cat(
            [scene_gs._features_dc, out_gs._features_dc], dim=0
        )
        scene_gs._scaling = torch.cat([scene_gs._scaling, out_gs._scaling], dim=0)
        scene_gs._rotation = torch.cat([scene_gs._rotation, out_gs._rotation], dim=0)
        scene_gs._opacity = torch.cat([scene_gs._opacity, out_gs._opacity], dim=0)

    return scene_gs


def load_image(path):
    image = Image.open(path)
    image = np.array(image)
    image = image.astype(np.uint8)
    return image


def load_mask(path):
    mask = load_image(path)
    mask = mask > 0
    if mask.ndim == 3:
        mask = mask[..., -1]
    return mask


def load_single_mask(folder_path, index=0, extension=".png"):
    masks = load_masks(folder_path, [index], extension)
    return masks[0]


def load_masks(folder_path, indices_list=None, extension=".png"):
    masks = []
    indices_list = [] if indices_list is None else list(indices_list)
    if not len(indices_list) > 0:  # get all all masks if not provided
        idx = 0
        while os.path.exists(os.path.join(folder_path, f"{idx}{extension}")):
            indices_list.append(idx)
            idx += 1

    for idx in indices_list:
        mask_path = os.path.join(folder_path, f"{idx}{extension}")
        assert os.path.exists(mask_path), f"Mask path {mask_path} does not exist"
        mask = load_mask(mask_path)
        masks.append(mask)
    return masks