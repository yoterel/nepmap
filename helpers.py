import torch
import numpy as np
import torch.nn.functional as F
import logging
from pathlib import Path
from scipy.spatial.transform import Rotation as R
from collections import defaultdict
import random
from typing import Optional
import collections
import tqdm
import gsoup
import scipy.ndimage
import subprocess

from nerfacc import (
    OccupancyGrid,
    ray_marching,
    render_weight_from_density,
    accumulate_along_rays,
    render_transmittance_from_density,
    render_visibility,
    ray_aabb_intersect
)

Rays = collections.namedtuple("Rays", ("origins", "viewdirs"))

def namedtuple_map(fn, tup):
    """Apply `fn` to each element of `tup` and cast to `tup`'s namedtuple."""
    return type(tup)(*(None if x is None else fn(x) for x in tup))

def create_projector(K, v, t, W, H, textures, amp=4.0, gamma=2.2, device="cuda:0"):
    projector = {}
    # t_proj = torch.tensor([0., 0., 1.], device=device)
    projector["W"] = torch.ones(1, dtype=torch.float32, device=device) * W
    projector["H"] = torch.ones(1, dtype=torch.float32, device=device) * H
    projector["t"] = t.clone().detach().requires_grad_(True)
    projector["v"] = v.clone().detach().requires_grad_(True)
    projector["f"] = (K[0, 0] / W)[None].clone().detach().requires_grad_(True)
    projector["cx"] = (K[0, 2] / W)[None].clone().detach().requires_grad_(True)
    projector["cy"] = (K[1, 2] / W)[None].clone().detach().requires_grad_(True)
    projector["amp"] = torch.tensor([amp], dtype=torch.float32, device=device, requires_grad=True)
    projector["gamma"] = torch.tensor([gamma], dtype=torch.float32, device=device, requires_grad=True)
    # projector["textures"] = torch.ones((3, proj_h, proj_w), dtype=torch.float32).to(device)
    projector["textures"] = textures
    return projector

def get_projector_stats(projector):
    """
    converts projector dictionary to Rt and K matrices
    """
    device = projector["v"].device
    if projector["v"].shape[-1] == 4:
        R_proj = gsoup.qvec2mat(projector["v"])  # c2w
    else:
        R_proj = gsoup.rotvec2mat(projector["v"])  # c2w
    T_proj = projector["t"]  # c2w
    Rt = torch.cat((R_proj, T_proj[:, None]), axis=1)  # c2w
    Rt = Rt.detach().cpu().numpy()
    K_proj = torch.eye(3).to(device)
    K_proj[0, 0] = projector["f"]
    K_proj[0, 2] = projector["cx"]
    K_proj[1, 1] = projector["f"]
    K_proj[1, 2] = projector["cy"]
    K_proj = K_proj.detach().cpu().numpy()
    return Rt, K_proj

def create_camera(K, t, v, W, H, device="cuda:0"):
    camera = {}
    camera["K"] = K.clone().detach().requires_grad_(True)
    camera["W"] = W
    camera["H"] = H
    camera["t"] = t.clone().detach().requires_grad_(True)
    camera["v"] = v.clone().detach().requires_grad_(True)
    return camera

def create_cam_rays(K, v, t, W, H, device):
    R_vcam = gsoup.qvec2mat(v)  # c2w
    T_vcam = t  # c2w
    vc2w = gsoup.to_torch(gsoup.compose_rt(gsoup.to_np(R_vcam[None, :]), gsoup.to_np(T_vcam[None, :])), device=device)
    x, y = torch.meshgrid(
        torch.arange(W, device=device),
        torch.arange(H, device=device),
        indexing="xy",
    )
    x = x.flatten()
    y = y.flatten()
    vcamera_dirs = (torch.inverse(K) @ gsoup.to_hom(torch.stack((x, y), dim=-1)).T).T
    directions = (vc2w[:, :3, :3] @ vcamera_dirs[:, :, None]).squeeze()
    origins = torch.broadcast_to(vc2w[:, :3, -1], directions.shape)
    viewdirs = directions / torch.linalg.norm(
        directions, dim=-1, keepdims=True
    )
    origins = torch.reshape(origins, (H, W, 3))
    viewdirs = torch.reshape(viewdirs, (H, W, 3))
    rays = Rays(origins=origins, viewdirs=viewdirs)
    return rays

def create_light_field(projectors=None, coloc_light=None, inverse_square=False):
    light_field = {}
    if projectors is not None:
        light_field["projectors"] = projectors
    if coloc_light is not None:
        light_field["coloc_light"] = coloc_light
    if inverse_square:
        light_field["inverse_square"] = True
    else:
        light_field["inverse_square"] = False
    return light_field

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def save_results_to_images(result, prefix, dst_folder, image_dims, w2c=None):
    for key in result:
        image = result[key].view(*image_dims, result[key].shape[-1])
        name = prefix + "_" + key
        if w2c is not None and image.shape[-1] == 3:
            image = gsoup.draw_gizmo_on_image(image[None, ...].cpu().detach().numpy(), w2c[None, ...].cpu().detach().numpy())
        else:
            image = image[None, ...]
        gsoup.save_images(image, dst_folder, [name])

def save_results_to_gif(result, n_frames, prefix, dst_folder, image_dims, w2c=None):
    for key in result:
        image = result[key].view(n_frames, *image_dims, result[key].shape[-1])
        name = prefix + "_" + key + ".gif"
        if w2c is not None and image.shape[-1] == 3:
            image = gsoup.draw_gizmo_on_image(gsoup.to_numpy(image), gsoup.to_numpy(w2c))
        gsoup.save_animation(image, Path(dst_folder, name))

def march_and_extract(
    # scene
    radiance_field: torch.nn.Module,
    rays,  # of type Rays
    scene_aabb: torch.Tensor,
    occupancy_grid: OccupancyGrid = None,
    # rendering options
    near_plane: Optional[float] = None,
    far_plane: Optional[float] = None,
    render_step_size: float = 1e-3,
    render_bkgd: Optional[torch.Tensor] = None,
    cone_angle: float = 0.0,
    alpha_thre: float = 0.0,
    # test options
    test_chunk_size: int = 8192,
    ret_vals = None,
    is_relightable: bool = False,
    light_field = None,
    texture_ids= None,
    # filter_unit_sphere: bool = False,
    cur_step=None,
    only_transmittance: bool = False,
    vanilla_radiance_field = None,
    trans_thre: float = 1e-4,
):
    """Render the pixels of an image."""
    rays_shape = rays.origins.shape
    if len(rays_shape) == 3:
        height, width, _ = rays_shape
        num_rays = height * width
        rays = namedtuple_map(
            lambda r: r.reshape([num_rays] + list(r.shape[2:])), rays
        )
    else:
        num_rays, _ = rays_shape

    def sigma_fn(t_starts, t_ends, ray_indices):
        ray_indices = ray_indices.long()
        t_origins = chunk_rays.origins[ray_indices]
        t_dirs = chunk_rays.viewdirs[ray_indices]
        positions = t_origins + t_dirs * (t_starts + t_ends) / 2.0
        return radiance_field.query_density(positions)

    def rgb_sigma_fn(t_starts, t_ends, ray_indices, calc_norms=False):
        ray_indices = ray_indices.long()
        t_origins = chunk_rays.origins[ray_indices]
        t_dirs = chunk_rays.viewdirs[ray_indices]
        positions = t_origins + t_dirs * (t_starts + t_ends) / 2.0
        if is_relightable:
            if texture_ids is not None:
                tex=texture_ids[ray_indices]
            else:
                tex=None
            return radiance_field(positions, t_dirs, texture_ids=tex, light_field=light_field, cur_step=cur_step, calc_norms=calc_norms)
        return radiance_field(positions, t_dirs)

    def sigma_vanilla_fn(t_starts, t_ends, ray_indices):
        ray_indices = ray_indices.long()
        t_origins = chunk_rays.origins[ray_indices]
        t_dirs = chunk_rays.viewdirs[ray_indices]
        positions = t_origins + t_dirs * (t_starts + t_ends) / 2.0
        return vanilla_radiance_field.query_density(positions)
    
    chunk = (
        torch.iinfo(torch.int32).max
        if radiance_field.training
        else test_chunk_size
    )
    all_ret = {}
    for i in range(0, num_rays, chunk):
        chunk_rays = namedtuple_map(lambda r: r[i : i + chunk], rays)
        init_ray_indices, init_t_starts, init_t_ends = ray_marching(
            chunk_rays.origins,
            chunk_rays.viewdirs,
            scene_aabb=scene_aabb,
            grid=occupancy_grid,
            # sigma_fn=sigma_fn,
            near_plane=near_plane,
            far_plane=far_plane,
            render_step_size=render_step_size,
            stratified=radiance_field.training,
            cone_angle=cone_angle,
            alpha_thre=alpha_thre)
        initial_sigmas = sigma_fn(init_t_starts, init_t_ends, init_ray_indices.long())
        initial_alphas = 1.0 - torch.exp(-initial_sigmas * (init_t_ends - init_t_starts))
        n_rays=chunk_rays.origins.shape[0]
        if is_relightable:
            if "cam_transm" in ret_vals or "pred_cam_transm" in ret_vals or "n_pred_cam_transm" in ret_vals:
                long_init_ray_indices = init_ray_indices.long()
                t_origins = chunk_rays.origins[long_init_ray_indices]
                t_dirs = chunk_rays.viewdirs[long_init_ray_indices]
                positions = t_origins + t_dirs * (init_t_starts + init_t_ends) / 2.0
                pred_cam_transm = radiance_field.vis_network(positions.detach(), t_dirs.detach())
                cam_transm = render_transmittance_from_density(init_t_starts,
                                                            init_t_ends,
                                                            initial_sigmas,
                                                            ray_indices=init_ray_indices,
                                                            n_rays=n_rays)
                if "n_pred_cam_transm" in ret_vals:
                    noised_position = positions + torch.randn_like(positions) * 0.01
                    n_pred_cam_transm = radiance_field.vis_network(noised_position.detach(), t_dirs.detach())
                if only_transmittance:
                    if "cam_transm" not in all_ret:
                        all_ret["cam_transm"] = []
                        all_ret["pred_cam_transm"] = []
                        all_ret["n_pred_cam_transm"] = []
                    all_ret["cam_transm"].append(cam_transm)
                    all_ret["pred_cam_transm"].append(pred_cam_transm)
                    all_ret["n_pred_cam_transm"].append(n_pred_cam_transm)
                    continue
        # Compute visibility of the samples, and filter out invisible samples
        masks = render_visibility(
            initial_alphas,
            ray_indices=init_ray_indices,
            early_stop_eps=trans_thre,
            alpha_thre=alpha_thre,
            n_rays=n_rays,
        )
        ray_indices, t_starts, t_ends = (
            init_ray_indices[masks],
            init_t_starts[masks],
            init_t_ends[masks],
        )
        if vanilla_radiance_field is not None and "vanilla_diff" in ret_vals:
            with torch.no_grad():
                vanilla_sigma = sigma_vanilla_fn(t_starts, t_ends, ray_indices.long())
                vanilla_weights = render_weight_from_density(
                                                            t_starts,
                                                            t_ends,
                                                            vanilla_sigma,
                                                            ray_indices=ray_indices,
                                                            n_rays=n_rays)
                vanilla_weights = vanilla_weights.detach()
        if "diff_normals" in ret_vals or "normal_map" in ret_vals:
            calc_norms = True
        else:
            calc_norms = False
        rgbs1, rgbs2, sigmas, pred_proj_transm,\
        normals, pred_normals, predicted_albedo,\
        predicted_roughness, sampled_texture,\
        visible_texture=rgb_sigma_fn(t_starts, t_ends, ray_indices.long(), calc_norms=calc_norms)
        
        weights = render_weight_from_density(
            t_starts,
            t_ends,
            sigmas,
            ray_indices=ray_indices,
            n_rays=n_rays,
        )
        if "coloc_light" in light_field:
            if light_field["coloc_light"] is not None:
                if light_field["inverse_square"]:
                    weights_inverse_square = ((2.0 / (t_starts + t_ends + 1e-6)) ** 2)
                    rgbs2 = accumulate_along_rays(weights*weights_inverse_square, ray_indices, values=rgbs2, n_rays=n_rays)
                else:
                    rgbs2 = accumulate_along_rays(weights, ray_indices, values=rgbs2, n_rays=n_rays)
                rgbs1 = accumulate_along_rays(weights, ray_indices, values=rgbs1, n_rays=n_rays)
                rgb = rgbs1 + rgbs2
            else:
                rgb = accumulate_along_rays(weights, ray_indices, values=rgbs1, n_rays=n_rays)
        else:
            rgb = accumulate_along_rays(weights, ray_indices, values=rgbs1, n_rays=n_rays)
        opacity = accumulate_along_rays(weights, ray_indices, values=None, n_rays=n_rays)
        depth = accumulate_along_rays(
            weights,
            ray_indices,
            values=(t_starts + t_ends) / 2.0,
            n_rays=n_rays,
        )
        # Background composition.
        if render_bkgd is not None:
            rgb = rgb + render_bkgd * (1.0 - opacity)
        my_dict = {}
        if "rgb" in ret_vals:
            my_dict["rgb"] = rgb
        if "vanilla_diff" in ret_vals:
            my_dict["vanilla_diff"] = torch.abs(vanilla_weights - weights)
        if "opacity" in ret_vals:
            my_dict["opacity"] = opacity
        if "depth" in ret_vals:
            my_dict["depth"] = depth
        if "cam_transm" in ret_vals:
            # alphas = 1.0 - torch.exp(-sigmas * (t_ends - t_starts))
            # visibility = weights[:, None] / (alphas + 1e-6)
            my_dict["cam_transm"] = cam_transm
        if "pred_cam_transm" in ret_vals:
            my_dict["pred_cam_transm"] = pred_cam_transm
        if "n_pred_cam_transm" in ret_vals:
            my_dict["n_pred_cam_transm"] = n_pred_cam_transm
        if "pred_proj_transm_map" in ret_vals:
            ptm = accumulate_along_rays(weights, ray_indices, values=pred_proj_transm, n_rays=n_rays) 
            if render_bkgd is not None:
                ptm = ptm + render_bkgd * (1.0 - opacity)
            my_dict["pred_proj_transm_map"] = ptm
        if "pred_normals" in ret_vals:
            my_dict["pred_normals"] = accumulate_along_rays(weights, ray_indices, values=pred_normals, n_rays=n_rays)
        if "diff_normals" in ret_vals:
            my_dict["diff_normals"] = accumulate_along_rays(weights, ray_indices, values=(pred_normals - normals.detach()).norm(dim=-1, keepdim=True), n_rays=n_rays)
        if "pred_normal_map" in ret_vals:
            est_normal_map = (pred_normals + 1) / 2
            est_normal_map = accumulate_along_rays(weights, ray_indices, values=est_normal_map, n_rays=n_rays)
            if render_bkgd is not None:
                est_normal_map = est_normal_map + render_bkgd * (1.0 - opacity)
            my_dict["pred_normal_map"] = est_normal_map
        if "normal_map" in ret_vals:
            normal_map = (normals + 1) / 2
            normal_map = accumulate_along_rays(weights, ray_indices, values=normal_map, n_rays=n_rays)
            my_dict["normal_map"] = normal_map
        if "albedo" in ret_vals:
            alb =  accumulate_along_rays(weights, ray_indices, values=predicted_albedo, n_rays=n_rays)
            if render_bkgd is not None:
                alb = alb + render_bkgd * (1.0 - opacity)
            my_dict["albedo"] = alb
        if "roughness" in ret_vals:
            rough = accumulate_along_rays(weights, ray_indices, values=predicted_roughness, n_rays=n_rays)
            if render_bkgd is not None:
                rough = rough + render_bkgd * (1.0 - opacity)
            my_dict["roughness"] = rough
        if "sampled_texture_map" in ret_vals:
            stm = accumulate_along_rays(weights, ray_indices, values=sampled_texture, n_rays=n_rays)
            if render_bkgd is not None:
                stm = stm + render_bkgd * (1.0 - opacity)
            my_dict["sampled_texture_map"] = stm
        if "visible_texture_map" in ret_vals:
            vtm = accumulate_along_rays(weights, ray_indices, values=visible_texture, n_rays=n_rays)
            if render_bkgd is not None:
                vtm = vtm + render_bkgd * (1.0 - opacity)
            my_dict["visible_texture_map"] = vtm
        if "n_rendering_samples" in ret_vals:
            my_dict["n_rendering_samples"] = torch.tensor([len(t_starts)])
        for k in my_dict:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(my_dict[k])
    all_ret = {k: torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret

def vcams_from_cams(cams, n=30, plane="xz", center_pose=None, radius_factor=1.0):
    """
    creates a set of virtual cameras that are evenly distributed on a circle around the scene
    :param cams the original camera poses (used for determining radius of circle)
    :param n the number of virtual cameras
    :param plane the plane on which the virtual cameras are distributed in a circle
    :param center_pose the pose of the center of the circle
    :param radius_factor a factor that is multiplied with the radius of the circle
    """
    cams_np = cams.cpu().numpy()
    radius = np.mean(np.linalg.norm(cams_np[:, :3, 3], axis=-1))  # mean distance to origin used as radius
    radius *= radius_factor
    mean_loc = np.mean(cams_np[:, :3, 3], axis=0) #+ np.array([-0.3, 0.0, 0.0])
    t = np.linspace(0, 2*np.pi, n, endpoint=False)
    if center_pose is not None:
        origin = center_pose[:3, -1]
    else:
        origin = mean_loc
    if plane == "xz" or plane == "zx":
        x = radius * np.cos(t) + origin[0]
        y = np.broadcast_to(origin[1], t.shape)
        z = radius * np.sin(t) + origin[2]
    elif plane == "xy" or plane == "yx":
        x = radius * np.cos(t) + origin[0]
        y = radius * np.sin(t) + origin[1]
        z = np.broadcast_to(origin[2], t.shape)
    elif plane == "yz" or plane == "zy":
        raise NotImplementedError
    locs = np.array([x, y, z]).T
    v_poses = gsoup.look_at_np(locs, np.array([[0, 0, 0.0]]), np.array([[0, 0, 1.0]]))
    vcamera_poses = torch.tensor(v_poses, dtype=cams.dtype, device=cams.device)
    return vcamera_poses

def CDC(prompt, src_image, src_mask, tmp_input, tmp_output, output_path,
        T_in=None, T_out=None,
        CDC_env_path=None,
        CDC_src_path=None):
    """
    run cross-domain composition
    :prompt: the text prompt
    :src_image: the source image (m x n x 3), best resolution is 512x512 to avoid upsampling.
        note: current implementation will not upsample but instead crop from center if resolution is larger, and pad with zeros if it is smaller.
    :src_mask: the source mask, square image with res (m x n x 1), will be cropped to 512x512
    :tmp_input: the input directory for the CDC model
    :tmp_output: the output directory for the CDC model
    :output_path: the output path for the final image(s)
    :T_in: a list of values between 0 and 1 for CDC relating to how much to take into account image content inside mask
        note: this will control the amount of images produced in output_path.
    :T_out: a list of values between 0 and 1 for CDC relating to how much to take into account image content outside mask
        note: this will control the amount of images produced in output_path.
    CDC_env_path: path to the CDC conda environment i.e. /path/to/conda_env/root/folder
        note: if CDC dependencies are installed in the current environment, this can point at it
    CDC_src_path: path to the CDC source code i.e. /path/to/cdc/git/repo
        note: assumes source code is valid and sd-v1-5-inpainting model are installed properly according to CDC instructions.
    :return: a torch tensor of shape (b x m x n x 3) where b is len(T_in) * len(T_out) of generated outputs 
    """
    images_path = Path(tmp_input, "images")
    images_path.mkdir(parents=True, exist_ok=True)
    image_path = Path(images_path, "0000.png")
    masks_path = Path(tmp_input, "masks")
    masks_path.mkdir(parents=True, exist_ok=True)
    mask_path = Path(masks_path, "0000.png")
    if CDC_env_path is None or CDC_src_path is None:
        raise ValueError("CDC_env_path and CDC_src_path must be specified")
    h, w = src_image.shape[:2]
    if not image_path.exists():
        src_image = gsoup.crop_center(src_image[None, ...], min(512, h), min(512, w))
        src_image = gsoup.pad_to_res(src_image, 512, 512)
        gsoup.save_image(src_image[0], image_path)
    if not mask_path.exists():
        src_mask = gsoup.crop_center(src_mask[None, ...], min(512, h), min(512, w))
        src_mask = gsoup.pad_to_res(src_mask, 512, 512)
        gsoup.save_image(src_mask[0], mask_path)
    output_files = [x for x in Path(tmp_output).glob("**/*.png") if "grid" not in x.name]
    if len(output_files) == 0:
        for j, p in enumerate(prompt):
            print("{}: {}".format(j, p))
            prompt_path = Path(tmp_output, "{:02d}".format(j))
            prompt_path.mkdir(parents=True, exist_ok=True)
            cmd = "cd {0}; export PYTHONPATH=$PYTHONPATH:{0}:{0}/ResizeRight; conda run --prefix {1} python {0}/scripts_cdc/img2img_inpaint.py".format(CDC_src_path, CDC_env_path)
            cmd += " --config configs/stable-diffusion/v1-inpainting-inference.yaml --ckpt models/ldm/stable-diffusion-v1/sd-v1-5-inpainting.ckpt --n_samples 1 --ddim_steps 50 --strength_in 1.0 --T_out 1 --down_N_in 1 --down_N_out 1 --blend_pix 0 --seed 42 --repaint_start 0"
            cmd += " --prompt '{}'".format(p)
            cmd += " --init_img '{}' --mask '{}'".format(str(image_path.absolute()), str(mask_path.absolute()))
            cmd += " --outdir '{}'".format(str(prompt_path.absolute()))
            cmd += " --T_in {}".format(" ".join([str(i) for i in T_in]))  # 0: ignore input structure, 1: follow structure of input completely
            if T_out is not None:
                cmd += " --T_out {}".format(" ".join([str(i) for i in T_out]))
            stdout, stderr = subprocess.Popen(f"{cmd}", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()
            print("Standard Output:", stdout.decode('utf-8'))
            print("Standard Error:", stderr.decode('utf-8'))
            output_files = [x for x in Path(tmp_output).glob("**/*.jpg") if "grid" not in x.name]
    if h == w == 400:  # hard coded for 400x400 (syntehtic scenes): crop the center as it was just padding
        result = gsoup.load_images(output_files, float=True, to_torch=True, device=src_image.device)
        result = gsoup.crop_center(result, 400, 400)
    else:  # hard coded for 480x640 (bunny, teapot): pad to 512x640, then crop the center to reach 480x640
        result = gsoup.load_images(output_files, float=True, to_torch=True, device=src_image.device)
        result = gsoup.pad_image_to_res(result, 512, w)
        half_pad = (512 - h) // 2
        result = result[:, half_pad:h+half_pad, :, :]
    gsoup.save_images(result, output_path)
    return result


def dual_photography(desired_appearance, retvals, projector, camera,
                     radiance_field, occupancy_grid, light_field, scene_aabb,
                     render_step_size, test_chunk_size, dst=None, bg_color=None):
    """
    returns the projector image that best obtain desired appearance from camera viewpoint
    using a 2-pass rendering approach
    :param desired_apperance float tensor channels first, [C, H, W]
    :param retvals: return values from the render function
    :param projector: projector dictionary
    :param camera: camera dictionary
    :param radiance_field: radiance field
    :param occupancy_grid: occupancy grid
    :param light_field: light field
    :param scene_aabb: scene aabb
    :param render_step_size: render step size
    :param test_chunk_size: test chunk size
    :param dst: destination path to save result to
    :param bg_color: background color to render with
    """
    device = desired_appearance.device
    with torch.no_grad():
        ##### get vprojector
        vprojector = create_projector(camera["K"], camera["v"], camera["t"],
                                    camera["W"], camera["H"],
                                    desired_appearance,
                                    amp=7.0,
                                    gamma=1.0, device=device)  # light_field["projectors"][0]["textures"]
        new_light_field = create_light_field(projectors=[vprojector])
        ##### get vcam rays
        K_proj = torch.eye(3, device=device)
        K_proj[0, 0] = projector["f"] * projector["W"]
        K_proj[0, 2] = projector["cx"] * projector["W"]
        K_proj[1, 1] = projector["f"] * projector["W"]
        K_proj[1, 2] = projector["cy"] * projector["W"]
        # train_dataset.K_proj
        dual_rays = create_cam_rays(K_proj, projector["v"], projector["t"],
                                    int(projector["W"].item()), int(projector["H"].item()),
                                    device)
        dual_result = march_and_extract(
                radiance_field,
                dual_rays,
                scene_aabb,
                occupancy_grid=occupancy_grid,
                render_step_size=render_step_size,
                render_bkgd=bg_color,
                test_chunk_size=test_chunk_size,
                ret_vals=retvals,
                is_relightable=True,
                light_field=new_light_field,
                texture_ids=None
            )
        if dst is not None:
            save_results_to_images(dual_result, "", dst, dual_rays.viewdirs.shape[:2])
        return dual_result


def optimize_texture(target_images, rays, radiance_field, occupancy_grid, light_field,
                     scene_aabb, render_step_size, dst, intermediate_results=False, mode="sigmoid"):
    """
    optimize projector texture given a batch of target images
    todo: create randomized batches of rays instead of differentiating through whole image which is slow and wasteful 
    :param target_image: target image h x w x 3
    :param rays: rays (with origins and directions same shape as target_image)
    :param radiance_field: radiance field
    :param occupancy_grid: occupancy grid
    :param light_field: light field
    :param scene_aabb: scene aabb
    :param render_step_size: render step size
    :param dst: destination path to save intermediate results to
    :param intermediate_results: whether to save intermediate results
    :param mode: sigmoid or clip (sigmoid yields slightly better results, clip is faster)
    :return: optimized projector texture (proj_h x proj_w x 3)
    """
    projector = light_field["projectors"][0]
    mypath = Path(dst)
    mypath.mkdir(parents=True, exist_ok=True)
    step = 0
    if mode == "sigmoid":
        init_value = 0.0
        steps = 201
    else:
        init_value = 0.01
        steps = 101
    texture_map = torch.full((3, int(projector["H"]), int(projector["W"])), init_value,
                            dtype=torch.float32, device=projector["H"].device, requires_grad=True)
    optmizer = torch.optim.Adam([texture_map], lr=0.05)
    for i in range(step, steps):
        print("step {:03d}/{:03d}".format(i, steps))
        if mode == "sigmoid":
            light_field["projectors"][0]["textures"] = torch.sigmoid(texture_map)
        else:
            light_field["projectors"][0]["textures"] = torch.clamp(texture_map, min=0.0, max=1.0)
        # todo: create randomized batches of rays instead of differentiating through whole image which is slow and wasteful
        primal_result = march_and_extract(
                radiance_field,
                rays,
                scene_aabb,
                occupancy_grid=occupancy_grid,
                render_step_size=render_step_size / 2,
                render_bkgd=torch.zeros(3, device=target_images.device),
                test_chunk_size=512,
                ret_vals=["rgb"],
                is_relightable=True,
                light_field=light_field,
                texture_ids=None,
                # trans_thre=5e-1  # increase threshold for not allowing penetration
            )
        image = primal_result["rgb"].view(*rays.viewdirs.shape[:2], -1)
        loss = F.smooth_l1_loss(image, target_images)
        optmizer.zero_grad()
        loss.backward()
        optmizer.step()
        if i % 10 == 0:
            logging.info("step: {}, loss: {}".format(i, loss.item()))
            if intermediate_results:
                if mode == "sigmoid":
                    gsoup.save_image(torch.sigmoid(texture_map).permute(1, 2, 0), Path(mypath, "opt_texture_{:03d}.png".format(i)))
                else:
                    gsoup.save_image(torch.clamp(texture_map, min=0.0, max=1.0).permute(1, 2, 0), Path(mypath, "opt_texture_{:03d}.png".format(i)))
                gsoup.save_image(image, Path(mypath, "opt_image_{:03d}.png".format(i)))
    if mode == "sigmoid":
        final_texture = torch.sigmoid(texture_map).permute(1, 2, 0).detach()
        final_texture[(final_texture == 0.5).all(dim=-1)] = 0.0 # set unoptimized pixels to black
    else:
        final_texture = torch.clamp(texture_map, min=0.0, max=1.0).permute(1, 2, 0).detach()
    return final_texture

def render_sandbox(radiance_field, occupancy_grid, scene_aabb,
                render_step_size, test_retvals, light_field,
                test_dataset, train_dataset, args, dst=None,
                prefix=None, mode="test_set", extra_info=None):
    """
    monster function for rendering a scene in various modes, see modes for details
    :param radiance_field: the radiance field
    :param occupancy_grid: the occupancy grid
    :param scene_aabb: the scene bounding box
    :param render_step_size: the render step size
    :param test_retvals: the return values
    :param light_field: the light field
    :param test_dataset: the test dataset
    :param train_dataset: the train dataset
    :param args: command line arguments
    :param dst: destination folder
    :param prefix: prefix for generated files
    :param mode: string describing the mode
        multi_t2t: text -> projection of multiple views (optimized jointly).
        t2t: text -> projection of a single view, or multiple views optimized **sequentially**.
        compensate: perform projector compensation
        projector_calib: evaluate projector parameters effect on image
        dual_photo: render a dual photo, currently set up to produce XRAY result from paper
        train_set, test_set: renders decompositions of frames from train/test set, with the exact lighting as in the dataset
        test_set_movie: renders all test set as RGB, sequentially as a .gif (kind of useless)
        train_set_movie: renders all train set as RGB, sequentially as a .gif
        move_camera: renders the scene using a circular camera path from novel views.
        move_projector: renders the scene using a circular projector path from any view.
        play_vid: streams a raw video onto the scene. projector is fixed while camera moves in circular motion.
    :param extra_info: extra information for render
    :return: the psnrs if applicaple
    todo: refactor this function
    """
    logging.info("sandbox mode: {}".format(mode))
    radiance_field.eval()
    train_dataset.training = False
    if light_field is not None:
        if "proj_texture" in extra_info and "projectors" in light_field:
            cur_textures = light_field["projectors"][0]["textures"]
            if type(extra_info["proj_texture"]) == str:
                texture_index = np.where(test_dataset.texture_names == extra_info["proj_texture"])[0]
                light_field["projectors"][0]["textures"] = test_dataset.textures[texture_index][0]
                texture_ids = torch.full((test_dataset.HEIGHT*test_dataset.WIDTH, 2), texture_index[0], device=args.device)
            else:
                light_field["projectors"][0]["textures"] = extra_info["proj_texture"] # 3 x 1080 x 1920
                texture_ids = None
        else:
            texture_ids = None
        if "proj_amp" in extra_info and "projectors" in light_field:
            cur_amp = light_field["projectors"][0]["amp"]
            light_field["projectors"][0]["amp"] = torch.full((1, ), extra_info["proj_amp"], dtype=cur_amp.dtype, device=args.device)
        if "coloc_light" in extra_info:
            if extra_info["coloc_light"] == False:
                if "coloc_light" in light_field:
                    cur_coloc = light_field["coloc_light"]
                    light_field["coloc_light"] = None
        psnrs = []
        if mode == "multi_t2t":  # text -> projection of multiple views (optimized jointly).
            mypath = Path(args.experiment_folder , "multi_t2t")
            mypath.mkdir(parents=True, exist_ok=True)
            for i, viewpoint in enumerate(extra_info["cam_index"]):
                # render from each viewpoint and generate image
                with torch.no_grad():
                    primary_rays = train_dataset[viewpoint]["rays"]
                    rgb_path = Path(mypath, "rgb_{}.png".format(i))
                    mask_path = Path(mypath, "masks_{}".format(i))
                    mask_path.mkdir(parents=True, exist_ok=True)
                    mask1_path = Path(mask_path, "mask1.png")
                    mask2_path = Path(mask_path, "mask2.png")
                    mask3_path = Path(mask_path, "mask3.png")
                    normals_path = Path(mypath, "normals_{}.png".format(i))
                    if rgb_path.exists() and mask1_path.exists() and mask2_path.exists() and mask3_path.exists():
                        image = gsoup.load_image(rgb_path, float=True, to_torch=True, device=args.device)
                        mask1 = gsoup.load_image(mask1_path, float=True, to_torch=True, device=args.device).to(torch.bool).view(-1, 1)
                        mask2 = gsoup.load_image(mask2_path, float=True, to_torch=True, device=args.device).to(torch.bool).view(-1, 1)
                        mask3 = gsoup.load_image(mask3_path, float=True, to_torch=True, device=args.device).to(torch.bool).view(-1, 1)
                    else:
                        result = march_and_extract(
                                radiance_field,
                                primary_rays,
                                scene_aabb,
                                occupancy_grid=occupancy_grid,
                                render_step_size=render_step_size,
                                render_bkgd=None,
                                test_chunk_size=args.test_chunk_size,
                                ret_vals=test_retvals,
                                is_relightable=args.relightable,
                                light_field=light_field,
                                texture_ids=texture_ids
                            )
                        image = result["rgb"]
                        image = image.view(*primary_rays.viewdirs.shape[:2], -1)
                        pred_normals = result["pred_normals"]
                        # define static inpainting masks
                        mask1 = result["opacity"] > 0
                        mask2 = result["pred_proj_transm_map"] > 0.2
                        mask3 = (pred_normals[:, None, :] @ primary_rays.viewdirs.view(-1, 3)[:, :, None])[:, :, 0] < -0.3
                        mask3 = torch.ones_like(mask2)  # not used eventually
                        gsoup.save_image(image, rgb_path) 
                        gsoup.save_image(mask1.view(*primary_rays.viewdirs.shape[:2], -1), mask1_path)
                        gsoup.save_image(mask2.view(*primary_rays.viewdirs.shape[:2], -1), mask2_path)
                        gsoup.save_image(mask3.view(*primary_rays.viewdirs.shape[:2], -1), mask3_path)
                        gsoup.save_image((pred_normals.view(*primary_rays.viewdirs.shape[:2], -1)+1) / 2, normals_path)
                    mask_path = Path(mypath, "mask_{}.png".format(i))
                    if mask_path.exists():
                        mask = gsoup.load_image(mask_path, float=True, to_torch=True, device=args.device).to(torch.bool)[..., None]
                    else:
                        mask = (mask1 & mask2 & mask3).view(*primary_rays.viewdirs.shape[:2], -1)
                        gsoup.save_image(mask, mask_path)
                    diffuse_path = Path(mypath, "diffuse_{}".format(i))
                    diffuse_path.mkdir(parents=True, exist_ok=True)
                    diffuse_result = Path(diffuse_path, "diffuse_final.png")
                    if diffuse_result.exists():
                        result = gsoup.load_image(diffuse_result, float=True, to_torch=True, device=args.device)
                    else:
                        CDC([extra_info["prompt"][i]], image,
                            mask.to(torch.float32),
                            Path(diffuse_path, "tmp_input"),
                            Path(diffuse_path, "tmp_output"),
                            Path(diffuse_path, "output"),
                            extra_info["t_in"],
                            extra_info["t_out"],
                            extra_info["cdc_conda"],
                            extra_info["cdc_src"])
                        input("place diffuse_final.png in diffuse_{} and press enter".format(i))
            # load all generated images and reduce brightness
            images = []
            for i in range(len(extra_info["cam_index"])):
                image_path = Path(mypath, "diffuse_{}".format(i), "diffuse_final.png")
                image = gsoup.load_image(image_path, float=True, to_torch=True, device=args.device)
                if image.shape != (400, 400, 3):
                    image = gsoup.crop_center(image[None, ...], 400, 400)[0]
                new_brightness = gsoup.change_brightness(image, extra_info["brightness"])
                images.append(new_brightness)
            reduced_brightness = torch.cat(images, dim=0)

            # optmize over multiple views at once
            all_ray_origins = []
            all_ray_viewpoints = []
            for i, viewpoint in enumerate(extra_info["cam_index"]):
                with torch.no_grad():
                    all_ray_origins.append(train_dataset[viewpoint]["rays"].origins)
                    all_ray_viewpoints.append(train_dataset[viewpoint]["rays"].viewdirs)
            all_rays = Rays(torch.cat(all_ray_origins, dim=0), torch.cat(all_ray_viewpoints, dim=0))
            projector_texture_path = Path(mypath, "projector_texture.png")
            intermeds = Path(mypath, "intermediate")
            intermeds.mkdir(parents=True, exist_ok=True)
            if projector_texture_path.exists():
                dual_image = gsoup.load_image(projector_texture_path, float=True, to_torch=True, device=args.device)
            else:
                dual_image = optimize_texture(reduced_brightness, all_rays, radiance_field,
                                            occupancy_grid, light_field,
                                            scene_aabb, render_step_size,
                                            intermeds, intermediate_results=True)
                gsoup.save_image(dual_image, projector_texture_path)
        elif mode == "t2t":  # text -> projection of a single view, or multiple views optimized **sequentially**.
            mypath = Path(args.experiment_folder , "t2t")
            mypath.mkdir(parents=True, exist_ok=True)
            dual_mask_aggregate = None
            texture_aggregate = None
            for i, viewpoint in enumerate(extra_info["cam_index"]):
                # render from viewpoint
                with torch.no_grad():
                    primary_rays = train_dataset[viewpoint]["rays"]
                    rgb_path = Path(mypath, "rgb_{}.png".format(i))
                    mask1_path = Path(mypath, "mask1_{}.png".format(i))
                    mask2_path = Path(mypath, "mask2_{}.png".format(i))
                    mask3_path = Path(mypath, "mask3_{}.png".format(i))
                    # mask4_path = Path(mypath, "mask4_{}.png".format(i))
                    normals_path = Path(mypath, "normals_{}.png".format(i))
                    if rgb_path.exists() and mask1_path.exists() and mask2_path.exists():
                        image = gsoup.load_image(rgb_path, float=True, to_torch=True, device=args.device)
                        mask1 = gsoup.load_image(mask1_path, float=True, to_torch=True, device=args.device).to(torch.bool).view(-1, 1)
                        mask2 = gsoup.load_image(mask2_path, float=True, to_torch=True, device=args.device).to(torch.bool).view(-1, 1)
                        # n_dot_v = gsoup.load_image(mask4_path, float=True, to_torch=True, device=args.device).to(torch.bool).view(-1, 1)
                    else:
                        primal_result = march_and_extract(
                                radiance_field,
                                primary_rays,
                                scene_aabb,
                                occupancy_grid=occupancy_grid,
                                render_step_size=render_step_size,
                                render_bkgd=None,
                                test_chunk_size=args.test_chunk_size,
                                ret_vals=test_retvals,
                                is_relightable=args.relightable,
                                light_field=light_field,
                                texture_ids=texture_ids
                            )
                        image = primal_result["rgb"]
                        image = image.view(*primary_rays.viewdirs.shape[:2], -1)
                        pred_normals = primal_result["pred_normals"]
                        # define static inpainting masks
                        mask1 = primal_result["opacity"] > 0
                        mask2 = primal_result["pred_proj_transm_map"] > 0.1
                        # n_dot_v = (pred_normals[:, None, :] @ primary_rays.viewdirs.view(-1, 3)[:, :, None])[:, :, 0]
                        gsoup.save_image(image, rgb_path) 
                        gsoup.save_image(mask1.view(*primary_rays.viewdirs.shape[:2], -1), mask1_path)
                        gsoup.save_image(mask2.view(*primary_rays.viewdirs.shape[:2], -1), mask2_path)
                        # gsoup.save_image((torch.abs(n_dot_v) >= 0.3).view(*primary_rays.viewdirs.shape[:2], -1), mask4_path)
                        gsoup.save_image((pred_normals.view(*primary_rays.viewdirs.shape[:2], -1)+1) / 2, normals_path)
                    if mask3_path.exists():
                        mask3 = gsoup.load_image(mask3_path, float=True, to_torch=True, device=args.device).to(torch.bool).view(-1, 1)
                    else:
                        if dual_mask_aggregate is not None:
                            light_field["projectors"][0]["textures"] = dual_mask_aggregate.repeat(1,1,3).permute(2, 0, 1).to(torch.float32)
                            with torch.no_grad():
                                mask_result = march_and_extract(
                                    radiance_field,
                                    primary_rays,
                                    scene_aabb,
                                    occupancy_grid=occupancy_grid,
                                    render_step_size=render_step_size,
                                    render_bkgd=None,
                                    test_chunk_size=args.test_chunk_size,
                                    ret_vals=["visible_texture_map"],
                                    is_relightable=args.relightable,
                                    light_field=light_field,
                                    texture_ids=texture_ids
                                )
                            # ideally use result, but surface response changes image, so use a threhold.
                            mask3 = (mask_result["visible_texture_map"] <= 0.1).all(dim=-1, keepdims=True)
                            mask3 = mask3.view(*primary_rays.viewdirs.shape[:2], -1)
                            dilate = scipy.ndimage.binary_dilation(gsoup.to_numpy(mask3), iterations=10)
                            dilate_mask = torch.tensor(dilate, device=args.device, dtype=torch.bool)
                            # checkboard = gsoup.generate_checkerboard(dilate_mask.shape[0], dilate_mask.shape[1], 4)
                            # checkboard = torch.tensor(checkboard, device=args.device, dtype=torch.bool)
                            # dont_trust_mask = dilate_mask & ~mask3
                            # final_mask3 = (dont_trust_mask & checkboard) | (~dont_trust_mask & dilate_mask)
                            gsoup.save_image(dilate_mask, mask3_path)
                            mask3 = dilate_mask.view(-1, 1)
                        else:
                            mask3 = torch.ones_like(mask2)
                            gsoup.save_image(mask3.view(*primary_rays.viewdirs.shape[:2], -1), mask3_path)
                    mask_path = Path(mypath, "mask_{}.png".format(i))
                    if mask_path.exists():
                        mask = gsoup.load_image(mask_path, float=True, to_torch=True, device=args.device).to(torch.bool)[..., None]
                    else:
                        mask = (mask1 & mask2 & mask3).view(*primary_rays.viewdirs.shape[:2], -1)  # 
                        gsoup.save_image(mask, mask_path)
                    distortion_path = Path(mypath, "mask_distortion_{}.png".format(i))
                    if distortion_path.exists():
                        distortion_mask = gsoup.load_image(distortion_path, float=True, to_torch=True, device=args.device)[..., None].to(torch.bool)
                    else:
                        erode = scipy.ndimage.binary_erosion(gsoup.to_numpy(mask)[:, :, 0], iterations=5)
                        distortion_mask = torch.tensor(erode[..., None], device=args.device, dtype=torch.bool)
                        gsoup.save_image(distortion_mask, distortion_path)
                    # color compensation of image
                    # new_white_level = 150/255
                    # compensation_white = 1 - torch.clamp(image, 0, 1)
                    # compensation_white_normalized = (compensation_white * (1-new_white_level)) + new_white_level
                    # gsoup.save_image(compensation_white_normalized, Path(mypath, "compensate_{}.png".format(i)))
                diffuse_path = Path(mypath, "diffuse_{}".format(i))
                diffuse_path.mkdir(parents=True, exist_ok=True)
                diffuse_result = Path(diffuse_path, "diffuse_final.png")
                if not diffuse_result.exists():
                    CDC([extra_info["prompt"][i]], image,
                                mask.to(torch.float32),
                                Path(diffuse_path, "tmp_input"),
                                Path(diffuse_path, "tmp_output"),
                                Path(diffuse_path, "output"),
                                extra_info["t_in"],
                                extra_info["t_out"])
                    input("place diffuse_final.png in diffuse_{} and press enter".format(i))
                image = gsoup.load_image(diffuse_result, float=True, to_torch=True, device=args.device)
                if image.shape != (400, 400, 3):
                    image = gsoup.crop_center(image[None, ...], 400, 400)[0]
                if image.shape[-1] == 4:
                    image = image[..., :3]
                new_brightness = gsoup.change_brightness(image, extra_info["brightness"])
                new_brightness_path = Path(mypath, "reduced_{}.png".format(i))
                gsoup.save_image(new_brightness, new_brightness_path)
                projector_texture_path = Path(mypath, "projector_texture_{}.png".format(i))
                intermeds = Path(mypath, "intermediate_{}".format(i))
                intermeds.mkdir(parents=True, exist_ok=True)
                if projector_texture_path.exists():
                    dual_image = gsoup.load_image(projector_texture_path, float=True, to_torch=True, device=args.device)
                else:
                    with torch.no_grad():
                        opt_rays = Rays(origins=primary_rays.origins.detach(), viewdirs=primary_rays.viewdirs.detach())
                    dual_image = optimize_texture(new_brightness, opt_rays, radiance_field,
                                                occupancy_grid, light_field,
                                                scene_aabb, render_step_size,
                                                intermeds, intermediate_results=True)
                    gsoup.save_image(dual_image, projector_texture_path)
                    # mark mask on projector plane
                # dual_mask = ~(dual_image <= 0.01).all(dim=-1, keepdims=True).repeat(1, 1, 3)
                # get nv mask on projector using dual photography
                projector = light_field["projectors"][0]
                t_cam = train_dataset.camtoworlds[viewpoint, :3, -1]
                rot_cam = R.from_matrix(train_dataset.camtoworlds[viewpoint, :3, :3].cpu().numpy())
                v_cam = gsoup.to_torch(rot_cam.as_rotvec().astype(np.float32), device=args.device)
                camera = create_camera(train_dataset.K, t_cam, v_cam, train_dataset.WIDTH, train_dataset.HEIGHT, device=args.device)
                dual_retvals = ["visible_texture_map"]
                dual_folder = Path(mypath, "dual_photo_{}".format(i))
                dual_mask_path = Path(dual_folder, "dual_mask.png")
                if dual_mask_path.exists():
                    hard_dual_mask = gsoup.load_image(dual_mask_path, float=True, to_torch=True, device=args.device)[..., None].to(torch.bool)
                else:
                    result = dual_photography(mask.repeat(1,1,3).to(torch.float32).permute(2, 0, 1),
                                            dual_retvals, projector, camera,
                                            radiance_field, occupancy_grid,
                                            light_field, scene_aabb, render_step_size,
                                            test_chunk_size=args.test_chunk_size)
                    soft_dual_mask = result["visible_texture_map"].view(int(projector["H"].item()), int(projector["W"].item()), -1)
                    hard_dual_mask = (soft_dual_mask.mean(dim=-1, keepdim=True) >= 1.0)
                    gsoup.save_image(hard_dual_mask, dual_mask_path)
                dual_dist_mask_path = Path(dual_folder, "dual_distortion_mask.png")
                if dual_dist_mask_path.exists():
                    hard_dual_distortion_mask = gsoup.load_image(dual_dist_mask_path, float=True, to_torch=True, device=args.device)[..., None].to(torch.bool)
                else:
                    # distortion_mask = (torch.abs(n_dot_v) >= 0.3).view(*primary_rays.viewdirs.shape[:2], -1)
                    result = dual_photography(distortion_mask.repeat(1,1,3).to(torch.float32).permute(2, 0, 1),
                                            dual_retvals, projector, camera,
                                            radiance_field, occupancy_grid,
                                            light_field, scene_aabb, render_step_size,
                                            test_chunk_size=args.test_chunk_size)
                    soft_dual_distortion_mask = result["visible_texture_map"].view(int(projector["H"].item()), int(projector["W"].item()), -1) 
                    hard_dual_distortion_mask = (soft_dual_distortion_mask.mean(dim=-1, keepdim=True) >= 1.0)
                    hard_dual_distortion_mask = scipy.ndimage.binary_opening(gsoup.to_numpy(hard_dual_distortion_mask[:, :, 0]),
                                                                             structure=np.ones((10,10)).astype(int))
                    hard_dual_distortion_mask = torch.tensor(hard_dual_distortion_mask[..., None], device=args.device, dtype=torch.bool)
                    gsoup.save_image(hard_dual_distortion_mask, dual_dist_mask_path)
                dual_mask = hard_dual_mask & hard_dual_distortion_mask
                texture_fixed_path = Path(mypath, "projector_texture_fixed_{}.png".format(i))
                texture_fixed_mask_path = Path(mypath, "projector_texture_mask_fixed_{}.png".format(i))
                if texture_fixed_path.exists():
                    texture_aggregate = gsoup.load_image(texture_fixed_path, float=True, to_torch=True, device=args.device)
                    dual_mask_aggregate = gsoup.load_image(texture_fixed_mask_path, float=True, to_torch=True, device=args.device)[..., None].to(torch.bool)
                else:
                    if i == 0:
                        dual_mask_aggregate = dual_mask
                        texture_aggregate = (dual_mask_aggregate * dual_image) + ((~dual_mask_aggregate) * torch.ones_like(dual_image))
                    else:
                        texture_aggregate = dual_mask_aggregate * texture_aggregate + (~dual_mask_aggregate) * dual_image
                    # dual_mask_aggregate = dual_mask_aggregate | dual_mask
                    gsoup.save_image(texture_aggregate, texture_fixed_path)
                    gsoup.save_image(dual_mask_aggregate, texture_fixed_mask_path)
                #### set projector texture to dual image ####
                light_field["projectors"][0]["textures"] = texture_aggregate.permute(2, 0, 1)
            # show final result from all views
            with torch.no_grad():
                for i, viewpoint in enumerate(extra_info["cam_index"]):
                    primary_rays = train_dataset[viewpoint]["rays"]
                    primal_result = march_and_extract(
                            radiance_field,
                            primary_rays,
                            scene_aabb,
                            occupancy_grid=occupancy_grid,
                            render_step_size=render_step_size,
                            render_bkgd=torch.zeros(3, device=args.device),
                            test_chunk_size=args.test_chunk_size,
                            ret_vals=test_retvals,
                            is_relightable=args.relightable,
                            light_field=light_field,
                            texture_ids=texture_ids
                    )
                    # define mask as opacity > 0 m1
                    image = primal_result["rgb"]
                    image = image.view(*primary_rays.viewdirs.shape[:2], -1)
                    gsoup.save_image(image, Path(mypath, "final_rgb_{}.png".format(i)))
        elif mode == "compensate":  # perform projector compensation
            brightness_values = [-150]
            with torch.no_grad():
                primary_rays = train_dataset[extra_info["cam_index"]]["rays"]
            opt_rays = Rays(origins=primary_rays.origins.detach(), viewdirs=primary_rays.viewdirs.detach())
            for target_path in extra_info["image_paths"]:
                target_image = gsoup.load_image(target_path, float=True, to_torch=True,
                                                resize_wh=(640, 480), device=args.device)
                if target_image.shape[-1] == 4:
                    target_image = target_image[..., :3]
                for i, brightness in enumerate(brightness_values):
                    cur_path = Path(args.experiment_folder, "compensation", "{}_{}".format(target_path.stem, i))
                    cur_path.mkdir(parents=True, exist_ok=True)
                    final_texture_path = Path(cur_path, "final_texture_{}.png".format(i))
                    if final_texture_path.exists():
                        continue
                    cur_target = target_image.clone()
                    cur_target = gsoup.change_brightness(cur_target, brightness)
                    result_texture = optimize_texture(cur_target, opt_rays, radiance_field,
                                                    occupancy_grid, light_field,
                                                    scene_aabb, render_step_size,
                                                    cur_path,
                                                    intermediate_results=True)
                    gsoup.save_image(cur_target, Path(cur_path, "target.png"))
                    gsoup.save_image(result_texture, final_texture_path)
        elif mode == "projector_calib":  # evaluate projector parameters effect on image
            with torch.no_grad():
                num_images = 20
                cur_gamma = light_field["projectors"][0]["gamma"]
                cur_amp = light_field["projectors"][0]["amp"]
                cur_f = light_field["projectors"][0]["f"]
                cur_t = light_field["projectors"][0]["t"]
                cur_v = light_field["projectors"][0]["v"]
                data = train_dataset[extra_info["cam_index"]]
                render_bkgd = data["color_bkgd"]
                rays = data["rays"]
                if not "proj_texture" in extra_info:
                    texture_ids = data["texture_ids"]
                pixels = data["pixels"]
                t_values = np.linspace(-1, 1, num_images)
                for i, t in tqdm.tqdm(enumerate(t_values)):
                    c2w = gsoup.qvec2mat(light_field["projectors"][0]["v"])
                    z_dir = c2w[:, 2]
                    new_t = cur_t + t*z_dir
                    light_field["projectors"][0]["t"] = new_t
                    result = march_and_extract(
                            radiance_field,
                            rays,
                            scene_aabb,
                            occupancy_grid=occupancy_grid,
                            render_step_size=render_step_size,
                            render_bkgd=torch.zeros(3, device=args.device),
                            test_chunk_size=args.test_chunk_size,
                            ret_vals=test_retvals,
                            is_relightable=args.relightable,
                            light_field=light_field,
                            texture_ids=texture_ids
                        )
                    if prefix is None:
                        myprefix = "t_{:02d}".format(i)
                    else:
                        myprefix = prefix + "_t_{:02d}".format(i)
                    if dst is not None:
                        mypath = Path(dst, args.experiment_folder)
                        mypath.mkdir(parents=True, exist_ok=True)
                    else:
                        mypath = args.experiment_folder
                    save_results_to_images(result, myprefix, mypath, rays.viewdirs.shape[:2])
                light_field["projectors"][0]["t"] = cur_t
                f_values = np.linspace(1.1, 2.1, num_images)
                for i, f in tqdm.tqdm(enumerate(f_values)):
                    light_field["projectors"][0]["f"] = torch.full((1, ), f, dtype=torch.float32, device=args.device)
                    result = march_and_extract(
                            radiance_field,
                            rays,
                            scene_aabb,
                            occupancy_grid=occupancy_grid,
                            render_step_size=render_step_size,
                            render_bkgd=torch.zeros(3, device=args.device),
                            test_chunk_size=args.test_chunk_size,
                            ret_vals=test_retvals,
                            is_relightable=args.relightable,
                            light_field=light_field,
                            texture_ids=texture_ids
                        )
                    if prefix is None:
                        myprefix = "f_{:02d}".format(i)
                    else:
                        myprefix = prefix + "_f_{:02d}".format(i)
                    if dst is not None:
                        mypath = Path(dst, args.experiment_folder)
                        mypath.mkdir(parents=True, exist_ok=True)
                    else:
                        mypath = args.experiment_folder
                    save_results_to_images(result, myprefix, mypath, rays.viewdirs.shape[:2])
                light_field["projectors"][0]["f"] = cur_f
                gamma_values = np.linspace(0.1, 3.0, num_images)
                light_field["projectors"][0]["amp"] = torch.full((1, ), 25.0, dtype=torch.float32, device=args.device)
                for i, gamma in tqdm.tqdm(enumerate(gamma_values)):
                    light_field["projectors"][0]["gamma"] = torch.full((1, ), gamma, dtype=torch.float32, device=args.device)
                    result = march_and_extract(
                            radiance_field,
                            rays,
                            scene_aabb,
                            occupancy_grid=occupancy_grid,
                            render_step_size=render_step_size,
                            render_bkgd=torch.zeros(3, device=args.device),
                            test_chunk_size=args.test_chunk_size,
                            ret_vals=test_retvals,
                            is_relightable=args.relightable,
                            light_field=light_field,
                            texture_ids=texture_ids
                        )
                    if prefix is None:
                        myprefix = "gamma_{:02d}".format(i)
                    else:
                        myprefix = prefix + "_gamma_{:02d}".format(i)
                    if dst is not None:
                        mypath = Path(dst, args.experiment_folder)
                        mypath.mkdir(parents=True, exist_ok=True)
                    else:
                        mypath = args.experiment_folder
                    save_results_to_images(result, myprefix, mypath, rays.viewdirs.shape[:2])
                    mse = F.mse_loss(result["rgb"].view(pixels.shape), pixels)
                    psnr = -10.0 * torch.log(mse) / np.log(10.0)
                    psnrs.append(psnr.item())
                light_field["projectors"][0]["gamma"] = torch.full((1, ), 2.2, dtype=torch.float32, device=args.device)
                cur_amp = light_field["projectors"][0]["amp"]
                amp_values = np.linspace(5., 30.0, num_images)
                for i, amp in tqdm.tqdm(enumerate(amp_values)):
                    light_field["projectors"][0]["amp"] = torch.full((1, ), amp, dtype=torch.float32, device=args.device)
                    result = march_and_extract(
                            radiance_field,
                            rays,
                            scene_aabb,
                            occupancy_grid=occupancy_grid,
                            render_step_size=render_step_size,
                            render_bkgd=torch.zeros(3, device=args.device),
                            test_chunk_size=args.test_chunk_size,
                            ret_vals=test_retvals,
                            is_relightable=args.relightable,
                            light_field=light_field,
                            texture_ids=texture_ids
                        )
                    if prefix is None:
                        myprefix = "amp_{:02d}".format(i)
                    else:
                        myprefix = prefix + "_amp_{:02d}".format(i)
                    save_results_to_images(result, myprefix, mypath, rays.viewdirs.shape[:2])
                    mse = F.mse_loss(result["rgb"].view(pixels.shape), pixels)
                    psnr = -10.0 * torch.log(mse) / np.log(10.0)
                    psnrs.append(psnr.item())
                light_field["projectors"][0]["gamma"] = cur_gamma
                light_field["projectors"][0]["amp"] = cur_amp
                light_field["projectors"][0]["f"] = cur_f
                print("f values: ", f_values)
                print("gamma values: ", gamma_values)
                print("psnrs gamma: ", psnrs[:num_images])
                print("amp values: ", amp_values)
                print("psnrs amp: ", psnrs[num_images:])
        elif mode == "dual_photo":  # render a dual photo, currently set up to produce XRAY result from paper
            # obtain some desired view (beyond an occluder for example)
            # note: change this to your desired view, currently it is set up to produce paper xray result
            # select particular view from teapot-neko scene
            orig_c2w = train_dataset.camtoworlds[extra_info["cam_index"]].clone().detach()
            view = orig_c2w.clone().detach()
            x_dir = view[:3, 0].clone()
            y_dir = view[:3, 1].clone()
            z_dir = view[:3, 2].clone()
            at = view[:3, -1].clone() + 0.95 * z_dir  # advance camera in view direction
            view[:3, -1] += 0.2*y_dir  # and down a bit
            z_dir_new = at - view[:3, -1]
            z_dir_new = z_dir_new / torch.norm(z_dir_new)
            x_dir_new = x_dir
            y_dir_new = torch.cross(z_dir_new, x_dir_new)
            y_dir_new = y_dir_new / torch.norm(y_dir_new)
            view[:3, 0] = x_dir_new
            view[:3, 1] = y_dir_new
            view[:3, 2] = z_dir_new
            # rot = gsoup.look_at_torch(view[:3, -1],
            #                         at,
            #                         torch.tensor([0.0, 0.0, 1.0], device=args.device))
            train_dataset.camtoworlds[extra_info["cam_index"]] = view  # set new viewpoint
            primary_rays = train_dataset[extra_info["cam_index"]]["rays"]  # get rays
            primal_result = march_and_extract(  # march but use a near plane further away than 0 to render beyond an occluder
                    radiance_field,
                    primary_rays,
                    scene_aabb,
                    near_plane=1.0 if extra_info["xray"] else None,
                    occupancy_grid=occupancy_grid,
                    render_step_size=render_step_size,
                    render_bkgd=torch.zeros(3, device=args.device),
                    test_chunk_size=args.test_chunk_size,
                    ret_vals=test_retvals,
                    is_relightable=args.relightable,
                    light_field=light_field,
                    texture_ids=texture_ids
                )
            desired = primal_result["rgb"].view(*primary_rays.viewdirs.shape[:2], -1)  # final desired view
            dual_retvals = ["rgb"]
            projector = light_field["projectors"][0]
            t_cam = train_dataset.camtoworlds[extra_info["cam_index"], :3, -1]
            rot_cam = R.from_matrix(train_dataset.camtoworlds[extra_info["cam_index"], :3, :3].cpu().numpy())
            v_cam = gsoup.to_torch(rot_cam.as_rotvec().astype(np.float32), device=args.device)
            camera = create_camera(train_dataset.K, t_cam, v_cam, train_dataset.WIDTH, train_dataset.HEIGHT, device=args.device)
            # pass parameters to dual_photography function
            result = dual_photography(desired.permute(2, 0, 1),
                                      dual_retvals, projector, camera,
                                      radiance_field, occupancy_grid,
                                      light_field, scene_aabb, render_step_size,
                                      test_chunk_size=args.test_chunk_size)
            dual_photo = result["rgb"].view(1080, 1920, -1)
            dual_path = Path(args.experiment_folder, "dual_photo")
            dual_path.mkdir(parents=True, exist_ok=True)
            gsoup.save_image(desired, Path(dual_path, "desired.png"))
            gsoup.save_image(dual_photo, Path(dual_path, "dual_photo.png"))
            # lets also reproject the dual photo back to the original view, just as sanity check
            light_field["projectors"][0]["textures"] = dual_photo.permute(2, 0, 1)
            reprojected = march_and_extract(
                    radiance_field,
                    primary_rays,
                    scene_aabb,
                    occupancy_grid=occupancy_grid,
                    render_step_size=render_step_size,
                    render_bkgd=torch.zeros(3, device=args.device),
                    test_chunk_size=args.test_chunk_size,
                    ret_vals=test_retvals,
                    is_relightable=args.relightable,
                    light_field=light_field,
                    texture_ids=texture_ids
                )
            reprojected = reprojected["rgb"].view(*primary_rays.viewdirs.shape[:2], -1)
            gsoup.save_image(reprojected, Path(dual_path, "reprojected.png"))
        elif mode == "train_set" or mode == "test_set":  # renders decompositions of frames from train or test set, with the exact lighting as in the dataset
            with torch.no_grad():
                dataset = train_dataset if mode == "train_set" else test_dataset
                if "projectors" in light_field:
                    light_field["projectors"][0]["textures"] = dataset.textures
                for i in tqdm.tqdm(range(len(extra_info["frames_to_render"]))):
                    frame_index = extra_info["frames_to_render"][i]
                    data = dataset[frame_index]
                    render_bkgd = data["color_bkgd"]
                    rays = data["rays"]
                    if not "proj_texture" in extra_info:
                        texture_ids = data["texture_ids"]
                    pixels = data["pixels"]
                    # rendering
                    grid = occupancy_grid
                    if "no_grid" in extra_info:
                        if extra_info["no_grid"]:
                            grid = None
                    result = march_and_extract(
                        radiance_field,
                        rays,
                        scene_aabb,
                        occupancy_grid=grid,
                        render_step_size=render_step_size,
                        render_bkgd=render_bkgd,
                        test_chunk_size=args.test_chunk_size,
                        ret_vals=test_retvals,
                        is_relightable=args.relightable,
                        light_field=light_field,
                        texture_ids=texture_ids
                    )
                    mse = F.mse_loss(result["rgb"].view(pixels.shape), pixels)
                    psnr = -10.0 * torch.log(mse) / np.log(10.0)
                    psnrs.append(psnr.item())
                    c2w = dataset.get_current_cameras()[frame_index]
                    c2w = gsoup.to_44(c2w)
                    w2c = dataset.K @ torch.inverse(c2w)[:3, :]
                    if dst is not None:
                        mypath = Path(dst, args.experiment_folder)
                        mypath.mkdir(parents=True, exist_ok=True)
                    else:
                        mypath = args.experiment_folder
                    if prefix is None:
                        myprefix = "{:04d}".format(frame_index)
                    else:
                        myprefix = prefix + "_{:04d}".format(frame_index)
                    if mode == "test_set":
                        w2c = None
                    save_results_to_images(result, myprefix, mypath, rays.viewdirs.shape[:2], w2c)
                if "projectors" in light_field:
                    light_field["projectors"][0]["textures"] = train_dataset.textures
        elif mode == "test_set_movie":  # renders all test set as RGB, sequentially as a .gif (kind of useless)
            with torch.no_grad():
                if "stride" in extra_info:
                    stride=extra_info["stride"]
                else:
                    stride=20
                results = defaultdict(list)
                for i in tqdm.tqdm(range(0, len(test_dataset), stride)):
                    frame_index = i
                    data = test_dataset[frame_index]
                    if not "proj_texture" in extra_info:
                        texture_ids = data["texture_ids"]
                    # rendering
                    result = march_and_extract(
                        radiance_field,
                        data["rays"],
                        scene_aabb,
                        occupancy_grid=occupancy_grid,
                        render_step_size=render_step_size,
                        render_bkgd=data["color_bkgd"],
                        test_chunk_size=args.test_chunk_size,
                        ret_vals=test_retvals,
                        is_relightable=args.relightable,
                        light_field=light_field,
                        texture_ids=texture_ids
                    )
                    for key in result.keys():
                        results[key].append(result[key])
                for key in results.keys():
                    results[key] = torch.stack(results[key], 0)
                w2c = test_dataset.K[None, :, :] @ torch.inverse(test_dataset.camtoworlds[::stride])[:, :3, :]
                if prefix is None:
                    myprefix = "test_set_movie"
                else:
                    myprefix = prefix
                save_results_to_gif(results, len(w2c), myprefix, args.experiment_folder, data["rays"].viewdirs.shape[:2], w2c)
        elif mode == "train_set_movie":  # renders all train set as RGB, sequentially as a .gif
            with torch.no_grad():
                if "stride" in extra_info:
                    stride=extra_info["stride"]
                else:
                    stride=20
                results = defaultdict(list)
                for i in tqdm.tqdm(range(0, len(train_dataset), stride)):
                    frame_index = i
                    data = train_dataset[frame_index]
                    if not "proj_texture" in extra_info:
                        texture_ids = data["texture_ids"]
                    # rendering
                    result = march_and_extract(
                        radiance_field,
                        data["rays"],
                        scene_aabb,
                        occupancy_grid=occupancy_grid,
                        render_step_size=render_step_size,
                        render_bkgd=data["color_bkgd"],
                        test_chunk_size=args.test_chunk_size,
                        ret_vals=test_retvals,
                        is_relightable=args.relightable,
                        light_field=light_field,
                        texture_ids=texture_ids
                    )
                    for key in result.keys():
                        results[key].append(result[key])
                for key in results.keys():
                    results[key] = torch.stack(results[key], 0)
                # t_cam = train_dataset.cameras[0][::stride]
                # v_cam = train_dataset.cameras[1][::stride]
                # r_cam = batch_rotvec2mat(v_cam)  # c2w
                # c2w = torch.cat((r_cam, t_cam[:, :, None]), axis=-1)
                c2w = train_dataset.get_current_cameras()[::stride]
                c2w = torch.cat((c2w, torch.tensor([0, 0, 0, 1.], device=c2w.device)[None, None, :].repeat(len(c2w), 1, 1)), dim=1)
                w2c = train_dataset.K[None, :, :] @ torch.inverse(c2w)[:, :3, :]
                if prefix is None:
                    myprefix = "train_set_movie"
                else:
                    myprefix = prefix
                save_results_to_gif(results, len(w2c), myprefix, args.experiment_folder, data["rays"].viewdirs.shape[:2], w2c)
        elif mode == "move_camera":  # renders the scene using a circular camera path from novel views.
            with torch.no_grad():
                c2w = test_dataset.camtoworlds
                if "plane" in extra_info:
                    plane = extra_info["plane"]
                else:
                    plane = "xz"
                vc2w = vcams_from_cams(test_dataset.camtoworlds, extra_info["n_frames"], plane=plane)
                test_dataset.camtoworlds = vc2w
                results = defaultdict(list)
                for i in tqdm.tqdm(range(len(vc2w))):  # len(test_dataset)
                    data = test_dataset[i]
                    render_bkgd = data["color_bkgd"]
                    rays = data["rays"]
                    if texture_ids is None:
                        texture_ids = data["texture_ids"]
                    # rendering
                    result = march_and_extract(
                        radiance_field,
                        rays,
                        scene_aabb,
                        occupancy_grid=occupancy_grid,
                        render_step_size=render_step_size,
                        render_bkgd=render_bkgd,
                        test_chunk_size=args.test_chunk_size,
                        ret_vals=test_retvals,
                        is_relightable=args.relightable,
                        light_field=light_field,
                        texture_ids=texture_ids,
                    )
                    for key in result.keys():
                        results[key].append(result[key])
                for key in results.keys():
                    results[key] = torch.stack(results[key], 0)
                w2c = test_dataset.K[None, :, :] @ torch.inverse(test_dataset.camtoworlds)[:, :3, :]
                if prefix is None:
                    myprefix = "move_camera"
                else:
                    myprefix = prefix
                save_results_to_gif(results, len(vc2w), myprefix, args.experiment_folder, rays.viewdirs.shape[:2], w2c)
                test_dataset.camtoworlds = c2w
        elif mode == "move_projector":  # renders the scene using a circular projector path from any view.
            with torch.no_grad():
                if "plane" in extra_info:
                    plane = extra_info["plane"]
                else:
                    plane = "xz"
                vc2w = vcams_from_cams(test_dataset.camtoworlds, extra_info["n_frames"], plane=plane)
                results = defaultdict(list)
                cur_t = light_field["projectors"][0]["t"]
                cur_v = light_field["projectors"][0]["v"]
                for i in tqdm.tqdm(range(len(vc2w))):  # len(test_dataset)
                    data = test_dataset[extra_info["cam_index"]]
                    render_bkgd = data["color_bkgd"]
                    rays = data["rays"]
                    if texture_ids is None:
                        texture_ids = data["texture_ids"]
                    rot = R.from_matrix(vc2w[i, :3, :3].cpu().numpy())
                    rot_vec = rot.as_rotvec()
                    light_field["projectors"][0]["t"] = vc2w[i, :3, -1]
                    light_field["projectors"][0]["v"] = torch.tensor(rot_vec, dtype=torch.float32, device=args.device)
                    # rendering
                    result = march_and_extract(
                        radiance_field,
                        rays,
                        scene_aabb,
                        occupancy_grid=occupancy_grid,
                        render_step_size=render_step_size,
                        render_bkgd=render_bkgd,
                        test_chunk_size=args.test_chunk_size,
                        ret_vals=test_retvals,
                        is_relightable=args.relightable,
                        light_field=light_field,
                        texture_ids=texture_ids,
                    )
                    for key in result.keys():
                        results[key].append(result[key])
                for key in results.keys():
                    results[key] = torch.stack(results[key], 0)
                w2c = test_dataset.K[None, :, :] @ torch.inverse(test_dataset.camtoworlds)[:, :3, :]
                if prefix is None:
                    myprefix = "move_projector"
                else:
                    myprefix = prefix
                save_results_to_gif(results, len(vc2w), prefix, args.experiment_folder, rays.viewdirs.shape[:2], w2c[extra_info["cam_index"]:extra_info["cam_index"]+1])
                light_field["projectors"][0]["t"] = cur_t
                light_field["projectors"][0]["v"] = cur_v
        elif mode == "play_vid":  # streams a raw video onto the scene. projector is fixed while camera moves in circular motion.
            from gsoup.video import VideoReader
            reader = VideoReader(Path(extra_info["vid_path"]),
                                int(light_field["projectors"][0]["H"].item()),
                                int(light_field["projectors"][0]["W"].item()),
                                verbose=True)
            frames = gsoup.to_torch(np.array([frame for frame in reader]), device=args.device) / 255
            if "stride" in extra_info:
                stride=extra_info["stride"]
            else:
                stride=1
            frames = frames[::stride]
            with torch.no_grad():
                cur_t = light_field["projectors"][0]["t"]
                cur_v = light_field["projectors"][0]["v"]
                cur_c2w = test_dataset.camtoworlds
                proj_c2w = train_dataset.get_current_cameras(select=torch.tensor([extra_info["proj_index"]], device=args.device))[0]
                results = defaultdict(list)
                if "animate_cam" in extra_info:
                    plane = extra_info["animate_cam"]
                    vc2w = vcams_from_cams(train_dataset.camtoworlds,
                                           len(frames) // 2,
                                           plane=plane,
                                           center_pose=proj_c2w.cpu().numpy(),
                                           radius_factor=0.25)
                    test_dataset.camtoworlds = vc2w
                else:
                    cam_c2w = train_dataset.get_current_cameras(select=torch.tensor([extra_info["cam_index"]], device=args.device))
                    test_dataset.camtoworlds = cam_c2w.repeat(len(frames), 1, 1)
                for i in tqdm.tqdm(range(len(frames))):
                    data = test_dataset[i % (len(frames) // 2)]
                    render_bkgd = data["color_bkgd"]
                    # render_bkgd = torch.tensor([0.0, 0.0, 0.0], device=args.device)
                    rays = data["rays"]
                    texture_ids = None
                    light_field["projectors"][0]["textures"] = frames[i].permute(2, 0, 1)
                    R_mat = proj_c2w[:3, :3]
                    # R_mat = train_dataset.camtoworlds[extra_info["proj_index"], :3, :3]
                    rot = R.from_matrix(R_mat.cpu().numpy())
                    forward_vector = R_mat @ torch.tensor([0., 0., 1.], device=args.device)
                    qvec = rot.as_quat().astype(np.float32)
                    qvec = np.array([qvec[3], qvec[0], qvec[1], qvec[2]])  # flip real to be first
                    light_field["projectors"][0]["t"] = proj_c2w[:3, -1] + forward_vector * 0.75
                    light_field["projectors"][0]["v"] = torch.tensor(qvec, dtype=torch.float32, device=args.device)
                    result = march_and_extract(
                        radiance_field,
                        rays,
                        scene_aabb,
                        occupancy_grid=occupancy_grid,
                        render_step_size=render_step_size,
                        render_bkgd=render_bkgd,
                        test_chunk_size=args.test_chunk_size,
                        ret_vals=test_retvals,
                        is_relightable=args.relightable,
                        light_field=light_field,
                        texture_ids=texture_ids,
                    )
                    for key in result.keys():
                        results[key].append(result[key])
                for key in results.keys():
                    results[key] = torch.stack(results[key], 0)
                if prefix is None:
                    myprefix = "play_vid"
                else:
                    myprefix = prefix
                save_results_to_gif(results, len(frames), prefix, args.experiment_folder, rays.viewdirs.shape[:2])
                light_field["projectors"][0]["t"] = cur_t
                light_field["projectors"][0]["v"] = cur_v
                test_dataset.camtoworlds = cur_c2w
        else:
            raise NotImplementedError
        if light_field is not None:
            if "proj_texture" in extra_info and "projectors" in light_field:
                light_field["projectors"][0]["textures"] = cur_textures
            if "proj_amp" in extra_info and "projectors" in light_field:
                light_field["projectors"][0]["amp"] = cur_amp
            if "coloc_light" in extra_info and "coloc_light" in light_field:
                if extra_info["coloc_light"] == False:
                    light_field["coloc_light"] = cur_coloc
        train_dataset.training = True
    return psnrs
