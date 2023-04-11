"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""
from pathlib import Path
import numpy as np
import collections
import json
import logging
import shutil
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import gsoup
from data import pose_format_converter
from collections import defaultdict

def get_pattern_names(amount, length, voronoi_amount=3):
    pattern_amount = amount
    pattern_length = length
    sync_time = length
    x = np.arange(0, pattern_amount).repeat(pattern_length)
    indices = np.where(np.ediff1d(x, to_begin=1, to_end=1))[0]
    x = np.insert(x, np.repeat(indices, pattern_length), -1)
    indices = np.where(np.ediff1d(x) > 0)[0] + 1
    x = np.insert(x, np.repeat(indices, sync_time), -2)
    x = np.concatenate((x, np.repeat(-2, sync_time)))
    y = np.empty(len(x), dtype='<U16')
    y[x==-2] = "all_black"
    y[x==-1] = "all_white"
    #lollipop
    for i in range(pattern_amount):
        y[x==i] = "lollipop_{}".format(i)
    # for i in range(voronoi_amount):
    #     y[x==i] = "voronoi_{}".format(i)
    # for i in range(voronoi_amount, pattern_amount):
    #     y[x==i] = "gray_{}".format(i)
    selected_textures = y
    # if len(selected_textures) < N_VIEWS:
    #     selected_textures = np.concatenate((selected_textures, np.full(N_VIEWS - len(selected_textures), "all_black")))
    return selected_textures

def blender_pose_to_cv(blender_pose):
    assert blender_pose.shape == (4, 4)
    transform = np.array([[1, 0, 0, 0],  # flip y and z
                          [0, -1, 0, 0],
                          [0, 0, -1, 0],
                          [0, 0, 0, 1]])
    cv_pose = np.matmul(blender_pose, transform)
    return cv_pose

def process_rawdata_folder(folder, colmap_mode):
    cam_path = Path(folder, "camera")
    proj_path = Path(folder, "projector")
    trim_file = Path(cam_path, "trim.txt")
    if trim_file.exists():
        with open(trim_file, "r") as f:
            trim = int(f.readlines()[0])
    else:
        trim = 0
    if proj_path.exists():  # the project
        assert Path(proj_path, "raw_patterns").exists()
        if colmap_mode == "video":
            pattern_length = 6
            pattern_amount = 102
        else:
            pattern_length = 1
            pattern_amount = 102
        texture_ids, texture_names = get_pattern_names(pattern_amount, pattern_length)
        _, unique_indices = np.unique(texture_names, return_index=True)
        texture_file_names = ["{:04d}.png".format(x) for x in np.arange(len(texture_ids))]
        unique_file_src = np.array(texture_file_names)[unique_indices]
        unique_file_dst = texture_names[unique_indices]
        for src, dst in zip(unique_file_src, unique_file_dst):
            shutil.copy(Path(proj_path, "raw_patterns", src), Path(proj_path, dst + ".png"))
        # trimmed_texture_ids = texture_ids[start_index:]
        if colmap_mode == "video":
            squeezed_texture_indices = np.where(np.ediff1d(texture_ids, to_begin=1))[0][1:]
            actual_texture_names = texture_names[squeezed_texture_indices]
            raw_frames = gsoup.slice_from_video(next(cam_path.glob("*.avi")),
                                                    every_n_frames=pattern_length*2,
                                                    start_frame=trim+pattern_length,
                                                    verbose=True)
            raw_frames = raw_frames[:len(squeezed_texture_indices)]
            file_names = ["{:04d}.png".format(x) for x in range(len(raw_frames))]
            gsoup.save_images(raw_frames, folder, file_names=file_names)
            write_img2tex_file({x: y for x, y in zip(file_names, actual_texture_names)},
                           {x: i for i, x in enumerate(file_names)},
                           Path(folder, "img2tex.json"))
        else:
            squeezed_texture_indices = np.where(np.ediff1d(texture_ids, to_end=1))[0]
            actual_texture_names = texture_names[squeezed_texture_indices]
            file_names = [x.name for x in folder.glob("*.png")]
            actual_texture_names = actual_texture_names[:len(file_names)]
            write_img2tex_file({x: y for x, y in zip(file_names, actual_texture_names)},
                                {x: i // 3 for i, x in enumerate(file_names)},
                                Path(folder, "img2tex.json"))
    else:
        raise FileNotFoundError("No projector folder found")

def write_img2tex_file(img2tex, img2view, output_path):
    my_dict = {}
    for i, key in enumerate(img2tex.keys()):
        assert key in img2view.keys()
        my_dict[key] = [img2tex[key], img2view[key]]
    with open(output_path, "w") as fp:
        json.dump(my_dict, fp, indent=4)

def parse_img2tex_file(tex_path):
    with open(tex_path, "r") as fp:
        meta = json.load(fp)
    img2view = {}
    view2img = defaultdict(list)
    image_names = []
    all_texture_names = []
    for i, key in enumerate(meta.keys()):
        image_name = key
        texture_names = meta[key][0]
        view_id = meta[key][1]
        image_names.append(image_name)
        all_texture_names.append(texture_names)
        img2view[image_name] = view_id
        view2img[view_id].append(image_name)
    all_texture_names = np.array(all_texture_names).squeeze()
    unique_tex_names = np.unique(all_texture_names)
    image_names = np.array(image_names)
    tex2img = {}
    for i, tex_name in enumerate(unique_tex_names):
        indices = np.where(all_texture_names == tex_name)[0]
        tex2img[tex_name] = image_names[indices]
    return tex2img, img2view, view2img

def approximate_bg(images_path):
    """
    approximate background
    """
    imgs, paths = gsoup.load_images(images_path, return_paths=True)
    image_names = [x.name for x in paths]
    orig_imgs_path = images_path / "orig_images"
    debug_imgs_path = images_path / "debug_images"
    debug_imgs_path.mkdir(exist_ok=True, parents=True)
    orig_imgs_path.mkdir(exist_ok=True, parents=True)
    all_exist = True
    for i, image in enumerate(imgs):
        if not Path(orig_imgs_path, image_names[i]).exists():
            all_exist = False
    if all_exist:
        return
    from rembg import remove, new_session
    session = new_session()
    for i, image in enumerate(imgs):
        if Path(orig_imgs_path, image_names[i]).exists():
            continue
        shutil.copy(paths[i], Path(orig_imgs_path, image_names[i]))
        logging.info("Removing bg {} / {}".format(i, len(imgs)))
        output = remove(image, session=session).copy()
        alpha_channel = output[..., -1]
        # mask = alpha_channel > 10
        # mask = mask.astype(np.uint8) * 255
        mask = alpha_channel
        new_image = np.concatenate([image, mask[..., None]], axis=-1)
        gsoup.save_image(new_image, paths[i])
        # gsoup.save_image(new_image, Path(debug_imgs_path, image_names[i]))

def load_renderings(data_dir: str, optimize_cams: bool, colmap_mode="video", colmap_views="all_black", post_added_views=None, interpolate=True):
    """Load images from disk."""
    json_path = Path(data_dir, 'transforms.json')
    if not json_path.is_file():
        if colmap_mode=="video" or colmap_mode=="random_views":
            colmap_dir = Path(data_dir, "colmap_output")
            colmap_dir_actual = Path(data_dir, "colmap_output", "merged")  # ewww colmap...
            if not colmap_dir_actual.exists():
                colmap_dir_actual = Path(data_dir, "colmap_output", "0")
            # if not colmap_dir.is_dir():
            if not Path(data_dir, "img2tex.json").is_file():
                process_rawdata_folder(data_dir, colmap_mode)
            fail = pose_format_converter.run_colmap(data_dir, only_files_with=colmap_views)
            assert not fail
            pose_format_converter.colmap_to_nerf(colmap_dir_actual, json_path, post_added_views)
        else:
            colmap_dir = Path(data_dir, "colmap_output")
            if not colmap_dir.is_dir():
                pose_format_converter.run_colmap(data_dir, only_files_with=colmap_views)  #  only_files_with="all_white"
            pose_format_converter.colmap_to_nerf(colmap_dir, json_path)
    with open(json_path, "r") as fp:
        meta = json.load(fp)
    if not "blender_matrix_world_proj" in meta:  # if this isnt blender dataset then we need to approximate the bg
        approximate_bg(data_dir)
        is_blender = False
    else:
        is_blender = True
    textures_path = Path(data_dir, "projector")
    textures, texture_paths = gsoup.load_images(textures_path, to_float=True, channels_last=False, return_paths=True)
    texture_names = np.array([x.stem for x in texture_paths])
    images = []
    camtoworlds = []
    texture_ids = []
    view_ids = []
    for i in range(len(meta["frames"])):
        logging.info("loading: {} / {}".format(i, len(meta["frames"])))
        frame = meta["frames"][i]
        fname = Path(data_dir, frame["file_path"])
        if not fname.exists():
            continue
        if "transform_matrix" in frame:
            camtoworlds.append(frame["transform_matrix"])
        elif "blender_matrix_world" in frame:
            # camtoworlds.append(frame["blender_matrix_world"])
            camtoworlds.append(blender_pose_to_cv(np.array(frame["blender_matrix_world"])))
        else:
            if optimize_cams or interpolate:
                camtoworlds.append(np.eye(4)*-1)
            else:
                continue
            # raise ValueError("Unknown transform matrix")
        if "patterns" in frame:
            cur_texture_names = frame["patterns"]
            cur_texture_ids = np.where(np.isin(texture_names, cur_texture_names))[0]
            if len(cur_texture_ids) != 2:  # two or more identical patterns, or just one proejctor in data...
                cur_texture_ids = np.concatenate((cur_texture_ids, cur_texture_ids))
            # if frame["patterns"][0] != "all_black":
                # continue
            texture_ids.append(cur_texture_ids)
        
        rgba = gsoup.load_image(fname)
        view_ids.append(int(frame["view_id"]))
        images.append(rgba)
    images = np.stack(images, axis=0)
    camtoworlds = np.stack(camtoworlds, axis=0)
    # camtoworlds2 = np.stack(camtoworlds2, axis=0)
    texture_ids = np.stack(texture_ids, axis=0)
    view_ids = np.stack(view_ids, axis=0)
    if not optimize_cams and interpolate:  # interpolate missing cameras, but everything is static. remove anything that can't be interpolated
        opt_mask = (camtoworlds == np.eye(4)*-1).all(axis=(1, 2)) # true if needs optimization
        up_to_first = np.where(~opt_mask)[0][0]  # find first camera that is not missing
        up_to_last = np.where(~opt_mask)[0][-1]  # find last camera that is not missing
        to_keep_mask = np.zeros(len(images), dtype=np.bool)
        to_keep_mask[up_to_first:up_to_last+1] = True
        images = images[to_keep_mask]
        camtoworlds = camtoworlds[to_keep_mask]
        texture_ids = texture_ids[to_keep_mask]
        view_ids = view_ids[to_keep_mask]
    h, w = images.shape[1:3]
    camera_angle_x = float(meta["camera_angle_x"])
    focal = 0.5 * w / np.tan(0.5 * camera_angle_x)
    if "K_proj" in meta:
        K_proj = np.array(meta["K_proj"]).astype(np.float32)
    else:
        # K_proj = np.array([  # blender
        #     [800, 0, 400],
        #     [0, 800, 400],
        #     [0, 0, 1]
        # ]).astype(np.float32)
        # K_proj = np.array([  # real projector 
        #     [2208, 0, 960],
        #     [0, 2700, 864],
        #     [0, 0, 1]
        # ]).astype(np.float32)
        K_proj = np.array([  # real projector 
            [2800, 0, 960],
            [0, 2800, 1010],
            [0, 0, 1]
        ]).astype(np.float32)
    if "blender_matrix_world_proj" in meta:
        RT_proj = blender_pose_to_cv(np.array(meta["blender_matrix_world_proj"]))
        t_proj = RT_proj[:3, -1].astype(np.float32)
        R_proj = RT_proj[:3, :3]
        r = R.from_matrix(R_proj)
        v_proj = r.as_quat().astype(np.float32)
        v_proj = np.array([v_proj[3], v_proj[0], v_proj[1], v_proj[2]])  # flip real to be first
    elif "transform_matrix_proj" in meta:
        RT_proj = np.array(meta["transform_matrix_proj"]).astype(np.float32)
        t_proj = RT_proj[:3, -1].astype(np.float32)
        R_proj = RT_proj[:3, :3]
        r = R.from_matrix(R_proj)
        # v_proj = r.as_rotvec().astype(np.float32)
        v_proj = r.as_quat().astype(np.float32)
        v_proj = np.array([v_proj[3], v_proj[0], v_proj[1], v_proj[2]])  # flip real to be first
        
    else:
        t_proj = np.array([1.0, -2.0, 0.08]).astype(np.float32)
        v_proj = np.array([-1.62, -0.42, 0.17]).astype(np.float32)
    return images, camtoworlds, focal, K_proj, t_proj, v_proj, texture_ids, textures, texture_names, h, w, view_ids, is_blender


class SubjectLoader(torch.utils.data.Dataset):
    """Single subject data loader for training and evaluation."""
    def __init__(
        self,
        data_dir: str,
        split: str,
        color_bkgd_aug: str = "white",
        num_rays: int = None,
        near: float = None,
        far: float = None,
        batch_over_images: bool = True,
        device: str = "cpu",
        divide_res: int = 1,
        opt_cam = {"optimize_cams": False, "opt_over": None,
                   "dont_opt_over": None, "force_opt_all": False,
                   "force_opt_none": False, "interpolate": False},
        colmap_views = None,
        colmap_mode = "video",
        post_added_views = None,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.device = device
        self.NEAR, self.FAR = 2.0, 6.0
        self.divide_res = divide_res
        self.Rays = collections.namedtuple("Rays", ("origins", "viewdirs"))
        assert color_bkgd_aug in ["white", "black", "random"]
        self.split = split
        self.num_rays = num_rays
        self.near = self.NEAR if near is None else near
        self.far = self.FAR if far is None else far
        self.training = (num_rays is not None) and (split =="train")
        self.use_random_cams = False
        self.only_static_views = True
        self.only_black_views = True
        self.only_white_views = False
        self.no_color_views = False
        if self.training or self.split == "test":
            self.all_foreground = False
        else:
            self.all_foreground = True
        self.color_bkgd_aug = color_bkgd_aug
        self.batch_over_images = batch_over_images
        self.images, self.camtoworlds, self.focal, \
        K_proj, t_proj, v_proj, texture_ids, textures, texture_names, \
        self.HEIGHT, self.WIDTH, view_ids, is_blender = load_renderings(data_dir,
                                                            optimize_cams=opt_cam["optimize_cams"],
                                                            colmap_mode=colmap_mode,
                                                            colmap_views=colmap_views,
                                                            post_added_views=post_added_views,
                                                            interpolate=opt_cam["interpolate"])
        self.is_blender = is_blender
        self.images = self.post_process_images(self.images)
        self.orig_camtoworlds = torch.tensor(self.camtoworlds, dtype=torch.float32, device=device)
        rand_w2c, _ = gsoup.create_random_cameras_on_unit_sphere(len(self.images), 1.0,
                                                                 normal=torch.tensor([0.0, 0.0, 1.0], device=device),
                                                                 device=device)
        self.rand_c2w = torch.inverse(rand_w2c)
        self.view_ids = torch.tensor(view_ids, dtype=torch.int64, device=device)
        self.texture_names = texture_names
        self.texture_ids = torch.tensor(texture_ids, dtype=torch.int64, device=device)
        self.textures = torch.tensor(textures, dtype=torch.float32, device=device)
        self.cam_opt_mask = self.post_process_c2w(opt_over=opt_cam["opt_over"],
                                                  dont_opt_over=opt_cam["dont_opt_over"],
                                                  force_opt_all=opt_cam["force_opt_all"],
                                                  force_opt_none=opt_cam["force_opt_none"],
                                                  interpolate=opt_cam["interpolate"])
        self.cam_opt_mask = torch.tensor(self.cam_opt_mask, device=device)
        try:  # save all_black frames for (possible) usage later
            index_black = np.where(self.texture_names == "all_black")[0]
            self.all_black_mask = (self.texture_ids[:, 0] == index_black.item()).to(device)
        except ValueError:
            self.all_black_mask = torch.zeros_like(self.texture_ids[:, 0], dtype=torch.bool, device=device)
        try:
            index_white = np.where(self.texture_names == "all_white")[0]
            self.all_white_mask = (self.texture_ids[:, 0] == index_white.item()).to(device)
        except ValueError:
            self.all_white_mask = torch.zeros_like(self.texture_ids[:, 0], dtype=torch.bool, device=device)
        self.special_mask = torch.ones_like(self.texture_ids[:, 0], dtype=torch.bool)
        self.images = torch.tensor(self.images, dtype=torch.uint8, device=device)
        self.camtoworlds = torch.tensor(self.camtoworlds, dtype=torch.float32, device=device)
        self.cameras = None
        self.K = torch.tensor(
            [
                [self.focal, 0, self.WIDTH / 2.0],
                [0, self.focal, self.HEIGHT / 2.0],
                [0, 0, 1],
            ],
            dtype=torch.float32,
            device=device
        )  # (3, 3)
        self.K_proj = torch.tensor(K_proj, dtype=torch.float32, device=device)
        self.t_proj = torch.tensor(t_proj, dtype=torch.float32, device=device)
        self.v_proj = torch.tensor(v_proj, dtype=torch.float32, device=device)
        if self.divide_res != 1:
            self.HEIGHT = self.HEIGHT // self.divide_res
            self.WIDTH = self.WIDTH // self.divide_res
            self.focal = self.focal / self.divide_res
            self.K = self.K / self.divide_res
            self.K[2, 2] = 1
            # channels_first = torch.from_numpy(imgs.transpose(0, 3, 1, 2))
            # collapsed = channels_first.reshape(-1, *channels_first.shape[2:])
            channels_first = self.images.permute(0, 3, 1, 2) / 255.0
            imgs_low_res = torch.nn.functional.interpolate(channels_first, size=(self.HEIGHT, self.WIDTH), mode="bilinear")
            # uncollapsed = imgs_low_res.reshape(*channels_first.shape[:2], *imgs_low_res.shape[1:])
            channels_last = imgs_low_res.permute(0, 2, 3, 1) * 255
            # if channels_last.shape[-1] > 3:
                # channels_last[..., -1] = (channels_last[..., -1] > 0.5).astype(np.float32)  # set alpha mask to be 0 or 1
            self.images = channels_last.to(torch.uint8)
        assert self.images.shape[1:3] == (self.HEIGHT, self.WIDTH)

    def __len__(self):
        return len(self.images)

    #@torch.no_grad()
    def __getitem__(self, index):
        data = self.fetch_data(index)
        data = self.preprocess(data)
        return data

    def post_process_images(self, imgs):
        # add naive alpha channel to images if it doesn't exist
        alpha_channel = np.ones((imgs.shape[0], imgs.shape[1], imgs.shape[2], 1), dtype=np.uint8) * 255
        if imgs.shape[-1] == 3:
            imgs = np.concatenate([imgs, alpha_channel], axis=-1)
        if not self.is_blender:
            imgs[:, :, :, -1] = ((imgs[:, :, :, -1] > 10)*255)
        return imgs

    def find_interpolation_indices(self, to_opt_mask):
        """
        """
        indices_start = []
        indices_end = []
        prev_state = 0
        for i in range(len(to_opt_mask)):
            if to_opt_mask[i] != prev_state:
                if to_opt_mask[i]:
                    indices_start.append((i-1))
                else:
                    indices_end.append((i))
                prev_state = to_opt_mask[i]
        if len(indices_start) > len(indices_end):
            indices_start.pop()
        if indices_start[0] == -1:
            indices_start = indices_start[1:]
            indices_end = indices_end[1:]
        return np.stack([indices_start, indices_end], axis=1)

    def post_process_c2w(self, opt_over=None, dont_opt_over=None,
                         force_opt_all=False, force_opt_none=False,
                         interpolate=False):
        """
        given a list of camtoworlds, return a mask of which ones to optimize over
        if any camtoworld is marked with -1*eye, the initial guess will be estimated using interpolation
        :param opt_over: if not None, force optimize over these views
        :param dont_opt_over: if not None, force not to optimize over these views
        :param force_opt_all: if True, force optimize over all views
        :param force_opt_none: if True, force not to optimize over any views
        :param interpolate: if True, interpolate between missing views. 
        egde case: if first or last view is missing, it will be interpolated with the second or second to last view
        :return: a boolean mask of which views to optimize over
        """
        mask = (self.camtoworlds == np.eye(4)*-1).all(axis=(1, 2)) # true if needs optimization
        if dont_opt_over is not None:
            try:
                tex_index = np.where(self.texture_names == dont_opt_over)[0].item()
                tex_mask = (self.texture_ids == tex_index).all(dim=-1).cpu().numpy()
                mask = np.logical_or(mask, ~tex_mask)
            except ValueError:
                pass
        if opt_over is not None:
            tex_index = np.array([i for i, x in enumerate(self.texture_names) if opt_over in x])
            tex_mask = np.isin(self.texture_ids[:, 0].cpu().numpy(), tex_index)
            mask = np.logical_or(mask, tex_mask)
        if self.training and interpolate and mask.any():  # interpolate between missing views
            # deal with views before first anchor and after last anchor
            first_anchor = np.where(~mask)[0][0]
            last_anchor = np.where(~mask)[0][-1]
            self.camtoworlds[:first_anchor] = self.camtoworlds[first_anchor]
            self.camtoworlds[last_anchor+1:] = self.camtoworlds[last_anchor]
            indices = self.find_interpolation_indices(mask)
            # interpolate between all the rest of the missing views
            # indices = np.where(np.ediff1d(mask.astype(np.int)))[0].reshape(-1, 2)
            # indices[:, 1] += 1
            for i in range(indices.shape[0]):
                initial_pose = self.camtoworlds[indices[i, 0]]
                final_pose = self.camtoworlds[indices[i, 1]]
                num_samples_to_interpolate = indices[i, 1] - indices[i, 0] - 1
                i_qvec = R.from_matrix(np.concatenate((initial_pose[:3, :3][None, :], final_pose[:3, :3][None, :]), axis=0))
                full_matrices = []
                for t in np.linspace(start=0, stop=1, num=num_samples_to_interpolate+1, endpoint=False)[1:]:
                    slerp = Slerp(np.array([0, 1]), i_qvec)
                    rot = slerp(t).as_matrix()
                    loc = (1 - t) * initial_pose[:3, 3] + t * final_pose[:3, 3]
                    full_matrix = gsoup.to_44(gsoup.compose_rt(rot[None, :], loc[None, :]))[0]
                    full_matrices.append(full_matrix)
                full_matrices = np.stack(full_matrices, axis=0)
                self.camtoworlds[indices[i, 0]+1:indices[i, 1]] = full_matrices
        if force_opt_all:
            mask =  np.ones((len(self.camtoworlds),), dtype=bool)
        if force_opt_none:
            mask = np.zeros((len(self.camtoworlds),), dtype=bool)
        return mask


    def preprocess(self, data):
        """Process the fetched / cached data with randomness."""
        rgba, rays = data["rgba"], data["rays"]
        pixels, alpha = torch.split(rgba, [3, 1], dim=-1)

        if self.training:
            if self.color_bkgd_aug == "random":
                color_bkgd = torch.rand(3, device=self.device)
            elif self.color_bkgd_aug == "white":
                color_bkgd = torch.ones(3, device=self.device)
            elif self.color_bkgd_aug == "black":
                color_bkgd = torch.zeros(3, device=self.device)
        else:
            # just use black during inference
            if self.split == "train":
                color_bkgd = torch.tensor([0, 0, 0], device=self.device)
            else:
                color_bkgd = torch.tensor([1.0, 1.0, 1.0], device=self.device)

        pixels = pixels * alpha + color_bkgd * (1.0 - alpha)
        return {
            "pixels": pixels,  # [n_rays, 3] or [h, w, 3]
            "rays": rays,  # [n_rays,] or [h, w]
            "color_bkgd": color_bkgd,  # [3,]
            **{k: v for k, v in data.items() if k not in ["rgba", "rays"]},
        }

    def update_num_rays(self, num_rays):
        self.num_rays = num_rays

    def get_current_cameras(self, select=None, random_cams=False):
        """
        hot swaps the optimizable cameras (if any) with static/random cameras
        """
        if select is None:
            select = torch.arange(len(self.camtoworlds), device=self.device)
        if random_cams:
            return self.rand_c2w[select]
        if self.cameras is not None:  # some cameras are optimizable, replace the correct indices with opt parameters
            c2w = torch.zeros((len(select), 3, 4), dtype=self.camtoworlds.dtype, device=self.device)  # placeholder
            toopt = self.cam_opt_mask[select]  # mask of cameras that are optimizable
            # view_ids = self.opt_maskview_ids[self.cam_opt_mask]
            c2w[~toopt] = self.camtoworlds[select[~toopt], :3, :]  # non-optimizable cameras just get initial cam2world
            map_allcams_to_optcams = torch.full((len(self.camtoworlds),), -1, dtype=torch.int64, device=self.device)  # a map between all cameras indices and optimizable cameras indices (each entry is the index of the optimizable camera, or -1 if not optimizable)
            map_allcams_to_optcams[self.cam_opt_mask] = torch.arange(len(self.cameras[0]), device=self.device)
            t_cam = self.cameras[0][map_allcams_to_optcams[select[toopt]]]  # use the map and the selected image ids to get the optimizable camera parameters
            v_cam = self.cameras[1][map_allcams_to_optcams[select[toopt]]]
            r_cam = gsoup.batch_rotvec2mat(v_cam)  # c2w
            c2w_opt = torch.cat((r_cam, t_cam[:, :, None]), axis=-1)
            c2w[toopt] = c2w_opt  # finally replace in places where we can optimize
            if self.view_ids.shape[0] != torch.unique(self.view_ids).shape[0]: # same views will share the same camera
                indices_into_all_cams = torch.where(self.cam_opt_mask)[0][self.view_ids][select][~toopt]  # find the indices of cameras that are optimizable and share the same view
                t_cam = self.cameras[0][map_allcams_to_optcams[indices_into_all_cams]]  # map indices into optimizable cameras
                v_cam = self.cameras[1][map_allcams_to_optcams[indices_into_all_cams]]
                r_cam = gsoup.batch_rotvec2mat(v_cam)  # c2w
                c2w[~toopt] = torch.cat((r_cam, t_cam[:, :, None]), axis=-1)  # replace shared views with the same camera
        else:
            c2w = self.camtoworlds[select]  # (num_rays, 3, 4)
        return c2w

    def fetch_data(self, index):
        """Fetch the data (it maybe cached for multiple batches)."""
        num_rays = self.num_rays
        use_random_cams_flag = False
        if self.training:
            if self.batch_over_images:
                if self.use_random_cams:
                    use_random_cams_flag = True
                    high = len(self.rand_c2w)
                    image_id = torch.randint(0, high, size=(num_rays,), device=self.device)
                else:
                    mask = torch.ones(len(self.images), device=self.device, dtype=torch.bool)
                    mask = mask & self.special_mask
                    if self.no_color_views:
                        color_mask = ~(self.all_black_mask | self.all_white_mask)
                        mask = mask & ~(color_mask)
                    if self.only_black_views:
                        mask = mask & self.all_black_mask
                    if self.only_static_views:
                        mask = mask & ~self.cam_opt_mask
                    if self.only_white_views:
                        mask = mask & self.all_white_mask
                    high = torch.count_nonzero(mask)
                    image_id = torch.randint(0, high, size=(num_rays,), device=self.device)
                    # image_id = torch.randint(1, 5, size=(num_rays,), device=self.device)
                    # image_id[image_id == 2] = 1
                    # image_id[image_id == 3] = 4
                    image_id = torch.arange(len(self.images), device=self.device)[mask][image_id]
            else:
                image_id = torch.tensor([index], dtype=torch.int64, device=self.device)
            x = torch.randint(
                0, self.WIDTH, size=(num_rays,), device=self.device
            )
            y = torch.randint(
                0, self.HEIGHT, size=(num_rays,), device=self.device
            )
        else:
            image_id = torch.tensor([index], dtype=torch.int64, device=self.device)
            x, y = torch.meshgrid(
                torch.arange(self.WIDTH, device=self.device),
                torch.arange(self.HEIGHT, device=self.device),
                indexing="xy",
            )
            x = x.flatten()
            y = y.flatten()

        # generate rays
        rgba = self.images[image_id, y, x] / 255.0  # (num_rays, 4)
        if self.all_foreground:
            rgba[..., -1] = 1.0
        texture_ids = self.texture_ids[image_id]
        # c2w = self.camtoworlds[image_id]  # (num_rays, 3, 4)
        c2w = self.get_current_cameras(select=image_id, random_cams=use_random_cams_flag)
        # camera_dirs = F.pad(
        #     torch.stack(
        #         [
        #             (x - self.K[0, 2] + 0.5) / self.K[0, 0],
        #             (y - self.K[1, 2] + 0.5)
        #             / self.K[1, 1]
        #             * (-1.0 if self.OPENGL_CAMERA else 1.0),
        #         ],
        #         dim=-1,
        #     ),
        #     (0, 1),
        #     value=(-1.0 if self.OPENGL_CAMERA else 1.0),
        # )  # [num_rays, 3]  # equivalent to inv(K) @ [x, y ,1] and y*=-1, z*=-1
        camera_dirs = (torch.inverse(self.K) @ gsoup.to_hom(torch.stack((x, y), dim=-1)).T).T
        directions = (c2w[:, :3, :3] @ camera_dirs[:, :, None]).squeeze()
        # [n_cams, height, width, 3]
        # directions = (camera_dirs[:, None, :] * c2w[:, :3, :3]).sum(dim=-1)
        origins = torch.broadcast_to(c2w[:, :3, -1], directions.shape)
        viewdirs = directions / torch.linalg.norm(
            directions, dim=-1, keepdims=True
        )  # viewdirs are oriented away from camera

        if self.training:
            origins = torch.reshape(origins, (num_rays, 3))
            viewdirs = torch.reshape(viewdirs, (num_rays, 3))
            rgba = torch.reshape(rgba, (num_rays, 4))
        else:
            origins = torch.reshape(origins, (self.HEIGHT, self.WIDTH, 3))
            viewdirs = torch.reshape(viewdirs, (self.HEIGHT, self.WIDTH, 3))
            rgba = torch.reshape(rgba, (self.HEIGHT, self.WIDTH, 4))
            texture_ids = texture_ids.repeat(self.HEIGHT*self.WIDTH, 1)
        rays = self.Rays(origins=origins, viewdirs=viewdirs)

        return {
            "rgba": rgba,  # [h, w, 4] or [num_rays, 4]
            "rays": rays,  # [h, w, 3] or [num_rays, 3]
            "texture_ids": texture_ids  # [num_rays, num_projectors]
        }
