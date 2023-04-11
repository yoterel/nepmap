import math
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from radiance_fields.mlp import VanillaNeRFRadianceField
from radiance_fields.relight_mlp import ProjectorRadianceField
from helpers import set_random_seed, get_projector_stats, march_and_extract, render_sandbox, create_projector, create_light_field
from options import config_parser
import logging
import gsoup
from data.data_loader import SubjectLoader
from nerfacc import ContractionType, OccupancyGrid
from scipy.spatial.transform import Rotation as R
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

if __name__ == "__main__":
    args = config_parser()
    logging.basicConfig(level="INFO")
    set_random_seed(args.seed)
    # setup the scene bounding box.
    contraction_type = ContractionType.AABB
    scene_aabb = torch.tensor(args.aabb, dtype=torch.float32, device=args.device)
    near_plane = None
    far_plane = None
    render_step_size = (
        (scene_aabb[3:] - scene_aabb[:3]).max()
        * math.sqrt(3)
        / args.render_n_samples
    ).item()

    # setup the dataset
    opt_cam = {"optimize_cams": args.opt_cams, "force_opt_all": False}
    opt_cam["opt_over"] = "lollipop" if args.opt_cams else None
    opt_cam["dont_opt_over"] = None
    opt_cam["interpolate"] = args.colmap_mode == "video"
    opt_cam["force_opt_none"] = not args.opt_cams
    train_dataset = SubjectLoader(
        data_dir=Path(args.datadir, args.scene),
        split="train",
        device=args.device,
        num_rays=args.target_sample_batch_size // args.render_n_samples,
        color_bkgd_aug="random",
        divide_res=args.divide_res,
        opt_cam=opt_cam,
        colmap_views=args.colmap_views,
        colmap_mode=args.colmap_mode,
        post_added_views=args.post_added_views,
    )
    opt_cam_test = {"optimize_cams": False, "opt_over": None, "dont_opt_over": None,
                    "force_opt_all": False, "force_opt_none": False,
                    "interpolate": False}
    test_dataset = SubjectLoader(
        data_dir=Path(args.datadir, args.test_scene),
        split="test",
        num_rays=None,
        device=args.device,
        divide_res=args.divide_res,
        opt_cam=opt_cam_test
    )
    all_black_index = np.where(train_dataset.texture_names == "all_black")[0]
    if all_black_index.size != 0:
        all_black_index = all_black_index.item()
    if args.nepmap:
        radiance_field = ProjectorRadianceField(final_step=args.max_step,
                                                geo_freq=args.geo_freq,
                                                mat_freq=args.mat_freq,
                                                net_width=args.net_width).to(args.device)
        train_retvals = ["rgb", "opacity", "cam_transm", "n_rendering_samples"]  # "normal_map", "pred_normal_map", "depth", "roughness"
        # if not train_dataset.is_blender:
        test_retvals = ["rgb", "opacity", "depth", "pred_normal_map", "albedo", "roughness"]
        train_retvals += ["diff_normals", "pred_normals"]
        if args.projectors:
            train_retvals += ["pred_cam_transm", "n_pred_cam_transm"]
            test_retvals += ["pred_proj_transm_map", "sampled_texture_map", "visible_texture_map"]
    else:
        radiance_field = VanillaNeRFRadianceField().to(args.device)
        train_retvals = ["rgb", "opacity", "n_rendering_samples"]  # "normal_map",
        test_retvals = ["rgb", "opacity"]  # "normal_map",
    
    # setup the radiance field we want to train. 
    opt_group = {"net": 0}
    grad_vars = list(radiance_field.mlp.parameters())
    if args.nepmap:
        grad_vars += list(radiance_field.mat_mlp.parameters())
    optimizer = torch.optim.Adam(grad_vars, lr=args.lr)
    if args.nepmap:
        opt_group["vis"] = len(opt_group.keys())
        optimizer.add_param_group({'params': list(radiance_field.vis_network.parameters()),
                                'lr': args.lr, 'name': 'vis'})
    step = 0
    projectors = None
    cameras = None
    coloc_light = None
    ckpt = None
    ckpts = [str(file_path) for file_path in sorted(args.experiment_folder.glob("*.tar"))]
    if args.checkpoint:
        ckpts = [args.checkpoint]
    if len(ckpts) > 0:
        ckpt_path = ckpts[-1]
        logging.info('Reloading from: {}'.format(ckpt_path))
        ckpt = torch.load(ckpt_path, map_location=args.device)
        if not args.just_load_networks:
            if "step" in ckpt.keys():
                step = ckpt['step'] + 1
            if "projectors" in ckpt.keys():
                projectors = ckpt["projectors"]
                projectors[0]["textures"] = train_dataset.textures  # override
                # projectors[0]["v"] = train_dataset.v_proj  # override
                # projectors[0]["t"] = train_dataset.t_proj  # override
            if "coloc_light" in ckpt.keys():
                coloc_light = ckpt["coloc_light"]
                if args.freeze_coloc:
                    coloc_light.requires_grad = False
            if "cameras" in ckpt.keys():
                cameras = ckpt["cameras"]
    vanilla_radiance_field = None
    if args.nerf_checkpoint:
        nerf_ckpt = torch.load(args.nerf_checkpoint, map_location=args.device)
        vanilla_radiance_field = VanillaNeRFRadianceField().to(args.device)
        vanilla_radiance_field.load_state_dict(nerf_ckpt["radiance_field"])
        for param in vanilla_radiance_field.parameters():
            param.requires_grad = False
        train_retvals += ["vanilla_diff"]
    
    if args.render_only:  # change exp dir to render only
        args.experiment_folder = Path(args.experiment_folder, "render_only")
    if args.force_step >= 0:
        step = args.force_step
    initial_step = step
    if args.projectors:
        if projectors is None:
            projector = {}
            projector = create_projector(train_dataset.K_proj, train_dataset.v_proj, train_dataset.t_proj,
                                         args.proj_w, args.proj_h, train_dataset.textures, amp=args.proj_amp, device=args.device)
            projectors = [projector]
        if initial_step == 0:
            path = Path(args.experiment_folder, 'bundle_orig.tar')  # save the original camera poses & projector
            Rt, K_proj = get_projector_stats(projectors[0])
            np.savez(str(path),
                    cam_rt=np.stack(gsoup.to_np(train_dataset.orig_camtoworlds)[None, ...]),
                    proj_rt=Rt[None, ...], proj_k=K_proj[None, ...])
    if args.projectors:
        for projector in projectors:
            if args.projector_add_noise and not args.render_only:
                if train_dataset.is_blender:
                    projector["t"] = (projector["t"] + torch.randn_like(projector["t"])*0.1).detach().clone()
                    projector["v"] = (projector["v"] + torch.randn_like(projector["v"])*0.1).detach().clone()
                else:
                    projector["t"] = torch.tensor([0.5, -0.5, 0.5], device=args.device)  # place arbitrary in unit cube corner
                    rot = gsoup.look_at_torch(projector["t"],
                                            torch.zeros(3, device=args.device),
                                            torch.tensor([0.0, 0.0, 1.0], device=args.device))
                    projector["v"] = gsoup.mat2qvec(rot[:3, :3]).detach().clone()
                    # projector["t"] = (projector["t"] + torch.randn_like(projector["t"])*0.05).detach().clone()
                    # projector["v"] = (projector["v"] + torch.randn_like(projector["v"])*0.05).detach().clone()
            if args.projector_force_value:
                pass
                # projector["amp"] = torch.full((1, ), 15.0, dtype=torch.float32, device=args.device)
                # projector["gamma"] = torch.full((1, ), 2.2, dtype=torch.float32, device=args.device)
                # projector["f"] = torch.full_like(projector["f"], 1.9, device=args.device)
                # projector["cy"] = torch.full_like(projector["cy"], 0.7, device=args.device)
            projector["t"].requires_grad = True
            projector["v"].requires_grad = True
            projector["gamma"].requires_grad = True
            # projector["amp"].requires_grad = False
            # projector["amp"] = torch.full((1, ), 3.0, dtype=torch.float32, device=args.device)
            projector["amp"].requires_grad = True
            projector["cx"].requires_grad = True
            projector["cy"].requires_grad = True
            projector["f"].requires_grad = True
            # if args.rotation_is_qvec:
            #     R_proj = gsoup.rotvec2mat(projector["v"]) 
            #     projector["v"] = gsoup.mat2qvec(R_proj).detach().clone()
            #     projector["v"].requires_grad = True
            if args.freeze_projector:
                for key in projector.keys():
                    if type(projector[key]) == torch.Tensor:
                        projector[key].requires_grad = False
            if train_dataset.is_blender:
                geo_params = [projector["t"], projector["v"]]
            else:
                geo_params = [projector["t"], projector["v"], projector["cx"], projector["cy"], projector["f"]]
            col_params = [projector["gamma"], projector["amp"]]
            opt_group["proj_geo"] = len(opt_group.keys())
            opt_group["proj_col"] = len(opt_group.keys())
            optimizer.add_param_group({'params': geo_params, 'lr': 3e-3, 'name': 'projector_geo_params'})
            optimizer.add_param_group({'params': col_params, 'lr': 3e-3, 'name': 'projector_col_params'})
    if args.nepmap:
        if args.coloc:
            if coloc_light is None:
                coloc_light = torch.tensor([args.coloc_amp], dtype=torch.float32, device=args.device)
            if not args.freeze_coloc:
                coloc_light.requires_grad = True
            opt_group["coloc"] = len(opt_group.keys())
            optimizer.add_param_group({'params': [coloc_light], 'lr': 5e-3, 'name': 'coloc_light'})
    
    if args.opt_cams:
        if cameras is None:
            device = args.device
            t_cams = gsoup.to_numpy(train_dataset.camtoworlds[train_dataset.cam_opt_mask, :3, -1])
            v_cams = np.empty((train_dataset.camtoworlds[train_dataset.cam_opt_mask].shape[0], 3), dtype=np.float32)
            for i in range(train_dataset.camtoworlds[train_dataset.cam_opt_mask].shape[0]):
                R_cam = gsoup.to_numpy(train_dataset.camtoworlds[train_dataset.cam_opt_mask][i, :3, :3])
                r = R.from_matrix(R_cam)
                v_cam = r.as_rotvec().astype(np.float32)
                v_cams[i] = v_cam
            t_cams = torch.tensor(t_cams, dtype=torch.float32, device=device)
            v_cams = torch.tensor(v_cams, dtype=torch.float32, device=device)
            cameras = [t_cams, v_cams]
        if args.cameras_add_noise and not args.render_only:
            cameras[0] = (cameras[0] + torch.randn_like(cameras[0])*0.02).detach().clone()
        cameras[0].requires_grad = True
        cameras[1].requires_grad = True
        opt_group["cams"] = len(opt_group.keys())
        optimizer.add_param_group({'params': cameras, 'lr': args.lr, 'name': 'cameras'})
        train_dataset.cameras = cameras
    if args.projectors and initial_step == 0:
        path = Path(args.experiment_folder, 'bundle_init.tar')
        Rt, K_proj = get_projector_stats(projectors[0])
        np.savez(str(path),
                cam_rt=np.stack(gsoup.to_np(train_dataset.get_current_cameras())[None, ...]),
                proj_rt=Rt[None, ...], proj_k=K_proj[None, ...])
        # test_dataset.cameras = cameras
    light_field = create_light_field(projectors=projectors, coloc_light=coloc_light, inverse_square=args.inverse_square) # train_dataset.textures
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9998)
    milestones = [int(x) for x in args.sch_milestones.split()]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=milestones,
        gamma=0.25,
    )
    if args.freeze_radiance:
        for param in radiance_field.parameters():
            param.requires_grad = False
        if not args.freeze_vis:
            for param in radiance_field.vis_network.parameters():
                param.requires_grad = True
    if args.freeze_vis:
        for param in radiance_field.vis_network.parameters():
            param.requires_grad = False
    grad_scaler = torch.cuda.amp.GradScaler(1)
    occupancy_grid = OccupancyGrid(
        roi_aabb=args.aabb,
        resolution=args.grid_resolution,
        contraction_type=contraction_type,
    ).to(args.device)

    if ckpt is not None:
        if radiance_field is not None and "radiance_field" in ckpt:
            radiance_field.load_state_dict(ckpt["radiance_field"])
        if occupancy_grid is not None and "occupancy_grid" in ckpt:
            occupancy_grid.load_state_dict(ckpt["occupancy_grid"])
    if args.nerf_checkpoint:
        occupancy_grid.load_state_dict(nerf_ckpt["occupancy_grid"])
    if args.render_only:
        if args.render_modes is None:
            modes = ["train_set"] # "test_set", "train_set", "t2t", "compensate" , "dual_photo"
        else:
            modes = args.render_modes.strip().split(",")
        for param in radiance_field.parameters():
            param.requires_grad = False
        for mode in modes:
            if mode == "multi_t2t":
                texture = torch.ones((3, 800, 800), dtype=torch.float32, device=args.device)
                texture[1, :, :] = 0
                texture[2, :, :] = 0
                extra_info = {"coloc_light": False, "proj_texture": texture,  # "all_white"
                              "cam_index": [115, 61, 97],#batman:[28, 286]
                                "prompt": ["A beutiful red pear.",
                                           "A beutiful red pear.",
                                           "A beutiful red pear."],
                                "t_in": [0.0, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5],
                                "t_out": None,
                                "brightness": -50,
                                }
                proj_retvals = ["rgb", "pred_proj_transm_map", "opacity", "pred_normals"]
                render_sandbox(radiance_field, occupancy_grid, scene_aabb,
                            render_step_size / 4, proj_retvals, light_field,
                            test_dataset, train_dataset, args, prefix="ro_mt2t", mode="multi_t2t", extra_info=extra_info)
            if mode == "t2t":
                # color = torch.tensor([79, 40, 15], dtype=torch.float32, device=args.device) / 255
                # color = color[:, None, None].repeat(1, int(light_field["projectors"][0]["H"]), int(light_field["projectors"][0]["W"]))
                extra_info = {"coloc_light": False, "proj_texture": "all_white", "cam_index": [28, 37],
                                "prompt": ["Side profile of Abraham Lincoln",
                                           "Stone sculpture of Abraham Lincoln"],
                                "t_in": [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5],
                                "t_out": None,
                                "brightness": -50,
                                }
                proj_retvals = ["rgb", "pred_proj_transm_map", "opacity", "pred_normals"]
                render_sandbox(radiance_field, occupancy_grid, scene_aabb,
                            render_step_size / 4, proj_retvals, light_field,
                            test_dataset, train_dataset, args, prefix="ro_t2t", mode="t2t", extra_info=extra_info)
            elif mode == "compensate":
                extra_info = {"coloc_light": False,
                              "image_paths": [x for x in Path("./resource/compensation_raw").glob("*")],
                              "cam_index": 253}
                render_sandbox(radiance_field, occupancy_grid, scene_aabb,
                            render_step_size / 2, ["rgb"], light_field,
                            test_dataset, train_dataset, args, prefix="ro_compensate", mode="compensate", extra_info=extra_info)
            elif mode == "train_set":
                if args.frames_for_render is not None:
                    frames_to_render = np.array([int(x) for x in args.frames_for_render.split()])
                    mask = np.array(frames_to_render) >= len(train_dataset)
                    frames_to_render[mask] = 0
                else:
                    frames_to_render = [0]
                extra_info = {"frames_to_render": frames_to_render}
                train_set_retvals = test_retvals.copy()
                render_sandbox(radiance_field, occupancy_grid, scene_aabb,
                                render_step_size / 2, train_set_retvals, light_field,
                                test_dataset, train_dataset, args, prefix="ro_train_set", mode="train_set", extra_info=extra_info)
            elif mode == "test_set":
                if args.frames_for_render is not None:
                    frames_to_render = np.array([int(x) for x in args.frames_for_render.split()])
                    mask = np.array(frames_to_render) >= len(train_dataset)
                    frames_to_render[mask] = 0
                else:
                    frames_to_render = np.array([110,74,179,167,50,92,278,272,251,5,221,71,107,17,257,56,182,44,287,269,29,188,77,212,62,83,227,233,14,32,176,68,170,101,173,206])
                    # projector_on_frames = np.arange(2, len(test_dataset), 3)
                    # frames_to_render = np.random.choice(projector_on_frames,
                    #                                     size=36,
                    #                                     replace=False)
                print("frames_to_render: ", frames_to_render)
                extra_info = {"frames_to_render": frames_to_render,
                              "no_grid": False}
                train_set_retvals = test_retvals.copy()
                psnrs = render_sandbox(radiance_field, occupancy_grid, scene_aabb,
                                render_step_size / 4, train_set_retvals, light_field,
                                test_dataset, train_dataset, args, prefix="ro_test_set", mode="test_set", extra_info=extra_info)
                print("psnrs: ", np.mean(psnrs))
            elif mode == "test_set_movie":
                extra_info = {"stride": 10}
                test_set_movie_retvals = test_retvals.copy()
                render_sandbox(radiance_field, occupancy_grid, scene_aabb,
                            render_step_size / 2, test_set_movie_retvals, light_field,
                            test_dataset, train_dataset, args, prefix="ro_test_stream", mode="test_set_movie", extra_info=extra_info)
            elif mode == "move_projector":
                if args.projectors:
                    extra_info = {"coloc_light": False, "proj_texture": "all_white",
                                "cam_index": 28, "n_frames": 30, "plane": "xy"}
                    proj_retvals = ["rgb", "pred_proj_transm_map", "visible_texture_map"]
                    render_sandbox(radiance_field, occupancy_grid, scene_aabb,
                                render_step_size / 2, proj_retvals, light_field,
                                test_dataset, train_dataset, args, prefix="ro_move_proj", mode="move_projector", extra_info=extra_info)
            elif mode == "move_camera":
                extra_info = {"coloc_light": True,
                            "proj_texture": "all_white",
                            "n_frames": 30, "plane": "xy"}
                move_cam_retvals = test_retvals.copy()
                render_sandbox(radiance_field, occupancy_grid, scene_aabb,
                                render_step_size / 2, move_cam_retvals, light_field,
                                test_dataset, train_dataset, args, prefix="ro_move_cam", mode="move_camera", extra_info=extra_info)
            elif mode == "train_set_movie":
                if args.opt_cams:
                    extra_info = {"stride": 10}
                    train_set_movie_retvals = test_retvals.copy()
                    render_sandbox(radiance_field, occupancy_grid, scene_aabb,
                            render_step_size / 2, train_set_movie_retvals, light_field,
                            test_dataset, train_dataset, args, prefix="ro_train_stream", mode="train_set_movie", extra_info=extra_info)
            elif mode == "projector_calib":
                extra_info = {"cam_index": 35}
                gamma_and_amp_retvals = ["rgb"]
                render_sandbox(radiance_field, occupancy_grid, scene_aabb,
                                render_step_size / 2, gamma_and_amp_retvals, light_field,
                                test_dataset, train_dataset, args, prefix="proj_calib", mode="projector_calib", extra_info=extra_info)
            elif mode == "dual_photo":
                extra_info = {"coloc_light": False,
                            "proj_texture": "all_white",
                            "cam_index": 273,
                            "xray": True}
                dual_photo_retvals = test_retvals.copy()
                render_sandbox(radiance_field, occupancy_grid, scene_aabb,
                                render_step_size / 8, dual_photo_retvals, light_field,
                                test_dataset, train_dataset, args, prefix="ro_dp", mode="dual_photo", extra_info=extra_info)
            elif mode == "play_vid":
                if args.projectors:
                    extra_info = {"cam_index": 20, "proj_index": 20, "stride": 4, "vid_path": "./resource/test.mp4", "animate_cam": "xz"}
                    play_vid_retvals = ["rgb"]
                    render_sandbox(radiance_field, occupancy_grid, scene_aabb,
                                render_step_size / 4, play_vid_retvals, light_field,
                                test_dataset, train_dataset, args, prefix="ro_project_vid", mode="play_vid", extra_info=extra_info)
        exit(0)
    # training
    if ckpt is not None and not args.just_load_networks:
        try:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        except ValueError:
            print("Optimizer state not loaded")
    for i in range(step - 1):  # skip ahead step - 1 iterations
        scheduler.step()
    tic = time.time()
    proj_rt = []
    proj_k = []
    cam_rt = []
    stat_accum = {"imloss": [],
                  "amp": [],
                  "coloc": [],
                  "gamma": [],
                  "psnr": [],
                  "r": [],
                  "f": []}
    transm_loss = 0.0
    phase = -1
    phase_swap = False
    for epoch in range(10000000):
        for i in range(len(train_dataset)):
            radiance_field.train()
            ### todo: move to scheduler class
            if 0 <= step < args.phase1 and args.phase1 > 0:
                cur_phase = 0
            elif args.phase1 <= step < args.phase2 and args.phase2 > 0:
                cur_phase = 1
            elif args.phase2 <= step < args.phase3 and args.phase3 > 0:
                cur_phase = 2
            else:
                cur_phase = 3
            if cur_phase != phase:
                phase_swap = True
                phase = cur_phase
            if phase_swap:
                phase_swap = False
                if phase == 0 or phase == 1:  # optimize initial geometry
                    for param in optimizer.param_groups[opt_group["net"]]["params"]:
                        param.requires_grad = True
                    if "vis" in opt_group:
                        for param in optimizer.param_groups[opt_group["vis"]]["params"]:
                            param.requires_grad = True
                    if "coloc" in opt_group:
                        for param in optimizer.param_groups[opt_group["coloc"]]["params"]:
                            param.requires_grad = True
                    if "proj_geo" in opt_group:
                        for param in optimizer.param_groups[opt_group["proj_geo"]]["params"]:
                            param.requires_grad = False
                    if "proj_col" in opt_group:
                        for param in optimizer.param_groups[opt_group["proj_col"]]["params"]:
                            param.requires_grad = False
                    if "cams" in opt_group:
                        for param in optimizer.param_groups[opt_group["cams"]]["params"]:
                            param.requires_grad = False
                elif phase == 2:  # optimize optical elements
                    for param in optimizer.param_groups[opt_group["net"]]["params"]:
                        param.requires_grad = False
                    for param in optimizer.param_groups[opt_group["vis"]]["params"]:
                        param.requires_grad = False
                    for param in optimizer.param_groups[opt_group["coloc"]]["params"]:
                        param.requires_grad = False
                    for param in optimizer.param_groups[opt_group["proj_geo"]]["params"]:
                        param.requires_grad = True
                    for param in optimizer.param_groups[opt_group["proj_col"]]["params"]:
                        param.requires_grad = False
                    if "cams" in opt_group:
                        for param in optimizer.param_groups[opt_group["cams"]]["params"]:
                            param.requires_grad = True
                        optimizer.param_groups[opt_group["cams"]]['lr'] = args.lr / 4
                    train_dataset.use_random_cams = False
                    train_dataset.only_static_views = False
                    train_dataset.only_black_views = False
                    if train_dataset.is_blender:
                        optimizer.param_groups[opt_group["proj_geo"]]['lr'] = args.lr
                    else:
                        optimizer.param_groups[opt_group["proj_geo"]]['lr'] = args.lr
                elif phase == 3:  # finetune all
                    train_dataset.use_random_cams = False
                    train_dataset.only_static_views = False
                    train_dataset.only_black_views = False
                    # train_dataset.no_color_views = True
                    # train_dataset.only_white_views = True
                    for param in optimizer.param_groups[opt_group["net"]]["params"]:
                        param.requires_grad = True
                    # if not args.nerf_checkpoint:
                    for param in radiance_field.mlp.parameters():
                        param.requires_grad = True #False
                    for param in optimizer.param_groups[opt_group["vis"]]["params"]:
                        param.requires_grad = True
                    for param in optimizer.param_groups[opt_group["coloc"]]["params"]:
                        param.requires_grad = True
                    for param in optimizer.param_groups[opt_group["proj_geo"]]["params"]:
                        param.requires_grad = True
                    for param in optimizer.param_groups[opt_group["proj_col"]]["params"]:
                        param.requires_grad = True
                    if "cams" in opt_group:
                        for param in optimizer.param_groups[opt_group["cams"]]["params"]:
                            param.requires_grad = True
                        optimizer.param_groups[opt_group["cams"]]['lr'] = args.lr / 16
                    optimizer.param_groups[opt_group["net"]]['lr'] = args.lr / 32
                    optimizer.param_groups[opt_group["vis"]]['lr'] = args.lr / 32
                    optimizer.param_groups[opt_group["proj_geo"]]['lr'] = args.lr / 4
                    optimizer.param_groups[opt_group["proj_col"]]['lr'] = args.lr
                    optimizer.param_groups[opt_group["coloc"]]['lr'] = args.lr / 4
                else:
                    pass
            data = train_dataset[i]
            render_bkgd = data["color_bkgd"]
            rays = data["rays"]
            pixels = data["pixels"]
            texture_ids = data["texture_ids"]
            # update grid
            if not args.nerf_checkpoint:
                occupancy_grid.every_n_step(
                    step=step,
                    occ_eval_fn=lambda x: radiance_field.query_opacity(
                        x, render_step_size)
                    
                    )
            # render
            result = march_and_extract(
                radiance_field,
                rays,
                scene_aabb,
                occupancy_grid=occupancy_grid,
                render_step_size=render_step_size,
                render_bkgd=render_bkgd,
                ret_vals=train_retvals,
                is_relightable=args.relightable,
                light_field=light_field,
                texture_ids=texture_ids,
                cur_step=step,
                only_transmittance=False,
                vanilla_radiance_field=vanilla_radiance_field,
            )
            n_rendering_samples = result["n_rendering_samples"].sum().item()
            if n_rendering_samples == 0:
                print("step: {} no render samples".format(step))
                continue
            # dynamic batch size for rays to keep sample batch size constant.
            num_rays = len(pixels)
            num_rays = int(
                num_rays * (args.target_sample_batch_size / float(n_rendering_samples))
            )
            train_dataset.update_num_rays(num_rays)
            alive_ray_mask = result["opacity"].squeeze(-1) > 0
            if not alive_ray_mask.any() and phase == 0:
                print("step: {} No alive rays".format(step))
                random_indices = torch.randint(size=(alive_ray_mask.shape[0]//2,), high=alive_ray_mask.shape[0])
                alive_ray_mask[random_indices] = True
            # black_rays_mask = texture_ids[:, 0][alive_ray_mask] == all_black_index
            opt_weights = torch.ones_like(result["rgb"][alive_ray_mask][:, 0:1])
            # opt_weights[~black_rays_mask] /= 5
            ############################################## img loss ####################################################
            img_loss = F.smooth_l1_loss(result["rgb"][alive_ray_mask], pixels[alive_ray_mask], reduction='none')
            img_loss = (img_loss*opt_weights).mean()
            psnr = -10.0 * torch.log(img_loss) / np.log(10.0)
            stat_accum["imloss"].append(img_loss.item())
            stat_accum["psnr"].append(psnr.item())
            ############################################## nerf loss ###############################################
            vanilla_loss = 0.0
            if "vanilla_diff" in result.keys():
                vanilla_loss = args.vanilla_loss_coeff*torch.sum(result["vanilla_diff"])
            ############################################## fog loss ####################################################
            fog_loss = 0.0
            if "cam_transm" in result.keys():
                b = 8  # increase for steeper parbola
                fog_loss = args.fog_loss_coeff * torch.mean(-b*((result["cam_transm"]-0.5)**2) + b/4)
            ############################################## cam loss ####################################################
            camera_loss = 0.0
            if args.opt_cams:
                t = train_dataset.get_current_cameras()[:, :3, 3]
                t_orig = train_dataset.camtoworlds[:, :3, 3]
                camera_loss = args.cam_loss_coeff * torch.mean((t - t_orig).norm(dim=-1))
            ############################################## normal loss ####################################################
            normal_loss = 0.0
            if "pred_normals" in result.keys():
                dot_product = (result["pred_normals"][alive_ray_mask][:, None, :] @ rays.viewdirs[alive_ray_mask][:, :, None]).squeeze()
                normal_loss = args.normal_loss_coeff * torch.mean(torch.maximum(torch.zeros_like(dot_product), dot_product))
            ############################################## normal loss 2 ####################################################
            normal_loss2 = 0.0
            if "diff_normals" in result.keys():
                normal_loss2 = args.normal_loss2_coeff*torch.mean(result["diff_normals"][alive_ray_mask])
            ############################################## visibility loss ############################################# 
            transm_loss = 0.0
            if "pred_cam_transm" in result.keys():
                transm_loss = torch.mean(((result["pred_cam_transm"] - result["cam_transm"].detach())**2))#*transm_w[:, None, None])
                if "n_pred_cam_transm" in result.keys():
                    transm_loss += torch.mean(((result["n_pred_cam_transm"] - result["pred_cam_transm"])**2))
            # log stats
            if "projectors" in light_field:
                stat_accum["gamma"].append(light_field["projectors"][0]["gamma"].item())
                stat_accum["amp"].append(light_field["projectors"][0]["amp"].item())
                stat_accum["f"].append(light_field["projectors"][0]["f"].item())
                stat_accum["r"].append(light_field["projectors"][0]["t"].norm().item())
            if "coloc_light" in light_field:
                stat_accum["coloc"].append(light_field['coloc_light'].item())
            if projectors is not None:
                Rt, K_proj = get_projector_stats(projectors[0])
                proj_rt.append(Rt)
                proj_k.append(K_proj)
            if cameras is not None:
                c2w = train_dataset.get_current_cameras()
                cam_rt.append(gsoup.to_numpy(c2w))
            # compute loss according to phase
            if phase == 0:  # geometry
                if args.nerf_checkpoint:
                    loss = img_loss + transm_loss + normal_loss + normal_loss2 + fog_loss + vanilla_loss
                else:
                    loss = img_loss + transm_loss #+ normal_loss + normal_loss2 + fog_loss
            elif phase == 1:  # geometry
                loss = img_loss + transm_loss + normal_loss + normal_loss2 + fog_loss + vanilla_loss
            elif phase == 2:  # projector + cameras
                loss = img_loss + camera_loss
            elif phase == 3:  # refine
                if args.nerf_checkpoint:
                    loss = img_loss + transm_loss + normal_loss + normal_loss2 + fog_loss
                else:
                    loss = img_loss + transm_loss + normal_loss + normal_loss2 + fog_loss
            optimizer.zero_grad()
            grad_scaler.scale(loss).backward()
            optimizer.step()
            scheduler.step()
            # print & plot stats
            if step != initial_step and step % args.print_step == 0:
                elapsed_time = time.time() - tic
                for key in stat_accum.keys():
                    data = np.array(stat_accum[key])
                    if data.shape[0] % args.print_step != 0:
                        data = data[1:]
                    data = data.reshape(-1, args.print_step).mean(axis=1)
                    plt.plot(data)
                    plt.ylabel(key)
                    plt.savefig(str(Path(args.experiment_folder, "{}.png".format(key))))
                    plt.close()
                my_str = \
                    "step={} | elapsed_time={:.2f}s | phase={} |" \
                    " psnr={:.4f} |" \
                    " img_loss={:.5f} |" \
                    " lr_nets={:.2e} |" \
                    " transm_loss={:.5f} |" \
                    " fog_loss={:.5f} |" \
                    " vanilla_loss={:.5f} |" \
                    " normal_loss={:.5f} |" \
                    " alive_ray_mask={:d} |" \
                    " n_rendering_samples={:d} | num_rays={:d} |".format(
                        step, elapsed_time, phase, psnr,
                        img_loss, optimizer.param_groups[opt_group["net"]]['lr'],
                        transm_loss, fog_loss, vanilla_loss, normal_loss,
                        alive_ray_mask.long().sum(), n_rendering_samples, len(pixels)
                    )
                if args.nepmap and args.projectors:
                    my_str += "lr_vis={:.2e}|" \
                              "lr_proj_geo={:.2e} |" \
                              "lr_proj_col={:.2e} |" \
                              "lr_coloc={:.2e} |".format(
                        optimizer.param_groups[opt_group["vis"]]['lr'],
                        optimizer.param_groups[opt_group["proj_geo"]]['lr'],
                        optimizer.param_groups[opt_group["proj_col"]]['lr'],
                        optimizer.param_groups[opt_group["coloc"]]['lr']
                    )
                if "projectors" in light_field:
                    proj_t = light_field["projectors"][0]["t"]
                    proj_v = light_field["projectors"][0]["v"]
                    if proj_v.shape[0] == 4:
                        dir_str = f"projector_dir={proj_v[0]:.2f},{proj_v[1]:.2f},{proj_v[2]:.2f}, {proj_v[3]:.2f} | "
                    else:
                        dir_str = f"projector_dir={proj_v[0]:.2f},{proj_v[1]:.2f},{proj_v[2]:.2f} | "
                    gamma = light_field["projectors"][0]["gamma"][0]
                    amp = light_field["projectors"][0]["amp"][0]
                    f = light_field["projectors"][0]["f"][0]
                    cx = light_field["projectors"][0]["cx"][0]
                    cy = light_field["projectors"][0]["cy"][0]
                    my_str += f"projector_loc={proj_t[0]:.2f},{proj_t[1]:.2f},{proj_t[2]:.2f} | "\
                              + dir_str +\
                              f"projector_gamma={gamma:.2f} | "\
                              f"projector_amp={amp:.2f} | "\
                              f"projector_fcxcy={f:.3f},{cx:.3f},{cy:.3f} | "
                if "coloc_light" in light_field:
                    my_str += f"coloc_amp={light_field['coloc_light'][0]:.2f} | "
                logging.info(my_str)
            
            # save checkpoint
            if step != initial_step and step % args.save_step == 0:
                path = Path(args.experiment_folder, 'latest.tar')
                path_phase = Path(args.experiment_folder, '0_latest_{:02d}.tar'.format(phase))
                my_dict = {
                    'step': step,
                    'optimizer_state_dict': optimizer.state_dict()
                }
                if radiance_field is not None:
                    my_dict["radiance_field"] = radiance_field.state_dict()
                if occupancy_grid is not None:
                    my_dict["occupancy_grid"] = occupancy_grid.state_dict()
                if projectors is not None:
                    my_dict["projectors"] = projectors
                if coloc_light is not None:
                    my_dict["coloc_light"] = coloc_light
                if cameras is not None:
                    my_dict["cameras"] = cameras
                torch.save(my_dict, str(path))
                torch.save(my_dict, str(path_phase))
                logging.info('Saved checkpoints at {}'.format(path))
                bundle_path = Path(args.experiment_folder, 'bundle_stats_{:06d}.tar'.format(step))
                if proj_rt or cam_rt:
                    cam_stack = np.stack(cam_rt) if cam_rt else []
                    proj_stack = np.stack(proj_rt) if proj_rt else []
                    proj_k_stack = np.stack(proj_k) if proj_k else []
                    np.savez(str(bundle_path), cam_rt=cam_stack, proj_rt=proj_stack, proj_k=proj_k_stack)
                    cam_rt = []
                    proj_rt = []
                    proj_k = []
                # occ_path = Path(args.experiment_folder, 'occ_grid_{:06d}.tar'.format(step))
                # np.savez(str(occ_path), cords=occupancy_grid.grid_coords.cpu().numpy(), vals=occupancy_grid.occs.cpu().numpy())
            
            # run inference on some training views
            if step != initial_step and step % args.test_step == 0:
                if args.frames_for_render is not None:
                    frames_to_render = np.array([int(x) for x in args.frames_for_render.split()])
                    mask = np.array(frames_to_render) >= len(train_dataset)
                    frames_to_render[mask] = 0
                else:
                    frames_to_render = [0]
                extra_info = {"frames_to_render": frames_to_render}
                psnrs = render_sandbox(radiance_field, occupancy_grid, scene_aabb,
                                        render_step_size / 2, test_retvals, light_field,
                                        test_dataset, train_dataset, args, prefix="{:05d}".format(step), mode="train_set", extra_info=extra_info)
                psnr_avg = sum(psnrs) / len(psnrs)
                logging.info(f"evaluation: psnr_avg={psnr_avg}")
            if step >= args.max_step:
                logging.info("training stops")
                exit()
            step += 1