from pathlib import Path
import struct
from PIL import Image
import json
import numpy as np
import torch
import collections
import os
import shutil
import gsoup
import cv2

def scale_poses(poses, n=4.0):
    avglen = np.mean(np.linalg.norm(poses[:, 0:3, 3], axis=-1))
    print("avg camera distance from origin", avglen)
    poses[:, 0:3, 3] *= n / avglen  # scale to "nerf sized"
    return poses, avglen

def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file.
    :param fid:
    :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    :param endian_character: Any of {@, =, <, >, !}
    :return: Tuple of read and unpacked values.
    """
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)


def read_colmap_raw_images_binary(path_to_model_file):
    """
    see https://github.com/colmap/colmap
    """
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            binary_image_properties = read_next_bytes(
                fid, num_bytes=64, format_char_sequence="idddddddi")
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":   # look for the ASCII 0 entry
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            num_points2D = read_next_bytes(fid, num_bytes=8,
                                           format_char_sequence="Q")[0]
            x_y_id_s = read_next_bytes(fid, num_bytes=24*num_points2D,
                                       format_char_sequence="ddq"*num_points2D)
            xys = np.column_stack([tuple(map(float, x_y_id_s[0::3])),
                                   tuple(map(float, x_y_id_s[1::3]))])
            point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
            images[image_id] = {"id": image_id,
                                "qvec": qvec,
                                "tvec": tvec,
                                "camera_id": camera_id,
                                "name": image_name,
                                "xys": xys,
                                "point3D_ids": point3D_ids}
    return images


def read_colmap_raw_images_text(path):
    """
    see https://github.com/colmap/colmap
    """
    images = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                image_id = int(elems[0])
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                camera_id = int(elems[8])
                image_name = elems[9]
                elems = fid.readline().split()
                xys = np.column_stack([tuple(map(float, elems[0::3])),
                                       tuple(map(float, elems[1::3]))])
                point3D_ids = np.array(tuple(map(int, elems[2::3])))
                images[image_id] = {"id": image_id,
                                    "qvec": qvec,
                                    "tvec": tvec,
                                    "camera_id": camera_id,
                                    "name": image_name,
                                    "xys": xys,
                                    "point3D_ids": point3D_ids}
    return images


def read_cameras_text(path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::WriteCamerasText(const std::string& path)
        void Reconstruction::ReadCamerasText(const std::string& path)
    """
    cameras = []
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                camera_id = int(elems[0])
                model = elems[1]
                width = int(elems[2])
                height = int(elems[3])
                params = np.array(tuple(map(float, elems[4:])))
                cameras.append({"id": camera_id,
                                "model": model,
                                "width": width,
                                "height": height,
                                "params": params})
    return cameras


def read_cameras_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::WriteCamerasBinary(const std::string& path)
        void Reconstruction::ReadCamerasBinary(const std::string& path)
    """
    CameraModel = collections.namedtuple(
        "CameraModel", ["model_id", "model_name", "num_params"])
    CAMERA_MODELS = {
        CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
        CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
        CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
        CameraModel(model_id=3, model_name="RADIAL", num_params=5),
        CameraModel(model_id=4, model_name="OPENCV", num_params=8),
        CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
        CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
        CameraModel(model_id=7, model_name="FOV", num_params=5),
        CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
        CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
        CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12)
    }
    CAMERA_MODEL_IDS = dict([(camera_model.model_id, camera_model)
                             for camera_model in CAMERA_MODELS])
    cameras = []
    with open(path_to_model_file, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_properties = read_next_bytes(
                fid, num_bytes=24, format_char_sequence="iiQQ")
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            model_name = CAMERA_MODEL_IDS[camera_properties[1]].model_name
            width = camera_properties[2]
            height = camera_properties[3]
            num_params = CAMERA_MODEL_IDS[model_id].num_params
            params = read_next_bytes(fid, num_bytes=8*num_params,
                                     format_char_sequence="d"*num_params)
            params = np.array(params)
            cameras.append({"id": camera_id,
                            "model": model_name,
                            "width": width,
                            "height": height,
                            "params": params})
        assert len(cameras) == num_cameras
    return cameras

def read_points3d_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DBinary(const std::string& path)
        void Reconstruction::WritePoints3DBinary(const std::string& path)
    """
    points3D = {}
    with open(path_to_model_file, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_points):
            binary_point_line_properties = read_next_bytes(
                fid, num_bytes=43, format_char_sequence="QdddBBBd")
            point3D_id = binary_point_line_properties[0]
            xyz = np.array(binary_point_line_properties[1:4])
            rgb = np.array(binary_point_line_properties[4:7])
            error = np.array(binary_point_line_properties[7])
            track_length = read_next_bytes(
                fid, num_bytes=8, format_char_sequence="Q")[0]
            track_elems = read_next_bytes(
                fid, num_bytes=8*track_length,
                format_char_sequence="ii"*track_length)
            image_ids = np.array(tuple(map(int, track_elems[0::2])))
            point2D_idxs = np.array(tuple(map(int, track_elems[1::2])))
            points3D[point3D_id] = {"id": point3D_id, "xyz": xyz, "rgb": rgb,
                            "error": error, "image_ids": image_ids,
                            "point2D_idxs": point2D_idxs}
    return points3D

    
def get_colmap_camera_extrinsics(images):
    c2ws = []
    names = []
    # for id_im in range(1, len(images)+1):
        # image = images[id_im]
    for image in images.values():
        names.append(image["name"])
        rot_mat = gsoup.qvec2mat(image["qvec"])
        t = -rot_mat.T @ image["tvec"]
        r = rot_mat.T
        pose = np.concatenate((r, t[:, None]), axis=1)
        pose = np.concatenate((pose, np.array([0, 0, 0, 1])[None, :]), axis=0)
        c2ws.append(pose)
    c2ws = np.array(c2ws)
    return c2ws, names


def get_colmap_camera_intrinsics(raw_cameras_file):
    if raw_cameras_file.suffix == ".txt":
        cameras = read_cameras_text(raw_cameras_file)
    elif raw_cameras_file.suffix == ".bin":
        cameras = read_cameras_binary(raw_cameras_file)
    else:
        raise NotImplementedError
    for camera in cameras:
        camera["K"] = np.array([[camera["params"][0], 0, camera["params"][2]],
                               [0, camera["params"][1], camera["params"][3]],
                               [0, 0, 0]])
        camera["k1"] = camera["params"][4]
        camera["k2"] = camera["params"][5]
        camera["p1"] = camera["params"][6]
        camera["p2"] = camera["params"][7]
    return cameras


def create_dummy_images(n, width, height, dst_folder):
    paths = []
    Path(dst_folder).mkdir(exist_ok=True, parents=True)
    for i in range(n):
        my_dummy_image = Image.fromarray(np.zeros((height, width, 3)).astype('uint8'))
        my_path = Path(dst_folder, "{:003d}.png".format(i))
        my_dummy_image.save(str(my_path))
        paths.append(my_path)
    return paths


def intrinsics_to_dict(camera_parameters=None):
    # focalx = 0.5 * width / np.tan(0.5 * camera_angle_x)
    # focaly = 0.5 * height / np.tan(0.5 * camera_angle_y)
    if camera_parameters is None:
        camera_parameters = {
            "width": 1280,
            "height": 800,
            "K": np.array([[500, 0, 640],
                           [0, 500, 600],
                           [0, 0, 1]]),
            "k1": 0.10011838101684062,
            "k2": -0.17057238136555566,
            "p1": -0.0007332714727649372,
            "p2": -0.001884604696989078,
        }
    camera_angle_x = 2 * np.arctan(0.5 * camera_parameters["width"] / camera_parameters["K"][0, 0])
    camera_angle_y = 2 * np.arctan(0.5 * camera_parameters["height"] / camera_parameters["K"][1, 1])
    intrinsics = {
        "camera_angle_x": camera_angle_x,
        "camera_angle_y": camera_angle_y,
        "fl_x": camera_parameters["K"][0, 0],
        "fl_y": camera_parameters["K"][1, 1],
        "k1": camera_parameters["k1"],
        "k2": camera_parameters["k2"],
        "p1": camera_parameters["p1"],
        "p2": camera_parameters["p2"],
        "cx": camera_parameters["K"][0, 2],
        "cy": camera_parameters["K"][1, 2],
        "w": camera_parameters["width"],
        "h": camera_parameters["height"],
        "aabb_scale": 1,
    }
    # if camera_parameters is None:
    #     K = np.array([
    #         [2369.0594062889395, 0, 1062.6341554841224],
    #         [0, 2370.0587198993017, 825.6182841325392],
    #         [0, 0, 1]
    #     ])
    #     camera_angle_x = 0.8273254395026632
    #     camera_angle_y = 0.6328347680237548
    # intrinsics = {
    #     "camera_angle_x": 0.8273254395026632,
    #     "camera_angle_y": 0.6328347680237548,
    #     "fl_x": 2369.0594062889395,
    #     "fl_y": 2370.0587198993017,
    #     "k1": 0.10011838101684062,
    #     "k2": -0.17057238136555566,
    #     "p1": -0.0007332714727649372,
    #     "p2": -0.001884604696989078,
    #     "cx": 1062.6341554841224,
    #     "cy": 825.6182841325392,
    #     "w": 2080.0,
    #     "h": 1552.0,
    #     "aabb_scale": 1,
    # }
    return intrinsics


def dump_dict_to_json(dict, output_file):
    with open(output_file, "w") as f:
        json.dump(dict, f, indent=4)


def nerf_to_np(file):
    with open(file, 'r') as fp:
        meta = json.load(fp)
    all_poses = []
    for i, frame in enumerate(sorted(meta['frames'], key=lambda x: x["file_path"])):
        if "transform_matrix" in frame:
            all_poses.append(np.array(frame['transform_matrix']))
        elif "RT" in frame:
            all_poses.append(np.array(frame['RT']))
        else:
            raise NotImplementedError
    camera_poses = np.array(all_poses).astype(np.float32)
    return camera_poses


def load_sensor_depth(images, path_to_points3d):
    # data_file = Path(basedir) / 'colmap_depth.npy'
    points = read_points3d_binary(path_to_points3d)

    Errs = np.array([x["error"] for x in points.values()])
    Err_mean = np.mean(Errs)
    print("Mean Projection Error:", Err_mean)
    poses, _ = get_colmap_camera_extrinsics(images)
    # _, bds_raw, _ = _load_data(basedir, factor=factor) # factor=8 downsamples original imgs by 8x
    # bds_raw = np.moveaxis(bds_raw, -1, 0).astype(np.float32)
    # print(bds_raw.shape)
    # Rescale if bd_factor is provided
    sc = 1.
    
    # near = np.ndarray.min(bds_raw) * .9 * sc
    # far = np.ndarray.max(bds_raw) * 1. * sc
    # print('near/far:', near, far)

    # depthfiles = [Path(basedir) / 'depth' / f for f in sorted(os.listdir(Path(basedir) / 'depth')) if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
    # depths = [imageio.imread(f) for f in depthfiles]
    # depths = np.stack(depths, 0)

    data_list = []
    for id_im in range(1, len(images)+1):
        depth_list = []
        coord_list = []
        weight_list = []
        for i in range(len(images[id_im]["xys"])):
            point2D = images[id_im]["xys"][i]
            id_3D = images[id_im]["point3D_ids"][i]
            if id_3D == -1:
                continue
            point3D = points[id_3D]["xyz"]
            depth = (poses[id_im-1,:3,2].T @ (point3D - poses[id_im-1,:3,3])) * sc
            # if depth < bds_raw[id_im-1,0] * sc or depth > bds_raw[id_im-1,1] * sc:
                # continue
            err = points[id_3D]["error"]
            weight = 2 * np.exp(-(err/Err_mean)**2)
            depth_list.append(depth)
            coord_list.append(point2D)
            weight_list.append(weight)
        if len(depth_list) > 0:
            print(id_im, len(depth_list), np.min(depth_list), np.max(depth_list), np.mean(depth_list))
            data_list.append({"depth":np.array(depth_list), "coord":np.array(coord_list), "weight":np.array(weight_list), "im_name":images[id_im]["name"]})
        else:
            print(id_im, len(depth_list))
    return data_list

def join_colmap_outputs(per_pattern_dir, img2view, fails, output_file):
    """
    joins colmap outputs for multiple patterns
    :param per_pattern_dir:
    :param output_file:
    :return:
    """
    my_dict = {}
    all_poses = {}
    all_ids = {}
    all_cameras = {}
    first_view_names = {}
    pattern_names = []
    # all_points = {}
    for pattern_dir in sorted(per_pattern_dir.glob("*")):
        pattern_name = pattern_dir.name
        if fails[pattern_name]:
            continue
        models = []
        for folder in Path(pattern_dir, "colmap_output").glob("*"):
            models.append(folder)
        
        if len(models) > 1:
            merged_model_path = Path(pattern_dir, "colmap_output", "merged")
            merged_model_path.mkdir(parents=True, exist_ok=True)
            if not Path(merged_model_path, "cameras.bin").exists():
                fail = do_system(f"colmap model_merger --input_path1 {str(models[0])} --input_path2 {str(models[1])} --output_path {str(merged_model_path)}")
                if fail:
                    print("Failed to merge colmap models")
                else:
                    fail = do_system(f"colmap bundle_adjuster --input_path {str(merged_model_path)} --output_path {str(merged_model_path)} --BundleAdjustment.refine_principal_point 1")
                    if fail:
                        print("Failed to refine colmap models")
        else:
            merged_model_path = models[0]
        images_path = Path(merged_model_path, "images.bin")
        cameras_path = Path(merged_model_path, "cameras.bin")
        # points_path = Path(pattern_dir, "colmap_output", "0", "points3D.bin")
        cameras = get_colmap_camera_intrinsics(cameras_path)
        images = read_colmap_raw_images_binary(images_path)
        image_key_name_pair = [[key, images[key]["name"]] for key in sorted(images.keys())]
        views_key_id_pair = [[x[0], img2view[x[1]]] for x in image_key_name_pair]
        views_only = [x[1] for x in views_key_id_pair]
        if not 0 in views_only:
            continue
        first_view_image_name = image_key_name_pair[views_only.index(0)][1]
        first_view_names[pattern_name] = first_view_image_name
        # first_view_key = views_key_id_pair[views_only.index(0)][0]
        # first_view_keys[pattern_dir.name] = first_view_key
        poses, ids = get_colmap_camera_extrinsics(images)
        all_cameras[pattern_name] = cameras
        all_poses[pattern_name] = poses
        all_ids[pattern_name] = ids
        pattern_names.append(pattern_name)
    
    first_view_poses = []
    for pattern in pattern_names:
        first_view_name = first_view_names[pattern]
        id = all_ids[pattern].index(first_view_name)
        pose = all_poses[pattern][id]
        first_view_poses.append(pose)
    new_transforms = [np.eye(4, dtype=np.float32)]
    for i in range(1, len(first_view_poses)):
        new_transforms.append(first_view_poses[0] @ np.linalg.inv(first_view_poses[i]))
    for i, pattern in enumerate(pattern_names):
        all_poses[pattern] = new_transforms[i] @ all_poses[pattern]
    
    poses = np.concatenate([all_poses[x] for x in pattern_names], axis=0)
    poses_patterns = np.concatenate([[x]*len(all_poses[x]) for x in pattern_names], axis=0)
    poses_ids = np.concatenate([all_ids[x] for x in pattern_names], axis=0)
    coa = gsoup.get_center_of_attention(poses)
    poses[:, :3, 3] -= coa
    # make z axis up (currently -y is up)
    xrot_mat = gsoup.rotx(-np.pi/2, degrees=False)
    poses = xrot_mat @ poses
    # scale to unit length ( or n times unit length)
    poses, factor = scale_poses(poses, n=1.0)
    my_dict.update(intrinsics_to_dict(all_cameras[pattern_names[0]][0]))
    my_dict.update({"transform_matrix_proj": poses[0].tolist()})
    # my_dict.update({"transform_matrix_proj": poses[cam_aprox_for_projector].tolist()})
    frames = []
    for i, pose in enumerate(poses):
        entry = {
            "file_path": poses_ids[i],
            "transform_matrix": pose.tolist(),
            "patterns": [poses_patterns[i]],
            "view_id": img2view[poses_ids[i]]
        }
        frames.append(entry)
    my_dict.update({"frames": frames})
    dump_dict_to_json(my_dict, output_file)

def colmap_to_nerf(colmap_root_dir, output_file, post_added_views=None):
    """
    converts colmap output to our format, considering there may be multiple projected patterns per view
    :param colmap_root_dir:
    :param output_file:

    :return:
    """
    my_dict = {}
    cameras_path = Path(colmap_root_dir, "cameras.bin")
    points_path = Path(colmap_root_dir, "points3D.bin")
    images_path = Path(colmap_root_dir, "images.bin")

    cameras = get_colmap_camera_intrinsics(cameras_path)
    if images_path.suffix == ".txt":
        images = read_colmap_raw_images_text(images_path)
    elif images_path.suffix == ".bin":
        images = read_colmap_raw_images_binary(images_path)
    else:
        raise NotImplementedError
    # depths_output_file = str(output_file).replace("transforms.json", "depths")
    # depths_struct = load_sensor_depth(images, points_path)
    poses, ids = get_colmap_camera_extrinsics(images)
    heldout_poses = None  # poses that were added post training, and shouldnt effect transforms below
    heldout_ids = None
    if post_added_views is not None:
        for view in post_added_views.split():
            if view in ids:
                index = ids.index(view)
                heldout_poses.append(poses[index])
                heldout_ids.append(ids[index])
                poses = np.concatenate([poses[:index], poses[index+1:]], axis=0)
                ids = ids[:index]+ ids[index+1:]
        heldout_poses = np.stack(heldout_poses, axis=0)
    # move center of attention to origin
    coa = gsoup.get_center_of_attention(poses)
    poses[:, :3, 3] -= coa
    if heldout_poses is not None:
        heldout_poses[:, :3, 3] -= coa
    # make z axis up (currently -y is up)
    xrot_mat = gsoup.rotx(-np.pi/2, degrees=False)
    poses = xrot_mat @ poses
    if heldout_poses is not None:
        heldout_poses = xrot_mat @ heldout_poses
    # scale to unit length ( or n times unit length)
    poses, factor = scale_poses(poses, n=1.0)
    if heldout_poses is not None:
        heldout_poses[:, 0:3, 3] *= 1.0 / factor
    # move such that cameras are looking towards y+ axis
    y_plus = gsoup.look_at_np(np.array([0, -1, 0]), np.array([0, 0, 0]), np.array([0, 0, 1]))
    cur_pos = gsoup.look_at_np(poses[:, :3, -1].mean(axis=0), np.array([0, 0, 0]), -poses[:, :3, 1].mean(axis=0))
    poses = y_plus @ np.linalg.inv(cur_pos) @ poses
    if heldout_poses is not None:
        heldout_poses = y_plus @ np.linalg.inv(cur_pos) @ heldout_poses
    # another scaling round
    coa = gsoup.get_center_of_attention(poses)
    poses[:, :3, 3] -= coa
    if heldout_poses is not None:
        heldout_poses[:, :3, 3] -= coa
    poses, factor = scale_poses(poses, n=1.0)
    if heldout_poses is not None:
        heldout_poses[:, 0:3, 3] *= 1.0 / factor
    # scale depths
    # for entry in depths_struct:
        # entry["depth"] *= 1.0 /factor
    frames = []
    my_dict.update(intrinsics_to_dict(cameras[0]))
    # cam_aprox_for_projector = np.where("0008" == np.array([Path(id).stem.split("-")[0] for id in ids]))[0].item()
    # projector_estimator = sorted(ids)[-1]
    # proj_transform = poses[ids.index(projector_estimator)
    average_loc = poses[:, :3, -1].mean(axis=0)
    average_loc += np.array([0.5, 0, 0.5])
    proj_transform = gsoup.look_at_np(average_loc, np.array([0, 0, 0.0]), np.array([0, 0, 1.0]))[0]
    my_dict.update({"transform_matrix_proj": proj_transform.tolist()})
    with open(Path(output_file.parent, "img2tex.json"), 'r') as fp:
        img2tex = json.load(fp)
    known_viewpoints = []
    if heldout_poses is not None:
        ids = ids + heldout_ids
        poses = np.concatenate([poses, heldout_poses], axis=0)
    for id in ids:
        known_viewpoints.append(img2tex[id][1])
    for file in sorted(output_file.parent.glob("*.png")):
        file_textures = img2tex[file.name][0]
        file_viewpoint = img2tex[file.name][1]
        entry = {
                "file_path": str(file.name),
                "patterns": file_textures,
                "view_id": file_viewpoint,
                # "depths": depths_list[i]["depth"].tolist(),
                # "coords": depths_list[i]["coord"].tolist(),
                # "weights": depths_list[i]["weight"].tolist()
            }
        try:
            matching_white_file_index = known_viewpoints.index(file_viewpoint)
            entry["transform_matrix"] = poses[matching_white_file_index].tolist()
        except ValueError:
            pass
        frames.append(entry)
    # for i in range(len(poses)):
    #     file_name = Path(ids[i]).name
    #     # index = file_stem.split("-")[0]
    #     view_id = img2tex[file_name][1]
    #     for pattern_files in output_file.parent.glob("{}*".format(index)):
    #         pattern = "-".join(pattern_files.stem.split("-")[1:])
    #         entry = {
    #             "file_path": str(pattern_files.name),
    #             "sharpness": 0,
    #             "transform_matrix": poses[i].tolist(),
    #             "patterns": [pattern],
    #             "view_id": view_id,
    #             # "depths": depths_list[i]["depth"].tolist(),
    #             # "coords": depths_list[i]["coord"].tolist(),
    #             # "weights": depths_list[i]["weight"].tolist()
    #         }
    #         frames.append(entry)
    my_dict.update({"frames": frames})
    dump_dict_to_json(my_dict, output_file)
    # np.save(depths_output_file, depths_struct)


def poses_to_nerf(poses, output_file, width=512, height=512):
    """
    :param poses:
    :param output_file:
    :return:
    """
    assert poses.ndim == 3
    assert poses.shape[1:] == (4, 4)
    poses_dict = {}
    poses_dict.update(intrinsics_to_dict())
    frames = []
    paths = create_dummy_images(len(poses), width, height, Path("dummy_images"))
    for i in range(len(poses)):
        entry = {
            "file_path": str(paths[i]),
            "sharpness": 13,
            "transform_matrix": poses[i].tolist()
        }
        frames.append(entry)
    poses_dict.update({"frames": frames})
    # assert not Path(output_file).is_file()
    dump_dict_to_json(poses_dict, output_file)


def do_system(arg):
    print(f"==== running: {arg}")
    err = os.system(arg)
    if err:
        print("the following command failed:")
        print(arg)
        print("the error was:")
        print(err)
        return True
    return False


def run_colmap(img_directory, matcher="exhaustive", only_files_with=None, model_combine=True, histo_equalize=True):  # choices=["exhaustive", "sequential", "spatial", "transitive", "vocab_tree"]
    img_directory_to_process = img_directory
    if only_files_with is not None or histo_equalize:
        sub_img_directory = Path(img_directory, "for_colmap")
        sub_img_directory.mkdir(exist_ok=True, parents=True)
        # open a img2tex file that contains mapping between images and textures
        with open(Path(img_directory, "img2tex.json"), 'r') as fp:
            img2tex = json.load(fp)
            for key in img2tex.keys():
                if only_files_with is not None:
                    if not img2tex[key][0] in only_files_with.split(","):
                        continue
                src = Path(img_directory, key)
                dst = Path(sub_img_directory, src.name)
                # shutil.copy(str(src), str(dst))
                if histo_equalize:
                    if not dst.exists():
                        # Contrast Limited Adaptive Histogram Equalization
                        img = gsoup.load_image(str(src), as_grayscale=True)
                        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                        cl1 = clahe.apply(img)
                        gsoup.save_image(cl1, str(dst))
        img_directory_to_process = sub_img_directory
    output_folder = Path(img_directory, "colmap_output")
    db_path = Path(img_directory, "colmap_db")
    # if (input("folder {} will be deleted, continue?".format(sparse_folder)).lower().strip() + "y")[:1] != "y":
        # sys.exit(1)
    if not Path(output_folder, "0").is_dir():
        if not Path(output_folder, "0", "images.bin").is_file():
            fail = do_system(
                f"colmap feature_extractor --ImageReader.camera_model OPENCV --SiftExtraction.estimate_affine_shape=true --SiftExtraction.domain_size_pooling=true --ImageReader.single_camera 1 --database_path {str(db_path)} --image_path {str(img_directory_to_process)}")
            if not fail:
                fail = do_system(f"colmap {matcher}_matcher --SiftMatching.guided_matching=true --database_path {str(db_path)}")
            try:
                shutil.rmtree(output_folder)
            except:
                pass
            output_folder.mkdir(parents=True, exist_ok=False)
            if not fail:
                fail = do_system(f"colmap mapper --database_path {str(db_path)} --image_path {str(img_directory_to_process)} --output_path {str(output_folder)}")
            if not fail:
                fail = do_system(f"colmap bundle_adjuster --input_path {str(output_folder)}/0 --output_path {str(output_folder)}/0 --BundleAdjustment.refine_principal_point 1")
    
    if model_combine:
        model_folders = []
        for folder in Path(output_folder).glob("*"):
            if folder.is_dir():
                model_folders.append(folder)
        if len(model_folders) > 1:
            max_ids = 0
            for model_folder in model_folders:
                _, cur_ids = get_colmap_camera_extrinsics(read_colmap_raw_images_binary(Path(model_folder, "images.bin")))
                if len(cur_ids) > max_ids:
                    max_ids = len(cur_ids)
                    max_model_folder = model_folder
            merged_model_path = Path(output_folder, "merged")
            if not merged_model_path.exists():
                merged_model_path.mkdir(parents=True, exist_ok=True)
                for model_folder in model_folders:
                    if model_folder == max_model_folder:
                        continue
                    else:
                        if Path(merged_model_path, "cameras.bin").exists():
                            max_model_folder = merged_model_path
                        fail = do_system(f"colmap model_merger --input_path1 {str(max_model_folder)} --input_path2 {str(model_folder)} --output_path {str(merged_model_path)}")
                        if fail:
                            print("Failed to merge colmap models")
                        else:
                            fail = do_system(f"colmap bundle_adjuster --input_path {str(merged_model_path)} --output_path {str(merged_model_path)} --BundleAdjustment.refine_principal_point 1")
                            if fail:
                                print("Failed to refine colmap models")
    return False
