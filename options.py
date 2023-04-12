import configargparse
from pathlib import Path

def config_parser(cmd=None):
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, help='config file path')
    parser.add_argument("--basedir", type=str, default='./logs/', help='root dir to store all ckpts, logs, etc.')
    parser.add_argument("--expname", type=str, help='experiment name')
    parser.add_argument("--datadir", type=str, default='./datasets', help='input data directory')
    parser.add_argument("--scene", type=str, default="bunny", help="which scene to use")
    parser.add_argument("--test_scene", type=str, help="which test scene to use")
    parser.add_argument("--seed", type=int, default=27, help='random seed')
    parser.add_argument("--checkpoint", type=str, help='checkpoint to load and use as starting point')
    parser.add_argument("--nerf_checkpoint", type=str, help='nerf checkpoint to load and use as starting point')
    parser.add_argument("--just_load_networks", action="store_true", help='if load ckpp, only load network weights')
    parser.add_argument("--freeze_radiance", action="store_true", help='freeze radiance field params')
    parser.add_argument("--freeze_vis", action="store_true", help='freeze vis network params')
    parser.add_argument("--freeze_coloc", action="store_true", help='freeze coloc light')
    parser.add_argument("--freeze_projector", action="store_true", help='freeze projectors')
    parser.add_argument("--projector_add_noise", action="store_true", help='add noise to projector')
    parser.add_argument("--projector_force_value", action="store_true", help='force projector to have certain params')
    parser.add_argument("--cameras_add_noise", action="store_true", help='add noise to camera locs')
    parser.add_argument("--rotation_is_qvec", action="store_true", help='rotations are rep. by quaternions')
    parser.add_argument("--datatype", type=str, default='random', choices=["random", "movie"])
    parser.add_argument("--resource_dir", type=str, default='./resource', help='resource data directory')
    parser.add_argument("--cdc_conda", type=str, help='path to conda env with cdc installed')  # required for text to projection
    parser.add_argument("--cdc_src", type=str, help='path to cdc source code folder')  # required for text to projection
    parser.add_argument("--device", type=str, default='cuda:0')
    parser.add_argument("--aabb", type=lambda s: [float(item) for item in s.split(",")],
        default="-1.5,-1.5,-1.5,1.5,1.5,1.5",
        help="delimited list input",
    )
    parser.add_argument("--net_width", type=int, default=128)
    parser.add_argument("--geo_freq", type=int, default=7, help="geo freq")
    parser.add_argument("--mat_freq", type=int, default=8, help="geo freq")
    parser.add_argument("--test_res_wh", type=str, help='test resolution')
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--render_only", default=False, action="store_true")
    parser.add_argument("--render_modes", type=str, help="render only modes passed as a comma delimited string")
    parser.add_argument("--test_chunk_size", type=int, default=8192)  # 8192
    parser.add_argument("--target_sample_batch_size", type=int, default=2**16)
    parser.add_argument("--unbounded", action="store_true", help="whether to use unbounded rendering")
    parser.add_argument("--cone_angle", type=float, default=0.0)
    parser.add_argument("--relightable", default=False, action="store_true", help="whether to use relightable rendering")
    parser.add_argument("--nepmap", action="store_true", default=False, help="whether to use nepmap or classical nerf")
    parser.add_argument("--coloc", default=False, action="store_true", help="whether to use colocated light")
    parser.add_argument("--coloc_amp", type=float, default=2.0)
    parser.add_argument("--proj_amp", type=float, default=4.0)
    parser.add_argument("--inverse_square", default=False, help="whether to use inverse square law for coloc light")
    parser.add_argument("--projectors", default=False, action="store_true", help="whether to optimize projectors")
    parser.add_argument("--opt_cams", default=False, action="store_true", help="whether to optimize cameras")
    parser.add_argument("--colmap_views", type=str, help="which type of views to use for colmap")
    parser.add_argument("--colmap_mode", default="video", type=str)
    parser.add_argument("--render_n_samples", type=int, default=1024)
    parser.add_argument("--grid_resolution", type=int, default=128)
    parser.add_argument("--divide_res", type=int, default=1)
    parser.add_argument("--frames_for_render", type=str)
    parser.add_argument("--post_added_views", type=str)
    parser.add_argument("--sch_milestones", type=str)
    parser.add_argument("--proj_w", type=int, default=800)
    parser.add_argument("--proj_h", type=int, default=800)
    parser.add_argument("--max_step", type=int, default=30000)
    parser.add_argument("--force_step", type=int, default=-1)
    parser.add_argument("--phase1", type=int, default=-1)
    parser.add_argument("--phase2", type=int, default=-1)
    parser.add_argument("--phase3", type=int, default=-1)
    parser.add_argument("--vanilla_loss_coeff", type=float, default=0.0)
    parser.add_argument("--fog_loss_coeff", type=float, default=0.005)
    parser.add_argument("--cam_loss_coeff", type=float, default=1.0, help="camera loss coeff")
    parser.add_argument("--normal_loss_coeff", type=float, default=0.01, help="normal loss coeff")
    parser.add_argument("--normal_loss2_coeff", type=float, default=0.1, help="normal 2 loss coeff")
    parser.add_argument("--surface_loss", default=False, action="store_true", help="whether to use surface loss")
    parser.add_argument("--save_step", type=int, default=2500)
    parser.add_argument("--test_step", type=int, default=2500)
    parser.add_argument("--video_step", type=int, default=5000)
    parser.add_argument("--print_step", type=int, default=100)
    if cmd:
        args = parser.parse_args(cmd.split())
    else:
        args = parser.parse_args()
    if args.test_res_wh:
        args.test_res_wh = [int(x) for x in args.test_res_wh.split(",")]
    if args.test_scene is None:
        args.test_scene = args.scene
    args.datadir = Path(args.datadir)
    args.experiment_folder = Path(args.basedir, args.expname)
    args.experiment_folder.mkdir(exist_ok=True, parents=True)
    if args.render_only:
        to_save_path = Path(args.basedir, args.expname, "render_only")
        to_save_path.mkdir(exist_ok=True, parents=True)
    else:
        to_save_path = args.experiment_folder
    f_args = Path(to_save_path, 'args.txt')
    f_config = Path(to_save_path, 'config.txt')
    with open(f_args, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        with open(f_config, 'w') as file:
            file.write(open(args.config, 'r').read())
    return args