import bpy
from pathlib import Path
import json
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import Voronoi

####################################################################################
##################################### Settings #####################################
####################################################################################

RESULTS_PATH = 'test'
RANDOM_TEXTURES = False  # projectors use different patterns each time
SYNC_TEXTURES = True  # projectors will projector a "video-like" sequence
RANDOM_VIEWS = False  # views are randomly sampled on a hemisphere
TEXTURE_TYPE = "lollipop" #"all_black", "lollipop" # "voronoi", "circles", "dots", "stripes"
######## only for SYNC_TEXTURES = True
pattern_amount = 99
pattern_length = 1
sync_time = 1
######## only for SYNC_TEXTURES = True
N_VIEWS = 99  # total number of views (overidden if SYNC_TEXTURES=True)
N_TEXTURES_PER_VIEW = 3
DEBUG = False  # will quit after setting a texture
DEBUG_VIEW_INDEX = 161
AMBIENT_LIGHT = False
COLOC_LIGHT = True
CONSTANT_BG = False
OUTPUT_DEPTH = False
OUTPUT_NORMALS = False
SINGLE_BOUNCE = True
TEXTURES = np.array(["all_black", "all_white", "random_lollipop"]) 
RESOLUTION = 800
PROJ_RES_W = 800
PROJ_RES_H = 800
COLOR_DEPTH = 8
FORMAT = 'PNG'
CAM_DIST_FROM_ZERO = 1
PROJECTOR_LOC = (-0.3, -0.3, 0.9)
PROJECTOR2_LOC = (0.0, 0.5, 0.5)
BLENDER_SUFFIX = 161  # eww blender...
if RANDOM_VIEWS and SYNC_TEXTURES:
    raise ValueError("Cannot have both RANDOM_VIEWS and SYNC_TEXTURES")
normal_layer_name = "Normal"  # AVG_SHADING_NORMAL for luxcore
denoised_layer_name = "Image"  # DENOISED for luxcore
results_dir = Path(bpy.path.abspath(f"//{RESULTS_PATH}"))
patterns_dir = Path(results_dir, "projector")
####################################################################################
############################### Helper Functions ###################################
####################################################################################


def get_scene_resolution(scene):
    resolution_scale = (scene.render.resolution_percentage / 100.0)
    resolution_x = scene.render.resolution_x * resolution_scale  # [pixels]
    resolution_y = scene.render.resolution_y * resolution_scale  # [pixels]
    return int(resolution_x), int(resolution_y)


def get_sensor_size(sensor_fit, sensor_x, sensor_y):
    if sensor_fit == 'VERTICAL':
        return sensor_y
    return sensor_x


def get_sensor_fit(sensor_fit, size_x, size_y):
    if sensor_fit == 'AUTO':
        if size_x >= size_y:
            return 'HORIZONTAL'
        else:
            return 'VERTICAL'
    return sensor_fit


def get_camera_parameters_extrinsic(scene, camera):
    """ 
    Get extrinsic camera parameters w2c (opencv) from blender c2w
      There are 3 coordinate systems involved:
         1. The World coordinates: "world"
            - right-handed
         2. The Blender camera coordinates: "bcam"
            - x is horizontal
            - y is up
            - right-handed: negative z look-at direction
            - note: matrix_world is c2w in this coordinate system
         3. The desired computer vision camera coordinates: "cv" (w2c)
            - x is horizontal
            - y is down (to align to the actual pixel coordinates
               used in digital images)
            - right-handed: positive z look-at direction
      ref: https://blender.stackexchange.com/questions/38009/3x4-camera-matrix-from-blender-camera
    """
    # bcam stands for blender camera
    bcam = camera
    R_bcam2cv = np.array([[1, 0, 0],
                          [0, -1, 0],
                          [0, 0, -1]])

    # Transpose since the rotation is object rotation,
    # and we want coordinate rotation
    # R_world2bcam = cam.rotation_euler.to_matrix().transposed()
    # T_world2bcam = -1*R_world2bcam * location
    #
    # Use matrix_world instead to account for all constraints
    location = np.array([bcam.matrix_world.decompose()[0]]).T
    R_world2bcam = np.array(bcam.matrix_world.decompose()[1].to_matrix().transposed())

    # Convert camera location to translation vector used in coordinate changes
    # T_world2bcam = -1*R_world2bcam*bcam.location
    # Use location from matrix_world to account for constraints:
    T_world2bcam = np.matmul(R_world2bcam.dot(-1), location)

    # Build the coordinate transform matrix from world to computer vision camera
    R_world2cv = np.matmul(R_bcam2cv, R_world2bcam)
    T_world2cv = np.matmul(R_bcam2cv, T_world2bcam)
    extr = np.concatenate((R_world2cv, T_world2cv), axis=1)  # c2w
    return extr


def get_c2w_opencv(camera):
    """
    converts a c2w matrix of blender, into opencv c2w (x is right, y is down, z goes into view direction)
    """
    R_bcam2cv = np.array([[1, 0, 0, 0],
                          [0, -1, 0, 0],
                          [0, 0, -1, 0],
                          [0, 0, 0, 1]])
    c2w = np.matmul(np.array(camera.matrix_world), R_bcam2cv)
    return c2w
    
    
def get_camera_parameters_intrinsic(scene, camera):
    """ Get intrinsic camera parameters: focal length and principal point. """
    # ref: https://blender.stackexchange.com/questions/38009/3x4-camera-matrix-from-blender-camera/120063#120063
    focal_length = camera.data.lens  # [mm]
    res_x, res_y = get_scene_resolution(scene)
    cam_data = camera.data
    sensor_size_in_mm = get_sensor_size(cam_data.sensor_fit, cam_data.sensor_width, cam_data.sensor_height)
    sensor_fit = get_sensor_fit(
        cam_data.sensor_fit,
        scene.render.pixel_aspect_x * res_x,
        scene.render.pixel_aspect_y * res_y
    )
    pixel_aspect_ratio = scene.render.pixel_aspect_y / scene.render.pixel_aspect_x
    if sensor_fit == 'HORIZONTAL':
        view_fac_in_px = res_x
    else:
        view_fac_in_px = pixel_aspect_ratio * res_y
    pixel_size_mm_per_px = (sensor_size_in_mm / focal_length) / view_fac_in_px
    skew = 0
    f_x = 1.0 / pixel_size_mm_per_px
    f_y = (1.0 / pixel_size_mm_per_px) / pixel_aspect_ratio
    c_x = (res_x) / 2.0 - cam_data.shift_x * view_fac_in_px
    c_y = (res_y) / 2.0 + (cam_data.shift_y * view_fac_in_px) / pixel_aspect_ratio
    K = np.array([[f_x, skew, c_x],
                  [0, f_y, c_y],
                  [0, 0, 1]])
    # return f_x, f_y, c_x, c_y
    return K

def parent_obj_to_camera(object, a_parent):
    object.parent = a_parent  # setup parenting
    return b_empty


def listify_matrix(matrix):
    matrix_list = []
    for row in matrix:
        matrix_list.append(list(row))
    return matrix_list
    

def generate_voronoi_diagram(width, height, num_cells=1000, dst=None):
    nx = np.random.rand(num_cells) * width
    ny = np.random.rand(num_cells) * height
    nxy = np.stack((nx, ny), axis=-1)
    img = Image.new("RGB", (width, height), "white")
    vor = Voronoi(nxy)
    polys = vor.regions
    vertices = vor.vertices
    for poly in polys:
        polygon = vertices[poly]
        if len(poly) > 0 and np.all(np.array(poly) > 0):
            img1 = ImageDraw.Draw(img)
            img1.polygon(list(map(tuple, polygon)), fill=tuple(np.random.randint(0, 255, size=(3,))))
    if dst is not None:
        img.save(str(dst))
    return np.array(img)

def generate_dot_pattern(height, width, background="black", radius=5, spacing=30, dst=None):
    img = Image.new("RGB", (width, height), background)
    img1 = ImageDraw.Draw(img)
    for i in range(0, width, spacing):
        for j in range(0, height, spacing):
            img1.ellipse([i, j, i + radius*2, j + radius*2], fill=tuple(np.random.randint(0, 255, size=(3,))))
    if dst is not None:
        img.save(str(dst))
    return np.array(img)

def generate_stripe_pattern(height, width, background="black", direction="hor", thickness=5, spacing=20, dst=None):
    img = Image.new("RGB", (width, height), background)
    img1 = ImageDraw.Draw(img)
    if direction == "vert":
        for i in range(0, width, spacing):
            img1.rectangle([i, 0, i + thickness, height], fill=tuple(np.random.randint(0, 255, size=(3,))))
    elif direction == "hor":
        for i in range(0, height, spacing):
            img1.rectangle([0, i, width, i + thickness], fill=tuple(np.random.randint(0, 255, size=(3,))))
    elif direction == "both":
        for i in range(0, width, spacing):
            img1.rectangle([i, 0, i + thickness, height], fill=tuple(np.random.randint(0, 255, size=(3,))))
        for i in range(0, height, spacing):
            img1.rectangle([0, i, width, i + thickness], fill=tuple(np.random.randint(0, 255, size=(3,))))
    else:
        raise ValueError("direction must be either 'vert', 'hor' or 'both'")
    if dst is not None:
        img.save(str(dst))
    return np.array(img)

def generate_concentric_circles(height, width, background="black", n=30, dst=None):
    spacing_x = width // (2*n)
    spacing_y = height // (2*n)
    img = Image.new("RGB", (width, height), background)
    img1 = ImageDraw.Draw(img)
    for i in range(n):
        x0 = spacing_x * i
        y0 = spacing_y * i
        img1.ellipse([x0, y0, width - x0, height - y0], fill=tuple(np.random.randint(0, 255, size=(3,))))
    if dst is not None:
        img.save(str(dst))
    return np.array(img)

def generate_all_black(height, width, dst=None):
    img = Image.new("RGB", (width, height), "black")
    if dst is not None:
        img.save(str(dst))
    return np.array(img)

def generate_all_white(height, width, dst=None):
    img = Image.new("RGB", (width, height), "white")
    if dst is not None:
        img.save(str(dst))
    return np.array(img)

def generate_lollipop_pattern(height, width, background="black", n=30, m=8, dst=None):
    spacing_x = width // (2*n)
    spacing_y = height // (2*n)
    spacing_angle = 360 // m
    img = Image.new("RGB", (width, height), background)
    img1 = ImageDraw.Draw(img)
    for j in range(m):
        for i in range(n):
            x0 = spacing_x * i
            y0 = spacing_y * i
            start_angle = j * spacing_angle
            end_angle = 360
            img1.pieslice([x0, y0, width - x0, height - y0], start=start_angle, end=end_angle, fill=tuple(np.random.randint(0, 255, size=(3,))))
    if dst is not None:
        img.save(str(dst))
    return np.array(img)

def remove_texures():
    # remove current textures
    exceptions = ["marble.jpg", "bob_diffuse.png", "spot_texture.png", "fabric.jpg", "blub_texture.png"]
    string_exceptions = ["Wall"]
    for image in bpy.data.images:
        if image.name not in exceptions:
            if ~np.any(np.array([x in image.name for x in string_exceptions])):
                print("removing texture: {}".format(image))
                bpy.data.images.remove(image)

def load_texture(texture_path, texture_key, proj_h, proj_w):
    pilImage = Image.open(str(texture_path))
    image = np.asarray(pilImage)
    if image.shape[0] != proj_h or image.shape[1] != proj_w:
        pilImage = pilImage.resize((proj_w, proj_h))
        image = np.asarray(pilImage)
    float_texture = (image / 255).astype(np.float32)
    # flipped_texture = np.flip(float_texture, axis=0)
    padded_texture = np.concatenate((float_texture, np.ones_like(float_texture)[:, :, 0:1]), axis=-1)
    bpy_image = bpy.data.images.new(texture_key, width=proj_w, height=proj_h, alpha=False)
    bpy_image.pixels.foreach_set(padded_texture.ravel())
    bpy_image.pack()

def save_texture(texture_key, proj_width, proj_height, dst):
    # save current texture
    if dst.is_file():
        return
    print("saving to: {}".format(str(dst)))
    image = np.array(bpy.data.images[texture_key].pixels).reshape(proj_height, proj_width, 4)
    image = Image.fromarray((image[:, :, :3]*255).astype(np.uint8))
    image.save(dst)

def pack_textures(textures, proj_width, proj_height):
    # add new textures
    for key in textures.keys():
        if key not in bpy.data.images.keys():
            # flipped_texture = np.flip(textures[key], axis=0)
            padded_texture = np.concatenate((textures[key], np.ones_like(textures[key])[:, :, 0:1]), axis=-1)
            bpy_image = bpy.data.images.new(key, width=proj_width, height=proj_height, alpha=False)
            bpy_image.pixels.foreach_set(padded_texture.ravel())
            bpy_image.pack()
        
def swap_projector_texture(texture_name, secondary_projector=False):
    if secondary_projector:
        projector_name = "Projector2"
    else:
        projector_name = "Projector"
    bpy.data.images[texture_name].colorspace_settings.name = 'Linear'
    bpy.data.lights[projector_name].node_tree.nodes["Image Texture"].image = bpy.data.images[texture_name]


def hide_object_and_children(obj, hide=True):
    # hide the children
    obj.hide_viewport = hide
    obj.hide_render = hide
    for child in obj.children:
        child.hide_viewport = hide
        child.hide_render = hide

   
####################################################################################
################################ Configure Scene ###################################
####################################################################################
if not patterns_dir.is_dir():
    patterns_dir.mkdir(exist_ok=True, parents=True)

# data to store in JSON file
proj1_intrinsics = get_camera_parameters_intrinsic(bpy.context.scene, bpy.context.scene.objects["Projector_Camera"])
proj1_intrinsics[0, :] /= RESOLUTION
proj1_intrinsics[0, :] *= PROJ_RES_W
proj1_intrinsics[1, :] /= RESOLUTION
proj1_intrinsics[1, :] *= PROJ_RES_H
proj2_intrinsics = get_camera_parameters_intrinsic(bpy.context.scene, bpy.context.scene.objects["Projector2_Camera"])
proj2_intrinsics[0, :] /= RESOLUTION
proj2_intrinsics[0, :] *= PROJ_RES_W
proj2_intrinsics[1, :] /= RESOLUTION
proj2_intrinsics[1, :] *= PROJ_RES_H
out_data = {
    'camera_angle_x': bpy.data.objects['Camera'].data.angle_x,
    'K_cam': listify_matrix(get_camera_parameters_intrinsic(bpy.context.scene, bpy.context.scene.camera)),
    'K_proj': listify_matrix(proj1_intrinsics),
    'K_proj2': listify_matrix(proj2_intrinsics),
    'blender_matrix_world_proj': listify_matrix(bpy.context.scene.objects["Projector"].matrix_world),
    'blender_matrix_world_proj2': listify_matrix(bpy.context.scene.objects["Projector2"].matrix_world)
}
# get textures
remove_texures()
texture_names = np.array(TEXTURES)

bpy.context.scene.objects['Projector'].data.energy = 50
bpy.context.scene.objects['Projector2'].data.energy = 50
bpy.context.scene.objects["Camera_Light"].data.energy = 10
bpy.context.scene.render.use_persistent_data = True
if SINGLE_BOUNCE:
    bpy.context.scene.cycles.max_bounces = 0
else:
    bpy.context.scene.cycles.max_bounces = 3
if AMBIENT_LIGHT:
    bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[1].default_value = 0.01
else:
    bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[1].default_value = 0
if COLOC_LIGHT:
    bpy.context.scene.objects["Camera_Light"].hide_render = False
    bpy.context.scene.objects["Camera_Light"].hide_viewport = False
else:
    bpy.context.scene.objects["Camera_Light"].hide_render = True
    bpy.context.scene.objects["Camera_Light"].hide_viewport = True
# Set up rendering of depth map.
bpy.context.scene.use_nodes = True
tree = bpy.context.scene.node_tree
links = tree.links
# Add passes for additionally dumping albedo and normals.
bpy.context.scene.view_layers["RenderLayer"].use_pass_normal = True
bpy.context.scene.render.image_settings.file_format = str(FORMAT)
bpy.context.scene.render.image_settings.color_depth = str(COLOR_DEPTH)

# Remove all nodes from current compositor
for node in tree.nodes:
    tree.nodes.remove(node)

# Add from scratch nodes in compositor
if 'Custom Outputs' not in tree.nodes:
    # Create input render layer node.
    render_layers = tree.nodes.new('CompositorNodeRLayers')
    render_layers.label = 'Custom Outputs'
    render_layers.name = 'Custom Outputs'

    depth_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
    depth_file_output.label = 'Depth Output'
    depth_file_output.name = 'Depth Output'
    if OUTPUT_DEPTH:
        if FORMAT == 'OPEN_EXR':
            links.new(render_layers.outputs['Depth'], depth_file_output.inputs[0])
        else:
            # Remap as other types can not represent the full range of depth.
            map = tree.nodes.new(type="CompositorNodeMapRange")
            # Size is chosen kind of arbitrarily, try out until you're satisfied with resulting depth map.
            map.inputs['From Min'].default_value = 0
            map.inputs['From Max'].default_value = 8
            map.inputs['To Min'].default_value = 1
            map.inputs['To Max'].default_value = 0
            links.new(render_layers.outputs['Depth'], map.inputs[0])

            links.new(map.outputs[0], depth_file_output.inputs[0])

    normal_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
    normal_file_output.label = 'Normal Output'
    normal_file_output.name = 'Normal Output'
    if OUTPUT_NORMALS:
        links.new(render_layers.outputs[normal_layer_name], normal_file_output.inputs[0])

    if CONSTANT_BG:
        alpha_over = tree.nodes.new(type="CompositorNodeAlphaOver")
        alpha_over.label = 'Alpha Over'
        alpha_over.name = 'Alpha Over'
        #alpha_over.premul = 1
        alpha_over.inputs[1].default_value = (0, 0, 0, 1)
        links.new(render_layers.outputs[denoised_layer_name], alpha_over.inputs[2])

        image_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
        image_file_output.label = 'Image Output'
        image_file_output.name = 'Image Output'
        links.new(alpha_over.outputs[0], image_file_output.inputs[0])
    else:
        image_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
        image_file_output.label = 'Image Output'
        image_file_output.name = 'Image Output'
        links.new(render_layers.outputs[denoised_layer_name], image_file_output.inputs[0])
    image_file_output.format.color_mode = 'RGBA'
# Background

bpy.context.scene.render.dither_intensity = 0.0
bpy.context.scene.render.film_transparent = True

# Create collection for objects not to render with background

objs = [ob for ob in bpy.context.scene.objects if ob.type in ('EMPTY') and 'Empty' in ob.name]
bpy.ops.object.delete({"selected_objects": objs})
scene = bpy.context.scene
scene.render.resolution_percentage = 100
cam = scene.objects['Camera']
cam_light = scene.objects["Camera_Light"]
projector = scene.objects['Projector']
projector_camera = scene.objects['Projector_Camera']
for c in cam.constraints:
    cam.constraints.remove(c)
scene.render.resolution_x = RESOLUTION
scene.render.resolution_y = RESOLUTION
cam.location = (CAM_DIST_FROM_ZERO, 0, 0)
cam_light.location = (CAM_DIST_FROM_ZERO, 0, 0)
projector.location = PROJECTOR_LOC
projector_camera.location = PROJECTOR_LOC
cam_constraint = cam.constraints.new(type='TRACK_TO')
cam_constraint.track_axis = 'TRACK_NEGATIVE_Z'
cam_constraint.up_axis = 'UP_Y'
proj_constraint = projector.constraints[0]
proj_cam_constraint = projector_camera.constraints[0]
origin = (0, 0, 0)
b_empty = bpy.data.objects.new("Empty", None)
b_empty.location = origin
parent_obj_to_camera(cam, b_empty)
parent_obj_to_camera(cam_light, b_empty)
#parent_obj_to_camera(light, b_empty)
scene.collection.objects.link(b_empty)
bpy.context.view_layer.objects.active = b_empty
# scene.objects.active = b_empty
cam_constraint.target = b_empty
proj_constraint.target = b_empty
proj_cam_constraint.target = b_empty
scene.render.image_settings.file_format = 'PNG'  # set output format to .png

for output_node in [tree.nodes['Depth Output'], tree.nodes['Normal Output'], tree.nodes['Image Output']]:
    output_node.base_path = ''

out_data['frames'] = []

# prepare texture sequence
if RANDOM_TEXTURES:
    selected_textures = np.random.choice(texture_names, size=N_VIEWS*N_TEXTURES_PER_VIEW, replace=True)

elif SYNC_TEXTURES:
    x = np.arange(0, pattern_amount).repeat(pattern_length)
    indices = np.where(np.ediff1d(x, to_begin=1, to_end=1))[0]
    x = np.insert(x, np.repeat(indices, sync_time), -1)
    indices = np.where(np.ediff1d(x) > 0)[0] + 1
    x = np.insert(x, np.repeat(indices, sync_time), -2)
    y = np.empty(len(x), dtype='<U16')
    y[x==-2] = "all_white"
    y[x==-1] = "all_black"
    for i in range(pattern_amount):
        y[x==i] = "{}_{}".format(TEXTURE_TYPE, i)
    selected_textures = y
    if TEXTURE_TYPE == "all_black":
        selected_textures = ["all_black"] * len(selected_textures)
    if len(selected_textures) < N_VIEWS:
        selected_textures = np.concatenate((selected_textures, np.full(N_VIEWS - len(selected_textures), "all_black")))
else:
    selected_textures = np.tile(texture_names, N_VIEWS)

# prepare camera locations sequence
if RANDOM_VIEWS:
    cam_locations = []
    for i in range(len(selected_textures) // N_TEXTURES_PER_VIEW):
        rot = np.random.uniform(0, 1, size=3) * (1, 0, 2*np.pi)
        rot[0] = np.arccos(2 * rot[0] - 1) / 2
        #b_empty.rotation_euler = rot
        r = CAM_DIST_FROM_ZERO
        new_loc = (r*np.sin(rot[0])*np.cos(rot[2]), r*np.sin(rot[0])*np.sin(rot[2]), r*np.cos(rot[0]))
        cam_locations.append(new_loc)
    cam_locations = np.stack(cam_locations)
    cam_locations = cam_locations.repeat(N_TEXTURES_PER_VIEW, axis=0)
else:
    t = np.linspace(0, 1, len(selected_textures), endpoint=False)
    phi = t*8*np.pi
    theta1 = np.flip(t)*np.pi/2 #np.arccos(2 * t - 1) / 2
    theta2 = np.flip(theta1)
    theta = np.concatenate((theta1[:-1], theta2))[::2]
    r = CAM_DIST_FROM_ZERO
    cam_locations = np.vstack((r*np.sin(theta)*np.cos(phi), r*np.sin(theta)*np.sin(phi), r*np.cos(theta))).T
view_ids = np.arange(len(cam_locations))

print("view_ids: {}".format(view_ids))
print("selected_textures: {}".format(selected_textures))
####################################################################################
############################### Main Render Loop ###################################
####################################################################################
frame_counter = 0
img2tex = {}
if DEBUG:
    cam.location = cam_locations[DEBUG_VIEW_INDEX]
    cam_light.location = cam_locations[DEBUG_VIEW_INDEX]
    raise NotImplementedError("DEBUG")
for i in range(0, len(selected_textures)):
    cam.location = cam_locations[i]
    cam_light.location = cam_locations[i]
    print("view: {} / {}, pattern_name: {}".format(i, len(selected_textures), selected_textures[i]))
    selected_texture = selected_textures[i]
    if "voronoi" in selected_texture:
        dst = Path(patterns_dir, selected_texture + ".png")
        if not dst.exists():
            generate_voronoi_diagram(PROJ_RES_H, PROJ_RES_W, dst=dst)
            load_texture(dst, selected_texture, PROJ_RES_H, PROJ_RES_W)
    elif "dots" in selected_texture:
        dst = Path(patterns_dir, selected_texture + ".png")
        if not dst.exists():
            generate_dot_pattern(PROJ_RES_H, PROJ_RES_W, dst=dst)
            load_texture(dst, selected_texture, PROJ_RES_H, PROJ_RES_W)
    elif "stripes" in selected_texture:
        dst = Path(patterns_dir, selected_texture + ".png")
        if not dst.exists():
            generate_stripe_pattern(PROJ_RES_H, PROJ_RES_W, direction="hor", dst=dst)
            load_texture(dst, selected_texture, PROJ_RES_H, PROJ_RES_W)
    elif "circles" in selected_texture:
        dst = Path(patterns_dir, selected_texture + ".png")
        if not dst.exists():
            generate_concentric_circles(PROJ_RES_H, PROJ_RES_W, dst=dst)
            load_texture(dst, selected_texture, PROJ_RES_H, PROJ_RES_W)
    elif "lollipop" in selected_texture:
        if "random" in selected_texture:
            selected_texture = "lollipop_{:02d}".format(i//N_TEXTURES_PER_VIEW)
        dst = Path(patterns_dir, selected_texture + ".png")
        if not dst.exists():
            generate_lollipop_pattern(PROJ_RES_H, PROJ_RES_W, dst=dst)
            load_texture(dst, selected_texture, PROJ_RES_H, PROJ_RES_W)
    elif "all_black" in selected_texture:
        dst = Path(patterns_dir, selected_texture + ".png")
        if not dst.exists():
            generate_all_black(PROJ_RES_H, PROJ_RES_W, dst=dst)
            load_texture(dst, selected_texture, PROJ_RES_H, PROJ_RES_W)
    elif "all_white" in selected_texture:
        dst = Path(patterns_dir, selected_texture + ".png")
        if not dst.exists():
            generate_all_white(PROJ_RES_H, PROJ_RES_W, dst=dst)
            load_texture(dst, selected_texture, PROJ_RES_H, PROJ_RES_W)
    swap_projector_texture(selected_texture)
    bpy.context.view_layer.update()
    dst = Path(patterns_dir, selected_texture + ".png")
    my_path_str = str(Path(results_dir, "{:04d}".format(frame_counter)))
    if OUTPUT_DEPTH:
        tree.nodes['Depth Output'].file_slots[0].path = str(my_path_str) + "_depth"
    if OUTPUT_NORMALS:
        tree.nodes['Normal Output'].file_slots[0].path = str(my_path_str) + "_normal"
    tree.nodes['Image Output'].file_slots[0].path = str(my_path_str)
    frame_data = {
        'file_path': Path(my_path_str).stem + ".png",
        'blender_matrix_world': listify_matrix(cam.matrix_world),
        'RT': listify_matrix(get_c2w_opencv(cam)),
        'patterns': [str(selected_texture)],
        'view_id': "{:04d}".format(view_ids[i])
    }
    out_data['frames'].append(frame_data)
    img2tex[str(Path(my_path_str).stem + ".png")] = [[str(selected_texture)], view_ids[i].item()]

    with open(Path(results_dir, 'transforms.json'), 'w') as out_file:
        json.dump(out_data, out_file, indent=4)
    with open(Path(results_dir, 'img2tex.json'), 'w') as out_file:
        json.dump(img2tex, out_file, indent=4)
    if not DEBUG:
        bpy.ops.render.render()
        #  blender hacks to change output name.
        outRenderFileNamePadded = Path(my_path_str + "depth{:04d}.png".format(BLENDER_SUFFIX))
        outRenderFileName = Path(my_path_str + "depth.png")
        if outRenderFileName.is_file():
            outRenderFileName.unlink()
        if outRenderFileNamePadded.is_file():
            outRenderFileNamePadded.rename(outRenderFileName)
        outRenderFileNamePadded = Path(my_path_str + "normal{:04d}.png".format(BLENDER_SUFFIX))
        outRenderFileName = Path(my_path_str + "normal.png")
        if outRenderFileName.is_file():
            outRenderFileName.unlink()
        if outRenderFileNamePadded.is_file():
            outRenderFileNamePadded.rename(outRenderFileName)
        outRenderFileNamePadded = Path(my_path_str + "{:04d}.png".format(BLENDER_SUFFIX))
        outRenderFileName = Path(my_path_str + ".png")
        if outRenderFileName.is_file():
            outRenderFileName.unlink()
        if outRenderFileNamePadded.is_file():
            outRenderFileNamePadded.rename(outRenderFileName)
        save_texture(selected_texture, PROJ_RES_W, PROJ_RES_H, dst)
    frame_counter += 1
