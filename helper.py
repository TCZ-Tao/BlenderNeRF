import os
import shutil
import random
import math
import mathutils
import bpy
from bpy.app.handlers import persistent


# global addon script variables
EMPTY_NAME = 'BlenderNeRF Sphere'
CAMERA_NAME = 'BlenderNeRF Camera'

## property poll and update functions

# camera pointer property poll function
def poll_is_camera(self, obj):
    return obj.type == 'CAMERA'

def visualize_sphere(self, context):
    scene = context.scene

    if EMPTY_NAME not in scene.objects.keys() and not scene.sphere_exists:
        # if empty sphere does not exist, create
        bpy.ops.object.empty_add(type='SPHERE') # non default location, rotation and scale here are sometimes not applied, so we enforce them manually below
        empty = context.active_object
        empty.name = EMPTY_NAME
        empty.location = scene.sphere_location
        empty.rotation_euler = scene.sphere_rotation
        empty.scale = scene.sphere_scale
        empty.empty_display_size = scene.sphere_radius

        scene.sphere_exists = True

    elif EMPTY_NAME in scene.objects.keys() and scene.sphere_exists:
        if CAMERA_NAME in scene.objects.keys() and scene.camera_exists:
            delete_camera(scene, CAMERA_NAME)

        objects = bpy.data.objects
        objects.remove(objects[EMPTY_NAME], do_unlink=True)

        scene.sphere_exists = False

def visualize_camera(self, context):
    scene = context.scene

    if CAMERA_NAME not in scene.objects.keys() and not scene.camera_exists:
        if EMPTY_NAME not in scene.objects.keys():
            scene.show_sphere = True

        bpy.ops.object.camera_add()
        camera = context.active_object
        camera.name = CAMERA_NAME
        camera.data.name = CAMERA_NAME
        camera.location = sample_from_sphere(scene)
        bpy.data.cameras[CAMERA_NAME].lens = scene.focal

        cam_constraint = camera.constraints.new(type='TRACK_TO')
        cam_constraint.track_axis = 'TRACK_Z' if scene.outwards else 'TRACK_NEGATIVE_Z'
        cam_constraint.up_axis = 'UP_Y'
        cam_constraint.target = bpy.data.objects[EMPTY_NAME]

        scene.camera_exists = True

    elif CAMERA_NAME in scene.objects.keys() and scene.camera_exists:
        objects = bpy.data.objects
        objects.remove(objects[CAMERA_NAME], do_unlink=True)

        for block in bpy.data.cameras:
            if CAMERA_NAME in block.name:
                bpy.data.cameras.remove(block)

        scene.camera_exists = False

def delete_camera(scene, name):
    objects = bpy.data.objects
    objects.remove(objects[name], do_unlink=True)

    scene.show_camera = False
    scene.camera_exists = False

    for block in bpy.data.cameras:
        if name in block.name:
            bpy.data.cameras.remove(block)

# non uniform sampling when stretched or squeezed sphere
def sample_from_sphere(scene):
    seed = (2654435761 * (scene.seed + 1)) ^ (805459861 * (scene.frame_current + 1))
    rng = random.Random(seed) # random number generator

    # sample random angles
    theta = rng.random() * 2 * math.pi
    phi = math.acos(1 - 2 * rng.random()) # ensure uniform sampling from unit sphere

    # uniform sample from unit sphere, given theta and phi
    unit_x = math.cos(theta) * math.sin(phi)
    unit_y = math.sin(theta) * math.sin(phi)
    unit_z = abs( math.cos(phi) ) if scene.upper_views else math.cos(phi)
    unit = mathutils.Vector((unit_x, unit_y, unit_z))

    # ellipsoid sample : center + rotation @ radius * unit sphere
    point = scene.sphere_radius * mathutils.Vector(scene.sphere_scale) * unit
    rotation = mathutils.Euler(scene.sphere_rotation).to_matrix()
    point = mathutils.Vector(scene.sphere_location) + rotation @ point

    return point

## two way property link between sphere and ui (property and handler functions)
# https://blender.stackexchange.com/questions/261174/2-way-property-link-or-a-filtered-property-display

def properties_ui_upd(self, context):
    can_scene_upd(self, context)

@persistent
def properties_desgraph_upd(scene):
    can_properties_upd(scene)

def properties_ui(self, context):
    scene = context.scene

    if EMPTY_NAME in scene.objects.keys():
        upd_off()
        bpy.data.objects[EMPTY_NAME].location = scene.sphere_location
        bpy.data.objects[EMPTY_NAME].rotation_euler = scene.sphere_rotation
        bpy.data.objects[EMPTY_NAME].scale = scene.sphere_scale
        bpy.data.objects[EMPTY_NAME].empty_display_size = scene.sphere_radius
        upd_on()

    if CAMERA_NAME in scene.objects.keys():
        upd_off()
        bpy.data.cameras[CAMERA_NAME].lens = scene.focal
        bpy.context.scene.objects[CAMERA_NAME].constraints['Track To'].track_axis = 'TRACK_Z' if scene.outwards else 'TRACK_NEGATIVE_Z'
        upd_on()

# if empty sphere modified outside of ui panel, edit panel properties
def properties_desgraph(scene):
    if scene.show_sphere and EMPTY_NAME in scene.objects.keys():
        upd_off()
        scene.sphere_location = bpy.data.objects[EMPTY_NAME].location
        scene.sphere_rotation = bpy.data.objects[EMPTY_NAME].rotation_euler
        scene.sphere_scale = bpy.data.objects[EMPTY_NAME].scale
        scene.sphere_radius = bpy.data.objects[EMPTY_NAME].empty_display_size
        upd_on()

    if scene.show_camera and CAMERA_NAME in scene.objects.keys():
        upd_off()
        scene.focal = bpy.data.cameras[CAMERA_NAME].lens
        scene.outwards = (bpy.context.scene.objects[CAMERA_NAME].constraints['Track To'].track_axis == 'TRACK_Z')
        upd_on()

    if EMPTY_NAME not in scene.objects.keys() and scene.sphere_exists:
        if CAMERA_NAME in scene.objects.keys() and scene.camera_exists:
            delete_camera(scene, CAMERA_NAME)

        scene.show_sphere = False
        scene.sphere_exists = False

    if CAMERA_NAME not in scene.objects.keys() and scene.camera_exists:
        scene.show_camera = False
        scene.camera_exists = False

        for block in bpy.data.cameras:
            if CAMERA_NAME in block.name:
                bpy.data.cameras.remove(block)

    if CAMERA_NAME in scene.objects.keys():
        scene.objects[CAMERA_NAME].location = sample_from_sphere(scene)

def empty_fn(self, context): pass

can_scene_upd = properties_ui
can_properties_upd = properties_desgraph

def upd_off():  # make sub function to an empty function
    global can_scene_upd, can_properties_upd
    can_scene_upd = empty_fn
    can_properties_upd = empty_fn
def upd_on():
    global can_scene_upd, can_properties_upd
    can_scene_upd = properties_ui
    can_properties_upd = properties_desgraph


## blender handler functions

# nerf_job_status: 0 idle, 1 running, 2 done, 3 cancelled
JOB_IDLE = 0
JOB_RUNNING = 1
JOB_DONE = 2
JOB_CANCELLED = 3

def wants_test_render(scene):
    return scene.test_data and scene.render_frames and not (scene.splats and scene.splats_test_dummy)

def dataset_output_path(scene):
    dataset_names = (scene.sof_dataset_name, scene.ttc_dataset_name, scene.cos_dataset_name)
    method_dataset_name = dataset_names[list(scene.rendering).index(True)]
    output_dir = bpy.path.clean_name(method_dataset_name)
    return os.path.join(scene.save_path, output_dir)

def maybe_compress_dataset(scene, output_path):
    '''Optionally zip the dataset folder and delete the uncompressed copy.'''
    if not scene.compress_to_zip:
        return
    shutil.make_archive(output_path, 'zip', output_path)
    shutil.rmtree(output_path)

def invoke_animation_render():
    '''Start an animation render with a VIEW_3D override when possible.'''
    wm = bpy.context.window_manager
    for window in wm.windows:
        screen = window.screen
        if screen is None:
            continue
        for area in screen.areas:
            if area.type != 'VIEW_3D':
                continue
            region = next((r for r in area.regions if r.type == 'WINDOW'), None)
            override = {'window': window, 'screen': screen, 'area': area}
            if region is not None:
                override['region'] = region
            with bpy.context.temp_override(**override):
                bpy.ops.render.render('INVOKE_DEFAULT', animation=True, write_still=True)
            return
    bpy.ops.render.render('INVOKE_DEFAULT', animation=True, write_still=True)

def begin_test_render(scene):
    output_test = os.path.join(dataset_output_path(scene), 'test')
    os.makedirs(output_test, exist_ok=True)

    if scene.rendering[0]:  # SOF : restore default frame step, keep full timeline
        scene.frame_step = scene.init_frame_step
    elif scene.rendering[1]:  # TTC : switch to test camera over the scene timeline
        scene.camera = scene.camera_test_target
        scene.frame_end = scene.init_frame_end
    elif scene.rendering[2]:  # COS : selected camera, Test Frames count
        scene.camera = scene.init_active_camera
        scene.frame_end = scene.frame_start + scene.cos_nb_test_frames - 1

    scene.render.filepath = os.path.join(output_test, '')
    invoke_animation_render()

def finalize_render(scene):
    if not any(scene.rendering):
        scene.nerf_job_status = JOB_IDLE
        return

    dataset_names = (scene.sof_dataset_name, scene.ttc_dataset_name, scene.cos_dataset_name)
    method_dataset_name = dataset_names[list(scene.rendering).index(True)]

    if scene.rendering[0]:
        scene.frame_step = scene.init_frame_step

    if scene.rendering[1]:
        scene.frame_end = scene.init_frame_end

    if scene.rendering[2]:
        if not scene.init_camera_exists:
            delete_camera(scene, CAMERA_NAME)
        if not scene.init_sphere_exists:
            objects = bpy.data.objects
            objects.remove(objects[EMPTY_NAME], do_unlink=True)
            scene.show_sphere = False
            scene.sphere_exists = False

        scene.camera = scene.init_active_camera
        scene.frame_end = scene.init_frame_end

    scene.rendering = (False, False, False)
    scene.nerf_job_status = JOB_IDLE
    scene.render.filepath = scene.init_output_path

    output_dir = bpy.path.clean_name(method_dataset_name)
    output_path = os.path.join(scene.save_path, output_dir)
    maybe_compress_dataset(scene, output_path)

@persistent
def post_render_complete(scene):
    if any(scene.rendering):
        scene.nerf_job_status = JOB_DONE

@persistent
def post_render_cancel(scene):
    if any(scene.rendering) and scene.nerf_job_status != JOB_DONE:
        scene.nerf_job_status = JOB_CANCELLED

# set initial property values (bpy.data and bpy.context require a loaded scene)
@persistent
def set_init_props(scene):
    filepath = bpy.data.filepath
    filename = bpy.path.basename(filepath)
    default_save_path = filepath[:-len(filename)] # remove file name from blender file path = directoy path

    scene.save_path = default_save_path
    scene.init_frame_step = scene.frame_step
    scene.init_output_path = scene.render.filepath

    bpy.app.handlers.depsgraph_update_post.remove(set_init_props)

# update cos camera when changing frame
@persistent
def cos_camera_update(scene):
    if CAMERA_NAME in scene.objects.keys():
        scene.objects[CAMERA_NAME].location = sample_from_sphere(scene)