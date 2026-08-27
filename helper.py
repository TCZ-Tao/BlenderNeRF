import os
import shutil
import random
import math
import traceback
import mathutils
import bpy
from bpy.app.handlers import persistent
from . import gbuffer


# global addon script variables
EMPTY_NAME = 'BlenderNeRF Sphere'
CAMERA_NAME = 'BlenderNeRF Camera'
SPIRAL_PATH_NAME = 'BlenderNeRF Spiral Path'
SPIRAL_TRACK_NAME = 'BlenderNeRF Spiral Track'

# NeRF blender-synthetic test path: two azimuth turns, one elevation cycle
SPIRAL_REVOLUTIONS = 2
SPIRAL_ELEV_MAX = math.radians(47.23205869382562)
SPIRAL_ELEV_MIN = math.radians(7.125016255700351)

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

        delete_spiral_path()
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

def delete_spiral_path():
    if SPIRAL_PATH_NAME not in bpy.data.objects:
        return
    obj = bpy.data.objects[SPIRAL_PATH_NAME]
    data = obj.data
    bpy.data.objects.remove(obj, do_unlink=True)
    if data is not None and data.users == 0:
        bpy.data.curves.remove(data)

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

def sphere_point_from_unit(scene, unit):
    '''Map a unit-sphere direction onto the BlenderNeRF Sphere (radius, scale, rotation, location).'''
    point = scene.sphere_radius * mathutils.Vector(scene.sphere_scale) * unit
    rotation = mathutils.Euler(scene.sphere_rotation).to_matrix()
    return mathutils.Vector(scene.sphere_location) + rotation @ point

def spiral_unit_on_sphere(i, n):
    '''Unit direction for frame i of n along the NeRF synthetic spherical spiral.

    Azimuth starts at +Y and completes two revolutions (endpoint-exclusive, like
    nerf_synthetic transforms_test.json). Elevation completes one cosine cycle
    between SPIRAL_ELEV_MAX and SPIRAL_ELEV_MIN.
    '''
    t = i / float(n)
    theta = math.pi / 2.0 + 2.0 * math.pi * SPIRAL_REVOLUTIONS * t
    elev = 0.5 * (SPIRAL_ELEV_MAX + SPIRAL_ELEV_MIN) + 0.5 * (SPIRAL_ELEV_MAX - SPIRAL_ELEV_MIN) * math.cos(2.0 * math.pi * t)
    cy = math.cos(elev)
    return mathutils.Vector((cy * math.cos(theta), cy * math.sin(theta), math.sin(elev)))

def spiral_positions_on_sphere(scene, n):
    return [sphere_point_from_unit(scene, spiral_unit_on_sphere(i, n)) for i in range(n)]

def iter_action_fcurves(id_data):
    '''Yield fcurves from a legacy or layered (Blender 4.4+/5) Action.'''
    ad = getattr(id_data, 'animation_data', None)
    if ad is None or ad.action is None:
        return
    action = ad.action
    fcurves = getattr(action, 'fcurves', None)
    if fcurves is not None and len(fcurves) > 0:
        for fc in fcurves:
            yield fc
        return
    for layer in getattr(action, 'layers', []):
        for strip in layer.strips:
            bags = getattr(strip, 'channelbags', None)
            if bags:
                for bag in bags:
                    for fc in bag.fcurves:
                        yield fc
            else:
                bag = getattr(strip, 'channelbag', None)
                if bag is not None:
                    for fc in bag.fcurves:
                        yield fc

def set_location_keyframe_interpolation(id_data, interpolation='LINEAR'):
    for fc in iter_action_fcurves(id_data):
        if fc.data_path != 'location':
            continue
        for kp in fc.keyframe_points:
            kp.interpolation = interpolation

def world_to_local_location(obj, world_location):
    if obj.parent is None:
        return world_location.copy()
    return obj.parent.matrix_world.inverted() @ world_location

def ensure_spiral_track_to(camera, scene):
    for c in list(camera.constraints):
        if c.type != 'TRACK_TO' or c.name == SPIRAL_TRACK_NAME:
            continue
        tgt = getattr(c, 'target', None)
        if tgt is not None and tgt.name == EMPTY_NAME:
            camera.constraints.remove(c)

    track = camera.constraints.get(SPIRAL_TRACK_NAME)
    if track is None or track.type != 'TRACK_TO':
        if track is not None:
            camera.constraints.remove(track)
        track = camera.constraints.new(type='TRACK_TO')
        track.name = SPIRAL_TRACK_NAME
    track.target = bpy.data.objects[EMPTY_NAME]
    track.track_axis = 'TRACK_Z' if scene.outwards else 'TRACK_NEGATIVE_Z'
    track.up_axis = 'UP_Y'
    return track

def update_spiral_path_curve(scene, positions):
    delete_spiral_path()

    curve_data = bpy.data.curves.new(SPIRAL_PATH_NAME, type='CURVE')
    curve_data.dimensions = '3D'
    curve_data.resolution_u = 2
    curve_data.bevel_depth = max(0.002, 0.005 * scene.sphere_radius)
    curve_data.bevel_resolution = 2
    curve_data.use_fill_caps = True

    spline = curve_data.splines.new('POLY')
    n = len(positions)
    spline.points.add(max(0, n - 1))
    for i, p in enumerate(positions):
        spline.points[i].co = (p.x, p.y, p.z, 1.0)

    curve_obj = bpy.data.objects.new(SPIRAL_PATH_NAME, curve_data)
    scene.collection.objects.link(curve_obj)
    curve_obj.hide_render = True
    curve_obj.show_in_front = True
    curve_obj.color = (1.0, 0.45, 0.08, 1.0)
    return curve_obj

def apply_spherical_spiral(scene, camera):
    '''Keyframe camera along a spherical spiral on the BlenderNeRF Sphere.'''
    n = scene.cos_nb_test_frames
    positions = spiral_positions_on_sphere(scene, n)

    if camera.animation_data:
        camera.animation_data_clear()

    ensure_spiral_track_to(camera, scene)
    update_spiral_path_curve(scene, positions)

    frame_start = scene.frame_start
    frame_end = frame_start + n - 1
    scene.frame_end = frame_end

    for i, world_loc in enumerate(positions):
        camera.location = world_to_local_location(camera, world_loc)
        camera.keyframe_insert(data_path='location', frame=frame_start + i)

    set_location_keyframe_interpolation(camera, 'LINEAR')
    scene.frame_set(frame_start)
    return frame_start, frame_end, n

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

    camera = scene.camera
    if camera is not None and SPIRAL_TRACK_NAME in camera.constraints:
        upd_off()
        camera.constraints[SPIRAL_TRACK_NAME].track_axis = 'TRACK_Z' if scene.outwards else 'TRACK_NEGATIVE_Z'
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

        delete_spiral_path()
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

def wants_any_image_render(scene):
    return bool(scene.render_frames and gbuffer.selected_output_channels(scene))

def dataset_output_path(scene):
    dataset_names = (scene.sof_dataset_name, scene.ttc_dataset_name, scene.cos_dataset_name)
    method_dataset_name = dataset_names[list(scene.rendering).index(True)]
    output_dir = bpy.path.clean_name(method_dataset_name)
    return os.path.join(scene.save_path, output_dir)

def images_output_dir(scene, split, channel):
    base = os.path.join(dataset_output_path(scene), split)
    if scene.gbuffer:
        return os.path.join(base, channel)
    return base

def maybe_compress_dataset(scene, output_path):
    '''Optionally zip the dataset folder and delete the uncompressed copy.'''
    if not scene.compress_to_zip:
        return
    shutil.make_archive(output_path, 'zip', output_path)
    shutil.rmtree(output_path)

_pipeline_timer_registered = False


def is_render_busy():
    is_job = getattr(bpy.app, 'is_job_running', None)
    if callable(is_job):
        try:
            if is_job('RENDER') or is_job('RENDER_PREVIEW'):
                return True
        except (TypeError, ValueError):
            pass
    return bool(getattr(bpy.app, 'is_rendering', False))


def _keep_ui_display_type():
    '''Blender 5 renamed Keep User Interface from KEEP_UI to NONE.'''
    view = bpy.context.preferences.view
    items = {item.identifier for item in view.bl_rna.properties['render_display_type'].enum_items}
    if 'NONE' in items:
        return 'NONE'
    if 'KEEP_UI' in items:
        return 'KEEP_UI'
    return view.render_display_type


def apply_hide_render_view(scene):
    '''Switch Blender to Keep User Interface so the render result window does not open.'''
    if not getattr(scene, 'hide_render_view', False):
        return
    view = bpy.context.preferences.view
    if not getattr(scene, 'init_render_display_type', ''):
        scene.init_render_display_type = view.render_display_type
    view.render_display_type = _keep_ui_display_type()


def restore_hide_render_view(scene):
    stored = getattr(scene, 'init_render_display_type', '') or ''
    if not stored:
        return
    try:
        bpy.context.preferences.view.render_display_type = stored
    except TypeError:
        pass
    scene.init_render_display_type = ''


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
                return bpy.ops.render.render('INVOKE_DEFAULT', animation=True, write_still=True)
    return bpy.ops.render.render('INVOKE_DEFAULT', animation=True, write_still=True)

def configure_train_timeline(scene):
    '''Re-apply train camera and frame range. Animation renders must not inherit a drifted timeline.'''
    if scene.rendering[0]:  # SOF
        scene.frame_step = scene.train_frame_steps
    elif scene.rendering[1]:  # TTC
        if scene.camera_train_target is not None:
            scene.camera = scene.camera_train_target
        scene.frame_end = scene.frame_start + scene.ttc_nb_frames - 1
    elif scene.rendering[2]:  # COS
        if CAMERA_NAME in scene.objects:
            scene.camera = scene.objects[CAMERA_NAME]
        scene.frame_end = scene.frame_start + scene.cos_nb_frames - 1

def configure_test_timeline(scene):
    if scene.rendering[0]:  # SOF : restore default frame step, keep full timeline
        scene.frame_step = scene.init_frame_step
    elif scene.rendering[1]:  # TTC : switch to test camera over the scene timeline
        scene.camera = scene.camera_test_target
        scene.frame_end = scene.init_frame_end
    elif scene.rendering[2]:  # COS : selected camera, Test Frames count
        restore_user_camera(scene)
        scene.frame_end = scene.frame_start + scene.cos_nb_test_frames - 1

def restore_user_camera(scene):
    '''Put the COS Camera field back to the user camera, never leave BlenderNeRF Camera selected.'''
    name = getattr(scene, 'init_active_camera_name', '') or ''
    cam = scene.init_active_camera
    if name and name in scene.objects and scene.objects[name].type == 'CAMERA':
        scene.camera = scene.objects[name]
        return
    if cam is not None:
        scene.camera = cam

def start_render_pass(scene):
    '''Configure the current G-buffer/RGB pass and invoke an animation render.'''
    scene.nerf_job_status = JOB_RUNNING
    current = gbuffer.current_pass()
    if current is None:
        scene.nerf_job_status = JOB_DONE
        finalize_render(scene)
        return

    split, channel = current
    if split == 'test':
        configure_test_timeline(scene)
    else:
        configure_train_timeline(scene)

    print(
        f"[BlenderNeRF] {split}/{channel}  "
        f"frames {scene.frame_start}-{scene.frame_end} step {scene.frame_step}  "
        f"camera {scene.camera.name if scene.camera else None}"
    )

    out_dir = images_output_dir(scene, split, channel)
    gbuffer.apply_pass_settings(scene, channel, out_dir)
    apply_hide_render_view(scene)
    invoke_animation_render()

def schedule_next_render_pass():
    global _pipeline_timer_registered
    if _pipeline_timer_registered:
        return
    _pipeline_timer_registered = True
    bpy.app.timers.register(_continue_render_pipeline, first_interval=0.35)

def _continue_render_pipeline():
    global _pipeline_timer_registered
    scene = bpy.context.scene

    if is_render_busy():
        return 0.25

    _pipeline_timer_registered = False

    if not any(scene.rendering):
        return None

    if scene.nerf_job_status == JOB_CANCELLED:
        finalize_render(scene)
        return None

    if scene.nerf_job_status != JOB_DONE:
        return None

    try:
        if gbuffer.advance_pass():
            start_render_pass(scene)
        else:
            finalize_render(scene)
    except Exception:
        traceback.print_exc()
        finalize_render(scene)
    return None

def begin_test_render(scene):
    configure_test_timeline(scene)
    channels = gbuffer.selected_output_channels(scene)
    channel = channels[0] if channels else gbuffer.RGBA_CHANNEL
    output_test = images_output_dir(scene, 'test', channel)
    os.makedirs(output_test, exist_ok=True)
    scene.render.filepath = os.path.join(output_test, '')
    invoke_animation_render()

def finalize_render(scene):
    gbuffer.end_job(scene)
    restore_hide_render_view(scene)

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
        restore_user_camera(scene)
        if not scene.init_camera_exists:
            delete_camera(scene, CAMERA_NAME)
        if not scene.init_sphere_exists:
            delete_spiral_path()
            objects = bpy.data.objects
            objects.remove(objects[EMPTY_NAME], do_unlink=True)
            scene.show_sphere = False
            scene.sphere_exists = False

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
        schedule_next_render_pass()

@persistent
def post_render_cancel(scene):
    if any(scene.rendering) and scene.nerf_job_status == JOB_RUNNING:
        scene.nerf_job_status = JOB_CANCELLED
        finalize_render(scene)

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