import os
import json
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
AABB_EMPTY_NAME = 'BlenderNeRF AABB'
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

def _link_object(scene, obj):
    '''Link an object without bpy.ops so this works in blender -b.'''
    coll = getattr(bpy.context, 'collection', None) or scene.collection
    try:
        coll.objects.link(obj)
    except RuntimeError:
        if obj.name not in scene.collection.objects:
            scene.collection.objects.link(obj)


def visualize_sphere(self, context):
    scene = context.scene

    if EMPTY_NAME not in scene.objects.keys() and not scene.sphere_exists:
        empty = bpy.data.objects.get(EMPTY_NAME)
        if empty is None:
            empty = bpy.data.objects.new(EMPTY_NAME, None)
        empty.empty_display_type = 'SPHERE'
        empty.location = scene.sphere_location
        empty.rotation_euler = scene.sphere_rotation
        empty.scale = scene.sphere_scale
        empty.empty_display_size = scene.sphere_radius
        _link_object(scene, empty)
        view_layer = getattr(context, 'view_layer', None)
        if view_layer is not None:
            view_layer.objects.active = empty

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

        cam_data = bpy.data.cameras.get(CAMERA_NAME)
        if cam_data is None:
            cam_data = bpy.data.cameras.new(CAMERA_NAME)
        camera = bpy.data.objects.get(CAMERA_NAME)
        if camera is None:
            camera = bpy.data.objects.new(CAMERA_NAME, cam_data)
        else:
            camera.data = cam_data
        camera.location = sample_from_sphere(scene)
        cam_data.lens = scene.focal
        _link_object(scene, camera)
        view_layer = getattr(context, 'view_layer', None)
        if view_layer is not None:
            view_layer.objects.active = camera

        cam_constraint = next((c for c in camera.constraints if c.type == 'TRACK_TO'), None)
        if cam_constraint is None:
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

def aabb_empty_half_extent(scene):
    '''Instant NGP aabb_scale is the cube side length, centered at the world origin.'''
    return scene.aabb * 0.5

def apply_aabb_empty(scene, empty):
    empty.empty_display_type = 'CUBE'
    empty.empty_display_size = aabb_empty_half_extent(scene)
    empty.location = (0.0, 0.0, 0.0)
    empty.rotation_euler = (0.0, 0.0, 0.0)
    empty.scale = (1.0, 1.0, 1.0)
    empty.hide_render = True
    empty.show_in_front = True
    empty.show_name = True

def aabb_empty_needs_sync(scene, empty):
    half = aabb_empty_half_extent(scene)
    return (
        empty.empty_display_type != 'CUBE'
        or abs(empty.empty_display_size - half) > 1e-6
        or empty.location.length_squared > 1e-12
        or abs(empty.rotation_euler[0]) > 1e-6
        or abs(empty.rotation_euler[1]) > 1e-6
        or abs(empty.rotation_euler[2]) > 1e-6
        or abs(empty.scale.x - 1.0) > 1e-6
        or abs(empty.scale.y - 1.0) > 1e-6
        or abs(empty.scale.z - 1.0) > 1e-6
    )

def visualize_aabb(self, context):
    scene = context.scene
    empty = bpy.data.objects.get(AABB_EMPTY_NAME)
    in_scene = empty is not None and AABB_EMPTY_NAME in scene.objects.keys()

    if scene.show_aabb:
        if empty is None:
            empty = bpy.data.objects.new(AABB_EMPTY_NAME, None)
        apply_aabb_empty(scene, empty)
        if not in_scene:
            _link_object(scene, empty)
        view_layer = getattr(context, 'view_layer', None)
        if view_layer is not None:
            view_layer.objects.active = empty
        scene.aabb_exists = True
    else:
        if empty is not None:
            bpy.data.objects.remove(empty, do_unlink=True)
        scene.aabb_exists = False

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

def set_keyframe_interpolation(id_data, data_paths, interpolation='LINEAR'):
    paths = {data_paths} if isinstance(data_paths, str) else set(data_paths)
    for fc in iter_action_fcurves(id_data):
        if fc.data_path not in paths:
            continue
        for kp in fc.keyframe_points:
            kp.interpolation = interpolation

_POSE_PATHS = ('location', 'rotation_euler', 'rotation_quaternion', 'rotation_axis_angle', 'scale')

def rotation_data_path(obj):
    if obj.rotation_mode == 'QUATERNION':
        return 'rotation_quaternion'
    if obj.rotation_mode == 'AXIS_ANGLE':
        return 'rotation_axis_angle'
    return 'rotation_euler'

def _pose_fcurves(obj):
    return [fc for fc in iter_action_fcurves(obj) if fc.data_path in _POSE_PATHS]

def _has_nla_strips(obj):
    ad = obj.animation_data
    if ad is None:
        return False
    return any(track.strips for track in ad.nla_tracks)

def _has_transform_drivers(obj):
    ad = obj.animation_data
    if ad is None:
        return False
    return any(driver.data_path in _POSE_PATHS for driver in ad.drivers)

def pose_span(obj):
    '''Inclusive integer frame span of pose keys and NLA strips, or None.

    The third value is True when a key does not sit on an integer frame.
    '''
    times = [float(kp.co[0]) for fc in _pose_fcurves(obj) for kp in fc.keyframe_points]
    ad = obj.animation_data
    if ad is not None:
        for track in ad.nla_tracks:
            for strip in track.strips:
                times.append(float(strip.frame_start))
                times.append(float(strip.frame_end))
    if not times:
        return None
    subframe = any(abs(t - round(t)) > 1e-4 for t in times)
    start = int(math.floor(min(times) + 1e-8))
    end = int(math.ceil(max(times) - 1e-8))
    return start, max(start, end), subframe

def every_pose_frame_is_keyed(obj):
    '''True when every integer frame in the pose span has a key on each pose curve.'''
    if _has_nla_strips(obj):
        return False
    span = pose_span(obj)
    if span is None:
        return True
    start, end, subframe = span
    if subframe:
        return False
    needed = range(start, end + 1)
    for fc in _pose_fcurves(obj):
        got = {int(round(kp.co[0])) for kp in fc.keyframe_points}
        if any(frame not in got for frame in needed):
            return False
    return True

def poses_are_literal(obj):
    '''True when sampled integer frames are the keyed pose, with nothing else on top.'''
    if obj.constraints or _has_transform_drivers(obj) or _has_nla_strips(obj):
        return False
    return every_pose_frame_is_keyed(obj)

def split_test_count(n, ratio):
    '''How many of n pooled poses go to the test camera.

    Half-up rounding. When n >= 2 and the ratio is strictly between 0 and 1,
    both sides keep at least one pose.
    '''
    count = int(math.floor(n * float(ratio) + 0.5))
    if n >= 2 and 0.0 < ratio < 1.0:
        return min(n - 1, max(1, count))
    return min(n, max(0, count))

def split_pose_indices(n, ratio, seed):
    '''Random test indices. Both lists keep the pooled order.'''
    n_test = split_test_count(n, ratio)
    chosen = set(random.Random(int(seed)).sample(range(n), n_test)) if n_test else set()
    train_idx = [i for i in range(n) if i not in chosen]
    test_idx = [i for i in range(n) if i in chosen]
    return train_idx, test_idx

def _sample_world_matrices(scene, camera, frames):
    matrices = []
    for frame in frames:
        scene.frame_set(frame)
        bpy.context.view_layer.update()
        matrices.append(camera.matrix_world.copy())
    return matrices

def _apply_world_matrix(obj, world_matrix):
    if obj.parent is None:
        obj.matrix_basis = world_matrix.copy()
        return
    obj.matrix_basis = obj.matrix_parent_inverse.inverted() @ obj.parent.matrix_world.inverted() @ world_matrix

def write_world_poses(scene, camera, matrices, frame_start):
    '''Replace pose keys and constraints with one literal world pose per frame.'''
    while camera.constraints:
        camera.constraints.remove(camera.constraints[0])
    if camera.animation_data:
        camera.animation_data_clear()
    if not matrices:
        return
    rot_path = rotation_data_path(camera)
    for i, world in enumerate(matrices):
        frame = frame_start + i
        scene.frame_set(frame)
        bpy.context.view_layer.update()
        _apply_world_matrix(camera, world)
        camera.keyframe_insert(data_path='location', frame=frame)
        camera.keyframe_insert(data_path=rot_path, frame=frame)
        camera.keyframe_insert(data_path='scale', frame=frame)
    set_keyframe_interpolation(camera, ('location', rot_path, 'scale'), 'LINEAR')

def ensure_per_frame_pose_keys(scene, camera):
    '''Bake a visual pose onto every integer frame in the camera's own key span.

    A camera with no pose keys contributes nothing. Assign reads the result only
    after this, when every frame is a keyframe.
    '''
    span = pose_span(camera)
    if span is None or poses_are_literal(camera):
        return
    matrices = _sample_world_matrices(scene, camera, range(span[0], span[1] + 1))
    write_world_poses(scene, camera, matrices, span[0])

def _iter_pose_matrices(scene, camera):
    span = pose_span(camera)
    if span is None:
        return []
    return _sample_world_matrices(scene, camera, range(span[0], span[1] + 1))

def assign_ttc_split(scene, train_camera, test_camera, ratio, seed):
    '''Split already-literal per-frame poses. Train frames first, then test frames.'''
    for camera in (train_camera, test_camera):
        if pose_span(camera) is not None and not poses_are_literal(camera):
            raise RuntimeError('TTC split expects a key on every frame of ' + camera.name)
    pool = _iter_pose_matrices(scene, train_camera)
    pool += _iter_pose_matrices(scene, test_camera)
    train_idx, test_idx = split_pose_indices(len(pool), ratio, seed)
    write_world_poses(scene, train_camera, [pool[i] for i in train_idx], scene.frame_start)
    write_world_poses(scene, test_camera, [pool[i] for i in test_idx], scene.frame_start)
    return len(train_idx), len(test_idx)

def redistribute_ttc_frames(scene, train_camera, test_camera, ratio, seed):
    '''Bake both cameras to per-frame keys, pool those poses, then split by ratio.'''
    if train_camera == test_camera:
        raise ValueError('Train and test cameras must be different objects')
    ensure_per_frame_pose_keys(scene, train_camera)
    ensure_per_frame_pose_keys(scene, test_camera)
    counts = assign_ttc_split(scene, train_camera, test_camera, ratio, seed)
    scene.frame_set(scene.frame_start)
    return counts

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

    set_keyframe_interpolation(camera, 'location', 'LINEAR')
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

    if AABB_EMPTY_NAME in scene.objects.keys():
        upd_off()
        apply_aabb_empty(scene, bpy.data.objects[AABB_EMPTY_NAME])
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

    if scene.show_aabb and AABB_EMPTY_NAME in scene.objects.keys():
        empty = bpy.data.objects[AABB_EMPTY_NAME]
        if aabb_empty_needs_sync(scene, empty):
            upd_off()
            apply_aabb_empty(scene, empty)
            upd_on()

    if AABB_EMPTY_NAME not in scene.objects.keys() and scene.aabb_exists:
        scene.show_aabb = False
        scene.aabb_exists = False

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

def wants_test_json(scene):
    return bool(scene.test_data or scene.splats_test_dummy)

def wants_test_render(scene):
    return scene.test_data and scene.render_frames and not scene.splats_test_dummy

def wants_any_image_render(scene):
    return bool(scene.render_frames and gbuffer.selected_output_channels(scene))

def _image_filename(image):
    filepath = image.filepath_raw or image.filepath
    if filepath:
        name = os.path.basename(bpy.path.abspath(filepath).replace('\\', '/'))
        if name:
            return name
    return image.name

def _linked_from(socket):
    if socket is not None and socket.is_linked:
        return socket.links[0].from_node
    return None

def _downstream_nodes(start):
    visited = set()
    stack = [start]
    while stack:
        node = stack.pop()
        if node is None or node in visited:
            continue
        visited.add(node)
        yield node
        for out in node.outputs:
            for link in out.links:
                stack.append(link.to_node)

def _find_background_node(node_tree):
    output = next((n for n in node_tree.nodes if n.type == 'OUTPUT_WORLD' and getattr(n, 'is_active_output', True)), None)
    if output is None:
        output = next((n for n in node_tree.nodes if n.type == 'OUTPUT_WORLD'), None)

    start = _linked_from(output.inputs['Surface']) if output and 'Surface' in output.inputs else None
    stack = [start] if start else []
    visited = set()
    while stack:
        node = stack.pop()
        if node is None or node in visited:
            continue
        visited.add(node)
        if node.type == 'BACKGROUND':
            return node
        stack.extend(_linked_from(inp) for inp in node.inputs)

    return next((n for n in node_tree.nodes if n.type == 'BACKGROUND'), None)

def _world_env_nodes(world, require_image=True):
    '''Return (Environment Texture node, Background node) from a World shader.'''
    if world is None or not (world.use_nodes and world.node_tree):
        return None, None

    node_tree = world.node_tree
    env_texes = [
        n for n in node_tree.nodes
        if n.type == 'TEX_ENVIRONMENT' and (n.image or not require_image)
    ]
    env_tex = None
    background = None
    for candidate in env_texes:
        bg = next((n for n in _downstream_nodes(candidate) if n.type == 'BACKGROUND'), None)
        if bg is not None:
            env_tex = candidate
            background = bg
            break
    if env_tex is None and env_texes:
        env_tex = env_texes[0]
    if background is None:
        background = _find_background_node(node_tree)
    return env_tex, background


def world_envmap_info(scene):
    '''Return (envmap filename, background strength) from the World shader.'''
    envmap = ''
    envmap_inten = 1.0
    env_tex, background = _world_env_nodes(scene.world, require_image=True)

    if env_tex is not None and env_tex.image:
        envmap = _image_filename(env_tex.image)
    if background is not None:
        envmap_inten = float(background.inputs['Strength'].default_value)

    return envmap, envmap_inten


_relight_world_state = None


def resolved_envmap_path(filepath):
    '''Absolute path for a World HDRI; DIR/FILE_PATH values may be stored as // relative paths.'''
    return os.path.abspath(os.path.expanduser(bpy.path.abspath(filepath)))


def envmap_stem(filepath):
    '''Folder name for test_rli/<stem>/ : filename without extension, cleaned.'''
    return bpy.path.clean_name(os.path.splitext(os.path.basename(filepath))[0])


def method_dataset_name(scene, method):
    '''Dataset folder name for a method. The Name field as-is — no SOF/TTC/COS prefix.'''
    if method == 'SOF':
        return scene.sof_dataset_name
    if method == 'TTC':
        return scene.ttc_dataset_name
    return scene.cos_dataset_name


def rendering_flags_for_method(method):
    if method == 'SOF':
        return (True, False, False)
    if method == 'TTC':
        return (False, True, False)
    return (False, False, True)


def relight_output_dir(scene, envmap_path, method=None):
    '''<save_path>/<dataset_name>/test_rli/<envmap_stem> — dataset_name is the method Name field.'''
    method = method or scene.relight_method
    name = bpy.path.clean_name(method_dataset_name(scene, method))
    stem = envmap_stem(envmap_path)
    return os.path.join(resolved_save_path(scene), name, 'test_rli', stem)


def apply_world_envmap(scene, filepath):
    '''Point the World Environment Texture at filepath. Scene lights and Film are left alone.'''
    global _relight_world_state
    restore_world_envmap(scene)

    filepath = resolved_envmap_path(filepath)
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f'Environment map file not found: {filepath}')

    prev_scene_world = scene.world
    created_world = False
    world = scene.world
    if world is None:
        world = bpy.data.worlds.new('BlenderNeRF World')
        scene.world = world
        created_world = True

    prev_use_nodes = bool(world.use_nodes)
    world.use_nodes = True
    node_tree = world.node_tree

    env_tex, background = _world_env_nodes(world, require_image=False)
    created_env_tex = False
    created_background = False
    created_output = False

    output = next((n for n in node_tree.nodes if n.type == 'OUTPUT_WORLD' and getattr(n, 'is_active_output', True)), None)
    if output is None:
        output = next((n for n in node_tree.nodes if n.type == 'OUTPUT_WORLD'), None)
    if output is None:
        output = node_tree.nodes.new('ShaderNodeOutputWorld')
        created_output = True

    if background is None:
        background = node_tree.nodes.new('ShaderNodeBackground')
        created_background = True
        if 'Surface' in output.inputs:
            node_tree.links.new(background.outputs['Background'], output.inputs['Surface'])

    if env_tex is None:
        env_tex = node_tree.nodes.new('ShaderNodeTexEnvironment')
        created_env_tex = True
        if 'Color' in background.inputs:
            node_tree.links.new(env_tex.outputs['Color'], background.inputs['Color'])
    elif background is not None and 'Color' in background.inputs and not env_tex.outputs['Color'].is_linked:
        node_tree.links.new(env_tex.outputs['Color'], background.inputs['Color'])

    prev_image = env_tex.image
    env_tex.image = bpy.data.images.load(filepath, check_existing=True)

    _relight_world_state = {
        'prev_scene_world': prev_scene_world,
        'created_world': created_world,
        'world': world,
        'prev_use_nodes': prev_use_nodes,
        'env_tex': env_tex,
        'prev_image': prev_image,
        'created_env_tex': created_env_tex,
        'created_background': created_background,
        'created_output': created_output,
        'background': background,
        'output': output,
    }
    return filepath


def restore_world_envmap(scene):
    '''Undo apply_world_envmap. Safe to call when no swap is pending.'''
    global _relight_world_state
    state = _relight_world_state
    _relight_world_state = None
    if not state:
        return

    env_tex = state.get('env_tex')
    world = state.get('world')
    node_tree = world.node_tree if world is not None and getattr(world, 'node_tree', None) else None

    if env_tex is not None:
        try:
            env_tex.image = state.get('prev_image')
        except (ReferenceError, RuntimeError):
            env_tex = None

    if node_tree is not None:
        if state.get('created_env_tex') and env_tex is not None:
            try:
                node_tree.nodes.remove(env_tex)
            except (ReferenceError, RuntimeError):
                pass
        if state.get('created_background') and state.get('background') is not None:
            try:
                node_tree.nodes.remove(state['background'])
            except (ReferenceError, RuntimeError):
                pass
        if state.get('created_output') and state.get('output') is not None:
            try:
                node_tree.nodes.remove(state['output'])
            except (ReferenceError, RuntimeError):
                pass

    if state.get('created_world'):
        scene.world = state.get('prev_scene_world')
        if world is not None:
            try:
                bpy.data.worlds.remove(world)
            except (ReferenceError, RuntimeError):
                pass
    elif world is not None:
        try:
            world.use_nodes = state.get('prev_use_nodes', True)
        except (ReferenceError, RuntimeError):
            pass

def render_spp(scene):
    engine = scene.render.engine
    if engine == 'CYCLES':
        return int(scene.cycles.samples)
    eevee = getattr(scene, 'eevee', None)
    if eevee is not None:
        for attr in ('taa_render_samples', 'taa_samples'):
            if hasattr(eevee, attr):
                return int(getattr(eevee, attr))
    return 0

def camera_clip_fields(camera):
    '''Per-camera clip, lens shift, and depth of field; same units as Camera data.'''
    cam = camera.data
    fields = {
        'clip_start': round(float(cam.clip_start), 6),
        'clip_end': round(float(cam.clip_end), 6),
        'shift_x': round(float(cam.shift_x), 6),
        'shift_y': round(float(cam.shift_y), 6),
    }
    dof = getattr(cam, 'dof', None)
    fields['use_dof'] = bool(dof.use_dof) if dof is not None else False
    if fields['use_dof']:
        fields['focus_distance'] = round(float(dof.focus_distance), 6)
        fields['aperture_fstop'] = round(float(dof.aperture_fstop), 6)
    return fields

def write_scene_metadata(scene, output_path, dataset_name):
    '''Write a TensoIR-style scene metadata.json next to transforms_*.json.'''
    scale = scene.render.resolution_percentage / 100.0
    envmap, envmap_inten = world_envmap_info(scene)
    data = {
        'scene': dataset_name,
        'imw': int(round(scene.render.resolution_x * scale)),
        'imh': int(round(scene.render.resolution_y * scale)),
        'envmap': envmap,
        'envmap_inten': envmap_inten,
        'spp': render_spp(scene),
        'film_transparent': bool(scene.render.film_transparent),
    }
    filepath = os.path.join(output_path, 'metadata.json')
    with open(filepath, 'w') as file:
        json.dump(data, file, indent=4)

def resolved_save_path(scene):
    '''Absolute output directory; DIR_PATH values may be stored as // relative paths.'''
    return bpy.path.abspath(scene.save_path)

def dataset_output_path(scene):
    dataset_names = (scene.sof_dataset_name, scene.ttc_dataset_name, scene.cos_dataset_name)
    name = dataset_names[list(scene.rendering).index(True)]
    return os.path.join(resolved_save_path(scene), bpy.path.clean_name(name))

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
    if bpy.app.background or not getattr(scene, 'hide_render_view', False):
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
    '''Start an animation render. Blocking EXEC in blender -b, INVOKE in the UI.'''
    if bpy.app.background:
        return bpy.ops.render.render(animation=True, write_still=True)

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
    elif scene.rendering[1]:  # TTC : switch to test camera over the Test Frames range
        scene.camera = scene.camera_test_target
        scene.frame_end = scene.frame_start + scene.ttc_nb_test_frames - 1
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

def _file_is_present(path):
    try:
        return os.path.isfile(path) and os.path.getsize(path) > 0
    except OSError:
        return False


def expected_frame_path(scene, out_dir, frame, channel=None):
    '''Path Blender would write for this frame under out_dir, with the channel extension.'''
    prev = scene.render.filepath
    scene.render.filepath = os.path.join(out_dir, '')
    try:
        path = bpy.path.abspath(scene.render.frame_path(frame=frame))
    finally:
        scene.render.filepath = prev
    if channel is not None:
        ext = gbuffer.channel_extension(scene, channel)
        root, cur = os.path.splitext(path)
        if cur.lower() != ext.lower():
            path = root + ext
    return path


def count_missing_frames(scene, out_dir, channel=None):
    '''Return (missing, total) for the current timeline under out_dir.'''
    missing = 0
    total = 0
    step = max(1, int(scene.frame_step))
    for frame in range(int(scene.frame_start), int(scene.frame_end) + 1, step):
        total += 1
        path = expected_frame_path(scene, out_dir, frame, channel)
        if not _file_is_present(path):
            missing += 1
    return missing, total


def should_write_splats_ply(scene, output_path):
    if not scene.splats:
        return False
    if getattr(scene, 'supplement_mode', False):
        ply = os.path.join(output_path, 'points3d.ply')
        if _file_is_present(ply):
            print('[BlenderNeRF] skip points3d.ply (already exists)')
            return False
    return True


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

    out_dir = images_output_dir(scene, split, channel)
    print(
        f"[BlenderNeRF] {split}/{channel}  "
        f"frames {scene.frame_start}-{scene.frame_end} step {scene.frame_step}  "
        f"camera {scene.camera.name if scene.camera else None}  "
        f"-> {out_dir}"
    )

    if getattr(scene, 'supplement_mode', False):
        missing, total = count_missing_frames(scene, out_dir, channel)
        if total > 0 and missing == 0:
            print(f'[BlenderNeRF] skip {split}/{channel} ({total} frames already exist)')
            scene.nerf_job_status = JOB_DONE
            if not bpy.app.background:
                schedule_next_render_pass()
            return
        print(f'[BlenderNeRF] supplement {split}/{channel}: {missing} missing / {total} frames')

    gbuffer.apply_pass_settings(scene, channel, out_dir)
    if getattr(scene, 'supplement_mode', False):
        scene.render.use_overwrite = False
    apply_hide_render_view(scene)
    result = invoke_animation_render()

    if bpy.app.background:
        if result == {'CANCELLED'}:
            scene.nerf_job_status = JOB_CANCELLED
        elif scene.nerf_job_status == JOB_RUNNING:
            scene.nerf_job_status = JOB_DONE

def launch_render_pipeline(do_train, do_test):
    '''Start the train/test G-buffer pipeline. Blocking under blender -b.'''
    bpy.ops.object.blendernerf_render_pipeline(do_train=do_train, do_test=do_test)

def run_render_pipeline_sync(scene):
    '''Run every G-buffer/RGB pass with blocking renders (headless).'''
    try:
        while gbuffer.current_pass() is not None:
            start_render_pass(scene)
            if scene.nerf_job_status == JOB_CANCELLED:
                break
            if not gbuffer.advance_pass():
                break
        finalize_render(scene)
    except Exception:
        traceback.print_exc()
        finalize_render(scene)
        raise

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

    if getattr(scene, 'relight_active', False):
        return None

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

_relight_use_overwrite = None


def start_relight_render(scene, out_dir):
    '''Configure the test-camera timeline and invoke a single RGB animation render.'''
    global _relight_use_overwrite
    scene.nerf_job_status = JOB_RUNNING
    configure_test_timeline(scene)
    print(
        f"[BlenderNeRF] relight/{scene.relight_method}  "
        f"frames {scene.frame_start}-{scene.frame_end} step {scene.frame_step}  "
        f"camera {scene.camera.name if scene.camera else None}  "
        f"-> {out_dir}"
    )
    if getattr(scene, 'supplement_mode', False):
        missing, total = count_missing_frames(scene, out_dir)
        if total > 0 and missing == 0:
            print(f'[BlenderNeRF] skip relight ({total} frames already exist)')
            scene.nerf_job_status = JOB_DONE
            if not bpy.app.background:
                finalize_relight(scene)
            return {'FINISHED'}
        _relight_use_overwrite = getattr(scene.render, 'use_overwrite', True)
        scene.render.use_overwrite = False
        print(f'[BlenderNeRF] supplement relight: {missing} missing / {total} frames')
    apply_hide_render_view(scene)
    result = invoke_animation_render()

    if bpy.app.background:
        if result == {'CANCELLED'}:
            scene.nerf_job_status = JOB_CANCELLED
        elif scene.nerf_job_status == JOB_RUNNING:
            scene.nerf_job_status = JOB_DONE
    return result

def finalize_relight(scene):
    '''Restore World HDRI, camera, and timeline after a relight render. Does not zip.'''
    global _relight_use_overwrite
    if _relight_use_overwrite is not None and hasattr(scene.render, 'use_overwrite'):
        scene.render.use_overwrite = _relight_use_overwrite
        _relight_use_overwrite = None
    restore_world_envmap(scene)
    restore_hide_render_view(scene)

    if not getattr(scene, 'relight_active', False) and not any(scene.rendering):
        scene.nerf_job_status = JOB_IDLE
        return

    if scene.rendering[0]:
        scene.frame_step = scene.init_frame_step

    if scene.rendering[1] or scene.rendering[2]:
        scene.frame_end = scene.init_frame_end

    restore_user_camera(scene)

    out_dir = scene.render.filepath
    scene.rendering = (False, False, False)
    scene.relight_active = False
    scene.nerf_job_status = JOB_IDLE
    scene.render.filepath = scene.init_output_path
    print(f"[BlenderNeRF] relight done: {out_dir}")

def finalize_render(scene):
    if getattr(scene, 'relight_active', False):
        finalize_relight(scene)
        return

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
    output_path = os.path.join(resolved_save_path(scene), output_dir)
    maybe_compress_dataset(scene, output_path)
    print(f"[BlenderNeRF] done: {output_path}")

@persistent
def post_render_complete(scene):
    if getattr(scene, 'relight_active', False):
        scene.nerf_job_status = JOB_DONE
        if not bpy.app.background:
            finalize_relight(scene)
        return

    if any(scene.rendering):
        scene.nerf_job_status = JOB_DONE
        if not bpy.app.background:
            schedule_next_render_pass()

@persistent
def post_render_cancel(scene):
    if getattr(scene, 'relight_active', False):
        scene.nerf_job_status = JOB_CANCELLED
        if not bpy.app.background:
            finalize_relight(scene)
        return

    if any(scene.rendering) and scene.nerf_job_status == JOB_RUNNING:
        scene.nerf_job_status = JOB_CANCELLED
        if not bpy.app.background:
            finalize_render(scene)

# set initial property values (bpy.data and bpy.context require a loaded scene)
@persistent
def set_init_props(scene):
    filepath = bpy.data.filepath
    filename = bpy.path.basename(filepath)
    default_save_path = filepath[:-len(filename)] # remove file name from blender file path = directoy path

    # In blender -b the CLI (or the .blend) may already have set save_path.
    # Overwriting here sent images next to the .blend while JSON went to --save-path.
    if not bpy.app.background or not (scene.save_path or '').strip():
        scene.save_path = default_save_path
    scene.init_frame_step = scene.frame_step
    scene.init_output_path = scene.render.filepath

    handlers = bpy.app.handlers.depsgraph_update_post
    while set_init_props in handlers:
        handlers.remove(set_init_props)

# update cos camera when changing frame
@persistent
def cos_camera_update(scene):
    if CAMERA_NAME in scene.objects.keys():
        scene.objects[CAMERA_NAME].location = sample_from_sphere(scene)