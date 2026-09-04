import bpy
import gpu
import mathutils
from gpu_extras.batch import batch_for_shader
from bpy.app.handlers import persistent

from . import helper


_LAYER_KEYS = ('rays', 'frame', 'tri', 'path')
_LAYER_COLOR = {
    'rays': 'color',
    'frame': 'frame_color',
    'tri': 'tri_color',
    'path': 'path_color',
}

# Runtime only — never written to the .blend.
_draw_handler = None
_coords = {k: None for k in _LAYER_KEYS}
_batches = {k: None for k in _LAYER_KEYS}
_sample_count = 0
_rebuild_lock = False

# Up triangle sits on the local +Y edge of view_frame (order-independent).
_UP_TRI_HEIGHT = 0.25


def _ray_segments(origin, corners):
    coords = []
    for corner in corners:
        coords.append(origin)
        coords.append(corner)
    return coords


def _rect_segments(corners):
    coords = []
    for i in range(4):
        coords.append(corners[i])
        coords.append(corners[(i + 1) % 4])
    return coords


def _up_triangle_points(local_corners, height_fac=_UP_TRI_HEIGHT):
    '''Return (left_top, peak, right_top) in camera space. Up is local +Y.'''
    y_min = min(p[1] for p in local_corners)
    y_max = max(p[1] for p in local_corners)
    top = [p for p in local_corners if abs(p[1] - y_max) < 1e-6]
    a, b = top[0], top[1]
    if a[0] > b[0]:
        a, b = b, a
    peak = (
        (a[0] + b[0]) * 0.5,
        y_max + (y_max - y_min) * height_fac,
        (a[2] + b[2]) * 0.5,
    )
    return a, peak, b


def _up_triangle_segments_world(mw, local_corners):
    left_top, peak, right_top = _up_triangle_points(local_corners)
    left_w = _as3(mw @ mathutils.Vector(left_top))
    peak_w = _as3(mw @ mathutils.Vector(peak))
    right_w = _as3(mw @ mathutils.Vector(right_top))
    return [left_w, peak_w, right_w, peak_w]


def _path_segments(origins):
    coords = []
    for i in range(len(origins) - 1):
        coords.append(origins[i])
        coords.append(origins[i + 1])
    return coords


def _scale_view_frame(view_frame, depth):
    z = abs(view_frame[0].z)
    if z < 1e-8:
        return None
    factor = depth / z
    return [p * factor for p in view_frame]


def _as3(p):
    return (p[0], p[1], p[2])


def _color4(name, description, default):
    return bpy.props.FloatVectorProperty(
        name=name,
        description=description,
        subtype='COLOR',
        size=4,
        min=0.0,
        max=1.0,
        default=default,
        update=_style_update,
    )


def _self_check():
    origin = (0.0, 0.0, 0.0)
    lb, rb, rt, lt = (
        (-1.0, -0.5, -1.0),
        (1.0, -0.5, -1.0),
        (1.0, 0.5, -1.0),
        (-1.0, 0.5, -1.0),
    )
    corners = (lb, rb, rt, lt)
    rays = _ray_segments(origin, corners)
    assert len(rays) == 8
    rect = _rect_segments(corners)
    assert len(rect) == 8
    assert rect[0] is lb and rect[1] is rb
    a, peak, b = _up_triangle_points(corners)
    assert peak[1] > lt[1]
    assert abs(peak[0]) < 1e-6
    assert a[0] < b[0]
    path = _path_segments([origin, lb, rb])
    assert len(path) == 4


def _vis(context=None):
    scene = (context or bpy.context).scene
    return getattr(scene, 'camera_vis', None)


def tag_redraw_all_view3d():
    wm = bpy.context.window_manager
    if wm is None:
        return
    for window in wm.windows:
        screen = window.screen
        if screen is None:
            continue
        for area in screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()


def _clear_cache():
    global _sample_count
    for k in _LAYER_KEYS:
        _coords[k] = None
        _batches[k] = None
    _sample_count = 0


def _try_build_batch():
    if all(_batches[k] is not None or not _coords[k] for k in _LAYER_KEYS):
        return
    try:
        shader = gpu.shader.from_builtin('POLYLINE_UNIFORM_COLOR')
        for k in _LAYER_KEYS:
            if _batches[k] is None and _coords[k]:
                _batches[k] = batch_for_shader(shader, 'LINES', {'pos': _coords[k]})
    except SystemError:
        for k in _LAYER_KEYS:
            _batches[k] = None


def _ensure_source_camera(vis, context):
    if vis.source_camera is not None:
        return
    obj = getattr(context, 'object', None)
    if obj is not None and obj.type == 'CAMERA':
        vis.source_camera = obj
        return
    scene_cam = context.scene.camera
    if scene_cam is not None and scene_cam.type == 'CAMERA':
        vis.source_camera = scene_cam


def _camera_supported(camera):
    return camera is not None and camera.type == 'CAMERA' and camera.data.type == 'PERSP'


def build_camera_cache(context):
    global _sample_count
    _clear_cache()

    vis = _vis(context)
    if vis is None:
        return 0

    camera = vis.source_camera
    if not _camera_supported(camera):
        return 0

    scene = context.scene
    start = vis.frame_start
    end = vis.frame_end
    step = max(1, vis.frame_step)
    if end < start:
        return 0

    orig_frame = scene.frame_current
    orig_sub = scene.frame_subframe
    depth = vis.frustum_depth
    rays, frame, tri, origins = [], [], [], []
    show_path = vis.show_path

    try:
        f = start
        while f <= end:
            scene.frame_set(f)
            depsgraph = context.evaluated_depsgraph_get()
            eval_cam = camera.evaluated_get(depsgraph)
            if eval_cam.data.type != 'PERSP':
                f += 1 if show_path else step
                continue
            mw = eval_cam.matrix_world
            origin = _as3(mw.translation)
            if show_path:
                origins.append(origin)
            if (f - start) % step == 0:
                scaled = _scale_view_frame(eval_cam.data.view_frame(scene=scene), depth)
                if scaled is not None:
                    local_corners = tuple(_as3(p) for p in scaled)
                    corners = tuple(_as3(mw @ p) for p in scaled)
                    rays.extend(_ray_segments(origin, corners))
                    tri.extend(_up_triangle_segments_world(mw, local_corners))
                    frame.extend(_rect_segments(corners))
            f += 1 if show_path else step
    finally:
        scene.frame_set(orig_frame, subframe=orig_sub)

    path = _path_segments(origins) if show_path else []
    _sample_count = len(rays) // 8
    _coords['rays'] = rays or None
    _coords['frame'] = frame or None
    _coords['tri'] = tri or None
    _coords['path'] = path or None
    if not any(_coords[k] for k in _LAYER_KEYS):
        return 0

    _try_build_batch()
    return _sample_count


def _draw_batch(shader, batch, color, line_width, viewport):
    if batch is None:
        return
    shader.uniform_float('viewportSize', viewport)
    shader.uniform_float('lineWidth', line_width)
    shader.uniform_float('color', tuple(color))
    batch.draw(shader)


def draw_camera_overlay():
    _try_build_batch()
    if not any(_batches[k] for k in _LAYER_KEYS):
        return
    context = bpy.context
    vis = _vis(context)
    if vis is None or not vis.enabled:
        return
    if not _camera_supported(vis.source_camera):
        return

    shader = gpu.shader.from_builtin('POLYLINE_UNIFORM_COLOR')
    shader.bind()
    viewport = gpu.state.viewport_get()[2:]

    gpu.state.blend_set('ALPHA')
    gpu.state.depth_test_set('LESS_EQUAL')
    gpu.state.depth_mask_set(False)
    try:
        for key, prop in _LAYER_COLOR.items():
            _draw_batch(shader, _batches[key], getattr(vis, prop), vis.line_width, viewport)
    finally:
        gpu.state.depth_mask_set(True)
        gpu.state.depth_test_set('NONE')
        gpu.state.blend_set('NONE')


def enable_draw_handler():
    global _draw_handler
    if _draw_handler is None:
        _draw_handler = bpy.types.SpaceView3D.draw_handler_add(
            draw_camera_overlay, (), 'WINDOW', 'POST_VIEW'
        )


def disable_draw_handler():
    global _draw_handler
    if _draw_handler is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_draw_handler, 'WINDOW')
        _draw_handler = None
    _clear_cache()


def _geom_update(self, context):
    if _rebuild_lock or not self.enabled:
        return
    build_camera_cache(context)
    tag_redraw_all_view3d()


def _style_update(self, context):
    tag_redraw_all_view3d()


def _enabled_update(self, context):
    global _rebuild_lock
    if self.enabled:
        _rebuild_lock = True
        try:
            _ensure_source_camera(self, context)
        finally:
            _rebuild_lock = False
        enable_draw_handler()
        build_camera_cache(context)
        tag_redraw_all_view3d()
    else:
        disable_draw_handler()
        tag_redraw_all_view3d()


class CameraVisualizationProperties(bpy.types.PropertyGroup):
    enabled: bpy.props.BoolProperty(
        name='Enable Visualization',
        description='Draw cached camera frustums in the 3D viewport without creating objects',
        default=False,
        update=_enabled_update,
    )
    source_camera: bpy.props.PointerProperty(
        type=bpy.types.Object,
        name='Source Camera',
        description='Animated camera used to build the overlay',
        poll=helper.poll_is_camera,
        update=_geom_update,
    )
    frame_start: bpy.props.IntProperty(
        name='Frame Start',
        description='First frame of the overlay sequence',
        default=1,
        update=_geom_update,
    )
    frame_end: bpy.props.IntProperty(
        name='Frame End',
        description='Last frame of the overlay sequence',
        default=250,
        update=_geom_update,
    )
    frame_step: bpy.props.IntProperty(
        name='Frame Step',
        description='Show a frustum every N frames. The camera path still uses every frame',
        default=1,
        min=1,
        soft_min=1,
        update=_geom_update,
    )
    frustum_depth: bpy.props.FloatProperty(
        name='Frustum Depth',
        description='Length of the displayed view frustum',
        default=1.0,
        min=0.001,
        soft_min=0.01,
        unit='LENGTH',
        update=_geom_update,
    )
    show_path: bpy.props.BoolProperty(
        name='Show Camera Path',
        description='Connect every frame\'s camera center in the range, ignoring Frame Step',
        default=True,
        update=_geom_update,
    )
    color: _color4(
        'Frustum',
        'Color of the frustum rays from the camera to the image plane',
        (0.2, 0.8, 1.0, 0.85),
    )
    frame_color: _color4(
        'Frame',
        'Color of the image-plane rectangle',
        (0.75, 0.93, 1.0, 0.4),
    )
    tri_color: _color4(
        'Up Triangle',
        'Color of the camera-up triangle on the image plane',
        (1.0, 0.85, 0.15, 0.95),
    )
    path_color: _color4(
        'Path',
        'Color of the per-frame camera path',
        (1.0, 0.45, 0.15, 0.9),
    )
    line_width: bpy.props.FloatProperty(
        name='Line Width',
        description='Overlay line width in pixels',
        default=2.0,
        min=1.0,
        soft_max=10.0,
        update=_style_update,
    )


class BLENDERNErf_OT_refresh_camera_overlay(bpy.types.Operator):
    bl_idname = 'object.blendernerf_refresh_camera_overlay'
    bl_label = 'Refresh'
    bl_description = 'Rebuild the camera sequence overlay cache'
    bl_options = {'REGISTER'}

    def execute(self, context):
        vis = _vis(context)
        if vis is None or not vis.enabled:
            self.report({'WARNING'}, 'Enable visualization first')
            return {'CANCELLED'}
        if not _camera_supported(vis.source_camera):
            self.report({'WARNING'}, 'Select a perspective camera')
            return {'CANCELLED'}
        n = build_camera_cache(context)
        tag_redraw_all_view3d()
        self.report({'INFO'}, f'Cached {n} camera samples')
        return {'FINISHED'}


class CameraVisualizationUI(bpy.types.Panel):
    bl_idname = 'VIEW3D_PT_camera_vis_ui'
    bl_label = 'Camera Sequence Visualization'
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'BlenderNeRF'
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        vis = context.scene.camera_vis

        layout.prop(vis, 'enabled', toggle=True)

        col = layout.column()
        col.enabled = vis.enabled
        col.use_property_split = True
        col.prop(vis, 'source_camera')

        camera = vis.source_camera
        if camera is not None and camera.type == 'CAMERA' and camera.data.type != 'PERSP':
            col.label(text='Only perspective cameras are supported', icon='ERROR')

        col.prop(vis, 'frame_start')
        col.prop(vis, 'frame_end')
        col.prop(vis, 'frame_step')
        col.prop(vis, 'frustum_depth')
        col.prop(vis, 'show_path')

        row = layout.row()
        row.enabled = vis.enabled
        row.operator('object.blendernerf_refresh_camera_overlay', text='Refresh')

        if vis.enabled and _sample_count:
            layout.label(text=f'{_sample_count} samples cached')


class CameraVisualizationStyleUI(bpy.types.Panel):
    bl_idname = 'VIEW3D_PT_camera_vis_style_ui'
    bl_label = 'Appearance'
    bl_parent_id = 'VIEW3D_PT_camera_vis_ui'
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'BlenderNeRF'
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        vis = context.scene.camera_vis
        layout = self.layout
        layout.enabled = vis.enabled
        layout.use_property_split = True
        layout.prop(vis, 'color')
        layout.prop(vis, 'frame_color')
        layout.prop(vis, 'tri_color')
        layout.prop(vis, 'path_color')
        layout.prop(vis, 'line_width')


@persistent
def _load_post(_dummy):
    def _deferred():
        scene = getattr(bpy.context, 'scene', None)
        vis = getattr(scene, 'camera_vis', None) if scene is not None else None
        if vis is not None and vis.enabled:
            enable_draw_handler()
            build_camera_cache(bpy.context)
            tag_redraw_all_view3d()
        return None

    bpy.app.timers.register(_deferred, first_interval=0.0)


CLASSES = (
    CameraVisualizationProperties,
    BLENDERNErf_OT_refresh_camera_overlay,
    CameraVisualizationUI,
    CameraVisualizationStyleUI,
)


def register():
    _self_check()
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.camera_vis = bpy.props.PointerProperty(type=CameraVisualizationProperties)
    bpy.app.handlers.load_post.append(_load_post)


def unregister():
    disable_draw_handler()
    if _load_post in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(_load_post)
    if hasattr(bpy.types.Scene, 'camera_vis'):
        del bpy.types.Scene.camera_vis
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
