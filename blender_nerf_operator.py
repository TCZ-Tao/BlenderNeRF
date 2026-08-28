import os
import math
import json
import datetime
import mathutils
import numpy as np
import bpy
from . import helper, gbuffer


# global addon script variables
OUTPUT_TRAIN = 'train'
OUTPUT_TEST = 'test'
CAMERA_NAME = 'BlenderNeRF Camera'
# 3DGS SH0 coefficient used when converting random harmonics to RGB
SPLATS_SH_C0 = 0.28209479177387814


# blender nerf operator parent class
class BlenderNeRF_Operator(bpy.types.Operator):

    # camera intrinsics
    def get_camera_intrinsics(self, scene, camera):
        camera_angle_x = camera.data.angle_x
        camera_angle_y = camera.data.angle_y

        # camera properties
        f_in_mm = camera.data.lens # focal length in mm
        scale = scene.render.resolution_percentage / 100
        width_res_in_px = scene.render.resolution_x * scale # width
        height_res_in_px = scene.render.resolution_y * scale # height
        optical_center_x = width_res_in_px / 2
        optical_center_y = height_res_in_px / 2

        # pixel aspect ratios
        size_x = scene.render.pixel_aspect_x * width_res_in_px
        size_y = scene.render.pixel_aspect_y * height_res_in_px
        pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y

        # sensor fit and sensor size (and camera angle swap in specific cases)
        if camera.data.sensor_fit == 'AUTO':
            sensor_size_in_mm = camera.data.sensor_height if width_res_in_px < height_res_in_px else camera.data.sensor_width
            if width_res_in_px < height_res_in_px:
                sensor_fit = 'VERTICAL'
                camera_angle_x, camera_angle_y = camera_angle_y, camera_angle_x
            elif width_res_in_px > height_res_in_px:
                sensor_fit = 'HORIZONTAL'
            else:
                sensor_fit = 'VERTICAL' if size_x <= size_y else 'HORIZONTAL'

        else:
            sensor_fit = camera.data.sensor_fit
            if sensor_fit == 'VERTICAL':
                sensor_size_in_mm = camera.data.sensor_height if width_res_in_px <= height_res_in_px else camera.data.sensor_width
                if width_res_in_px <= height_res_in_px:
                    camera_angle_x, camera_angle_y = camera_angle_y, camera_angle_x

        # focal length for horizontal sensor fit
        if sensor_fit == 'HORIZONTAL':
            sensor_size_in_mm = camera.data.sensor_width
            s_u = f_in_mm / sensor_size_in_mm * width_res_in_px
            s_v = f_in_mm / sensor_size_in_mm * width_res_in_px * pixel_aspect_ratio

        # focal length for vertical sensor fit
        if sensor_fit == 'VERTICAL':
            s_u = f_in_mm / sensor_size_in_mm * width_res_in_px / pixel_aspect_ratio
            s_v = f_in_mm / sensor_size_in_mm * width_res_in_px

        camera_intr_dict = {
            'camera_angle_x': camera_angle_x,
            'camera_angle_y': camera_angle_y,
            'fl_x': s_u,
            'fl_y': s_v,
            'k1': 0.0,
            'k2': 0.0,
            'p1': 0.0,
            'p2': 0.0,
            'cx': optical_center_x,
            'cy': optical_center_y,
            'w': width_res_in_px,
            'h': height_res_in_px,
            'aabb_scale': scene.aabb
        }

        return {'camera_angle_x': camera_angle_x} if scene.nerf else camera_intr_dict

    # camera extrinsics (transform matrices)
    def get_camera_extrinsics(self, scene, camera, mode='TRAIN', method='SOF'):
        assert mode == 'TRAIN' or mode == 'TEST'
        assert method == 'SOF' or method == 'TTC' or method == 'COS'

        if scene.splats_test_dummy and mode == 'TEST':
            return []

        initFrame = scene.frame_current
        step = scene.train_frame_steps if (mode == 'TRAIN' and method == 'SOF') else scene.frame_step
        if method == 'COS' and mode == 'TRAIN':
            end = scene.frame_start + scene.cos_nb_frames - 1
        elif method == 'COS' and mode == 'TEST':
            end = scene.frame_start + scene.cos_nb_test_frames - 1
        elif method == 'TTC' and mode == 'TRAIN':
            end = scene.frame_start + scene.ttc_nb_frames - 1
        else:
            end = scene.frame_end

        camera_extr_dict = []
        for frame in range(scene.frame_start, end + 1, step):
            scene.frame_set(frame)
            filename = os.path.basename( scene.render.frame_path(frame=frame) )
            filedir = OUTPUT_TRAIN * (mode == 'TRAIN') + OUTPUT_TEST * (mode == 'TEST')
            if scene.gbuffer:
                filedir = os.path.join(filedir, gbuffer.RGBA_CHANNEL)

            frame_data = {
                'file_path': os.path.join(filedir, os.path.splitext(filename)[0] if scene.splats else filename),
                'transform_matrix': self.listify_matrix(camera.matrix_world)
            }

            camera_extr_dict.append(frame_data)

        scene.frame_set(initFrame) # set back to initial frame

        return camera_extr_dict

    def visible_meshes_world_aabb(self, scene):
        '''World-space AABB of render-visible meshes (evaluated, so modifiers count).'''
        depsgraph = bpy.context.evaluated_depsgraph_get()
        mins = None
        maxs = None

        for obj in scene.objects:
            if obj.type != 'MESH' or not self.is_object_visible(obj):
                continue

            eval_obj = obj.evaluated_get(depsgraph)
            matrix = eval_obj.matrix_world
            for corner in eval_obj.bound_box:
                world = matrix @ mathutils.Vector(corner)
                if mins is None:
                    mins = world.copy()
                    maxs = world.copy()
                else:
                    mins.x = min(mins.x, world.x)
                    mins.y = min(mins.y, world.y)
                    mins.z = min(mins.z, world.z)
                    maxs.x = max(maxs.x, world.x)
                    maxs.y = max(maxs.y, world.y)
                    maxs.z = max(maxs.z, world.z)

        return mins, maxs

    def save_splats_ply(self, scene, directory):
        '''Random point cloud in the scene AABB, matching 3DGS NeRF-synthetic init.'''
        mins, maxs = self.visible_meshes_world_aabb(scene)
        if mins is None:
            raise RuntimeError('Gaussian Points requires at least one visible mesh to compute the scene AABB.')

        n = scene.splats_nb_points
        lo = np.array(mins, dtype=np.float64)
        hi = np.array(maxs, dtype=np.float64)
        rng = np.random.default_rng(scene.seed)
        xyz = rng.random((n, 3)) * (hi - lo) + lo
        shs = rng.random((n, 3)) / 255.0
        rgb = np.clip(shs * SPLATS_SH_C0 + 0.5, 0.0, 1.0) * 255.0
        rgb = np.rint(rgb).astype(np.uint8)
        normals = np.zeros((n, 3), dtype=np.float64)

        filepath = os.path.join(directory, 'points3d.ply')
        with open(filepath, 'w', encoding='ascii', newline='\n') as file:
            file.write('ply\n')
            file.write('format ascii 1.0\n')
            file.write(f'element vertex {n}\n')
            file.write('property float x\n')
            file.write('property float y\n')
            file.write('property float z\n')
            file.write('property float nx\n')
            file.write('property float ny\n')
            file.write('property float nz\n')
            file.write('property uchar red\n')
            file.write('property uchar green\n')
            file.write('property uchar blue\n')
            file.write('end_header\n')
            np.savetxt(file, np.column_stack((xyz, normals, rgb)), fmt='%.6f %.6f %.6f %.6f %.6f %.6f %d %d %d')

    def save_json(self, directory, filename, data, indent=4):
        filepath = os.path.join(directory, filename)
        with open(filepath, 'w') as file:
            json.dump(data, file, indent=indent)

    def is_power_of_two(self, x):
        return math.log2(x).is_integer()

    # function from original nerf 360_view.py code for blender
    def listify_matrix(self, matrix):
        matrix_list = []
        for row in matrix:
            matrix_list.append(list(row))
        return matrix_list

    # check whether an object is visible in render
    def is_object_visible(self, obj):
        if obj.hide_render:
            return False

        for collection in obj.users_collection:
            if collection.hide_render:
                return False

        return True

    # assert messages
    def asserts(self, scene, method='SOF'):
        assert method == 'SOF' or method == 'TTC' or method == 'COS'

        camera = scene.camera
        train_camera = scene.camera_train_target
        test_camera = scene.camera_test_target

        sof_name = scene.sof_dataset_name
        ttc_name = scene.ttc_dataset_name
        cos_name = scene.cos_dataset_name

        error_messages = []

        if (method == 'SOF' or method == 'COS') and not camera.data.type == 'PERSP':
            error_messages.append('Only perspective cameras are supported!')

        if method == 'TTC' and not (train_camera.data.type == 'PERSP' and test_camera.data.type == 'PERSP'):
           error_messages.append('Only perspective cameras are supported!')

        if method == 'COS' and CAMERA_NAME in scene.objects.keys():
            sphere_camera = scene.objects[CAMERA_NAME]
            if not sphere_camera.data.type == 'PERSP':
                error_messages.append('BlenderNeRF Camera must remain a perspective camera!')

        if (method == 'SOF' and sof_name == '') or (method == 'TTC' and ttc_name == '') or (method == 'COS' and cos_name == ''):
            error_messages.append('Dataset name cannot be empty!')

        if method == 'COS' and any(x == 0 for x in scene.sphere_scale):
            error_messages.append('The BlenderNeRF Sphere cannot be flat! Change its scale to be non zero in all axes.')

        if not scene.nerf and not self.is_power_of_two(scene.aabb):
            error_messages.append('iNGP AABB scale needs to be a power of two!')

        if scene.save_path == '':
            error_messages.append('Save path cannot be empty!')

        if scene.splats:
            if self.visible_meshes_world_aabb(scene)[0] is None:
                error_messages.append('Gaussian Points requires at least one visible mesh to compute the scene AABB!')

        if scene.splats and not scene.test_data and not scene.splats_test_dummy:
            error_messages.append('Gaussian Splatting requires test data!')

        if scene.splats and scene.render.image_settings.file_format != 'PNG':
            error_messages.append('Gaussian Splatting requires PNG file extensions!')

        if scene.gbuffer and scene.render_frames:
            if not gbuffer.selected_output_channels(scene):
                error_messages.append('Select at least one G-buffer channel to render!')
            elif gbuffer.needs_mesh_materials(scene) and not gbuffer.mesh_materials(scene):
                error_messages.append('G-buffer maps require mesh objects with node materials!')

        return error_messages

    def save_log_file(self, scene, directory, method='SOF'):
        assert method == 'SOF' or method == 'TTC' or method == 'COS'
        now = datetime.datetime.now()

        logdata = {
            'BlenderNeRF Version': scene.blendernerf_version,
            'Date and Time' : now.strftime("%d/%m/%Y %H:%M:%S"),
            'Train': scene.train_data,
            'Test': scene.test_data,
            'Dummy Test Camera File': scene.splats_test_dummy,
            'iNGP AABB': scene.aabb,
            'Gaussian Points': scene.splats,
            'Gaussian Points Count': scene.splats_nb_points if scene.splats else 0,
            'Render Frames': scene.render_frames,
            'G-buffer Maps': scene.gbuffer,
            'G-buffer Channels': gbuffer.selected_output_channels(scene) if scene.gbuffer else [],
            'File Format': 'NeRF' if scene.nerf else 'NGP',
            'Save Path': scene.save_path,
            'Compress to ZIP': scene.compress_to_zip,
            'Method': method
        }

        if method == 'SOF':
            logdata['Frame Step'] = scene.train_frame_steps
            logdata['Camera'] = scene.camera.name
            logdata['Dataset Name'] = scene.sof_dataset_name

        elif method == 'TTC':
            logdata['Train Camera Name'] = scene.camera_train_target.name
            logdata['Test Camera Name'] = scene.camera_test_target.name
            logdata['Frames'] = scene.ttc_nb_frames
            logdata['Dataset Name'] = scene.ttc_dataset_name

        else:
            logdata['Camera'] = scene.camera.name
            logdata['Location'] = str(list(scene.sphere_location))
            logdata['Rotation'] = str(list(scene.sphere_rotation))
            logdata['Scale'] = str(list(scene.sphere_scale))
            logdata['Radius'] = scene.sphere_radius
            logdata['Lens'] = str(scene.focal) + ' mm'
            logdata['Seed'] = scene.seed
            logdata['Train Frames'] = scene.cos_nb_frames
            logdata['Test Frames'] = scene.cos_nb_test_frames
            logdata['Upper Views'] = scene.upper_views
            logdata['Outwards'] = scene.outwards
            logdata['Dataset Name'] = scene.cos_dataset_name

        self.save_json(directory, filename='log.txt', data=logdata)


class BlenderNeRF_Relight(BlenderNeRF_Operator):
    '''Swap the World environment map and render test-camera frames into test_rli.'''
    bl_idname = 'object.blendernerf_relight'
    bl_label = 'Render Relight'
    bl_description = 'Use the selected HDRI as World lighting and render the method test-camera sequence into test_rli/<envmap>/'

    def relight_error_messages(self, scene, method):
        error_messages = []
        envmap = (scene.relight_envmap or '').strip()
        if not envmap:
            error_messages.append('Environment map path cannot be empty!')
        else:
            path = helper.resolved_envmap_path(envmap)
            if not os.path.isfile(path):
                error_messages.append(f'Environment map file not found: {path}')

        if scene.save_path == '':
            error_messages.append('Save path cannot be empty!')

        if helper.method_dataset_name(scene, method) == '':
            error_messages.append('Dataset name cannot be empty!')

        if method == 'TTC':
            test_camera = scene.camera_test_target
            if test_camera is None:
                error_messages.append('Be sure to have selected a test camera!')
            elif test_camera.data.type != 'PERSP':
                error_messages.append('Only perspective cameras are supported!')
        else:
            camera = scene.camera
            if camera is None:
                error_messages.append('Be sure to have a selected camera!')
            elif camera.data.type != 'PERSP':
                error_messages.append('Only perspective cameras are supported!')

        return error_messages

    def execute(self, context):
        scene = context.scene
        method = scene.relight_method

        if scene.relight_active or any(scene.rendering):
            self.report({'ERROR'}, 'A BlenderNeRF render is already running!')
            return {'CANCELLED'}

        error_messages = self.relight_error_messages(scene, method)
        if error_messages:
            self.report({'ERROR'}, error_messages[0])
            return {'CANCELLED'}

        envmap = helper.resolved_envmap_path(scene.relight_envmap)
        out_dir = helper.relight_output_dir(scene, envmap, method)
        os.makedirs(out_dir, exist_ok=True)

        scene.init_output_path = scene.render.filepath
        scene.init_frame_step = scene.frame_step
        scene.init_frame_end = scene.frame_end
        if scene.camera is not None:
            scene.init_active_camera = scene.camera
            scene.init_active_camera_name = scene.camera.name

        try:
            helper.apply_world_envmap(scene, envmap)
        except Exception as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}

        scene.rendering = helper.rendering_flags_for_method(method)
        scene.relight_active = True
        scene.render.filepath = os.path.join(out_dir, '')

        try:
            result = helper.start_relight_render(scene, out_dir)
        except Exception as exc:
            helper.finalize_relight(scene)
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}

        if bpy.app.background:
            cancelled = result == {'CANCELLED'} or scene.nerf_job_status == helper.JOB_CANCELLED
            helper.finalize_relight(scene)
            if cancelled:
                self.report({'ERROR'}, 'Relight render was cancelled')
                return {'CANCELLED'}
            return {'FINISHED'}

        if result == {'CANCELLED'}:
            helper.finalize_relight(scene)
            self.report({'ERROR'}, 'Could not start relight render')
            return {'CANCELLED'}

        return {'FINISHED'}


class BlenderNeRF_RenderPipeline(bpy.types.Operator):
    '''Train-then-test (and G-buffer) render pipeline.'''
    bl_idname = 'object.blendernerf_render_pipeline'
    bl_label = 'BlenderNeRF Render Pipeline'
    bl_options = {'INTERNAL'}

    do_train: bpy.props.BoolProperty(default=True)
    do_test: bpy.props.BoolProperty(default=False)

    def invoke(self, context, event):
        return self.execute(context)

    def execute(self, context):
        scene = context.scene
        passes = gbuffer.build_passes(self.do_train, self.do_test, scene)
        if not passes:
            helper.finalize_render(scene)
            return {'FINISHED'}

        gbuffer.begin_job(scene, passes)
        try:
            if bpy.app.background:
                helper.run_render_pipeline_sync(scene)
            else:
                helper.start_render_pass(scene)
        except Exception as exc:
            self.report({'ERROR'}, str(exc))
            helper.finalize_render(scene)
        return {'FINISHED'}