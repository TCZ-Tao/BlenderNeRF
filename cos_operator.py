import os
import bpy
from . import helper, blender_nerf_operator, gbuffer


# global addon script variables
EMPTY_NAME = 'BlenderNeRF Sphere'
CAMERA_NAME = 'BlenderNeRF Camera'

# camera on sphere operator class
class CameraOnSphere(blender_nerf_operator.BlenderNeRF_Operator):
    '''Camera on Sphere Operator'''
    bl_idname = 'object.camera_on_sphere'
    bl_label = 'Camera on Sphere COS'

    def execute(self, context):
        scene = context.scene
        camera = scene.camera

        # check if camera is selected : next errors depend on an existing camera
        if camera == None:
            self.report({'ERROR'}, 'Be sure to have a selected camera!')
            return {'FINISHED'}

        # if there is an error, print first error message
        error_messages = self.asserts(scene, method='COS')
        if len(error_messages) > 0:
           self.report({'ERROR'}, error_messages[0])
           return {'FINISHED'}

        output_data = self.get_camera_intrinsics(scene, camera)

        # clean directory name (unsupported characters replaced) and output path
        output_dir = bpy.path.clean_name(scene.cos_dataset_name)
        output_path = os.path.join(helper.resolved_save_path(scene), output_dir)
        os.makedirs(output_path, exist_ok=True)

        if scene.logs: self.save_log_file(scene, output_path, method='COS')
        if scene.splats: self.save_splats_ply(scene, output_path)
        helper.write_scene_metadata(scene, output_path, scene.cos_dataset_name)
        gbuffer.write_material_id_json(scene, output_path)

        # initial property might have changed since set_init_props update
        scene.init_output_path = scene.render.filepath

        # other intial properties
        scene.init_sphere_exists = scene.show_sphere
        scene.init_camera_exists = scene.show_camera
        scene.init_frame_end = scene.frame_end
        scene.init_active_camera = camera
        scene.init_active_camera_name = camera.name

        if scene.test_data:
            # testing transforms (selected camera, Test Frames)
            output_data['frames'] = self.get_camera_extrinsics(scene, camera, mode='TEST', method='COS')
            self.save_json(output_path, 'transforms_test.json', output_data)

        needs_train_render = scene.train_data and helper.wants_any_image_render(scene)
        needs_test_render = helper.wants_test_render(scene) and helper.wants_any_image_render(scene)

        if scene.train_data:
            if not scene.show_camera: scene.show_camera = True

            # train camera on sphere
            sphere_camera = scene.objects[CAMERA_NAME]
            sphere_output_data = self.get_camera_intrinsics(scene, sphere_camera)
            scene.camera = sphere_camera

            # training transforms
            sphere_output_data['frames'] = self.get_camera_extrinsics(scene, sphere_camera, mode='TRAIN', method='COS')
            self.save_json(output_path, 'transforms_train.json', sphere_output_data)

            if needs_train_render:
                output_train = os.path.join(output_path, 'train')
                os.makedirs(output_train, exist_ok=True)
                scene.rendering = (False, False, True)
                scene.frame_end = scene.frame_start + scene.cos_nb_frames - 1 # update end frame
                first_channel = gbuffer.selected_output_channels(scene)[0]
                scene.render.filepath = os.path.join(helper.images_output_dir(scene, 'train', first_channel), '')

        if needs_train_render or needs_test_render:
            if not any(scene.rendering):
                scene.rendering = (False, False, True)
            helper.launch_render_pipeline(needs_train_render, needs_test_render)
            return {'FINISHED'}

        if not any(scene.rendering):
            # reset camera settings
            if not scene.init_camera_exists: helper.delete_camera(scene, CAMERA_NAME)
            if not scene.init_sphere_exists:
                helper.delete_spiral_path()
                objects = bpy.data.objects
                objects.remove(objects[EMPTY_NAME], do_unlink=True)
                scene.show_sphere = False
                scene.sphere_exists = False

            scene.camera = scene.init_active_camera

            helper.maybe_compress_dataset(scene, output_path)

        return {'FINISHED'}


class ApplySphericalSpiral(bpy.types.Operator):
    '''Keyframe the COS camera along a two-revolution spherical spiral on the BlenderNeRF Sphere'''
    bl_idname = 'object.blendernerf_spherical_spiral'
    bl_label = 'Apply Spherical Spiral'
    bl_description = 'Keyframe the selected COS camera along a two-revolution spherical spiral on the BlenderNeRF Sphere (NeRF synthetic test path). Uses Test Frames for the sample count'
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        scene = context.scene
        camera = scene.camera

        if camera is None:
            self.report({'ERROR'}, 'Be sure to have a selected camera!')
            return {'CANCELLED'}

        if camera.type != 'CAMERA':
            self.report({'ERROR'}, 'The selected object must be a camera!')
            return {'CANCELLED'}

        if camera.name == CAMERA_NAME:
            self.report({'ERROR'}, 'The BlenderNeRF training camera cannot use a spiral path. Choose another camera in the COS Camera field.')
            return {'CANCELLED'}

        if any(x == 0 for x in scene.sphere_scale):
            self.report({'ERROR'}, 'The BlenderNeRF Sphere cannot be flat! Change its scale to be non zero in all axes.')
            return {'CANCELLED'}

        if scene.cos_nb_test_frames < 2:
            self.report({'ERROR'}, 'Test Frames must be at least 2 to build a spiral path.')
            return {'CANCELLED'}

        if not scene.show_sphere:
            scene.show_sphere = True

        if EMPTY_NAME not in scene.objects:
            self.report({'ERROR'}, 'Could not create the BlenderNeRF Sphere.')
            return {'CANCELLED'}

        frame_start, frame_end, n = helper.apply_spherical_spiral(scene, camera)
        self.report({'INFO'}, f'Applied spherical spiral to {camera.name} ({n} frames, {frame_start}-{frame_end}) on the BlenderNeRF Sphere.')
        return {'FINISHED'}