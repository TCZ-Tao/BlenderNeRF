import os
import bpy
from . import helper, blender_nerf_operator, gbuffer


# subset of frames operator class
class SubsetOfFrames(blender_nerf_operator.BlenderNeRF_Operator):
    '''Subset of Frames Operator'''
    bl_idname = 'object.subset_of_frames'
    bl_label = 'Subset of Frames SOF'

    def execute(self, context):
        scene = context.scene
        camera = scene.camera

        # check if camera is selected : next errors depend on an existing camera
        if camera == None:
            self.report({'ERROR'}, 'Be sure to have a selected camera!')
            return {'FINISHED'}

        # if there is an error, print first error message
        error_messages = self.asserts(scene, method='SOF')
        if len(error_messages) > 0:
           self.report({'ERROR'}, error_messages[0])
           return {'FINISHED'}

        output_data = self.get_camera_intrinsics(scene, camera)

        # clean directory name (unsupported characters replaced) and output path
        output_dir = bpy.path.clean_name(scene.sof_dataset_name)
        output_path = os.path.join(scene.save_path, output_dir)
        os.makedirs(output_path, exist_ok=True)

        if scene.logs: self.save_log_file(scene, output_path, method='SOF')
        if scene.splats: self.save_splats_ply(scene, output_path)
        helper.write_scene_metadata(scene, output_path, scene.sof_dataset_name)
        gbuffer.write_material_id_json(scene, output_path)

        # initial properties might have changed since set_init_props update
        scene.init_frame_step = scene.frame_step
        scene.init_output_path = scene.render.filepath

        if scene.test_data:
            # testing transforms
            output_data['frames'] = self.get_camera_extrinsics(scene, camera, mode='TEST', method='SOF')
            self.save_json(output_path, 'transforms_test.json', output_data)

        needs_train_render = scene.train_data and helper.wants_any_image_render(scene)
        needs_test_render = helper.wants_test_render(scene) and helper.wants_any_image_render(scene)

        if scene.train_data:
            # training transforms
            output_data['frames'] = self.get_camera_extrinsics(scene, camera, mode='TRAIN', method='SOF')
            self.save_json(output_path, 'transforms_train.json', output_data)

            if needs_train_render:
                output_train = os.path.join(output_path, 'train')
                os.makedirs(output_train, exist_ok=True)
                scene.rendering = (True, False, False)
                scene.frame_step = scene.train_frame_steps # update frame step
                first_channel = gbuffer.selected_output_channels(scene)[0]
                scene.render.filepath = os.path.join(helper.images_output_dir(scene, 'train', first_channel), '')

        if needs_train_render or needs_test_render:
            if not any(scene.rendering):
                scene.rendering = (True, False, False)
            bpy.ops.object.blendernerf_render_pipeline('INVOKE_DEFAULT', do_train=needs_train_render, do_test=needs_test_render)
            return {'FINISHED'}

        if not any(scene.rendering):
            helper.maybe_compress_dataset(scene, output_path)

        return {'FINISHED'}