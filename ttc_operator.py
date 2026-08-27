import os
import bpy
from . import helper, blender_nerf_operator, gbuffer


# train and test cameras operator class
class TrainTestCameras(blender_nerf_operator.BlenderNeRF_Operator):
    '''Train and Test Cameras Operator'''
    bl_idname = 'object.train_test_cameras'
    bl_label = 'Train and Test Cameras TTC'

    def execute(self, context):
        scene = context.scene
        train_camera = scene.camera_train_target
        test_camera = scene.camera_test_target

        # check if cameras are selected : next errors depend on existing cameras
        if train_camera == None or test_camera == None:
            self.report({'ERROR'}, 'Be sure to have selected a train and test camera!')
            return {'FINISHED'}

        # if there is an error, print first error message
        error_messages = self.asserts(scene, method='TTC')
        if len(error_messages) > 0:
           self.report({'ERROR'}, error_messages[0])
           return {'FINISHED'}

        output_train_data = self.get_camera_intrinsics(scene, train_camera)
        output_test_data = self.get_camera_intrinsics(scene, test_camera)

        # clean directory name (unsupported characters replaced) and output path
        output_dir = bpy.path.clean_name(scene.ttc_dataset_name)
        output_path = os.path.join(helper.resolved_save_path(scene), output_dir)
        os.makedirs(output_path, exist_ok=True)

        if scene.logs: self.save_log_file(scene, output_path, method='TTC')
        if scene.splats: self.save_splats_ply(scene, output_path)
        helper.write_scene_metadata(scene, output_path, scene.ttc_dataset_name)
        gbuffer.write_material_id_json(scene, output_path)

        # initial properties might have changed since set_init_props update
        scene.init_output_path = scene.render.filepath
        scene.init_frame_end = scene.frame_end

        if scene.test_data:
            # testing transforms
            output_test_data['frames'] = self.get_camera_extrinsics(scene, test_camera, mode='TEST', method='TTC')
            self.save_json(output_path, 'transforms_test.json', output_test_data)

        needs_train_render = scene.train_data and helper.wants_any_image_render(scene)
        needs_test_render = helper.wants_test_render(scene) and helper.wants_any_image_render(scene)

        if scene.train_data:
            # training transforms
            output_train_data['frames'] = self.get_camera_extrinsics(scene, train_camera, mode='TRAIN', method='TTC')
            self.save_json(output_path, 'transforms_train.json', output_train_data)

            if needs_train_render:
                output_train = os.path.join(output_path, 'train')
                os.makedirs(output_train, exist_ok=True)
                scene.camera = train_camera
                scene.rendering = (False, True, False)
                scene.frame_end = scene.frame_start + scene.ttc_nb_frames - 1 # update end frame
                first_channel = gbuffer.selected_output_channels(scene)[0]
                scene.render.filepath = os.path.join(helper.images_output_dir(scene, 'train', first_channel), '')

        if needs_train_render or needs_test_render:
            if not any(scene.rendering):
                scene.rendering = (False, True, False)
            helper.launch_render_pipeline(needs_train_render, needs_test_render)
            return {'FINISHED'}

        if not any(scene.rendering):
            helper.maybe_compress_dataset(scene, output_path)

        return {'FINISHED'}