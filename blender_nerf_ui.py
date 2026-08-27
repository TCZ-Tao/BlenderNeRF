import bpy


# blender nerf shared ui properties class
class BlenderNeRF_UI(bpy.types.Panel):
    '''BlenderNeRF UI'''
    bl_idname = 'VIEW3D_PT_blender_nerf_ui'
    bl_label = 'BlenderNeRF shared UI'
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'BlenderNeRF'

    def draw(self, context):
        layout = self.layout
        scene = context.scene

        layout.alignment = 'CENTER'

        row = layout.row(align=True)
        row.prop(scene, 'train_data', toggle=True)
        row.prop(scene, 'test_data', toggle=True)

        if not (scene.train_data or scene.test_data):
            layout.label(text='Nothing will happen!')

        else:
            layout.prop(scene, 'aabb')

            layout.separator()
            layout.prop(scene, 'render_frames')
            layout.prop(scene, 'hide_render_view')

            layout.prop(scene, 'logs')
            layout.prop(scene, 'splats', text='Gaussian Points (PLY file)')

            if scene.splats:
                layout.separator()
                layout.label(text='Gaussian Test Camera Poses')
                row = layout.row(align=True)
                row.prop(scene, 'splats_test_dummy', toggle=True, text='Dummy')
                row.prop(scene, 'splats_test_dummy', toggle=True, text='Full', invert_checkbox=True)

            layout.prop(scene, 'gbuffer', text='G-buffer Maps')

            if scene.gbuffer:
                box = layout.box()
                col = box.column(align=True)
                col.prop(scene, 'gbuffer_rgba')
                col.prop(scene, 'gbuffer_albedo')
                col.prop(scene, 'gbuffer_roughness')
                col.prop(scene, 'gbuffer_metallic')
                col.prop(scene, 'gbuffer_geometric_normal')
                col.prop(scene, 'gbuffer_shading_normal')
                col.prop(scene, 'gbuffer_linear_depth')
                col.prop(scene, 'gbuffer_material_id')
                col.prop(scene, 'gbuffer_material_id_vis')

            layout.separator()
            layout.label(text='File Format')

            row = layout.row(align=True)
            row.prop(scene, 'nerf', toggle=True, text='NGP', invert_checkbox=True)
            row.prop(scene, 'nerf', toggle=True)

            layout.separator()
            layout.use_property_split = True
            layout.prop(scene, 'save_path')
            layout.prop(scene, 'compress_to_zip')