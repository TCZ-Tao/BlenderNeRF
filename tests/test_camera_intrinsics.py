"""Run with blender --background --factory-startup --python-exit-code 1 --python <this file>."""

import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from BlenderNeRF.blender_nerf_operator import BlenderNeRF_Operator


class CameraIntrinsicsTests(unittest.TestCase):
    def setUp(self):
        self.scene = bpy.context.scene
        self.camera = self.scene.camera
        self.camera.data.type = 'PERSP'
        self.camera.data.lens = 50
        self.camera.data.sensor_width = 36
        self.camera.data.sensor_height = 24
        self.camera.data.sensor_fit = 'AUTO'
        self.camera.data.shift_x = 0
        self.camera.data.shift_y = 0
        self.scene.render.resolution_x = 800
        self.scene.render.resolution_y = 800
        self.scene.render.resolution_percentage = 100
        self.scene.render.pixel_aspect_x = 1
        self.scene.render.pixel_aspect_y = 1

    def export(self, nerf=True):
        bpy.context.view_layer.update()
        settings = SimpleNamespace(render=self.scene.render, nerf=nerf, aabb=16)
        return BlenderNeRF_Operator.get_camera_intrinsics(None, settings, self.camera)

    def test_cornell_square_render(self):
        for nerf in (True, False):
            with self.subTest(nerf=nerf):
                data = json.loads(json.dumps(self.export(nerf)))
                for axis in ('x', 'y'):
                    self.assertAlmostEqual(data['fl_' + axis], 1111.111111, delta=0.001)
                    self.assertAlmostEqual(data['camera_angle_' + axis], 0.691111161, places=6)
                self.assertEqual((data['cx'], data['cy']), (400, 400))
                self.assertEqual((data['w'], data['h']), (800, 800))
                for field in ('clip_start', 'clip_end', 'shift_x', 'shift_y', 'use_dof'):
                    self.assertIn(field, data)
                self.assertGreater(abs(data['camera_angle_y'] - self.camera.data.angle_y), 0.2)

    def test_projection_matches_blender_view_frame(self):
        # world_to_camera_view uses Camera.view_frame, an independent API path.
        for fit in ('AUTO', 'HORIZONTAL', 'VERTICAL'):
            for width, height in ((800, 800), (1200, 800), (800, 1200)):
                for aspect in ((1, 1), (2, 1), (1, 2)):
                    with self.subTest(fit=fit, size=(width, height), aspect=aspect):
                        self.camera.data.sensor_fit = fit
                        self.camera.data.shift_x = 0.13
                        self.camera.data.shift_y = -0.08
                        self.scene.render.resolution_x = width
                        self.scene.render.resolution_y = height
                        self.scene.render.resolution_percentage = 50
                        self.scene.render.pixel_aspect_x, self.scene.render.pixel_aspect_y = aspect
                        data = self.export()
                        for point in (Vector((0, 0, -2)), Vector((0.3, -0.2, -3)), Vector((-0.4, 0.5, -4))):
                            ndc = world_to_camera_view(self.scene, self.camera, self.camera.matrix_world @ point)
                            u = data['fl_x'] * point.x / -point.z + data['cx']
                            v = data['fl_y'] * -point.y / -point.z + data['cy']
                            self.assertAlmostEqual(u, ndc.x * data['w'], delta=0.001)
                            self.assertAlmostEqual(v, (1 - ndc.y) * data['h'], delta=0.001)

    def test_scaled_resolution_is_integer(self):
        self.scene.render.resolution_x = 803
        self.scene.render.resolution_y = 607
        self.scene.render.resolution_percentage = 37
        data = self.export()
        self.assertEqual((data['w'], data['h']), (297, 224))
        self.assertIsInstance(data['w'], int)
        self.assertIsInstance(data['h'], int)


if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(CameraIntrinsicsTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise RuntimeError('Camera intrinsics regression failed')
