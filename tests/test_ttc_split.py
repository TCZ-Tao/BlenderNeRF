"""Run with blender --background --factory-startup --python-exit-code 1 --python <this file>."""

import sys
from pathlib import Path
import unittest

import bpy
from mathutils import Matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from BlenderNeRF import helper


def link(obj):
    bpy.context.scene.collection.objects.link(obj)
    return obj


def new_camera(name):
    obj = bpy.data.objects.new(name, bpy.data.cameras.new(name))
    return link(obj)


def remove(obj):
    data = obj.data if obj.type == 'CAMERA' else None
    bpy.data.objects.remove(obj, do_unlink=True)
    if data is not None and data.users == 0:
        bpy.data.cameras.remove(data)


def sample(obj, frames):
    scene = bpy.context.scene
    matrices = []
    for frame in frames:
        scene.frame_set(frame)
        bpy.context.view_layer.update()
        matrices.append(obj.matrix_world.copy())
    return matrices


def assert_same_matrix(test, got, expected):
    for row in range(4):
        for col in range(4):
            test.assertAlmostEqual(got[row][col], expected[row][col], places=4)


class TTCSplitTests(unittest.TestCase):
    def test_counts_and_order(self):
        self.assertEqual(helper.split_test_count(100, 0.1), 10)
        self.assertEqual(helper.split_test_count(100, 0.125), 13)
        self.assertEqual(helper.split_test_count(2, 0.1), 1)
        self.assertEqual(helper.split_test_count(1, 0.5), 1)
        self.assertEqual(helper.split_test_count(5, 0.0), 0)
        self.assertEqual(helper.split_test_count(5, 1.0), 5)
        train_idx, test_idx = helper.split_pose_indices(20, 0.25, 0)
        self.assertEqual(train_idx, sorted(train_idx))
        self.assertEqual(test_idx, sorted(test_idx))
        self.assertEqual(len(test_idx), 5)
        self.assertEqual(set(train_idx) | set(test_idx), set(range(20)))

    def test_assign_refuses_sparse_keys(self):
        scene = bpy.context.scene
        train = new_camera('TTC Sparse Train')
        test = new_camera('TTC Sparse Test')
        try:
            train.location = (0, 0, 0)
            train.keyframe_insert('location', frame=1)
            train.location = (3, 0, 0)
            train.keyframe_insert('location', frame=4)
            test.location = (0, 1, 0)
            test.keyframe_insert('location', frame=1)
            with self.assertRaises(RuntimeError):
                helper.assign_ttc_split(scene, train, test, 0.25, 0)
        finally:
            remove(train)
            remove(test)

    def test_bake_pool_and_split(self):
        scene = bpy.context.scene
        scene.frame_start = 1
        scene.frame_set(1)
        train = new_camera('TTC Pool Train')
        test = new_camera('TTC Pool Test')
        empty = link(bpy.data.objects.new('TTC Look Target', None))
        try:
            for frame, x in ((1, 0.0), (4, 3.0)):
                train.location = (x, 0.0, 1.0)
                train.keyframe_insert('location', frame=frame)
            self.assertFalse(helper.every_pose_frame_is_keyed(train))
            for frame, y in ((1, 1.0), (2, 2.0)):
                test.location = (0.0, y, 4.0)
                test.rotation_euler = (0.2, 0.0, frame * 0.1)
                test.keyframe_insert('location', frame=frame)
                test.keyframe_insert('rotation_euler', frame=frame)
            track = test.constraints.new('TRACK_TO')
            track.target = empty
            track.track_axis = 'TRACK_NEGATIVE_Z'
            track.up_axis = 'UP_Y'
            pool = sample(train, range(1, 5)) + sample(test, range(1, 3))
            train_idx, test_idx = helper.split_pose_indices(len(pool), 0.25, 1)
            n_train, n_test = helper.redistribute_ttc_frames(scene, train, test, 0.25, 1)
            self.assertEqual((n_train, n_test), (len(train_idx), len(test_idx)))
            self.assertEqual(len(train.constraints), 0)
            self.assertEqual(len(test.constraints), 0)
            self.assertTrue(helper.poses_are_literal(train))
            self.assertTrue(helper.poses_are_literal(test))
            start = scene.frame_start
            for got, expected in zip(sample(train, range(start, start + n_train)), [pool[i] for i in train_idx]):
                assert_same_matrix(self, got, expected)
            for got, expected in zip(sample(test, range(start, start + n_test)), [pool[i] for i in test_idx]):
                assert_same_matrix(self, got, expected)
        finally:
            remove(train)
            remove(test)
            remove(empty)

    def test_unkeyed_camera_adds_nothing(self):
        scene = bpy.context.scene
        scene.frame_start = 1
        train = new_camera('TTC Keyed Train')
        test = new_camera('TTC Empty Test')
        try:
            for frame, x in ((1, 0.0), (2, 1.0), (3, 2.0), (4, 3.0)):
                train.location = (x, 0.0, 0.0)
                train.keyframe_insert('location', frame=frame)
            n_train, n_test = helper.redistribute_ttc_frames(scene, train, test, 0.25, 0)
            self.assertEqual(n_train + n_test, 4)
            self.assertEqual(n_test, 1)
        finally:
            remove(train)
            remove(test)

    def test_parent_compensation(self):
        scene = bpy.context.scene
        parent = link(bpy.data.objects.new('TTC Parent', None))
        cam = new_camera('TTC Parent Cam')
        try:
            parent.location = (0, 0, 0)
            scene.frame_set(1)
            bpy.context.view_layer.update()
            cam.parent = parent
            cam.matrix_parent_inverse = parent.matrix_world.inverted()
            parent.location = (5, 0, 0)
            parent.keyframe_insert('location', frame=10)
            scene.frame_set(10)
            bpy.context.view_layer.update()
            helper.write_world_poses(scene, cam, [Matrix.Translation((1, 2, 3))], 10)
            got = sample(cam, [10])[0].translation
            self.assertAlmostEqual(got.x, 1, places=4)
            self.assertAlmostEqual(got.y, 2, places=4)
            self.assertAlmostEqual(got.z, 3, places=4)
        finally:
            remove(cam)
            remove(parent)


if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TTCSplitTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise RuntimeError('TTC frame split regression failed')
