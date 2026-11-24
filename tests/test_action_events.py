import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from src.action_analysis import run, merge_samples, movement_speed, LABELS
from src.action_events import load_actions
from src.ai_assistant import AIAssistant
from src.action_labels import visible_actions
from src.office_vision import posture_from_keypoints, phone_near_hand, chair_support


class ActionsTest(unittest.TestCase):
    def test_two_office_cameras_run_concurrently_with_separate_models(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            cameras, detections = {}, {}
            for camera_id in (1, 2):
                name = f'cam{camera_id}.mp4'
                writer = cv2.VideoWriter(str(root / name), cv2.VideoWriter_fourcc(*'mp4v'),
                                         10, (64, 64))
                for _ in range(20):
                    writer.write(np.zeros((64, 64, 3), dtype=np.uint8))
                writer.release()
                cameras[str(camera_id)] = {'fps': 10, 'raw_video_file': name}
                box = {'gid': camera_id, 'bbox': [5, 5, 40, 60]}
                detections[str(camera_id)] = {'0': [box], '10': [box]}
            (root / 'tracking_log.json').write_text(json.dumps({
                'persons': {}, 'cameras': cameras, 'detections': detections,
            }))

            class FakeOfficeModel:
                lock = threading.Lock()
                instances = 0
                active = 0
                peak = 0

                def __init__(self, *args):
                    with self.lock:
                        type(self).instances += 1

                def predict(self, frame, boxes):
                    cls = type(self)
                    with cls.lock:
                        cls.active += 1
                        cls.peak = max(cls.peak, cls.active)
                    try:
                        time.sleep(0.1)
                        return [dict(posture='sitting', posture_score=.9,
                                     phone_score=None) for _ in boxes]
                    finally:
                        with cls.lock:
                            cls.active -= 1

            args = SimpleNamespace(output_dir=folder, data_dir=folder, camera=None,
                                   start=0, end=2, step=1, walk_speed=.25,
                                   device='cpu', workers=2)
            with patch('src.office_vision.OfficeVision', FakeOfficeModel):
                result = run(args)
            self.assertEqual(FakeOfficeModel.instances, 2)
            self.assertEqual(FakeOfficeModel.peak, 2)
            self.assertEqual(len(result['events']), 2)
            self.assertEqual([c['camera_id'] for c in result['coverage']], [1, 2])

    def test_overlay_filter_and_render(self):
        from src.review_ui import ReviewUI
        self.assertEqual(set(LABELS.values()), {'sitting', 'standing', 'walking', 'possible_phone_use'})
        event = dict(person_id=2, camera_id=1, action='walking', start_seconds=1, end_seconds=2)
        self.assertEqual(visible_actions([event], {'walking'}, 1, 1.5), [event])
        self.assertEqual(visible_actions([event], {'sitting'}, 1, 1.5), [])
        self.assertEqual(visible_actions([event], {'walking'}, 2, 1.5), [])
        self.assertEqual(visible_actions([event], {'walking'}, 1, 3), [])
        ui = ReviewUI.__new__(ReviewUI)
        ui.cameras = {1: {'fps': 10}}
        ui._overlay_actions = {'walking'}
        ui._tracked_gid = None
        ui.ai_assistant = SimpleNamespace(action_data={'events': [event]})
        box = {'gid': 2, 'bbox': [10, 10, 100, 180]}
        ui.detections = {'1': {'15': [box]}}
        image = np.zeros((200, 320, 3), dtype=np.uint8)
        ui._draw_action_labels(image, 1, 15, 1, 1)
        self.assertGreater(image.sum(), 0)
        ui.detections['1']['15'].append(box)
        image[:] = 0
        ui._draw_action_labels(image, 1, 15, 1, 1)
        self.assertEqual(image.sum(), 0)

    def test_pipeline_and_invalidation(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            writer = cv2.VideoWriter(str(root/'cam.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 10, (64, 64))
            for _ in range(40):
                writer.write(np.zeros((64, 64, 3), dtype=np.uint8))
            writer.release()
            detection = {'gid': 2, 'bbox': [5, 5, 40, 60]}
            data = {'persons': {'2': {}}, 'cameras': {'1': {'fps': 10, 'raw_video_file': 'cam.mp4', 'video_file': 'cam_tracked.mp4'}},
                    'detections': {'1': {'0': [detection], '10': [detection], '20': [detection, detection], '30': [detection]}}}
            (root/'tracking_log.json').write_text(json.dumps(data))
            class FakeModel:
                def predict(self, frame, boxes):
                    return [dict(posture='sitting', posture_score=.9, phone_score=None) for _ in boxes]
            args = SimpleNamespace(output_dir=folder, data_dir=folder, camera=None,
                                   start=0, end=4, step=1, walk_speed=.25)
            result = run(args, FakeModel())
            self.assertEqual(result['schema_version'], 2)
            self.assertEqual(len(result['events']), 1)
            self.assertEqual(result['events'][0]['action'], 'sitting')
            self.assertEqual(result['events'][0]['end_seconds'], 1)
            self.assertEqual(result['coverage'][0]['ambiguous_ids_skipped'], 1)
            self.assertEqual(load_actions(data, root)['events'], result['events'])
            assistant = AIAssistant(data, root)
            self.assertIn('Suggested action: sitting', assistant.build_system_context())
            original = (root/'action_events.json').read_bytes()
            class FailedModel:
                def predict(self, frame, boxes):
                    raise RuntimeError('inference failed')
            with self.assertRaises(RuntimeError):
                run(args, FailedModel())
            self.assertEqual((root/'action_events.json').read_bytes(), original)
            source = root/'cam.mp4'
            source.touch()
            self.assertEqual(load_actions(data, root)['events'], [])
            data['persons']['3'] = {}
            self.assertEqual(load_actions(data, root)['events'], [])

    def test_missing_or_corrupt_sidecar(self):
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(load_actions({}, folder)['events'], [])
            (Path(folder)/'action_events.json').write_text('invalid')
            self.assertEqual(load_actions({}, folder)['events'], [])
            (Path(folder)/'action_events.json').write_text(json.dumps({'schema_version': 1}))
            self.assertEqual(load_actions({}, folder)['events'], [])

    def test_pose_phone_and_motion_evidence(self):
        points = [[0, 0, 0] for _ in range(17)]
        for index, xy in {5: (0, 0), 11: (0, 10), 13: (10, 10), 15: (10, 20)}.items():
            points[index] = [*xy, .9]
        self.assertEqual(posture_from_keypoints(points), ('sitting', .9))
        self.assertTrue(chair_support([0, 0, 40, 60], points, [[0, 8, 30, 55]]))
        self.assertFalse(chair_support([0, 0, 40, 60], points, []))
        for index, xy in {13: (0, 20), 15: (0, 30)}.items():
            points[index] = [*xy, .9]
        self.assertEqual(posture_from_keypoints(points), ('standing', .9))
        points[9] = [15, 10, .9]
        self.assertTrue(phone_near_hand([14, 8, 18, 12], [0, 0, 40, 60], points))
        self.assertFalse(phone_near_hand([30, 8, 34, 12], [0, 0, 40, 60], points))
        self.assertGreater(movement_speed([0, 0, 20, 100], [30, 0, 50, 100], 1), .25)
        self.assertEqual(movement_speed([0, 0, 20, 100], [0, 0, 20, 100], 1), 0)

    def test_single_observation_is_not_an_interval(self):
        sample = dict(camera_id=1, person_id=0, action='possible_phone_use',
                      time=2, score=.8, frame=20, bbox=[1, 2, 3, 4])
        self.assertEqual(merge_samples([sample], 1), [])


if __name__ == '__main__':
    unittest.main()
