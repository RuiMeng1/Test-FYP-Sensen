import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

import cv2
import numpy as np

from src.action_analysis import run, merge_samples, LABELS
from src.action_events import load_actions
from src.ai_assistant import AIAssistant
from src.action_labels import visible_actions


class ActionsTest(unittest.TestCase):
    def test_overlay_filter_and_render(self):
        from src.review_ui import ReviewUI
        self.assertEqual(len(LABELS), 60)
        self.assertEqual(LABELS[57], 'texting_looking_at_phone')
        self.assertEqual(LABELS[64], 'fighting_hitting_person')
        event = dict(person_id=2, camera_id=1, action='running', start_seconds=1, end_seconds=2)
        self.assertEqual(visible_actions([event], {'running'}, 1, 1.5), [event])
        self.assertEqual(visible_actions([event], {'sitting'}, 1, 1.5), [])
        self.assertEqual(visible_actions([event], {'running'}, 2, 1.5), [])
        self.assertEqual(visible_actions([event], {'running'}, 1, 3), [])
        ui = ReviewUI.__new__(ReviewUI)
        ui.cameras = {1: {'fps': 10}}
        ui._overlay_actions = {'running'}
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
                def predict(self, frames, boxes):
                    assert len(frames) == 32
                    scores = np.zeros((len(boxes), 80))
                    scores[:, 10] = 0.9  # Official AVA label 11 = sit
                    return scores
            args = SimpleNamespace(output_dir=folder, data_dir=folder, camera=None, start=0, end=4, step=1, threshold=0.6)
            result = run(args, FakeModel())
            self.assertEqual(len(result['events']), 1)
            self.assertEqual(result['events'][0]['action'], 'sitting')
            self.assertEqual(result['events'][0]['end_seconds'], 1)
            self.assertEqual(result['coverage'][0]['ambiguous_ids_skipped'], 1)
            self.assertEqual(load_actions(data, root)['events'], result['events'])
            assistant = AIAssistant(data, root)
            self.assertIn('Suggested action: sitting', assistant.build_system_context())
            original = (root/'action_events.json').read_bytes()
            class FailedModel:
                def predict(self, frames, boxes):
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

    def test_brief_fall_preserved(self):
        self.assertEqual(LABELS[5], 'possible_fall')
        sample = dict(camera_id=1, person_id=0, action='possible_fall', time=2, score=.8, frame=50, bbox=[1, 2, 3, 4])
        self.assertEqual(len(merge_samples([sample], 1)), 1)


if __name__ == '__main__':
    unittest.main()
