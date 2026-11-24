import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from src.detector import Detection
from src.gallery import GlobalGallery
from src.pipeline import CameraWorker
from src.tracking_log import TrackingLog, saved_detection_counts
from src.video_loader import resolve_new_id_cameras


class IdCameraTest(unittest.TestCase):
    def test_saved_names_follow_input_order_and_missing_sources_fail(self):
        with tempfile.TemporaryDirectory() as folder:
            settings = Path(folder) / 'pipeline_settings.json'
            videos = [Path('back stairs.mp4'), Path('reception.mp4'), Path('entrance.mp4')]
            args = SimpleNamespace(all_id_cameras=False, use_saved_id_cameras=True, new_id_cameras=None)

            settings.write_text(json.dumps({'new_id_cameras': ['entrance', 'back stairs']}))
            self.assertEqual(resolve_new_id_cameras(args, videos, settings), {1, 3})
            self.assertEqual(resolve_new_id_cameras(args, [videos[1], videos[0], videos[2]], settings), {2, 3})
            with self.assertRaisesRegex(ValueError, 'Saved cameras are missing'):
                resolve_new_id_cameras(args, videos[:2], settings)

            args.use_saved_id_cameras = False
            args.new_id_cameras = ['entrance.mp4', '1']
            self.assertEqual(resolve_new_id_cameras(args, videos, settings), {1, 3})
            args.new_id_cameras = None
            with self.assertRaisesRegex(ValueError, 'Choose --new-id-cameras'):
                resolve_new_id_cameras(args, videos, settings)
            args.all_id_cameras = True
            self.assertIsNone(resolve_new_id_cameras(args, videos, settings))
            args.all_id_cameras = False
            args.use_saved_id_cameras = True
            settings.write_text(json.dumps({'new_id_cameras': None}))
            self.assertIsNone(resolve_new_id_cameras(args, videos, settings))

    def test_only_selected_cameras_register_and_other_cameras_can_match(self):
        person = np.array([1.0, 0.0])
        different = np.array([0.0, 1.0])
        gallery = GlobalGallery(new_id_camera_ids={1, 3})
        self.assertIsNone(gallery.match_or_register(person, 2, 10, 0))
        self.assertEqual(gallery.active_count, 0)
        self.assertEqual(gallery.match_batch(1, 1, [(1, person)]), {1: 0})
        self.assertEqual(gallery.match_batch(2, 2, [(10, person), (11, different)]), {10: 0})
        self.assertEqual(gallery.match_batch(2, 3, [(10, None)]), {10: 0})
        self.assertEqual(gallery.match_batch(3, 4, [(1, different)]), {1: 1})
        self.assertEqual(gallery.active_count, 2)
        self.assertEqual(GlobalGallery().match_batch(2, 0, [(10, different)]), {10: 0})

    def test_unknown_boxes_are_saved_and_matching_retries_at_the_reid_interval(self):
        encoder = MagicMock()
        person = np.array([1.0, 0.0])
        encoder.extract_batch.return_value = [person]
        gallery = GlobalGallery(new_id_camera_ids={1})
        dispatcher = MagicMock()
        cap = MagicMock()
        cap.get.side_effect = [160, 240, 25.0, 4]
        frame = np.ones((240, 160, 3), dtype=np.uint8)
        cap.read.side_effect = [(True, frame.copy()) for _ in range(4)] + [(False, None)]
        box = (10, 10, 90, 210)
        frames = iter(range(4))

        def detect(*_args):
            if next(frames) == 2:
                gallery.match_batch(1, 2, [(1, person)])
            return [Detection(box, 10, 0.9, 2)]

        dispatcher.detect.side_effect = detect
        log = TrackingLog()
        with tempfile.TemporaryDirectory() as folder, \
                patch('src.pipeline.OUTPUT_DIR', Path(folder)), \
                patch('src.pipeline.cv2.VideoCapture', return_value=cap), \
                patch('src.pipeline.cv2.VideoWriter'), \
                patch('src.config.ENABLE_AUTO_VISUAL_CAPTIONING', False):
            worker = CameraWorker(Path('interior.mp4'), 2, 0, dispatcher, encoder, gallery,
                                  reid_interval=2, tracking_log=log)
            worker._run()
            path = Path(folder) / 'tracking_log.json'
            log.save(str(path), gallery)
            data = json.loads(path.read_text())

        self.assertEqual(encoder.extract_batch.call_count, 2)
        boxes = data['detections']['2']
        self.assertEqual([boxes[str(i)][0]['gid'] for i in range(4)], [-1, -1, 0, 0])
        self.assertEqual(set(data['persons']), {'0'})
        self.assertEqual(saved_detection_counts(data['detections']), {0: 2})
