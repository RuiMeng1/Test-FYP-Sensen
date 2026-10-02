from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from src.occlusion import PartialOcclusionGuard
from src.detector import Detection, PersonDetector
from src.gallery import GlobalGallery


class PartialOcclusionTest(unittest.TestCase):
    def setUp(self):
        self.guard = PartialOcclusionGuard()

    def box(self, width=80, height=200, full_body=True, tid=1):
        self.guard.advance()
        return self.guard.check(tid, width, height, full_body)

    def test_shrink_freezes_baseline_until_three_recovery_frames(self):
        self.assertEqual(self.box(), (True, False))
        for _ in range(40):
            self.assertEqual(self.box(height=110), (True, True))
        self.assertEqual(self.box(), (True, True))
        self.assertEqual(self.box(), (True, True))
        self.assertEqual(self.box(), (True, False))

    def test_gradual_distance_change_is_allowed(self):
        for scale in (0.98 ** i for i in range(100)):
            self.assertEqual(self.box(80 * scale, 200 * scale), (True, False))

    def test_partial_box_requires_prior_reliable_detection(self):
        self.assertEqual(self.box(height=60, full_body=False), (False, False))
        self.assertEqual(self.box(), (True, False))
        self.assertEqual(self.box(height=60, full_body=False), (True, True))
        self.assertEqual(self.box(height=60, full_body=False, tid=2), (False, False))

    def test_missing_frames_reset_recovery_and_expire_history(self):
        self.box()
        self.box(height=100)
        self.box()
        self.guard.advance()
        self.assertEqual(self.box(), (True, True))
        self.assertEqual(self.box(), (True, True))
        self.assertEqual(self.box(), (True, False))
        for _ in range(31):
            self.guard.advance()
        self.assertEqual(self.box(height=100), (True, False))

    def test_detector_retains_and_marks_previously_rejected_partial_box(self):
        detector = PersonDetector.__new__(PersonDetector)
        detector.device = 'cpu'
        detector._occlusion = self.guard
        detector._model = MagicMock()
        box = MagicMock()
        box.id.item.return_value = 1
        box.conf.item.return_value = 0.9
        result = MagicMock()
        # A list-like result with the batch-level IDs YOLO exposes.
        result.boxes = MagicMock(id=True)
        result.boxes.__iter__.side_effect = lambda: iter([box])
        detector._model.track.return_value = [result]
        frame = np.zeros((300, 300, 3), dtype=np.uint8)
        box.xyxy[0].int().tolist.return_value = [10, 10, 90, 210]
        self.assertFalse(detector.detect(frame, 1)[0].partially_occluded)
        box.xyxy[0].int().tolist.return_value = [10, 10, 90, 70]
        self.assertTrue(detector.detect(frame, 1)[0].partially_occluded)
        box.id.item.return_value = 2
        self.assertEqual(detector.detect(frame, 1), [])

    def test_pipeline_preserves_id_without_learning_partial_crops(self):
        from src.pipeline import CameraWorker

        gallery = GlobalGallery()
        gallery.add_thumbnail_to_entry = MagicMock(wraps=gallery.add_thumbnail_to_entry)
        encoder = MagicMock()
        encoder.extract_batch.return_value = [np.array([1.0, 0.0], dtype=np.float32)]
        dispatcher = MagicMock()
        # Keep presenting a partial crop beyond the thumbnail update interval.
        flags = [False] + [True] * 6 + [False]
        dispatcher.detect.side_effect = [
            [Detection((10, 10, 90, 210 if not flag else 110), 1, 0.9, 1, flag)]
            for flag in flags
        ]
        capture = MagicMock()
        capture.get.side_effect = [300, 300, 25.0, len(flags)]
        frame = np.ones((300, 300, 3), dtype=np.uint8)
        capture.read.side_effect = [(True, frame.copy()) for _ in flags] + [(False, None)]
        log = MagicMock()
        with tempfile.TemporaryDirectory() as folder, \
                patch('src.pipeline.OUTPUT_DIR', Path(folder)), \
                patch('src.pipeline.cv2.VideoCapture', return_value=capture), \
                patch('src.pipeline.cv2.VideoWriter'):
            worker = CameraWorker(Path('occlusion-test.mp4'), 1, 0, dispatcher,
                                  encoder, gallery, reid_interval=1, tracking_log=log)
            worker._run()

        self.assertEqual(encoder.extract_batch.call_count, 2)
        self.assertEqual(gallery.add_thumbnail_to_entry.call_count, 2)
        self.assertEqual(gallery.active_count, 1)
        self.assertEqual(worker._id_cache, {1: 0})
        self.assertEqual(len(log.record_frame.call_args_list), len(flags))
        for call in log.record_frame.call_args_list:
            self.assertEqual(call.args[2], {0})


if __name__ == '__main__':
    unittest.main()
