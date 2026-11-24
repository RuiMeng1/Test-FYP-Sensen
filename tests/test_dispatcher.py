import threading
import time
import unittest
from unittest.mock import patch

import numpy as np

from src.dispatcher import DetectionDispatcher


class FakeDetector:
    lock = threading.Lock()
    active = 0
    peak = 0
    calls = {}
    fail = False

    def __init__(self, model_name, device):
        with self.lock:
            self.index = len(self.calls)
            self.calls[self.index] = []

    def detect(self, frame, camera_id):
        cls = type(self)
        with cls.lock:
            cls.active += 1
            cls.peak = max(cls.peak, cls.active)
            cls.calls[self.index].append(int(frame[0, 0, 0]))
        try:
            if self.fail:
                raise ValueError("inference failed")
            time.sleep(0.08)
            return [camera_id]
        finally:
            with cls.lock:
                cls.active -= 1


class DispatcherTest(unittest.TestCase):
    def setUp(self):
        FakeDetector.active = 0
        FakeDetector.peak = 0
        FakeDetector.calls = {}
        FakeDetector.fail = False
        self.patcher = patch("src.dispatcher.PersonDetector", FakeDetector)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def run_cameras(self, dispatcher):
        barrier = threading.Barrier(3)
        answers = {}

        def camera(index):
            barrier.wait()
            answers[index] = [
                dispatcher.detect(index, np.full((1, 1, 3), frame, dtype=np.uint8), index + 1)
                for frame in (0, 1)
            ]

        threads = [threading.Thread(target=camera, args=(index,)) for index in range(2)]
        for thread in threads:
            thread.start()
        barrier.wait()
        for thread in threads:
            thread.join(timeout=3)
            self.assertFalse(thread.is_alive(), "camera stalled waiting for detection")
        return answers

    def test_cpu_runs_two_cameras_concurrently_without_reordering_tracks(self):
        dispatcher = DetectionDispatcher(2, "cpu")
        try:
            self.assertEqual(dispatcher.n_workers, 2)
            self.assertEqual(self.run_cameras(dispatcher), {0: [[1], [1]], 1: [[2], [2]]})
            self.assertEqual(FakeDetector.peak, 2)
            self.assertEqual(FakeDetector.calls, {0: [0, 1], 1: [0, 1]})
        finally:
            dispatcher.stop()

    def test_one_worker_and_mps_remain_serial(self):
        for device, max_workers in (("cpu", 1), ("mps", 2)):
            with self.subTest(device=device):
                FakeDetector.peak = 0
                dispatcher = DetectionDispatcher(2, device, max_workers=max_workers)
                try:
                    self.assertEqual(dispatcher.n_workers, 1)
                    self.run_cameras(dispatcher)
                    self.assertEqual(FakeDetector.peak, 1)
                finally:
                    dispatcher.stop()
                FakeDetector.calls = {}

    def test_inference_error_wakes_waiting_camera(self):
        FakeDetector.fail = True
        dispatcher = DetectionDispatcher(1, "cpu")
        try:
            with self.assertRaisesRegex(RuntimeError, "Detection worker failed"):
                dispatcher.detect(0, np.zeros((1, 1, 3), dtype=np.uint8), 1)
        finally:
            dispatcher.stop()


if __name__ == "__main__":
    unittest.main()
