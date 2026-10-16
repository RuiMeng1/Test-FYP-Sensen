import json
import tempfile
import unittest
from types import SimpleNamespace
from src.occlusion_logger import OcclusionLogger


def detection(tid, box, partial=False):
    return SimpleNamespace(local_id=tid, bbox=box, confidence=.9,
                           partially_occluded=partial)


class OcclusionLoggerTests(unittest.TestCase):
    def test_shrink_missing_and_return(self):
        with tempfile.TemporaryDirectory() as root:
            log = OcclusionLogger(root, 6, 'camera.mp4', 10)
            full = detection(1, (0, 0, 50, 200))
            log.update(0, [full], {1: 4})
            log.update(1, [detection(1, (0, 0, 50, 100))], {1: 4})
            log.update(2, [], {})
            log.update(3, [], {})
            log.update(4, [full], {1: 4})
            log.finish(5)
            events = [json.loads(line) for line in log.path.read_text().splitlines()]
            self.assertEqual([e['event'] for e in events], [
                'session_start', 'suspected_occlusion', 'track_missing',
                'track_reappeared', 'occlusion_cues_cleared', 'session_end'])
            self.assertEqual(events[1]['height_ratio'], .5)
            self.assertEqual(events[3]['previous_global_id'], 4)

    def test_crossing_and_new_identity_candidate(self):
        with tempfile.TemporaryDirectory() as root:
            log = OcclusionLogger(root, 7, 'camera.mp4', 10)
            a = detection(1, (0, 0, 50, 100))
            b = detection(2, (20, 0, 70, 100))
            log.update(0, [a, b], {1: 4, 2: 5})
            log.update(1, [a, b], {1: 4, 2: 5})
            log.update(2, [b], {2: 5})
            log.update(3, [detection(3, a.bbox), b], {3: 8, 2: 5})
            events = [json.loads(line) for line in log.path.read_text().splitlines()]
            candidate = next(e for e in events if e['event']=='possible_reappearance_new_track')
            self.assertEqual(candidate['global_id'], 8)
            self.assertEqual(candidate['candidates'][0]['global_id'], 4)
            self.assertFalse(any(e['frame']==1 for e in events))

    def test_expiry_and_separate_runs(self):
        with tempfile.TemporaryDirectory() as root:
            log = OcclusionLogger(root, 1, 'camera.mp4', 10)
            log.update(0, [detection(1, (0, 0, 50, 100))], {1: 1})
            for frame in range(1, 40):
                log.update(frame, [], {})
            self.assertFalse(log.tracks)
            other = OcclusionLogger(root, 1, 'camera.mp4', 10)
            self.assertNotEqual(log.path, other.path)


if __name__ == '__main__':
    unittest.main()
