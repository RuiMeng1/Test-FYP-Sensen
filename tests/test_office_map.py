import json
from pathlib import Path
import tempfile
import unittest

from src.office_map import load_office_map, movement_history, overview_frames


class OfficeMapTest(unittest.TestCase):
    def setUp(self):
        self.cameras = {
            '1': dict(raw_video_file='1-entrance-stairs_000.mp4', fps=10, total_frames=100),
            '2': dict(raw_video_file='6-workspace-1_000.mp4', fps=10, total_frames=100),
        }
        self.box = dict(gid=7, bbox=[20, 25, 80, 50])

    def test_schematic_history_keeps_gaps_and_camera_handoffs_separate(self):
        layout, lookup = load_office_map(self.cameras)
        self.assertTrue(layout['schematic'])
        self.assertEqual(layout['zones'][0]['label'], 'Entrance Stairs')
        frames = {
            '1': {'0': [self.box], '1': [self.box], '2': [self.box, self.box],
                  '3': [self.box], '10': [self.box]},
            '2': {'11': [self.box]},
        }
        route = movement_history(7, self.cameras, frames, [], lookup, {})
        times = [(s['camera'], s['start'], s['end']) for s in route['segments']]
        self.assertEqual(times, [('1', 0, .2), ('1', .3, .4), ('1', 1, 1.1), ('2', 1.1, 1.2)])
        self.assertTrue(all(not segment['points'] for segment in route['segments']))

    def test_projection_stays_within_calibration_and_degrades_to_camera_zone(self):
        config = dict(floors=[dict(id='level', label='Level 1', elevation=3, bounds=[0, 0, 20, 30])],
                      zones=[dict(id='door', label='Door', floor='level', bounds=[0, 0, 10, 20])],
                      cameras={'1-entrance-stairs_000': dict(zone='door', calibration={
                          'image': [[.1, .1], [.9, .1], [.9, .9], [.1, .9]],
                          'world': [[0, 0], [10, 0], [10, 20], [0, 20]],
                      })})
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'layout.json'
            path.write_text(json.dumps(config))
            layout, lookup = load_office_map(self.cameras, path)
        self.assertFalse(layout['schematic'])
        self.assertEqual(layout['floors'][0]['elevation'], 3)
        self.assertIsNone(layout['cameras']['2']['zone'])
        frames = {'1': {'0': [self.box], '1': [{'gid': 7, 'bbox': [0, 0, 8, 50]}]}}
        legacy = [dict(camera_id=2, first_frame=4, last_frame=6)]
        route = movement_history(7, self.cameras, frames, legacy, lookup,
                                 {'1': {'width': 100, 'height': 100}})['segments']
        self.assertEqual(route[0]['points'], [[0, 5, 10]])
        self.assertEqual(route[1]['points'], [])
        self.assertEqual(route[1]['zone'], 'door')
        self.assertEqual((route[2]['zone'], route[2]['start'], route[2]['end']), (None, .4, .7))

    def test_overview_uses_current_frames_and_projects_calibrated_sightings(self):
        tracking = {'persons': {'7': {}, '8': {'appearances': [dict(camera_id=2, first_frame=1, last_frame=2)]}},
                    'cameras': self.cameras, 'detections': {'1': {
                        '0': [self.box, dict(gid=8, bbox=[10, 20, 20, 30]), dict(gid=-1, bbox=[1, 2, 3, 4])],
                        '1': [self.box, self.box],
                        '9': [self.box],
                    }}}
        transform = ([[0, 0], [1, 0], [1, 1], [0, 1]], [[10, 0, 0], [0, 20, 0], [0, 0, 1]])
        frames = overview_frames(tracking, {'1': (None, transform), '2': (None, None)},
                                 {'1': {'width': 100, 'height': 100}}, 0, .3)
        self.assertEqual(frames['1'], {0: [[7, 5, 10], [8, 1.5, 6]]})
        self.assertEqual(frames['2'], {1: [[8]], 2: [[8]]})
