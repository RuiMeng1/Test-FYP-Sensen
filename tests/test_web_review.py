import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
from src.action_events import fingerprint, load_actions, source_signature
from src.action_labels import LABELS
from src.web_review import create_app, prepare_video, overlay_boxes


class WebReviewTest(unittest.TestCase):
    def test_overlay_bridges_only_short_unambiguous_gaps(self):
        frames = {
            '0': [{'gid': 3, 'bbox': [0, 0, 20, 40]}],
            '3': [{'gid': 3, 'bbox': [6, 0, 26, 40]}],
            '15': [{'gid': 3, 'bbox': [30, 0, 50, 40]}],
        }
        boxes = overlay_boxes(frames, 0, 15, 25)
        self.assertEqual(boxes['1'][0], {'gid': 3, 'bbox': [2, 0, 22, 40], 'interpolated': True})
        self.assertEqual(boxes['2'][0]['bbox'], [4, 0, 24, 40])
        self.assertNotIn('4', boxes)
        frames['1'] = [{'gid': 3, 'bbox': [1, 0, 21, 40]}, {'gid': 3, 'bbox': [2, 0, 22, 40]}]
        self.assertNotIn('2', overlay_boxes(frames, 0, 3, 25))

    def test_overlay_keeps_explicit_unknown_boxes_without_assigning_ids(self):
        known = {'gid': 3, 'bbox': [0, 0, 20, 40]}
        unknown = {'gid': -1, 'bbox': [0, 0, 20, 40]}
        other_unknown = {'gid': -1, 'bbox': [22, 0, 42, 40]}
        frames = {'0': [known, unknown], '1': [unknown], '2': [other_unknown], '3': [known, unknown]}
        boxes = overlay_boxes(frames, 0, 3, 25)
        self.assertEqual(boxes['1'], [unknown])
        self.assertEqual([box['gid'] for box in boxes['2']], [-1, 3])

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        writer = cv2.VideoWriter(str(self.root/'cam.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 10, (64,64))
        for _ in range(20):
            writer.write(np.zeros((64,64,3), np.uint8))
        writer.release()
        self.data = {'cameras': {'1': {'fps': 10, 'total_frames': 20, 'video_file': 'tracked.mp4', 'raw_video_file': 'cam.mp4'}},
                     'persons': {str(g): {'global_id': g, 'appearances': [], 'cameras_seen': [1], 'total_detections': 1} for g in [0,1]},
                     'detections': {'1': {'0': [{'gid':0, 'bbox':[0,0,30,50]}]}}}
        (self.root/'tracking_log.json').write_text(json.dumps(self.data))
        self.app = create_app(self.root,self.root,prepare=False)
        self.client = self.app.test_client()
        state = self.client.get('/api/state').json
        self.headers = {'X-Review-Token': state['token']}
        self.revision = state['revision']

    def tearDown(self):
        self.directory.cleanup()

    def test_ranges_and_state(self):
        with self.client.get('/') as page:
            self.assertEqual(page.status_code,200)
        self.assertNotIn('detections', self.client.get('/api/state').json)
        response = self.client.get('/video/1', headers={'Range':'bytes=0-99'})
        self.assertEqual(response.status_code,206)
        self.assertEqual(len(response.data),100)
        response.close()
        self.assertIn('0',self.client.get('/api/boxes/1/0').json)
        self.assertEqual(self.client.get('/api/state',headers={'Host':'untrusted.example'}).status_code,403)

    def test_movement_history_reads_selected_person_and_updates_after_merge(self):
        self.assertTrue(self.client.get('/api/state').json['office_map']['schematic'])
        before = self.client.get('/api/office-map/0').json
        overview_before = self.client.get('/api/office-map/overview/0').json
        self.assertEqual(overview_before['frames'], {'1': {'0': [[0]]}})
        self.assertEqual(before['segments'], [dict(camera='1', zone='1', start=0, end=.1, points=[])])
        self.assertEqual(self.client.get('/api/office-map/1').json['segments'], [])
        body = {'source': 0, 'target': 1, 'revision': self.revision}
        self.assertEqual(self.client.post('/api/merge', json=body, headers=self.headers).status_code, 200)
        self.assertEqual(self.client.get('/api/office-map/1').status_code, 404)
        after = self.client.get('/api/office-map/0').json
        self.assertEqual(after['segments'], before['segments'])
        self.assertNotEqual(after['revision'], before['revision'])
        overview_after = self.client.get('/api/office-map/overview/0').json
        self.assertEqual(overview_after['revision'], after['revision'])
        self.assertEqual(overview_after['frames'], {'1': {'0': [[0]]}})

    def test_camera_editor_persists_layout_and_updates_movement_history(self):
        layout = self.client.get('/api/state').json['office_map']
        url = '/api/office-map/placements'
        pin = dict(zone=None, floor='zones', position=[2.5, 3], heading=90)
        body = {'revision': layout['revision'], 'cameras': {'1': pin}}
        self.assertEqual(self.client.post(url, json=body).status_code, 403)
        bad = {**body, 'cameras': {'1': {**pin, 'position': [-5, 3]}}}
        self.assertEqual(self.client.post(url, json=bad, headers=self.headers).status_code, 400)
        self.assertFalse((self.root / 'office_camera_placements.json').exists())

        result = self.client.post(url, json=body, headers=self.headers)
        self.assertEqual(result.status_code, 200, result.json)
        updated = result.json['office_map']
        self.assertEqual({key: updated['cameras']['1'][key] for key in pin}, pin)
        history = self.client.get('/api/office-map/0').json
        self.assertIsNone(history['segments'][0]['zone'])
        self.assertEqual(history['map_revision'], updated['revision'])
        self.assertEqual(self.client.post(url, json=body, headers=self.headers).status_code, 409)

        reopened = create_app(self.root, self.root, prepare=False).test_client()
        self.assertEqual(reopened.get('/api/state').json['office_map'], updated)

    def test_mutations_and_backup(self):
        body = {'source':0,'target':1,'revision':self.revision}
        self.assertEqual(self.client.post('/api/merge',json=body).status_code,403)
        self.assertEqual(self.client.post('/api/merge',json=body,headers=self.headers).status_code,200)
        saved = json.loads((self.root/'tracking_log.json').read_text())
        self.assertEqual(list(saved['persons']), ['0'])
        self.assertEqual(saved['detections']['1']['0'][0]['gid'], 0)
        backups = list(self.root.glob('tracking_log.before-merge-*.json'))
        self.assertEqual(json.loads(backups[0].read_text()),self.data)
        self.assertEqual(self.client.post('/api/merge',json=body,headers=self.headers).status_code,409)
        self.assertEqual(self.client.post('/api/settings',json={'actions':['walking']},headers=self.headers).status_code,200)
        self.assertEqual(self.client.get('/api/state').json['selected'],['walking'])

    def test_false_id_deletion_updates_overlays_and_preserves_other_activity(self):
        self.data['persons'] = {str(gid): dict(global_id=gid, appearances=[], cameras_seen=[1],
                                               thumbnail_b64=f'person-{gid}') for gid in range(14)}
        box = [1, 2, 30, 50]
        self.data['detections']['1'] = {'0': [dict(gid=12, bbox=box), dict(gid=-1, bbox=box)],
                                        '1': [dict(gid=13, global_id=13, bbox=box)]}
        log = self.root / 'tracking_log.json'
        log.write_text(json.dumps(self.data))
        events = [dict(person_id=gid, camera_id=1, action='walking', start_seconds=0, end_seconds=.5,
                       model_score=.8, evidence_frame=0, review_status='unreviewed') for gid in (12, 13)]
        (self.root / 'action_events.json').write_text(json.dumps(dict(
            schema_version=2, tracking_fingerprint=fingerprint(self.data),
            supported_actions=list(LABELS.values()), sources=[source_signature(self.root / 'cam.mp4')],
            status='Available', events=events,
        )))
        client = create_app(self.root, self.root, prepare=False).test_client()
        state = client.get('/api/state').json
        headers = {'X-Review-Token': state['token']}
        body = {'source': 12, 'revision': state['revision']}
        self.assertEqual(client.post('/api/delete-id', json=body).status_code, 403)
        self.assertEqual(client.post('/api/delete-id', json=body, headers=headers).status_code, 200)
        updated = json.loads(log.read_text())
        self.assertEqual(set(updated['persons']), {str(gid) for gid in range(13)})
        self.assertEqual(updated['persons']['12']['thumbnail_b64'], 'person-13')
        self.assertEqual(updated['detections']['1']['0'], [dict(gid=-1, bbox=box)])
        self.assertEqual(updated['detections']['1']['1'], [dict(gid=12, global_id=12, bbox=box)])
        self.assertEqual(client.get('/api/office-map/overview/0').json['frames'], {'1': {'1': [[12]]}})
        self.assertEqual(load_actions(updated, self.root)['events'], [dict(events[1], person_id=12)])
        self.assertEqual(client.post('/api/delete-id', json=body, headers=headers).status_code, 409)
        original = next(self.root.glob('tracking_log.before-delete-*.json'))
        self.assertEqual(json.loads(original.read_text()), self.data)

    def test_merge_closes_the_gap_and_keeps_one_same_frame_sighting(self):
        self.data['persons'] = {str(gid): dict(global_id=gid, appearances=[], cameras_seen=[1],
                                               thumbnail_b64=f'person-{gid}') for gid in range(14)}
        source_box, target_box = [2, 2, 30, 50], [1, 2, 30, 50]
        self.data['detections']['1'] = {'0': [dict(gid=11, bbox=target_box), dict(gid=12, bbox=source_box)],
                                        '1': [dict(gid=12, bbox=source_box), dict(gid=13, bbox=target_box)]}
        log = self.root / 'tracking_log.json'
        log.write_text(json.dumps(self.data))
        client = create_app(self.root, self.root, prepare=False).test_client()
        state = client.get('/api/state').json
        result = client.post('/api/merge', json={'source': 12, 'target': 11, 'revision': state['revision']},
                             headers={'X-Review-Token': state['token']})
        self.assertEqual(result.status_code, 200, result.json)
        self.assertEqual(result.json['kept_id'], 11)
        updated = json.loads(log.read_text())
        self.assertEqual(updated['persons']['12']['global_id'], 12)
        self.assertEqual(updated['persons']['12']['thumbnail_b64'], 'person-13')
        self.assertEqual(updated['detections']['1']['0'], [dict(gid=11, bbox=target_box)])
        self.assertEqual([box['gid'] for box in updated['detections']['1']['1']], [11, 12])

    def test_chat_and_pdf(self):
        with patch('src.ai_assistant.AIAssistant.ask', return_value='[Person 0] at [Cam 1 @ 0:01]'):
            response = self.client.post('/api/chat',json={'prompt':'Where is Person 0?'},headers=self.headers)
            self.assertIn('[Person 0]',response.json['reply'])
        pdf = self.client.get('/api/report.pdf')
        self.assertEqual(pdf.status_code,200)
        self.assertTrue(pdf.data.startswith(b'%PDF'))

    def test_chat_uses_current_map_and_playhead_and_renders_safe_markdown(self):
        path = self.root / 'map.json'
        path.write_text(json.dumps({
            'floors': [dict(id='office', label='Office level', elevation=0, bounds=[0, 0, 10, 5])],
            'zones': [dict(id='workspace', label='Workspace', floor='office', bounds=[0, 0, 5, 5]),
                      dict(id='hall', label='Hallway', floor='office', bounds=[5, 0, 5, 5])],
            'cameras': {'cam': dict(zone='workspace', floor='office', position=[6, 2], heading=270)},
        }))
        browser = create_app(self.root, self.root, prepare=False, office_map_path=path).test_client()
        state = browser.get('/api/state').json
        headers = {'X-Review-Token': state['token']}
        completion = MagicMock()
        model = completion.chat.completions.create
        model.return_value.choices[0].message.content = (
            '## Route\n\n- **Workspace** [Cam 1 @ 0:00]\n\n'
            '<img src=x onerror=alert(1)>\n\n[unsafe](javascript:alert%281%29)\n\n'
            '![image](https://example.test/image.png)'
        )
        prompt = "**Where** did ID0 go?\n\n- Use the map."
        with patch('src.ai_assistant.AIAssistant.is_configured', return_value=True), \
                patch('src.ai_assistant.AIAssistant._get_client', return_value=completion):
            reply = browser.post('/api/chat', json={
                'prompt': prompt, 'view': {'followed': '0', 'camera': '1', 'seconds': .04},
            }, headers=headers).json
            messages = model.call_args.kwargs['messages']
            self.assertEqual(messages[-1]['content'], prompt)
            context = messages[0]['content']
            self.assertIn('"coverage":{"area":"Workspace","floor":"Office level"}', context)
            self.assertIn('"pin":{"floor":"Office level","xy":[6,2],"heading_degrees":270}', context)
            self.assertIn('"followed_person":0,"focused_camera":1', context)
            self.assertIn('"sightings_at_playhead":[{"camera":1', context)
            self.assertIn('<strong>Where</strong>', reply['prompt_html'])
            self.assertIn('<h2>Route</h2>', reply['reply_html'])
            self.assertIn('<strong>Workspace</strong>', reply['reply_html'])
            self.assertIn('[Cam 1 @ 0:00]', reply['reply'])
            self.assertNotIn('<img', reply['reply_html'])
            self.assertNotIn('href="javascript:', reply['reply_html'])

            saved = browser.post('/api/office-map/placements', json={
                'revision': state['office_map']['revision'],
                'cameras': {'1': dict(zone='hall', floor='office', position=[6, 2], heading=270)},
            }, headers=headers)
            self.assertEqual(saved.status_code, 200)
            self.assertEqual(browser.post('/api/chat', json={
                'prompt': 'Where is ID0 now?', 'view': {'seconds': 1.5},
            }, headers=headers).status_code, 200)
            updated = model.call_args.kwargs['messages'][0]['content']
            self.assertIn('"coverage":{"area":"Hallway","floor":"Office level"}', updated)
            self.assertIn('"sightings_at_playhead":[]', updated)

    def test_browser_proxy(self):
        cache = self.root/'cache'
        cache.mkdir()
        path = prepare_video(self.root/'cam.mp4',cache)
        import av
        with av.open(str(path)) as container:
            self.assertEqual(container.streams.video[0].codec_context.name,'h264')
            self.assertEqual(len(list(container.decode(video=0))),20)
        self.assertEqual(prepare_video(self.root/'cam.mp4',cache),path)

    def test_chat_does_not_block_playback_metadata(self):
        entered, release = threading.Event(), threading.Event()

        def slow_reply(_prompt, *, office_context):
            entered.set()
            release.wait(5)
            return 'Done'

        def chat():
            with self.app.test_client() as client:
                client.post('/api/chat', json={'prompt': 'Describe the recording'}, headers=self.headers)

        with patch('src.ai_assistant.AIAssistant.ask', side_effect=slow_reply):
            worker = threading.Thread(target=chat)
            worker.start()
            try:
                self.assertTrue(entered.wait(2))
                self.assertEqual(self.client.get('/api/boxes/1/0').status_code, 200)
                self.assertEqual(self.client.post('/api/chat/clear', json={}, headers=self.headers).status_code, 409)
            finally:
                release.set()
                worker.join(6)

    def test_proxy_normalizes_near_duplicate_source_timestamps(self):
        import av
        from fractions import Fraction
        source = self.root / 'irregular.mkv'
        with av.open(str(source), 'w') as container:
            stream = container.add_stream('ffv1', rate=25)
            stream.width = stream.height = 64
            stream.pix_fmt = 'yuv420p'
            stream.codec_context.time_base = Fraction(1, 1000)
            for pts in (0, 1, 80, 120):
                frame = av.VideoFrame.from_ndarray(np.zeros((64,64,3), np.uint8), format='rgb24')
                frame.pts, frame.time_base = pts, Fraction(1, 1000)
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        cache = self.root / 'irregular-cache'
        cache.mkdir()
        proxy = prepare_video(source, cache, fps=25)
        with av.open(str(proxy)) as container:
            times = [float(frame.pts * frame.time_base) for frame in container.decode(video=0)]
        self.assertEqual(times, [0, .04, .08, .12])


if __name__ == '__main__':
    unittest.main()
