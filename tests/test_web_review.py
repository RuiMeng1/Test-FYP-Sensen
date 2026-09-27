import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from src.web_review import create_app, prepare_video


class WebReviewTest(unittest.TestCase):
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

    def test_mutations_and_backup(self):
        body = {'source':0,'target':1,'revision':self.revision}
        self.assertEqual(self.client.post('/api/merge',json=body).status_code,403)
        self.assertEqual(self.client.post('/api/merge',json=body,headers=self.headers).status_code,200)
        saved = json.loads((self.root/'tracking_log.json').read_text())
        self.assertNotIn('0',saved['persons'])
        self.assertEqual(saved['detections']['1']['0'][0]['gid'],1)
        backups = list(self.root.glob('tracking_log.before-merge-*.json'))
        self.assertEqual(json.loads(backups[0].read_text()),self.data)
        self.assertEqual(self.client.post('/api/merge',json=body,headers=self.headers).status_code,409)
        self.assertEqual(self.client.post('/api/settings',json={'actions':['running']},headers=self.headers).status_code,200)
        self.assertEqual(self.client.get('/api/state').json['selected'],['running'])

    def test_chat_and_pdf(self):
        with patch('src.ai_assistant.AIAssistant.ask', return_value='[Person 0] at [Cam 1 @ 0:01]'):
            response = self.client.post('/api/chat',json={'prompt':'Where is Person 0?'},headers=self.headers)
            self.assertIn('[Person 0]',response.json['reply'])
        pdf = self.client.get('/api/report.pdf')
        self.assertEqual(pdf.status_code,200)
        self.assertTrue(pdf.data.startswith(b'%PDF'))

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

        def slow_reply(_prompt):
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
