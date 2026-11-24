"""
test_ai_assistant.py — Tests for the OpenAI AI Assistant module.
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.ai_assistant import AIAssistant


class TestAIAssistant(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.mock_log = {
            "cameras": {
                "1": {"fps": 25.0, "total_frames": 1000, "video_file": "cam1.mp4"},
                "2": {"fps": 25.0, "total_frames": 1000, "video_file": "cam2.mp4"},
            },
            "persons": {
                "0": {
                    "global_id": 0,
                    "color_hex": "#10b981",
                    "thumbnail_b64": "dummy_b64",
                    "total_detections": 150,
                    "cameras_seen": [1, 2],
                    "visual_attributes": "Wearing a red jacket, black trousers, and white sneakers.",
                    "appearances": [
                        {
                            "camera_id": 1,
                            "first_frame": 100,
                            "last_frame": 200,
                            "first_timestamp": "0:04",
                            "last_timestamp": "0:08",
                            "duration_seconds": 4.0,
                        },
                        {
                            "camera_id": 2,
                            "first_frame": 250,
                            "last_frame": 400,
                            "first_timestamp": "0:10",
                            "last_timestamp": "0:16",
                            "duration_seconds": 6.0,
                        },
                    ],
                }
            },
        }
        self.assistant = AIAssistant(self.mock_log, Path(self.folder.name))

    def test_context_builder(self):
        context = self.assistant.build_system_context()
        self.assertIn("Camera 1: File=cam1.mp4", context)
        self.assertIn("[Person 0]", context)
        self.assertIn("0:04 to 0:08", context)

    def test_query_ai_without_key(self):
        with patch.object(self.assistant, 'reload_config'), patch.object(self.assistant, 'api_key', ''):
            reply = self.assistant.ask("Where was Person 0?")
        self.assertIn("OpenAI API Key Not Found or Not Configured", reply)


if __name__ == "__main__":
    unittest.main()
