"""
test_ai_assistant.py — Tests for the OpenAI AI Assistant module.
"""

import sys
import unittest
from pathlib import Path

# Ensure root directory is on python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ai_assistant import TrackingLogContextBuilder, query_tracking_ai


class TestAIAssistant(unittest.TestCase):
    def setUp(self):
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

    def test_context_builder(self):
        context = TrackingLogContextBuilder.build_system_context(self.mock_log)
        self.assertIn("Camera 1: File=cam1.mp4", context)
        self.assertIn("Person Global ID: 0", context)
        self.assertIn("Wearing a red jacket", context)
        self.assertIn("0:04 -> 0:08", context)

    def test_query_ai_without_key(self):
        # Without valid OpenAI API key, should return helpful message
        res = query_tracking_ai("Where was Person 0?", self.mock_log, api_key="")
        self.assertIn("OpenAI API Key Missing", res)


if __name__ == "__main__":
    unittest.main()
