"""Full MP4 evidence is independent of bounded, temporally sampled GIF previews."""

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from PIL import Image

from robot_tasks.backends.settings import SimulationSettings
from robot_tasks.types import RGBFrame
from robot_tasks.video import MotionVideo


class VideoPreviewTests(unittest.TestCase):
    def test_every_frame_reaches_mp4_while_preview_spans_full_episode(self):
        writer = Mock()
        cv2 = SimpleNamespace(VideoWriter=lambda *args: writer, VideoWriter_fourcc=lambda *args: 0,
                              cvtColor=lambda pixels, code: pixels, COLOR_RGB2BGR=1)
        writer.isOpened.return_value = True
        with tempfile.TemporaryDirectory() as directory, patch.dict("sys.modules", {"cv2": cv2}):
            settings = SimulationSettings(Path(directory), width=8, height=6, fps=10)
            video = MotionVideo(Path(directory), settings)
            video.preview_limit = 8
            for index in range(103):
                rgb = bytes([index, 255 - index, index // 2]) * 48
                video.add(RGBFrame("overview", index / 10, 8, 6, rgb))
                self.assertLessEqual(len(video.frames), 8)
            self.assertEqual(writer.write.call_count, 103)
            self.assertEqual(video.count, 103)
            self.assertEqual(video.preview_indices[0], 0)
            self.assertGreater(video.preview_indices[-1], 80)
            self.assertTrue(all(frame.mode == "P" for frame in video.frames))
            video.close()
            writer.release.assert_called_once()
            with Image.open(Path(directory) / "motion.gif") as gif:
                self.assertLessEqual(gif.n_frames, 9)
                duration = 0
                for index in range(gif.n_frames):
                    gif.seek(index)
                    duration += gif.info["duration"]
                self.assertEqual(duration, 10300)
                last = gif.convert("RGB").getpixel((0, 0))
                self.assertEqual(last, (102, 153, 51))
            self.assertEqual(video.frames, [])
            self.assertIsNone(video.last_pixels)


if __name__ == "__main__":
    unittest.main()
