"""Real camera-frame recording shared by tasks and integration checks."""

from pathlib import Path

from .backends.settings import SimulationSettings


class MotionVideo:
    """Stream every frame to MP4; bound the memory of a whole-episode GIF preview."""

    preview_limit = 240

    def __init__(self, output: Path, settings: SimulationSettings):
        import cv2
        self.output, self.settings = output, settings
        self.writer = cv2.VideoWriter(str(output / "motion.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                      settings.fps, (settings.width, settings.height))
        if not self.writer.isOpened():
            raise RuntimeError("Unable to open MP4 encoder")
        self.frames = []
        self.preview_indices = []
        self.preview_stride = 2
        self.last_pixels = None
        self.count = 0
        self.first_pixels = None
        self.maximum_pixel_change = 0.0

    def add(self, frame):
        import cv2
        import numpy as np
        from PIL import Image
        if (frame.width, frame.height) != (self.settings.width, self.settings.height):
            raise ValueError("Video dimensions must match the source camera; configure a separate writer per camera")
        if len(frame.data) != frame.width * frame.height * 3:
            raise ValueError("Video requires packed RGB8")
        pixels = np.frombuffer(frame.data, dtype=np.uint8).reshape(frame.height, frame.width, 3)
        if self.first_pixels is None:
            self.first_pixels = pixels.copy()
            Image.fromarray(pixels).save(self.output / "first_frame.png")
        self.maximum_pixel_change = max(self.maximum_pixel_change,
            float(np.abs(pixels.astype(float) - self.first_pixels.astype(float)).mean()))
        self.writer.write(cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR))
        self.last_pixels = pixels.copy()
        if self.count % self.preview_stride == 0:
            preview = Image.fromarray(pixels)
            preview.thumbnail((640, 480))
            self.frames.append(preview.quantize(colors=128))
            self.preview_indices.append(self.count)
            if len(self.frames) > self.preview_limit:
                # Increase sampling interval across the WHOLE episode. Never
                # discard only the beginning or select frames by task outcome.
                self.frames = self.frames[::2]
                self.preview_indices = self.preview_indices[::2]
                self.preview_stride *= 2
        self.count += 1
        if self.count % self.settings.fps == 0:
            Image.fromarray(pixels).save(self.output / "latest_frame.png")

    def close(self):
        from PIL import Image
        self.writer.release()
        if self.frames:
            if self.preview_indices[-1] != self.count - 1:
                preview = Image.fromarray(self.last_pixels)
                preview.thumbnail((640, 480))
                self.frames.append(preview.quantize(colors=128))
                self.preview_indices.append(self.count - 1)
            durations = [max(10, round(1000 * (b - a) / self.settings.fps))
                         for a, b in zip(self.preview_indices, self.preview_indices[1:] + [self.count])]
            self.frames[0].save(self.output / "motion.gif", save_all=True, append_images=self.frames[1:],
                                duration=durations, loop=0)
        self.frames.clear()
        self.preview_indices.clear()
        self.last_pixels = None
