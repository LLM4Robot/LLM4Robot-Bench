"""Bounded background recorder: ordered, complete frames and explicit failures."""

from dataclasses import replace
import queue
import threading


class VideoEncodingError(RuntimeError):
    pass


class BackgroundVideo:
    def __init__(self, output, settings, *, factory, timings, queue_frames=4):
        if type(queue_frames) is not int or queue_frames < 1:
            raise ValueError("Video queue must have positive bounded capacity")
        self.output, self.settings = output, settings
        self.timings = timings
        self.queue = queue.Queue(maxsize=queue_frames)
        self.ready = threading.Event()
        self.done = threading.Event()
        self.failure = None
        self.sink = None
        self.count = 0
        self.closed = False
        self.worker = threading.Thread(target=self._encode, args=(factory,),
                                       name="r5-video-" + output.name, daemon=True)
        self.worker.start()
        try:
            self.ready.wait()
            self._check()
        except BaseException:
            self.closed = True
            self.worker.join()
            raise

    def _check(self):
        if self.failure is not None:
            raise VideoEncodingError(f"Video encoding failed for {self.output.name}: {self.failure}") from self.failure

    def _encode(self, factory):
        try:
            with self.timings.measure("video.initialize"):
                self.sink = factory(self.output, self.settings)
            self.ready.set()
            while True:
                try:
                    frame = self.queue.get(timeout=.05)
                except queue.Empty:
                    if self.closed:
                        break
                    continue
                with self.timings.measure("video.encode"):
                    self.sink.add(frame)
            with self.timings.measure("video.finalize"):
                self.sink.close()
        except BaseException as exc:
            self.failure = exc
            # Release a partial MP4 even when add()/GIF finalization failed.
            # Do not manufacture a successful recording or suppress the error.
            writer = getattr(self.sink, "writer", None)
            if writer is not None:
                try:
                    writer.release()
                except Exception:
                    pass
        finally:
            self.ready.set()
            self.done.set()

    def _put(self, frame):
        while True:
            self._check()
            try:
                self.queue.put(frame, timeout=.05)
                return
            except queue.Full:
                self.timings.counted("video_backpressure_polls")

    def add(self, frame):
        if self.closed:
            raise RuntimeError("Video is already closed")
        if (frame.width, frame.height) != (self.settings.width, self.settings.height):
            raise ValueError("Video dimensions must match the source camera")
        if len(frame.data) != frame.width * frame.height * 3:
            raise ValueError("Video requires packed RGB8")
        # RGBFrame normally owns immutable bytes. Snapshot mutable inputs before
        # returning to the render thread, never retain a simulator-backed tensor.
        if not isinstance(frame.data, bytes):
            frame = replace(frame, data=bytes(frame.data))
        with self.timings.measure("video.enqueue"):
            self._put(frame)
            self.count += 1
            self._check()

    def close(self):
        with self.timings.measure("video.drain_wait"):
            if not self.closed:
                self.closed = True
                self.worker.join()
            self._check()
            if self.sink.count != self.count:
                raise VideoEncodingError("Not all submitted video frames were encoded")

    @property
    def maximum_pixel_change(self):
        return self.sink.maximum_pixel_change if self.sink is not None else 0.0
