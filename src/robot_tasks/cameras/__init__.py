"""Camera configuration and simulation rigs; hardware calibration is explicit."""
class RGBNotReady(RuntimeError):
    """A render annotator has not produced its first image yet."""
