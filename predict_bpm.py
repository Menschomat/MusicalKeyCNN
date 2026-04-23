import numpy as np
from beat_this.inference import File2Beats

_tracker = None


def _get_tracker():
    global _tracker
    if _tracker is None:
        _tracker = File2Beats(checkpoint_path="final0", device="cpu", dbn=False)
    return _tracker


def detect_bpm(audio_path):
    beats, _ = _get_tracker()(str(audio_path))
    if len(beats) > 1:
        bpm = 60.0 / np.median(np.diff(beats))
    else:
        bpm = 0.0
    return round(float(bpm), 1)
