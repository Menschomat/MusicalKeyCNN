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
    if len(beats) < 2:
        return 0.0
    # Fit t = t0 + n*T across all beats — slope T is the global constant period
    period, _ = np.polyfit(np.arange(len(beats)), beats, 1)
    return round(60.0 / period, 1)
