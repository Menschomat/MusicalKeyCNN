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
    if len(beats) < 4:
        return 0.0

    ibis = np.diff(beats)

    # Remove outliers relative to the median: long gaps from silent sections and
    # re-lock jitter after breakdowns produce IBIs far outside the true beat period.
    median_ibi = np.median(ibis)
    ibis = ibis[(ibis >= 0.4 * median_ibi) & (ibis <= 2.5 * median_ibi)]
    if len(ibis) < 4:
        return 0.0

    # Vectorised grid search: find the constant BPM whose beat grid best explains
    # all inter-beat intervals as integer multiples (tolerates missed/doubled beats).
    candidates = np.arange(60.0, 200.02, 0.02)
    periods = (60.0 / candidates)[:, np.newaxis]  # (n_candidates, 1)
    ratios = ibis / periods                        # (n_candidates, n_ibis)
    scores = np.mean(np.abs(ratios - np.round(ratios)), axis=1)
    return round(float(candidates[np.argmin(scores)]), 1)
