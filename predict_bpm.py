import numpy as np
from beat_this.inference import Audio2Beats

_tracker = None

def _get_tracker():
    global _tracker
    if _tracker is None:
        _tracker = Audio2Beats(checkpoint_path="final0", device="cpu", dbn=False)
    return _tracker

def detect_bpm(signal, sr, min_bpm=60.0, max_bpm=200.0):
    beats, _ = _get_tracker()(signal, sr)
    if len(beats) < 4:
        return 0.0

    # 1. Calculate Inter-Beat Intervals (IBIs)
    ibis = np.diff(beats)

    # 2. Find the dominant interval (median) to ignore missed/extra beats
    median_ibi = np.median(ibis)

    # 3. Filter for "straight" beats to get a cleaner local period
    # We only look at intervals that are very close to 1x the median
    # (e.g., +/- 15%) to exclude double-time or half-time gaps.
    inlier_ibis = ibis[(ibis >= 0.85 * median_ibi) & (ibis <= 1.15 * median_ibi)]

    if len(inlier_ibis) < 4:
        return 0.0

    # The mean of these inliers gives a very solid rough period
    rough_period = np.mean(inlier_ibis)

    # 4. Global Phase-Locked Grid Fit (The "DJ Software" method)
    # We assign an integer beat index to every detected beat timestamp.
    # This bridges gaps from missed beats perfectly.
    t0 = beats[0]
    shifted_beats = beats - t0
    
    # Calculate expected integer beat indices
    beat_indices = np.round(shifted_beats / rough_period)

    # Filter out beats that are wildly off the rough grid (e.g., syncopated transients)
    expected_times = beat_indices * rough_period
    errors = np.abs(shifted_beats - expected_times)
    
    # Keep beats within 30% of their expected grid position
    valid_mask = errors < (0.3 * rough_period)
    valid_indices = beat_indices[valid_mask]
    valid_beats = beats[valid_mask] # Use absolute timestamps for the regression

    if len(valid_beats) < 4:
        return 0.0

    # 5. Linear Regression: timestamp = period * index + offset
    # Fitting a line across the whole track guarantees a perfectly static, average BPM.
    # The slope (m) is the true global period; the intercept (c) is the grid offset.
    m, c = np.polyfit(valid_indices, valid_beats, 1)

    # Convert slope (seconds per beat) to BPM
    bpm = 60.0 / m

    # 6. Apply standard DJ software octave correction
    # If the tracker tapped at half-time or double-time, snap it back into standard ranges.
    while bpm < min_bpm:
        bpm *= 2.0
    while bpm > max_bpm:
        bpm /= 2.0

    # DJ software usually rounds to 2 decimal places for high-precision static grids
    return round(bpm, 2)