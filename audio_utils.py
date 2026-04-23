import librosa
import numpy as np
import torch

from constants import BASS_MAX_HZ, BINS_PER_OCTAVE, FMIN, HOP_LENGTH, MID_MAX_HZ, N_BINS, SAMPLE_RATE, WAVEFORM_NUM_POINTS


def load_audio(path, sample_rate=SAMPLE_RATE):
    """Load an audio file as a mono waveform at the given sample rate."""
    waveform, sr = librosa.load(path, sr=sample_rate, mono=True)
    return waveform, sr


def _chunk(waveform, num_points):
    """Split waveform into num_points equal chunks. Returns (chunks, chunk_size)."""
    chunk_size = len(waveform) // num_points
    trimmed = waveform[: chunk_size * num_points]
    return trimmed.reshape(num_points, chunk_size), chunk_size


def compute_waveform_basic(waveform, sr, num_points=WAVEFORM_NUM_POINTS):
    """
    Downsample a waveform to num_points RMS values with corresponding timestamps.
    Returns {"times": [...], "amplitudes": [...]}, both normalized to [0, 1].
    """
    chunks, chunk_size = _chunk(waveform, num_points)
    rms = np.sqrt(np.mean(chunks ** 2, axis=1))
    max_val = rms.max() or 1.0

    times = ((np.arange(num_points) + 0.5) * chunk_size / sr).tolist()
    return {"times": times, "amplitudes": (rms / max_val).tolist()}


def compute_waveform_rgb(waveform, sr, num_points=WAVEFORM_NUM_POINTS):
    """
    Compute a frequency-colored waveform: R=bass, G=mids, B=highs.
    Returns {"times": [...], "r": [...], "g": [...], "b": [...]}, all normalized to [0, 1].
    """
    chunks, chunk_size = _chunk(waveform, num_points)

    # Frequency of each FFT bin for this chunk size
    freqs = np.fft.rfftfreq(chunk_size, d=1.0 / sr)
    bass_mask = freqs < BASS_MAX_HZ
    mid_mask = (freqs >= BASS_MAX_HZ) & (freqs < MID_MAX_HZ)
    high_mask = freqs >= MID_MAX_HZ

    # Vectorised FFT across all chunks at once
    mags = np.abs(np.fft.rfft(chunks, axis=1))  # (num_points, chunk_size//2 + 1)
    r = mags[:, bass_mask].sum(axis=1)
    g = mags[:, mid_mask].sum(axis=1)
    b = mags[:, high_mask].sum(axis=1)

    def norm(arr):
        m = arr.max() or 1.0
        return (arr / m).tolist()

    times = ((np.arange(num_points) + 0.5) * chunk_size / sr).tolist()
    return {"times": times, "r": norm(r), "g": norm(g), "b": norm(b)}


def preprocess_from_waveform(waveform, sample_rate=SAMPLE_RATE, n_bins=N_BINS, hop_length=HOP_LENGTH):
    """
    Extracts a log-magnitude CQT spectrogram from a pre-loaded mono waveform.

    Args:
        waveform (np.ndarray): Mono audio signal.
        sample_rate (int): Sample rate of the waveform.
        n_bins (int): Number of CQT bins.
        hop_length (int): Hop length for CQT.

    Returns:
        torch.Tensor: Shape (1, freq_bins, time_frames), ready for model input.
    """
    cqt = librosa.cqt(waveform, sr=sample_rate, hop_length=hop_length, n_bins=n_bins, bins_per_octave=BINS_PER_OCTAVE, fmin=FMIN)
    spec = np.abs(cqt)
    spec = np.log1p(spec)

    # Remove last frequency bin
    chunk = spec[:, 0:-2]
    spec_tensor = torch.tensor(chunk, dtype=torch.float32)
    if spec_tensor.ndim == 2:
        spec_tensor = spec_tensor.unsqueeze(0)  # Shape: (1, freq, time)
    return spec_tensor
