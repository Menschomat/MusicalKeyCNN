# CQT spectrogram parameters — must be identical between preprocessing and inference
SAMPLE_RATE = 44100
N_BINS = 105
HOP_LENGTH = 8820        # ~0.2 s per frame at 44100 Hz (~5 FPS)
BINS_PER_OCTAVE = 24
FMIN = 65                # Lowest CQT frequency (Hz)

# Waveform visualization
WAVEFORM_NUM_POINTS = 1000
BASS_MAX_HZ = 250
MID_MAX_HZ = 2500

# Rainbow waveform — STFT settings and hue frequency range
STFT_N_FFT = 2048
STFT_HOP_LENGTH = 512
HUE_FREQ_MIN = 20     # Hz → hue 0.0 (red)
HUE_FREQ_MAX = 20000  # Hz → hue 0.667 (blue)
