# CQT spectrogram parameters — must be identical between preprocessing and inference
SAMPLE_RATE = 44100
N_BINS = 105
HOP_LENGTH = 8820        # ~0.2 s per frame at 44100 Hz (~5 FPS)
BINS_PER_OCTAVE = 24
FMIN = 65                # Lowest CQT frequency (Hz)

# Waveform visualization
WAVEFORM_NUM_POINTS = 1000
