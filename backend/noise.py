import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

# Load audio
sr, audio = wavfile.read("denoise_fft.wav")

# Take FFT
N = len(audio)
fft_vals = np.fft.fft(audio)
freqs = np.fft.fftfreq(N, 1/sr)

# Plot spectrum (magnitude)
plt.plot(freqs[:N//2], np.abs(fft_vals[:N//2]))
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.title("Frequency Spectrum of Audio")
plt.show()
