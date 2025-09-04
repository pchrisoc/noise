import numpy as np
import librosa, soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import librosa.display
from scipy.signal import iirnotch, filtfilt, butter, sosfiltfilt
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import os
import base64
import io
from typing import Dict, List

app = FastAPI(title="Audio Denoising API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Next.js dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SR_TARGET = 16000

# ------------------ IO & VIS ------------------

def load_audio(path, sr=SR_TARGET):
    y, sr = librosa.load(path, sr=sr, mono=True)
    y = y / (np.max(np.abs(y)) + 1e-9)
    return y, sr

def save_wav(path, y, sr):
    y = y / (np.max(np.abs(y)) + 1e-9)
    sf.write(path, y, sr)

def plot_waveform(y, sr, title, outpng):
    t = np.arange(len(y))/sr
    plt.figure()
    plt.plot(t, y)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpng, dpi=150)
    plt.close()

def plot_spectrogram(y, sr, title, outpng, n_fft=1024, hop=256):
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop))**2
    S_db = librosa.power_to_db(S, ref=np.max)
    plt.figure()
    librosa.display.specshow(S_db, sr=sr, hop_length=hop, x_axis='time', y_axis='linear')
    plt.title(title)
    plt.colorbar(format="%+2.0f dB")
    plt.tight_layout()
    plt.savefig(outpng, dpi=150)
    plt.close()

def load_audio_from_bytes(audio_bytes: bytes, sr=SR_TARGET):
    """Load audio from bytes data. Try soundfile from-memory first, fallback to temp file + librosa."""
    try:
        bio = io.BytesIO(audio_bytes)
        # soundfile can read various formats directly from bytes
        y, file_sr = sf.read(bio, dtype='float32')
        # convert to mono if needed
        if y.ndim > 1:
            y = np.mean(y, axis=1)
        # resample if sample rate differs
        if file_sr != sr:
            y = librosa.resample(y, orig_sr=file_sr, target_sr=sr)
        y = y / (np.max(np.abs(y)) + 1e-9)
        return y, sr
    except Exception:
        # fallback to temp file + librosa (maintain previous behavior)
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_file.write(audio_bytes)
            temp_path = temp_file.name
        
        try:
            y, _ = librosa.load(temp_path, sr=sr, mono=True)
            y = y / (np.max(np.abs(y)) + 1e-9)
            return y, sr
        finally:
            try:
                os.unlink(temp_path)
            except Exception:
                pass

def audio_to_base64(y, sr):
    """Convert audio array to base64 encoded wav data"""
    y_normalized = y / (np.max(np.abs(y)) + 1e-9)
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
        sf.write(temp_file.name, y_normalized, sr)
        
        with open(temp_file.name, 'rb') as f:
            audio_data = f.read()
        
        os.unlink(temp_file.name)
        return base64.b64encode(audio_data).decode()

def plot_to_base64(y, sr, title, plot_type="waveform", n_fft=1024, hop=256):
    """Generate plot and return as base64 encoded image"""
    plt.figure(figsize=(12, 6))
    
    if plot_type == "waveform":
        t = np.arange(len(y))/sr
        plt.plot(t, y)
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.title(title)
    elif plot_type == "spectrogram":
        S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop))**2
        S_db = librosa.power_to_db(S, ref=np.max)
        librosa.display.specshow(S_db, sr=sr, hop_length=hop, x_axis='time', y_axis='linear')
        plt.title(title)
        plt.colorbar(format="%+2.0f dB")
    
    plt.tight_layout()
    
    # Convert to base64
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
    buffer.seek(0)
    image_b64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    
    return image_b64

# ------------------ UTIL ------------------

def stft(y, n_fft=1024, hop=256, win="hann"):
    return librosa.stft(y, n_fft=n_fft, hop_length=hop, window=win)

def istft(Y, hop=256, win="hann", length=None):
    return librosa.istft(Y, hop_length=hop, window=win, length=length)

def estimate_noise_profile(y, sr, noise_sec=0.5, n_fft=1024, hop=256):
    # Use first 'noise_sec' seconds as noise-only
    n_samples = min(len(y), int(noise_sec*sr))
    yN = y[:n_samples]
    YN = stft(yN, n_fft=n_fft, hop=hop)
    noise_mag = np.mean(np.abs(YN), axis=1, keepdims=True)  # (freq_bins, 1)
    noise_psd = np.mean(np.abs(YN)**2, axis=1, keepdims=True)
    return noise_mag, noise_psd

def snr_db(ref, est):
    # Requires clean reference; if not available, skip.
    num = np.sum(ref**2) + 1e-12
    den = np.sum((ref - est)**2) + 1e-12
    return 10*np.log10(num/den)

def proxy_snr_from_noise_segment(y, yhat, sr, noise_sec=0.5):
    # Compare noise power before/after in the initial noise-only region
    n = min(len(y), int(noise_sec*sr))
    if n < 100: return None
    n_before = np.mean(y[:n]**2) + 1e-12
    n_after  = np.mean(yhat[:n]**2) + 1e-12
    return 10*np.log10(n_before / n_after)

# ------------------ (A) FFT / Notch ------------------

def notch_hum(y, sr, base=60.0, Q=35.0, max_hz=4000):
    yy = y.copy()
    k = 1
    while base*k < min(sr/2, max_hz):
        w0 = (base*k) / (sr/2)  # normalized
        b, a = iirnotch(w0, Q)
        yy = filtfilt(b, a, yy)
        k += 1
    return yy

def bandpass_speech(y, sr, f1=300, f2=3400, order=4):
    sos = butter(order, [f1, f2], btype='bandpass', fs=sr, output='sos')
    return sosfiltfilt(sos, y)

def denoise_fft(y, sr):
    # Notch 60 Hz hum + band-limit to speech
    y1 = notch_hum(y, sr, base=60.0, Q=35.0, max_hz=4000)
    y2 = bandpass_speech(y1, sr, 300, 3400, order=4)
    return y2

# ------------------ (B) Spectral Subtraction ------------------

def spectral_subtraction(y, sr, n_fft=1024, hop=256, alpha=1.0, beta=0.02, noise_sec=0.5):
    Y = stft(y, n_fft=n_fft, hop=hop)
    mag = np.abs(Y)
    phase = np.angle(Y)

    noise_mag, _ = estimate_noise_profile(y, sr, noise_sec, n_fft, hop)
    # Over-subtraction & flooring
    est_mag = np.maximum(mag - alpha*noise_mag, beta*mag)
    S_hat = est_mag * np.exp(1j*phase)
    y_hat = istft(S_hat, hop, length=len(y))
    return y_hat

# ------------------ (C) Wiener Filtering (decision-directed) ------------------

def wiener_filter(y, sr, n_fft=1024, hop=256, noise_sec=0.5, dd=0.98):
    Y = stft(y, n_fft=n_fft, hop=hop)
    mag = np.abs(Y); phase = np.angle(Y)
    _, noise_psd = estimate_noise_profile(y, sr, noise_sec, n_fft, hop)  # (F,1)

    powerY = mag**2
    # a posteriori SNR
    gamma = np.maximum(powerY / (noise_psd + 1e-12), 1.0)

    # decision-directed a priori SNR
    # initialize with (gamma-1)+
    xi_prev = np.maximum(gamma - 1.0, 0.0)
    G_prev = xi_prev / (1.0 + xi_prev)
    S_prev_mag2 = (G_prev**2) * powerY

    xi = dd*(S_prev_mag2/(noise_psd + 1e-12)) + (1.0 - dd)*np.maximum(gamma - 1.0, 0.0)
    G = xi / (1.0 + xi)

    S_hat = G * Y
    y_hat = istft(S_hat, hop, length=len(y))
    return y_hat

# ------------------ MAIN ------------------

@app.post("/denoise")
async def denoise_audio(file: UploadFile = File(...)):
    """
    Process uploaded audio file and return three denoised audio variants with visualizations
    """
    # Accept files that either have a .wav filename (case-insensitive) OR an audio/* content type
    is_wav_name = bool(file.filename) and file.filename.lower().endswith('.wav')
    is_audio_type = bool(file.content_type) and file.content_type.startswith('audio/')
    if not (is_wav_name or is_audio_type):
        raise HTTPException(status_code=400, detail="Only audio files are supported (preferably .wav)")

    try:
        # Load audio from uploaded file
        content = await file.read()
        y, sr = load_audio_from_bytes(content, SR_TARGET)
        
        # Apply three denoising methods
        y_fft = denoise_fft(y, sr)
        y_ss = spectral_subtraction(y, sr, alpha=1.2, beta=0.02)
        y_wien = wiener_filter(y, sr)
        
        # Calculate proxy metrics
        p_fft = proxy_snr_from_noise_segment(y, y_fft, sr)
        p_ss = proxy_snr_from_noise_segment(y, y_ss, sr)
        p_wien = proxy_snr_from_noise_segment(y, y_wien, sr)
        
        # Generate visualizations
        results = []
        
        # FFT/Notch method
        results.append({
            "name": "FFT/Notch Filter",
            "method": "fft_notch",
            "audio_data": audio_to_base64(y_fft, sr),
            "waveform_image": plot_to_base64(y_fft, sr, "FFT/Notch Denoised Waveform", "waveform"),
            "spectrogram_image": plot_to_base64(y_fft, sr, "FFT/Notch Spectrogram", "spectrogram"),
            "proxy_snr_db": p_fft if p_fft else None
        })
        
        # Spectral Subtraction method
        results.append({
            "name": "Spectral Subtraction",
            "method": "spectral_subtraction", 
            "audio_data": audio_to_base64(y_ss, sr),
            "waveform_image": plot_to_base64(y_ss, sr, "Spectral Subtraction Denoised Waveform", "waveform"),
            "spectrogram_image": plot_to_base64(y_ss, sr, "Spectral Subtraction Spectrogram", "spectrogram"),
            "proxy_snr_db": p_ss if p_ss else None
        })
        
        # Wiener Filter method
        results.append({
            "name": "Wiener Filter",
            "method": "wiener_filter",
            "audio_data": audio_to_base64(y_wien, sr),
            "waveform_image": plot_to_base64(y_wien, sr, "Wiener Filter Denoised Waveform", "waveform"),
            "spectrogram_image": plot_to_base64(y_wien, sr, "Wiener Filter Spectrogram", "spectrogram"),
            "proxy_snr_db": p_wien if p_wien else None
        })
        
        # Original for comparison
        original_data = {
            "waveform_image": plot_to_base64(y, sr, "Original Noisy Waveform", "waveform"),
            "spectrogram_image": plot_to_base64(y, sr, "Original Noisy Spectrogram", "spectrogram")
        }
        
        return JSONResponse({
            "success": True,
            "original_filename": file.filename if file.filename else "uploaded_audio",
            "sample_rate": sr,
            "original": original_data,
            "results": results
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.get("/")
async def root():
    return {
        "message": "Audio Denoising API",
        "endpoints": ["/denoise"],
        "methods": ["FFT/Notch Filter", "Spectral Subtraction", "Wiener Filter"]
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
