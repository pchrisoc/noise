# Audio Denoising Application

A full-stack audio denoising application with FastAPI backend and Next.js frontend.

## Quick Start

1. Make the startup script executable:
```bash
chmod +x start.sh
```

2. Run both backend and frontend:
```bash
./start.sh
```

3. Open http://localhost:3000 in your browser

## Manual Setup

### Backend
```bash
cd backend
pip install -r requirements.txt
python denoise.py
```

### Frontend
```bash
cd frontend
npm install
npm run dev
```

## Features

- Upload .wav files for denoising
- Three denoising algorithms:
  - FFT/Notch Filter
  - Spectral Subtraction
  - Wiener Filter
- Visual comparisons with waveforms and spectrograms
- Audio playback and download
- SNR metrics for quality assessment

## API Endpoints

- `POST /denoise` - Upload and process audio file
- `GET /` - API information
- `GET /health` - Health check
