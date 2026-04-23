import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from typing import List

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from audio_utils import compute_waveform_basic, compute_waveform_hmb, compute_waveform_rainbow, load_audio, preprocess_from_waveform
from eval import load_model
from predict_bpm import detect_bpm
from predict_keys import SUPPORTED_EXTENSIONS, camelot_output

MODEL_PATH = Path("checkpoints/keynet.pt")

_model = None
_device = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _device
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _model = load_model(str(MODEL_PATH), _device)
    _model.eval()
    yield


app = FastAPI(title="MusicalKeyCNN API", version="0.1.0", lifespan=lifespan)


class WaveformBasic(BaseModel):
    times: List[float]
    amplitudes: List[float]


class WaveformHMB(BaseModel):
    times: List[float]
    bass: List[float]
    mid: List[float]
    high: List[float]


class WaveformRGB(BaseModel):
    times: List[float]
    r: List[float]
    g: List[float]
    b: List[float]


class PredictResponse(BaseModel):
    filename: str
    class_id: int
    camelot: str
    key: str
    bpm: float
    waveform_basic: WaveformBasic
    waveform_hmb: WaveformHMB
    waveform_rainbow: WaveformRGB


@app.get("/health")
def health():
    return {"status": "ok", "model": "keynet", "device": str(_device)}


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    suffix = Path(file.filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported format '{suffix}'. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")

    contents = await file.read()

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(contents)
        tmp_path = Path(tmp.name)

    try:
        waveform, sr = load_audio(tmp_path)

        spec_tensor = preprocess_from_waveform(waveform, sr)
        spec_tensor = spec_tensor.unsqueeze(0).to(_device)

        with torch.no_grad():
            outputs = _model(spec_tensor)
            pred = int(torch.argmax(outputs, dim=1).cpu().item())

        camelot_str, key_text = camelot_output(pred)
        bpm = detect_bpm(waveform, sr)
        waveform_basic = compute_waveform_basic(waveform, sr)
        waveform_hmb = compute_waveform_hmb(waveform, sr)
        waveform_rainbow = compute_waveform_rainbow(waveform, sr)
    finally:
        tmp_path.unlink(missing_ok=True)

    return PredictResponse(
        filename=file.filename,
        class_id=pred,
        camelot=camelot_str,
        key=key_text,
        bpm=bpm,
        waveform_basic=WaveformBasic(**waveform_basic),
        waveform_hmb=WaveformHMB(**waveform_hmb),
        waveform_rainbow=WaveformRGB(**waveform_rainbow),
    )
