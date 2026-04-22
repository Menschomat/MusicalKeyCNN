import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from eval import load_model
from predict_keys import camelot_output, preprocess_mp3

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


class PredictResponse(BaseModel):
    filename: str
    class_id: int
    camelot: str
    key: str


@app.get("/health")
def health():
    return {"status": "ok", "model": "keynet", "device": str(_device)}


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".mp3"):
        raise HTTPException(status_code=400, detail="Only .mp3 files are supported.")

    contents = await file.read()

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp.write(contents)
        tmp_path = Path(tmp.name)

    try:
        spec_tensor = preprocess_mp3(tmp_path)
        spec_tensor = spec_tensor.unsqueeze(0).to(_device)

        with torch.no_grad():
            outputs = _model(spec_tensor)
            pred = int(torch.argmax(outputs, dim=1).cpu().item())

        camelot_str, key_text = camelot_output(pred)
    finally:
        tmp_path.unlink(missing_ok=True)

    return PredictResponse(
        filename=file.filename,
        class_id=pred,
        camelot=camelot_str,
        key=key_text,
    )
