# app.py
from fastapi import FastAPI
from pydantic import BaseModel
import base64, io
from PIL import Image
import torch
from embedder import ResNetEmbedder, preprocess

app = FastAPI()
model = ResNetEmbedder()  # loaded once at startup, not per-request

class Request(BaseModel):
    image_b64: str  # base64-encoded image bytes

# SageMaker requires these two specific paths:
@app.get("/ping")
def ping():
    return {"status": "ok"}

@app.post("/invocations")
def invocations(req: Request):
    img_bytes = base64.b64decode(req.image_b64)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    x = preprocess(img).unsqueeze(0)
    with torch.no_grad():
        emb = model(x).squeeze(0).tolist()
    return {"embedding": emb, "dim": len(emb)}