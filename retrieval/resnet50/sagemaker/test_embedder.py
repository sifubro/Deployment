import torch
from embedder import ResNetEmbedder, preprocess
from PIL import Image

def embed(model, path):
    img = Image.open(path).convert("RGB")
    x = preprocess(img).unsqueeze(0)
    return model(x).squeeze(0)   # shape: [2048]

# Load model ONCE — instantiating it downloads the weights, which is slow.
model = ResNetEmbedder()

paths = ["./images/cat1.jpg", "./images/cat2.jpg", "./images/car1.jpg", "./images/car2.jpg"]
embs = {p: embed(model, p) for p in paths}

def sim(a, b):
    # Vectors are already L2-normalised inside the model, so dot product == cosine similarity.
    return (embs[a] * embs[b]).sum().item()

print(f"cat1 vs cat2 (similar):    {sim('./images/cat1.jpg', './images/cat2.jpg'):.4f}")
print(f"car1 vs car2 (similar):    {sim('./images/car1.jpg', './images/car2.jpg'):.4f}")
print(f"cat1 vs car1 (different):  {sim('./images/cat1.jpg', './images/car1.jpg'):.4f}")
print(f"cat2 vs car2 (different):  {sim('./images/cat2.jpg', './images/car2.jpg'):.4f}")