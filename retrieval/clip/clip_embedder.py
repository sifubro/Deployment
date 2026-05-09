# clip_embedder.py
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch.nn.functional as F

class CLIPEmbedder:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model_name = "openai/clip-vit-base-patch32"
        self.model = CLIPModel.from_pretrained(model_name).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_name)

    @torch.no_grad()
    def embed_image(self, images):
        # images: list[PIL.Image] or single PIL.Image
        if not isinstance(images, list):
            images = [images]
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        feats = self.model.get_image_features(**inputs)
        return F.normalize(feats, dim=-1)  # [N, 512], L2-normalised

    @torch.no_grad()
    def embed_text(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        inputs = self.processor(text=texts, return_tensors="pt",
                                padding=True, truncation=True).to(self.device)
        feats = self.model.get_text_features(**inputs)
        return F.normalize(feats, dim=-1)  # [N, 512]


if __name__ == "__main__":
    emb = CLIPEmbedder()
    img = Image.open("jacket.jpg").convert("RGB")
    img_vec = emb.embed_image(img)
    txt_vec = emb.embed_text(["a red leather jacket", "a blue sneaker", "a wooden chair"])
    sims = (img_vec @ txt_vec.T).squeeze(0)
    for text, score in zip(["red leather jacket", "blue sneaker", "wooden chair"], sims):
        print(f"{text:25s} {score.item():.3f}")