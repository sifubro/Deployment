# export_to_torchscript.py
import torch
from transformers import CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval()

# Image encoder: takes pixel_values [B, 3, 224, 224], returns features [B, 512]
class ImageEncoder(torch.nn.Module):
    def __init__(self, clip):
        super().__init__()
        self.vision_model = clip.vision_model
        self.visual_projection = clip.visual_projection
    def forward(self, pixel_values):
        out = self.vision_model(pixel_values=pixel_values).pooler_output
        feats = self.visual_projection(out)
        return torch.nn.functional.normalize(feats, dim=-1)

# Text encoder: takes input_ids and attention_mask, returns features [B, 512]
class TextEncoder(torch.nn.Module):
    def __init__(self, clip):
        super().__init__()
        self.text_model = clip.text_model
        self.text_projection = clip.text_projection
    def forward(self, input_ids, attention_mask):
        out = self.text_model(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        feats = self.text_projection(out)
        return torch.nn.functional.normalize(feats, dim=-1)

img_enc = ImageEncoder(model).eval()
txt_enc = TextEncoder(model).eval()

# Trace with example inputs
img_example = torch.randn(1, 3, 224, 224)
img_traced = torch.jit.trace(img_enc, img_example)
img_traced.save("model_repository/clip_image/1/model.pt")

ids_example = torch.zeros(1, 77, dtype=torch.long)
mask_example = torch.ones(1, 77, dtype=torch.long)
txt_traced = torch.jit.trace(txt_enc, (ids_example, mask_example))
txt_traced.save("model_repository/clip_text/1/model.pt")

print("Done.")