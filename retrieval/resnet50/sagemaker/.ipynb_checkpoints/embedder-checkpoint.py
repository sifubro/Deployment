import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

class ResNetEmbedder(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        # Remove the final FC (classification) layer.
        # children() gives layers in order; we keep all but the last.
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.eval()

    @torch.no_grad()
    def forward(self, x):
        z = self.features(x)              # shape: [B, 2048, 1, 1]
        z = z.flatten(1)                  # shape: [B, 2048]
        z = nn.functional.normalize(z, dim=1)  # L2-normalise for cosine sim
        return z

# Standard ImageNet preprocessing
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def embed(image_path: str) -> torch.Tensor:
    model = ResNetEmbedder()
    img = Image.open(image_path).convert("RGB")
    x = preprocess(img).unsqueeze(0)  # add batch dim
    return model(x)