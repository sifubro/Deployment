import numpy as np, tritonclient.http as http
from transformers import CLIPProcessor
from PIL import Image

p = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
img = Image.open('jacket.jpg').convert('RGB')
pixel_values = p(images=img, return_tensors='np')['pixel_values']  # [1,3,224,224]

client = http.InferenceServerClient('localhost:8000')
inp = http.InferInput('pixel_values', pixel_values.shape, 'FP32')
inp.set_data_from_numpy(pixel_values.astype(np.float32))
out = http.InferRequestedOutput('embedding')
result = client.infer('clip_image', [inp], outputs=[out])
print('embedding shape:', result.as_numpy('embedding').shape)