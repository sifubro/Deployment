import io
import numpy as np
import triton_python_backend_utils as pb_utils
from PIL import Image
from transformers import CLIPProcessor

class TritonPythonModel:
    def initialize(self, args):
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def execute(self, requests):
        responses = []
        for request in requests:
            img_bytes_tensor = pb_utils.get_input_tensor_by_name(request, "image_bytes")
            img_bytes = img_bytes_tensor.as_numpy()[0]  # bytes
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            processed = self.processor(images=img, return_tensors="np")
            out_tensor = pb_utils.Tensor("pixel_values", processed["pixel_values"].astype(np.float32)[0])
            responses.append(pb_utils.InferenceResponse(output_tensors=[out_tensor]))
        return responses