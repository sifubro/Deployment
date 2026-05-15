import boto3
import base64
import json

runtime = boto3.client("sagemaker-runtime", region_name="eu-west-1")

# Read and encode your image
with open("test.jpg", "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode("utf-8")

payload = json.dumps({"image_b64": img_b64})

response = runtime.invoke_endpoint(
    EndpointName="depop-embedder-endpoint",
    ContentType="application/json",
    Body=payload,
)

result = json.loads(response["Body"].read().decode("utf-8"))
print(result)

#If you want to test from terminal (use GitBash on Windows!)
'''
IMG_B64=$(base64 -w 0 test_image.jpg)  # macOS: base64 -i test_image.jpg | tr -d '\n'
echo "{\"image_b64\": \"$IMG_B64\"}" > payload.json

aws sagemaker-runtime invoke-endpoint \
  --endpoint-name depop-embedder-endpoint \
  --region eu-west-1 \
  --content-type application/json \
  --body fileb://payload.json \
  output.json

cat output.json
'''

# How to stop/start SageMaker endpoint
# https://claude.ai/chat/4dea049c-01c9-4da4-b2e1-25c0a57949b0