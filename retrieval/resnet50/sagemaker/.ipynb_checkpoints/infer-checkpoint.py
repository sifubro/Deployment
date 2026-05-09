import boto3, base64, json

runtime = boto3.client("sagemaker-runtime", region_name="eu-west-1")

with open("test.jpg", "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode()

response = runtime.invoke_endpoint(
    EndpointName="depop-embedder-endpoint",
    ContentType="application/json",
    Body=json.dumps({"image_b64": img_b64}),
)
result = json.loads(response["Body"].read())
print(f"Embedding dim: {result['dim']}, first 5: {result['embedding'][:5]}")