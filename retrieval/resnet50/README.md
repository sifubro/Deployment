# Build (the "." means use the Dockerfile in the current directory)
docker build -t depop-embedder:latest .

# Run it
docker run -p 8080:8080 depop-embedder:latest

# In another terminal, test it
curl http://localhost:8080/ping
# Step 1: Encode the image (Git Bash or Linux Terminal)
IMG_B64=$(base64 -w 0 test.jpg)

# Step 2: Write the JSON payload to a temp file
echo "{\"image_b64\": \"$IMG_B64\"}" > payload.json

# Step 3: Send the file instead of inline data
curl -X POST http://localhost:8080/invocations \
  -H "Content-Type: application/json" \
  -d @payload.json

## Sagemaker install
>C:\Users\SiFuBrO\anaconda3\envs\py3.10_torch_2.2.2_cu118\python.exe -m pip install sagemaker --target C:\Users\SiFuBrO\anaconda3\envs\py3.10_torch_2.2.2_cu118\Lib\site-packages