# Rebuild with the serve script
docker build -t depop-embedder:latest .

# Verify locally that `serve` works
docker run --rm -p 8080:8080 depop-embedder:latest serve
# In another terminal: curl http://localhost:8080/ping

# Push to ECR
aws ecr get-login-password --region eu-west-1 | docker login --username AWS --password-stdin 586917955410.dkr.ecr.eu-west-1.amazonaws.com

docker tag depop-embedder:latest 586917955410.dkr.ecr.eu-west-1.amazonaws.com/depop-embedder:latest

docker push 586917955410.dkr.ecr.eu-west-1.amazonaws.com/depop-embedder:latest

# Then re-run
python deploy.py