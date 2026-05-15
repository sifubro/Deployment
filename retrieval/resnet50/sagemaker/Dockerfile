FROM python:3.11-slim

WORKDIR /opt/program

RUN apt-get update && apt-get install -y --no-install-recommends \
    libjpeg-dev zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install PyTorch with CUDA 11.8 from the official PyTorch index
RUN pip install --no-cache-dir numpy==1.26.3
RUN pip install --no-cache-dir torch==2.2.2+cu118 torchvision==0.17.2+cu118 \
    --index-url https://download.pytorch.org/whl/cu118
RUN pip install --no-cache-dir -r requirements.txt

COPY embedder.py app.py ./

RUN python -c "from embedder import ResNetEmbedder; ResNetEmbedder()"

EXPOSE 8080

# SageMaker invokes the container as `docker run <image> serve`.
# The serve script must be on PATH so that call resolves correctly.
RUN printf '#!/bin/bash\nexec uvicorn app:app --host 0.0.0.0 --port 8080 --workers 1\n' > /opt/program/serve \
    && chmod +x /opt/program/serve

ENV PATH="/opt/program:${PATH}"