FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (git required by GitPython)
RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Seed portable settings so config + cache stay under /app
RUN echo '{"use_portable_settings": true, "folder_paths": {"loras": ["/models/loras"], "checkpoints": ["/models/checkpoints"], "embeddings": ["/models/embeddings"]}}' > /app/settings.json

ENV LORA_MANAGER_STANDALONE=1

EXPOSE 8188

ENTRYPOINT ["python", "standalone.py"]
CMD ["--host", "0.0.0.0", "--port", "8188"]
