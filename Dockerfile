FROM python:3.11-slim
# TODO remplacer par une image cuda
# TODO example : nvidia/cuda:13.4.2-cudnn-devel-ubuntu26.04

WORKDIR /app

# 2. System Dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# 4 INSTALLATION (With Debugging)
RUN echo "===== CHECKING REQUIREMENTS =====" && \
    cat requirements.txt && \
    echo "=================================" && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONPATH=/app
CMD ["python", "app/train.py"]
