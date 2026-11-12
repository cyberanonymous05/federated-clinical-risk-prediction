# Reproducibility image (CPU only). Build: docker build -t hiperc-fl .   Smoke test: docker run --rm hiperc-fl ./reproduce.sh smoke
FROM python:3.13-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates && rm -rf /var/lib/apt/lists/*
COPY . .
ENV OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
CMD ["./reproduce.sh", "smoke"]
