# Stage 1: Frontend build (cache efficient)
FROM node:current AS frontend-build
WORKDIR /app/frontend

# Copy only lockfile/package manifests first for better npm ci cache
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm ci

# Copy frontend code and build last (invalidates cache ONLY when code changes)
COPY frontend/ ./
RUN npm run build

# Stage 2: Backend CUDA+Python
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04 AS backend
WORKDIR /app

# Install system dependencies & Python deps in cache-efficient order
COPY requirements.txt ./
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip python3-venv python3-dev git g++ \
 && python3 -m pip install --upgrade pip \
 && python3 -m pip install --no-cache-dir -r requirements.txt

# Copy only necessary backend/app files after deps are installed
COPY . .

# Copy built frontend static files into place (from frontend build stage)
COPY --from=frontend-build /app/frontend/dist /app/static

# Set environment
ENV PYTHONUNBUFFERED=1 TORCHINDUCTOR_DISABLE=1

EXPOSE 8000

# Start server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
