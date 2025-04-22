# Stage 1: Build frontend
FROM node:current AS frontend-build
WORKDIR /app/frontend
# Copy only manifest and install deps
COPY frontend/package.json ./ 
RUN npm install
# Copy frontend code and build
COPY frontend/ ./ 
RUN npm run build

# Stage 2: Backend (CUDA + Python)
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04 AS backend
WORKDIR /app

# System dependencies
COPY requirements.txt ./ 
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip python3-venv python3-dev git g++ \
    && python3 -m pip install --upgrade pip \
    && python3 -m pip install --no-cache-dir -r requirements.txt 
    # && apt-get remove -y python3-dev g++ \
    # && apt-get autoremove -y \
    # && apt-get clean \
    # && rm -rf /var/lib/apt/lists/* /root/.cache/pip

# Copy backend code
COPY . . 
# Copy frontend build
COPY --from=frontend-build /app/frontend/dist /app/static

# Listen on port 8000 for Azure Container Apps
ENV PYTHONUNBUFFERED=1 TORCHINDUCTOR_DISABLE=1
# Expose port 8000
EXPOSE 8000
# Start Uvicorn on port 8000, referencing main.py
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
