#!/bin/bash
set -e

# === CONFIGURE THESE ===
GCR_IMAGE="gcr.io/usat-460409/usat-generator:latest"

# === Authenticate Docker with GCR ===
echo "Authenticating Docker with GCR..."
gcloud auth configure-docker

# === Build Docker image ===
echo "Building Docker image..."
docker build --platform linux/amd64 -t "${GCR_IMAGE}" .

# === Push Docker image to GCR ===
echo "Pushing image to GCR..."
docker push "${GCR_IMAGE}"

