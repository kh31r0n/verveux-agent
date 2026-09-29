#!/usr/bin/env bash
# Build the yorchio-agent image and push it to Google Artifact Registry.
#
# Usage: ./scripts/build_and_push.sh [tag]
#
# If no tag is given, the git short SHA is used (e.g. a3f1c2d).
# The image is always also pushed as :latest.
#
# Examples:
#   ./scripts/build_and_push.sh               # tags: <sha> + latest
#   ./scripts/build_and_push.sh v1.2.0        # tags: v1.2.0 + latest
#
# Overridable via env: GCP_PROJECT_ID, GCP_REGION, AR_REPO

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_DIR="$(dirname "$SCRIPT_DIR")"

PROJECT_ID="${GCP_PROJECT_ID:-yorch-platform-prod}"
REGION="${GCP_REGION:-us-central1}"
AR_REPO="${AR_REPO:-yorch-prod-images}"

# Must match the Cloud Run service name: infra deploys resolve the image as
# <registry>/<service>:<tag>. See yorch-gcp-platform/scripts/deploy.sh.
SERVICE="yorchio-agent"

# Cloud Run migration jobs run the *builder* stage, tagged :migrate. The prod
# stage is `npm ci --omit=dev` and the Prisma CLI is a devDependency, so a
# prod-stage image would try to download Prisma at runtime.
BUILD_MIGRATE=false

REGISTRY="${REGION}-docker.pkg.dev"
REPO="${REGISTRY}/${PROJECT_ID}/${AR_REPO}/${SERVICE}"

GIT_SHA=$(git -C "$SERVICE_DIR" rev-parse --short HEAD 2>/dev/null || echo "")
TAG="${1:-${GIT_SHA:-latest}}"

echo "==> Service  : ${SERVICE}"
echo "==> Registry : ${REPO}"
echo "==> Tag      : ${TAG}"
echo ""

echo "==> Configuring docker auth for ${REGISTRY}"
gcloud auth configure-docker "${REGISTRY}" --quiet

echo "==> Building image ${REPO}:${TAG}"
docker build \
  --platform linux/amd64 \
  --file "${SERVICE_DIR}/Dockerfile" \
  --tag "${REPO}:${TAG}" \
  "${SERVICE_DIR}"

echo "==> Pushing ${REPO}:${TAG}"
docker push "${REPO}:${TAG}"

if [[ "${TAG}" != "latest" ]]; then
  echo "==> Tagging as ${REPO}:latest"
  docker tag "${REPO}:${TAG}" "${REPO}:latest"
  docker push "${REPO}:latest"
fi

if [[ "${BUILD_MIGRATE}" == "true" ]]; then
  echo ""
  echo "==> Building migration image ${REPO}:migrate (builder stage)"
  docker build \
    --platform linux/amd64 \
    --target builder \
    --file "${SERVICE_DIR}/Dockerfile" \
    --tag "${REPO}:migrate" \
    "${SERVICE_DIR}"

  echo "==> Pushing ${REPO}:migrate"
  docker push "${REPO}:migrate"
fi

echo ""
echo "Done. Images pushed:"
echo "  ${REPO}:${TAG}"
if [[ "${TAG}" != "latest" ]]; then
  echo "  ${REPO}:latest"
fi
if [[ "${BUILD_MIGRATE}" == "true" ]]; then
  echo "  ${REPO}:migrate"
fi

echo ""
echo "Deploy with:"
echo "  cd ~/yorch-gcp-platform && ./scripts/deploy.sh ${SERVICE}"
