#!/usr/bin/env bash
# Build the workspace-agent image and push it to ECR.
#
# Usage: ./scripts/build_and_push.sh [tag]
#
# If no tag is given, the git short SHA is used (e.g. a3f1c2d).
# A second push of the same image is always done as :latest.
#
# Examples:
#   ./scripts/build_and_push.sh               # tags: <sha> + latest
#   ./scripts/build_and_push.sh v1.2.0        # tags: v1.2.0 + latest

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_DIR="$(dirname "$SCRIPT_DIR")"

REGION="${AWS_REGION:-us-west-2}"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
REGISTRY="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"
REPO="${REGISTRY}/rockybot/workspace-agent"

GIT_SHA=$(git -C "$SERVICE_DIR" rev-parse --short HEAD 2>/dev/null || echo "")
TAG="${1:-${GIT_SHA:-latest}}"

echo "==> Service  : workspace-agent"
echo "==> Registry : ${REPO}"
echo "==> Tag      : ${TAG}"
echo ""

echo "==> Logging in to ECR"
aws ecr get-login-password --region "${REGION}" | \
  docker login --username AWS --password-stdin "${REGISTRY}"

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

echo ""
echo "Done. Image available at:"
echo "  ${REPO}:${TAG}"
[[ "${TAG}" != "latest" ]] && echo "  ${REPO}:latest"