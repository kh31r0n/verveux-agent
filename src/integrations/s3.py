"""
Minimal S3 read/write for the spend-file repair flow.

boto3 is imported lazily (and the client is cached) so importing this module — and
unit-testing the pure repair logic — does not require boto3 to be installed.
Credentials come from boto3's default chain: the static AWS_ACCESS_KEY_ID /
AWS_SECRET_ACCESS_KEY pair the backend's UploadService uses, or the pod's IAM role.
"""

from __future__ import annotations

from typing import Optional

import structlog

from ..config import settings

logger = structlog.get_logger(__name__)

_client = None


def _s3_client():
    global _client
    if _client is None:
        import boto3  # lazy — only needed at runtime in the agent image

        region = settings.aws_s3_region or None
        _client = boto3.client("s3", region_name=region)
        logger.info("s3_client_initialised", region=region or "default")
    return _client


def read_object(bucket: str, key: str) -> bytes:
    resp = _s3_client().get_object(Bucket=bucket, Key=key)
    return resp["Body"].read()


def write_object(
    bucket: str,
    key: str,
    body: bytes,
    content_type: Optional[str] = None,
) -> str:
    kwargs: dict = {"Bucket": bucket, "Key": key, "Body": body}
    if content_type:
        kwargs["ContentType"] = content_type
    _s3_client().put_object(**kwargs)
    logger.info("s3_object_written", bucket=bucket, key=key, bytes=len(body))
    return key
