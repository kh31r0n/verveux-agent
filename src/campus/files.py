"""Where the Campus agents keep files: the Game Master's voice
(``audio/{session}/{moment}.wav``) and the host's material
(``designs/{design}/{nn}-{name}``).

Cloud Storage when ``CAMPUS_BUCKET`` is set; a local folder otherwise, which
only works on one machine (development and tests). The bucket is private: the
gateway fetches the audio through ``GET /campus/sessions/:id/audio/:file`` and
the screens get it from the gateway. Objects expire by the bucket's lifecycle
rule, not here.
"""

import asyncio
from pathlib import Path
from typing import Protocol

from .config import campus_settings


class Files(Protocol):
    async def put(self, path: str, data: bytes, content_type: str) -> None: ...

    async def get(self, path: str) -> bytes | None:
        """The object, or ``None`` if it does not exist."""

    async def names(self, prefix: str) -> list[str]:
        """Paths under ``prefix``, sorted."""


class LocalFiles:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def _path(self, path: str) -> Path:
        full = (self.root / path).resolve()
        if not full.is_relative_to(self.root.resolve()):
            raise ValueError("path outside the data folder")
        return full

    async def put(self, path: str, data: bytes, content_type: str) -> None:
        full = self._path(path)
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_bytes(data)

    async def get(self, path: str) -> bytes | None:
        full = self._path(path)
        return full.read_bytes() if full.is_file() else None

    async def names(self, prefix: str) -> list[str]:
        folder = self._path(prefix)
        if not folder.is_dir():
            return []
        return sorted(str(p.relative_to(self.root.resolve())) for p in folder.iterdir() if p.is_file())


class GcsFiles:
    """Cloud Storage by ADC. The client is synchronous; calls go to a thread."""

    def __init__(self, bucket: str) -> None:
        from google.cloud import storage

        self._bucket = storage.Client().bucket(bucket)

    async def put(self, path: str, data: bytes, content_type: str) -> None:
        blob = self._bucket.blob(path)
        await asyncio.to_thread(blob.upload_from_string, data, content_type=content_type)

    async def get(self, path: str) -> bytes | None:
        from google.api_core.exceptions import NotFound

        try:
            return await asyncio.to_thread(self._bucket.blob(path).download_as_bytes)
        except NotFound:
            return None

    async def names(self, prefix: str) -> list[str]:
        prefix = prefix.rstrip("/") + "/"
        blobs = await asyncio.to_thread(lambda: list(self._bucket.list_blobs(prefix=prefix)))
        return sorted(blob.name for blob in blobs)


def default_files() -> Files:
    if campus_settings.bucket:
        return GcsFiles(campus_settings.bucket)
    return LocalFiles(campus_settings.data_dir)
