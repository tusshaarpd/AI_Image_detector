"""Tests for image preprocessing and validation."""

import base64
import io

import numpy as np
import pytest
from PIL import Image

from app.core.preprocessing import ImagePreprocessor


@pytest.fixture
def preprocessor() -> ImagePreprocessor:
    return ImagePreprocessor()


def _make_jpeg_bytes(width: int = 200, height: int = 200) -> bytes:
    arr = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    img = Image.fromarray(arr, "RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _make_png_bytes(width: int = 200, height: int = 200) -> bytes:
    arr = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    img = Image.fromarray(arr, "RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class TestImagePreprocessor:
    def test_load_jpeg(self, preprocessor: ImagePreprocessor):
        raw = _make_jpeg_bytes()
        img = preprocessor.load_from_bytes(raw)
        assert isinstance(img, Image.Image)
        assert img.size == (200, 200)

    def test_load_png(self, preprocessor: ImagePreprocessor):
        raw = _make_png_bytes()
        img = preprocessor.load_from_bytes(raw)
        assert isinstance(img, Image.Image)

    def test_decode_base64_plain(self, preprocessor: ImagePreprocessor):
        raw = _make_jpeg_bytes()
        b64 = base64.b64encode(raw).decode()
        img = preprocessor.decode_base64(b64)
        assert isinstance(img, Image.Image)

    def test_decode_base64_with_data_uri(self, preprocessor: ImagePreprocessor):
        raw = _make_jpeg_bytes()
        b64 = "data:image/jpeg;base64," + base64.b64encode(raw).decode()
        img = preprocessor.decode_base64(b64)
        assert isinstance(img, Image.Image)

    def test_rejects_invalid_bytes(self, preprocessor: ImagePreprocessor):
        with pytest.raises(ValueError, match="Invalid image"):
            preprocessor.load_from_bytes(b"not an image at all")

    def test_rejects_oversized(self, preprocessor: ImagePreprocessor):
        # Create bytes slightly over the limit (default 10 MB)
        oversized = b"\x00" * (11 * 1024 * 1024)
        with pytest.raises(ValueError, match="exceeds"):
            preprocessor.load_from_bytes(oversized)
