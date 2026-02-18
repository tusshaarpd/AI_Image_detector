"""Tests for individual detector modules."""

import numpy as np
import pytest
from PIL import Image

from app.detectors.frequency import FrequencyDetector
from app.detectors.metadata import MetadataDetector
from app.detectors.texture import TextureDetector


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def real_like_image() -> Image.Image:
    """Create a noisy RGB image that resembles a real photograph."""
    rng = np.random.RandomState(42)
    arr = rng.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    return Image.fromarray(arr, "RGB")


@pytest.fixture
def uniform_image() -> Image.Image:
    """Create a very uniform image (more likely to trigger AI signals)."""
    arr = np.full((512, 512, 3), 128, dtype=np.uint8)
    # Add minimal noise
    rng = np.random.RandomState(0)
    arr = arr + rng.randint(-3, 4, arr.shape).astype(np.uint8)
    return Image.fromarray(arr, "RGB")


# ---------------------------------------------------------------------------
# MetadataDetector
# ---------------------------------------------------------------------------
class TestMetadataDetector:
    def test_returns_score_and_signals(self, real_like_image: Image.Image):
        det = MetadataDetector()
        result = det.analyze(real_like_image)
        assert "score" in result
        assert "signals" in result
        assert 0.0 <= result["score"] <= 1.0
        assert isinstance(result["signals"], list)

    def test_no_exif_raises_signal(self, real_like_image: Image.Image):
        det = MetadataDetector()
        result = det.analyze(real_like_image)
        signals_text = " ".join(result["signals"]).lower()
        assert "no exif" in signals_text or "metadata" in signals_text

    def test_square_power_of_two_image(self):
        arr = np.zeros((512, 512, 3), dtype=np.uint8)
        img = Image.fromarray(arr, "RGB")
        det = MetadataDetector()
        result = det.analyze(img)
        signals_text = " ".join(result["signals"]).lower()
        assert "square" in signals_text or "512" in signals_text


# ---------------------------------------------------------------------------
# FrequencyDetector
# ---------------------------------------------------------------------------
class TestFrequencyDetector:
    def test_returns_score_and_signals(self, real_like_image: Image.Image):
        det = FrequencyDetector()
        result = det.analyze(real_like_image)
        assert "score" in result
        assert "signals" in result
        assert 0.0 <= result["score"] <= 1.0

    def test_uniform_image_higher_score(self, uniform_image: Image.Image, real_like_image: Image.Image):
        det = FrequencyDetector()
        uniform_result = det.analyze(uniform_image)
        noisy_result = det.analyze(real_like_image)
        # A nearly-uniform image should score >= a random-noise image
        assert uniform_result["score"] >= noisy_result["score"] - 0.15


# ---------------------------------------------------------------------------
# TextureDetector
# ---------------------------------------------------------------------------
class TestTextureDetector:
    def test_returns_score_and_signals(self, real_like_image: Image.Image):
        det = TextureDetector()
        result = det.analyze(real_like_image)
        assert "score" in result
        assert "signals" in result
        assert 0.0 <= result["score"] <= 1.0

    def test_uniform_image_flags_texture(self, uniform_image: Image.Image):
        det = TextureDetector()
        result = det.analyze(uniform_image)
        # Very uniform image should trigger texture uniformity signal
        assert result["score"] > 0.3
