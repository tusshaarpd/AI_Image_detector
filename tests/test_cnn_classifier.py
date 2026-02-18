"""Tests for the CNN classifier module."""

import numpy as np
import pytest
from PIL import Image

from app.models.cnn_classifier import CNNClassifier


@pytest.fixture(scope="module")
def classifier() -> CNNClassifier:
    """Load classifier once for all tests in this module."""
    return CNNClassifier()


@pytest.fixture
def sample_image() -> Image.Image:
    arr = np.random.randint(0, 256, (300, 300, 3), dtype=np.uint8)
    return Image.fromarray(arr, "RGB")


class TestCNNClassifier:
    def test_predict_returns_score_and_signals(self, classifier: CNNClassifier, sample_image: Image.Image):
        result = classifier.predict(sample_image)
        assert "score" in result
        assert "signals" in result
        assert 0.0 <= result["score"] <= 1.0
        assert isinstance(result["signals"], list)
        assert len(result["signals"]) >= 1

    def test_predict_grayscale_image(self, classifier: CNNClassifier):
        arr = np.random.randint(0, 256, (224, 224), dtype=np.uint8)
        img = Image.fromarray(arr, "L")
        result = classifier.predict(img)
        assert 0.0 <= result["score"] <= 1.0

    def test_predict_different_sizes(self, classifier: CNNClassifier):
        for size in [(100, 100), (1024, 768), (50, 200)]:
            arr = np.random.randint(0, 256, (size[1], size[0], 3), dtype=np.uint8)
            img = Image.fromarray(arr, "RGB")
            result = classifier.predict(img)
            assert 0.0 <= result["score"] <= 1.0
