"""Main detection service that orchestrates all detectors."""

import time

from PIL import Image

from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.metrics import (
    ACTIVE_REQUESTS,
    PREDICTION_CONFIDENCE,
    PREDICTION_LATENCY,
    PREDICTIONS_TOTAL,
)
from app.core.preprocessing import ImagePreprocessor
from app.detectors.ensemble import EnsembleEngine
from app.detectors.frequency import FrequencyDetector
from app.detectors.metadata import MetadataDetector
from app.detectors.texture import TextureDetector
from app.models.cnn_classifier import CNNClassifier
from app.schemas.detection import DetectionResult

logger = get_logger(__name__)


class DetectorService:
    """Orchestrates the full detection pipeline.

    Coordinates preprocessing, individual detectors, CNN classifier,
    and ensemble decision engine to produce a final detection result.
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self.preprocessor = ImagePreprocessor()
        self.metadata_detector = MetadataDetector()
        self.frequency_detector = FrequencyDetector()
        self.texture_detector = TextureDetector()
        self.cnn_classifier = CNNClassifier()
        self.ensemble = EnsembleEngine()
        logger.info("detector_service_initialized")

    def detect_from_base64(self, image_b64: str) -> DetectionResult:
        """Run full detection pipeline on a base64-encoded image."""
        image = self.preprocessor.decode_base64(image_b64)
        return self._run_pipeline(image)

    def detect_from_bytes(self, image_bytes: bytes) -> DetectionResult:
        """Run full detection pipeline on raw image bytes."""
        image = self.preprocessor.load_from_bytes(image_bytes)
        return self._run_pipeline(image)

    def _run_pipeline(self, image: Image.Image) -> DetectionResult:
        """Execute the full multi-signal detection pipeline."""
        ACTIVE_REQUESTS.inc()
        start_time = time.time()

        try:
            # Run all detectors
            metadata_result = self.metadata_detector.analyze(image)
            frequency_result = self.frequency_detector.analyze(image)
            texture_result = self.texture_detector.analyze(image)
            cnn_result = self.cnn_classifier.predict(image)

            # Ensemble decision
            decision = self.ensemble.decide(
                cnn_result=cnn_result,
                frequency_result=frequency_result,
                metadata_result=metadata_result,
                texture_result=texture_result,
            )

            result = DetectionResult(
                is_ai_generated=decision["is_ai_generated"],
                confidence_score=decision["confidence_score"],
                risk_level=decision["risk_level"],
                signals_detected=decision["signals_detected"],
                model_version=self.settings.app_version,
            )

            # Record metrics
            latency = time.time() - start_time
            PREDICTION_LATENCY.observe(latency)
            PREDICTION_CONFIDENCE.observe(result.confidence_score)
            PREDICTIONS_TOTAL.labels(
                result="ai_generated" if result.is_ai_generated else "real",
                risk_level=result.risk_level,
            ).inc()

            logger.info(
                "detection_complete",
                is_ai_generated=result.is_ai_generated,
                confidence=result.confidence_score,
                risk_level=result.risk_level,
                latency_ms=round(latency * 1000, 1),
            )
            return result

        finally:
            ACTIVE_REQUESTS.dec()
