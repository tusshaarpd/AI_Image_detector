"""Ensemble decision engine with confidence calibration."""

import numpy as np

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class EnsembleEngine:
    """Combines multiple detector outputs using weighted voting.

    Applies configurable weights and Platt scaling for calibrated
    confidence scores.
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        # Platt scaling parameters (can be updated with calibration data)
        self._platt_a: float = 1.0
        self._platt_b: float = 0.0

    def decide(
        self,
        cnn_result: dict,
        frequency_result: dict,
        metadata_result: dict,
        texture_result: dict,
    ) -> dict:
        """Combine detector results into a final decision.

        Args:
            cnn_result: CNN classifier output with score and signals
            frequency_result: Frequency detector output
            metadata_result: Metadata detector output
            texture_result: Texture detector output

        Returns:
            dict with final is_ai_generated, confidence_score,
            risk_level, and aggregated signals
        """
        # Weighted combination
        raw_score = (
            self.settings.weight_cnn * cnn_result["score"]
            + self.settings.weight_frequency * frequency_result["score"]
            + self.settings.weight_metadata * metadata_result["score"]
            + self.settings.weight_texture * texture_result["score"]
        )

        # Calibrate confidence using Platt scaling
        calibrated_score = self._platt_scale(raw_score)
        calibrated_score = float(np.clip(calibrated_score, 0.0, 1.0))

        # Determine classification
        threshold = self.settings.confidence_threshold
        is_ai_generated = calibrated_score >= threshold

        # Risk level
        risk_level = self._compute_risk_level(calibrated_score)

        # Aggregate signals from all detectors
        all_signals = (
            cnn_result["signals"]
            + frequency_result["signals"]
            + metadata_result["signals"]
            + texture_result["signals"]
        )
        # Filter to only meaningful signals
        all_signals = [s for s in all_signals if s]

        logger.info(
            "ensemble_decision",
            raw_score=raw_score,
            calibrated_score=calibrated_score,
            is_ai_generated=is_ai_generated,
            risk_level=risk_level,
            signal_count=len(all_signals),
        )

        return {
            "is_ai_generated": is_ai_generated,
            "confidence_score": round(calibrated_score, 4),
            "risk_level": risk_level,
            "signals_detected": all_signals,
        }

    def _platt_scale(self, score: float) -> float:
        """Apply Platt scaling for confidence calibration.

        P(y=1|s) = 1 / (1 + exp(a*s + b))
        With default parameters (a=1, b=0), this is just sigmoid.
        """
        z = self._platt_a * score + self._platt_b
        return 1.0 / (1.0 + np.exp(-z))

    def _compute_risk_level(self, score: float) -> str:
        """Map confidence score to risk level."""
        if score <= self.settings.risk_low_threshold:
            return "Low"
        elif score <= self.settings.risk_medium_threshold:
            return "Medium"
        else:
            return "High"

    def update_calibration(self, a: float, b: float) -> None:
        """Update Platt scaling parameters from calibration data.

        Args:
            a: Slope parameter
            b: Intercept parameter
        """
        self._platt_a = a
        self._platt_b = b
        logger.info("calibration_updated", platt_a=a, platt_b=b)
