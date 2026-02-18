"""Tests for the ensemble decision engine."""

import pytest

from app.detectors.ensemble import EnsembleEngine


@pytest.fixture
def engine() -> EnsembleEngine:
    return EnsembleEngine()


def _make_result(score: float, signals: list[str] | None = None) -> dict:
    return {"score": score, "signals": signals or []}


class TestEnsembleEngine:
    def test_all_high_scores_returns_ai(self, engine: EnsembleEngine):
        decision = engine.decide(
            cnn_result=_make_result(0.9, ["CNN: high"]),
            frequency_result=_make_result(0.8, ["FFT: abnormal"]),
            metadata_result=_make_result(0.9, ["No EXIF"]),
            texture_result=_make_result(0.7, ["Uniform texture"]),
        )
        assert decision["is_ai_generated"] is True
        assert decision["confidence_score"] > 0.5
        assert decision["risk_level"] in ("Medium", "High")

    def test_all_low_scores_returns_real(self, engine: EnsembleEngine):
        decision = engine.decide(
            cnn_result=_make_result(0.1),
            frequency_result=_make_result(0.1),
            metadata_result=_make_result(0.05),
            texture_result=_make_result(0.1),
        )
        assert decision["is_ai_generated"] is False
        assert decision["confidence_score"] < 0.5
        assert decision["risk_level"] == "Low"

    def test_signals_aggregated(self, engine: EnsembleEngine):
        decision = engine.decide(
            cnn_result=_make_result(0.5, ["sig_a"]),
            frequency_result=_make_result(0.5, ["sig_b"]),
            metadata_result=_make_result(0.5, ["sig_c"]),
            texture_result=_make_result(0.5, ["sig_d"]),
        )
        assert "sig_a" in decision["signals_detected"]
        assert "sig_d" in decision["signals_detected"]

    def test_confidence_bounded(self, engine: EnsembleEngine):
        decision = engine.decide(
            cnn_result=_make_result(1.0),
            frequency_result=_make_result(1.0),
            metadata_result=_make_result(1.0),
            texture_result=_make_result(1.0),
        )
        assert 0.0 <= decision["confidence_score"] <= 1.0

    def test_update_calibration(self, engine: EnsembleEngine):
        engine.update_calibration(a=2.0, b=-1.0)
        decision = engine.decide(
            cnn_result=_make_result(0.6),
            frequency_result=_make_result(0.6),
            metadata_result=_make_result(0.6),
            texture_result=_make_result(0.6),
        )
        assert 0.0 <= decision["confidence_score"] <= 1.0
