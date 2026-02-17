"""Prometheus metrics for monitoring and observability."""

from prometheus_client import Counter, Histogram, Gauge

PREDICTIONS_TOTAL = Counter(
    "detector_predictions_total",
    "Total number of predictions made",
    ["result", "risk_level"],
)

PREDICTION_LATENCY = Histogram(
    "detector_prediction_latency_seconds",
    "Time spent processing each prediction",
    buckets=[0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 2.0, 5.0],
)

PREDICTION_CONFIDENCE = Histogram(
    "detector_prediction_confidence",
    "Distribution of prediction confidence scores",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

MODEL_VERSION = Gauge(
    "detector_model_version_info",
    "Currently loaded model version",
    ["version"],
)

ACTIVE_REQUESTS = Gauge(
    "detector_active_requests",
    "Number of currently active requests",
)
