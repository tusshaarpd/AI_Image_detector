"""Application configuration using pydantic-settings."""

from pathlib import Path
from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Application
    app_name: str = "AI Image Detector"
    app_version: str = "1.0.0"
    debug: bool = False
    log_level: str = "INFO"

    # API Security
    api_key: str = "changeme-in-production"
    rate_limit: str = "100/minute"

    # Image Processing
    max_image_size_mb: int = 10
    allowed_formats: list[str] = ["JPEG", "PNG", "WEBP"]
    input_image_size: int = 224

    # Model
    model_name: str = "efficientnet_b4"
    model_weights_dir: Path = Path("models/weights")
    confidence_threshold: float = 0.5
    device: str = "cpu"

    # Risk Scoring
    risk_low_threshold: float = 0.40
    risk_medium_threshold: float = 0.70

    # Ensemble Weights
    weight_cnn: float = 0.45
    weight_frequency: float = 0.20
    weight_metadata: float = 0.20
    weight_texture: float = 0.15

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


@lru_cache
def get_settings() -> Settings:
    return Settings()
