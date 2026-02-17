"""Pydantic schemas for API request and response models."""

from pydantic import BaseModel, Field


class DetectionRequest(BaseModel):
    """Request body for image detection via base64."""

    image: str = Field(..., description="Base64-encoded image string")


class DetectionResult(BaseModel):
    """Detection result returned by the API."""

    is_ai_generated: bool = Field(..., description="Whether the image is AI-generated")
    confidence_score: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence score between 0 and 1"
    )
    risk_level: str = Field(..., description="Risk level: Low, Medium, or High")
    signals_detected: list[str] = Field(
        default_factory=list, description="Explainability signals detected"
    )
    model_version: str = Field(..., description="Model version used for detection")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str
    model_loaded: bool


class ErrorResponse(BaseModel):
    """Standard error response."""

    detail: str
