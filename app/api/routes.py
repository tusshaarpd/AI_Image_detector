"""API routes for image detection."""

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from app.core.detector_service import DetectorService
from app.core.logging import get_logger
from app.core.security import verify_api_key
from app.schemas.detection import DetectionRequest, DetectionResult, ErrorResponse, HealthResponse

logger = get_logger(__name__)

router = APIRouter()

# Singleton service instance (initialized on first use)
_service: DetectorService | None = None


def get_service() -> DetectorService:
    """Get or create the detector service singleton."""
    global _service
    if _service is None:
        _service = DetectorService()
    return _service


@router.post(
    "/detect",
    response_model=DetectionResult,
    responses={
        400: {"model": ErrorResponse},
        401: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
        413: {"model": ErrorResponse},
    },
    summary="Detect AI-generated image from base64",
    description="Analyze a base64-encoded image and return detection results.",
)
async def detect_image(
    request: DetectionRequest,
    _api_key: str = Depends(verify_api_key),
    service: DetectorService = Depends(get_service),
) -> DetectionResult:
    """Detect whether a base64-encoded image is AI-generated."""
    try:
        return service.detect_from_base64(request.image)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error("detection_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal detection error",
        )


@router.post(
    "/detect/upload",
    response_model=DetectionResult,
    responses={
        400: {"model": ErrorResponse},
        401: {"model": ErrorResponse},
        413: {"model": ErrorResponse},
    },
    summary="Detect AI-generated image from file upload",
    description="Upload an image file and get detection results.",
)
async def detect_image_upload(
    file: UploadFile = File(..., description="Image file (JPG, PNG, or WEBP)"),
    _api_key: str = Depends(verify_api_key),
    service: DetectorService = Depends(get_service),
) -> DetectionResult:
    """Detect whether an uploaded image is AI-generated."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be an image (JPG, PNG, or WEBP)",
        )

    try:
        image_bytes = await file.read()
        return service.detect_from_bytes(image_bytes)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error("detection_upload_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal detection error",
        )


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
)
async def health_check() -> HealthResponse:
    """Application health check endpoint."""
    service = get_service()
    return HealthResponse(
        status="healthy",
        version=service.settings.app_version,
        model_loaded=service.cnn_classifier.model is not None,
    )
