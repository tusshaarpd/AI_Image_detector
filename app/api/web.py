"""Web UI routes for the image detection application."""

from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.api.routes import get_service
from app.core.logging import get_logger

logger = get_logger(__name__)

web_router = APIRouter()
templates = Jinja2Templates(directory="app/templates")


@web_router.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    """Render the main web UI page."""
    return templates.TemplateResponse("index.html", {"request": request})


@web_router.post("/analyze", response_class=HTMLResponse)
async def analyze_upload(
    request: Request,
    file: UploadFile = File(...),
) -> HTMLResponse:
    """Handle web UI file upload and return results page."""
    service = get_service()
    error = None
    result = None

    try:
        if not file.content_type or not file.content_type.startswith("image/"):
            error = "Please upload a valid image file (JPG, PNG, or WEBP)."
        else:
            image_bytes = await file.read()
            result = service.detect_from_bytes(image_bytes)
    except ValueError as e:
        error = str(e)
    except Exception as e:
        logger.error("web_analysis_failed", error=str(e))
        error = "An error occurred during analysis. Please try again."

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "result": result,
            "error": error,
            "filename": file.filename,
        },
    )
