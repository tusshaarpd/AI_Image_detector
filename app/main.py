"""FastAPI application entry point."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from prometheus_client import make_asgi_app
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.api.routes import router as api_router
from app.api.web import web_router
from app.core.config import get_settings
from app.core.logging import setup_logging
from app.core.metrics import MODEL_VERSION


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: startup and shutdown events."""
    settings = get_settings()
    setup_logging(settings.log_level)
    MODEL_VERSION.labels(version=settings.app_version).set(1)
    yield


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    application = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "Production-grade AI-generated image detection system. "
            "Classifies images as AI-generated, real photograph, or uncertain "
            "with confidence scoring and explainability signals."
        ),
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Rate limiting
    limiter = Limiter(key_func=get_remote_address, default_limits=[settings.rate_limit])
    application.state.limiter = limiter
    application.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    # CORS
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Static files
    application.mount("/static", StaticFiles(directory="app/static"), name="static")

    # Prometheus metrics
    metrics_app = make_asgi_app()
    application.mount("/metrics", metrics_app)

    # API routes
    application.include_router(api_router, prefix="/api/v1", tags=["Detection"])

    # Web UI routes
    application.include_router(web_router, tags=["Web UI"])

    return application


app = create_app()
