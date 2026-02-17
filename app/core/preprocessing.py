"""Image preprocessing and validation utilities."""

import base64
import io

from PIL import Image

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class ImagePreprocessor:
    """Handles image validation, decoding, and preprocessing."""

    def __init__(self) -> None:
        self.settings = get_settings()

    def decode_base64(self, image_b64: str) -> Image.Image:
        """Decode a base64-encoded image string to PIL Image.

        Args:
            image_b64: Base64-encoded image data (with or without data URI prefix)

        Returns:
            PIL Image object

        Raises:
            ValueError: If decoding fails or image is invalid
        """
        try:
            # Strip data URI prefix if present
            if "," in image_b64:
                image_b64 = image_b64.split(",", 1)[1]

            image_bytes = base64.b64decode(image_b64)
            return self._load_and_validate(image_bytes)
        except Exception as e:
            logger.error("base64_decode_failed", error=str(e))
            raise ValueError(f"Failed to decode base64 image: {e}") from e

    def load_from_bytes(self, image_bytes: bytes) -> Image.Image:
        """Load and validate image from raw bytes.

        Args:
            image_bytes: Raw image data

        Returns:
            PIL Image object

        Raises:
            ValueError: If image is invalid or exceeds size limits
        """
        return self._load_and_validate(image_bytes)

    def _load_and_validate(self, image_bytes: bytes) -> Image.Image:
        """Load image from bytes and validate format and size."""
        # Size validation
        max_bytes = self.settings.max_image_size_mb * 1024 * 1024
        if len(image_bytes) > max_bytes:
            raise ValueError(
                f"Image size {len(image_bytes) / 1024 / 1024:.1f}MB exceeds "
                f"maximum {self.settings.max_image_size_mb}MB"
            )

        try:
            image = Image.open(io.BytesIO(image_bytes))
            image.load()  # Force load to verify integrity
        except Exception as e:
            raise ValueError(f"Invalid image data: {e}") from e

        # Format validation
        fmt = image.format or "UNKNOWN"
        if fmt not in self.settings.allowed_formats:
            raise ValueError(
                f"Unsupported format '{fmt}'. Allowed: {self.settings.allowed_formats}"
            )

        # Sanitize: remove potential embedded scripts
        if image.mode not in ("RGB", "RGBA", "L"):
            image = image.convert("RGB")

        logger.info("image_preprocessed", format=fmt, size=image.size, mode=image.mode)
        return image
