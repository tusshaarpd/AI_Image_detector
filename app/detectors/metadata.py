"""Metadata analysis detector - examines EXIF data for AI generation signals."""

from PIL import Image
from PIL.ExifTags import TAGS

from app.core.logging import get_logger

logger = get_logger(__name__)

# Known AI tool signatures found in metadata
AI_SOFTWARE_SIGNATURES = [
    "stable diffusion",
    "midjourney",
    "dall-e",
    "dall·e",
    "novelai",
    "comfyui",
    "automatic1111",
    "invoke ai",
    "dreamstudio",
    "leonardo.ai",
    "adobe firefly",
    "bing image creator",
    "craiyon",
    "nightcafe",
    "artbreeder",
    "runwayml",
    "playground ai",
    "deepai",
    "starryai",
    "jasper art",
]

# Camera-related EXIF tags that real photos typically have
CAMERA_TAGS = {
    "Make",
    "Model",
    "FocalLength",
    "ExposureTime",
    "FNumber",
    "ISOSpeedRatings",
    "Flash",
    "WhiteBalance",
    "LensModel",
    "LensMake",
    "GPSInfo",
    "DateTimeOriginal",
    "ShutterSpeedValue",
    "ApertureValue",
    "MeteringMode",
}


class MetadataDetector:
    """Analyzes image EXIF metadata to detect AI-generated images."""

    def analyze(self, image: Image.Image) -> dict:
        """Analyze image metadata and return detection signals.

        Returns:
            dict with keys:
                - score: float 0-1 (higher = more likely AI)
                - signals: list of string explanations
        """
        signals = []
        score_components = []

        exif_data = self._extract_exif(image)

        if not exif_data:
            signals.append("Metadata absence: no EXIF data found")
            score_components.append(0.6)
        else:
            # Check for AI software signatures
            ai_sig = self._check_ai_signatures(exif_data)
            if ai_sig:
                signals.append(f"AI tool signature detected: {ai_sig}")
                score_components.append(1.0)

            # Check for camera metadata
            camera_score, camera_signals = self._check_camera_metadata(exif_data)
            signals.extend(camera_signals)
            score_components.append(camera_score)

            # Check for suspicious software tags
            sw_score, sw_signals = self._check_software_tags(exif_data)
            signals.extend(sw_signals)
            if sw_score > 0:
                score_components.append(sw_score)

        # Check image format hints
        fmt_score, fmt_signals = self._check_format_hints(image)
        signals.extend(fmt_signals)
        if fmt_score > 0:
            score_components.append(fmt_score)

        final_score = sum(score_components) / max(len(score_components), 1)
        final_score = min(max(final_score, 0.0), 1.0)

        logger.info("metadata_analysis_complete", score=final_score, signal_count=len(signals))
        return {"score": final_score, "signals": signals}

    def _extract_exif(self, image: Image.Image) -> dict[str, str]:
        """Extract EXIF data from image."""
        try:
            exif = image.getexif()
            if not exif:
                return {}
            return {TAGS.get(k, str(k)): str(v) for k, v in exif.items()}
        except Exception:
            return {}

    def _check_ai_signatures(self, exif_data: dict[str, str]) -> str | None:
        """Check if EXIF data contains known AI tool signatures."""
        searchable = " ".join(exif_data.values()).lower()
        for sig in AI_SOFTWARE_SIGNATURES:
            if sig in searchable:
                return sig
        return None

    def _check_camera_metadata(self, exif_data: dict[str, str]) -> tuple[float, list[str]]:
        """Check for presence of camera-related metadata tags."""
        found_tags = CAMERA_TAGS.intersection(exif_data.keys())
        ratio = len(found_tags) / len(CAMERA_TAGS)

        signals = []
        if ratio == 0:
            signals.append("No camera metadata present")
            return 0.7, signals
        elif ratio < 0.3:
            signals.append(f"Sparse camera metadata: only {len(found_tags)} camera tags found")
            return 0.4, signals
        else:
            signals.append(f"Rich camera metadata present: {len(found_tags)} camera tags found")
            return 0.1, signals

    def _check_software_tags(self, exif_data: dict[str, str]) -> tuple[float, list[str]]:
        """Check software tags for image editing or generation tools."""
        signals = []
        score = 0.0

        software = exif_data.get("Software", "").lower()
        if software:
            editing_tools = ["photoshop", "gimp", "lightroom", "capture one"]
            for tool in editing_tools:
                if tool in software:
                    signals.append(f"Image editing software detected: {software}")
                    score = 0.2
                    break

        return score, signals

    def _check_format_hints(self, image: Image.Image) -> tuple[float, list[str]]:
        """Check image format-level hints."""
        signals = []
        score = 0.0

        # Check for unusual image dimensions (powers of 2, typical of AI outputs)
        w, h = image.size
        if w == h and w in (256, 512, 768, 1024, 2048):
            signals.append(f"Square dimensions typical of AI output: {w}x{h}")
            score = 0.3

        return score, signals
