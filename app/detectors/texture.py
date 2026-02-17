"""Texture inconsistency detector - analyzes micro-textures and spatial coherence."""

import cv2
import numpy as np
from PIL import Image

from app.core.logging import get_logger

logger = get_logger(__name__)


class TextureDetector:
    """Detects AI-generated images through texture and spatial coherence analysis.

    Examines skin micro-textures, background blur consistency,
    shadow direction, and local texture variance patterns.
    """

    def analyze(self, image: Image.Image) -> dict:
        """Analyze image textures for AI generation artifacts.

        Returns:
            dict with keys:
                - score: float 0-1 (higher = more likely AI)
                - signals: list of string explanations
        """
        signals = []
        score_components = []

        img_array = np.array(image.convert("RGB"))
        img_resized = cv2.resize(img_array, (512, 512))
        gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)

        # Local texture variance analysis
        tv_score, tv_signals = self._analyze_texture_variance(gray)
        signals.extend(tv_signals)
        score_components.append(tv_score)

        # Blur inconsistency
        blur_score, blur_signals = self._analyze_blur_consistency(gray)
        signals.extend(blur_signals)
        score_components.append(blur_score)

        # Edge coherence
        edge_score, edge_signals = self._analyze_edge_coherence(gray)
        signals.extend(edge_signals)
        score_components.append(edge_score)

        # Color channel correlation
        color_score, color_signals = self._analyze_color_correlation(img_resized)
        signals.extend(color_signals)
        score_components.append(color_score)

        final_score = sum(score_components) / max(len(score_components), 1)
        final_score = min(max(final_score, 0.0), 1.0)

        logger.info("texture_analysis_complete", score=final_score, signal_count=len(signals))
        return {"score": final_score, "signals": signals}

    def _analyze_texture_variance(self, gray: np.ndarray) -> tuple[float, list[str]]:
        """Analyze local texture variance patterns across the image.

        AI images often have unnaturally uniform texture patterns
        compared to real photographs.
        """
        signals = []
        block_size = 32
        h, w = gray.shape
        variances = []

        for y in range(0, h - block_size, block_size):
            for x in range(0, w - block_size, block_size):
                block = gray[y : y + block_size, x : x + block_size]
                variances.append(np.var(block.astype(np.float64)))

        if not variances:
            return 0.3, signals

        variances = np.array(variances)
        variance_of_variances = np.var(variances)
        mean_variance = np.mean(variances)

        # AI images tend to have lower variance-of-variance (more uniform textures)
        if mean_variance > 0:
            cv = np.sqrt(variance_of_variances) / mean_variance
        else:
            cv = 0.0

        if cv < 0.3:
            signals.append("Skin microtexture anomaly: unnaturally uniform texture distribution")
            return 0.7, signals
        elif cv < 0.5:
            signals.append("Moderate texture uniformity detected")
            return 0.4, signals

        return 0.15, signals

    def _analyze_blur_consistency(self, gray: np.ndarray) -> tuple[float, list[str]]:
        """Analyze spatial blur consistency.

        Real images have depth-dependent blur; AI images may have
        inconsistent blur patterns.
        """
        signals = []
        block_size = 64
        h, w = gray.shape
        laplacian_vars = []
        positions = []

        for y in range(0, h - block_size, block_size // 2):
            for x in range(0, w - block_size, block_size // 2):
                block = gray[y : y + block_size, x : x + block_size]
                lap_var = cv2.Laplacian(block, cv2.CV_64F).var()
                laplacian_vars.append(lap_var)
                positions.append((y + block_size // 2, x + block_size // 2))

        if len(laplacian_vars) < 4:
            return 0.3, signals

        laplacian_vars = np.array(laplacian_vars)

        # Check for abrupt blur transitions (AI artifact)
        sorted_vars = np.sort(laplacian_vars)
        q25 = sorted_vars[len(sorted_vars) // 4]
        q75 = sorted_vars[3 * len(sorted_vars) // 4]

        if q75 > 0 and q25 > 0:
            blur_ratio = q75 / q25
        else:
            blur_ratio = 1.0

        # Extremely high ratio suggests inconsistent blur
        if blur_ratio > 50:
            signals.append("Background blur inconsistency: abrupt blur transitions detected")
            return 0.65, signals
        elif blur_ratio > 20:
            signals.append("Moderate blur inconsistency detected")
            return 0.35, signals

        return 0.15, signals

    def _analyze_edge_coherence(self, gray: np.ndarray) -> tuple[float, list[str]]:
        """Analyze edge direction coherence for shadow/lighting consistency."""
        signals = []

        # Sobel edge detection
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

        # Compute gradient magnitudes and angles
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        angles = np.arctan2(grad_y, grad_x)

        # Focus on strong edges
        strong_edge_mask = magnitude > np.percentile(magnitude, 85)

        if strong_edge_mask.sum() < 100:
            return 0.3, signals

        strong_angles = angles[strong_edge_mask]

        # Histogram of edge directions
        hist, _ = np.histogram(strong_angles, bins=36, range=(-np.pi, np.pi))
        hist = hist.astype(np.float64)
        hist /= hist.sum() + 1e-10

        # Entropy of edge direction distribution
        entropy = -np.sum(hist * np.log2(hist + 1e-10))

        # AI images often have less diverse edge directions
        # (more uniform or overly regular patterns)
        if entropy < 3.0:
            signals.append("Shadow direction mismatch: low edge direction diversity")
            return 0.6, signals
        elif entropy < 4.0:
            signals.append("Moderate edge direction regularity")
            return 0.35, signals

        return 0.1, signals

    def _analyze_color_correlation(self, img: np.ndarray) -> tuple[float, list[str]]:
        """Analyze inter-channel color correlation patterns."""
        signals = []

        r, g, b = img[:, :, 0].ravel(), img[:, :, 1].ravel(), img[:, :, 2].ravel()

        # Compute correlation matrix between color channels
        rg_corr = np.corrcoef(r.astype(np.float64), g.astype(np.float64))[0, 1]
        rb_corr = np.corrcoef(r.astype(np.float64), b.astype(np.float64))[0, 1]
        gb_corr = np.corrcoef(g.astype(np.float64), b.astype(np.float64))[0, 1]

        correlations = [abs(rg_corr), abs(rb_corr), abs(gb_corr)]
        mean_corr = np.mean(correlations)

        # AI images sometimes show unnaturally high inter-channel correlation
        if mean_corr > 0.97:
            signals.append("Inconsistent lighting map: abnormally high color channel correlation")
            return 0.55, signals
        elif mean_corr > 0.95:
            signals.append("Elevated color channel correlation")
            return 0.3, signals

        return 0.1, signals
