"""Frequency domain analysis detector - FFT-based artifact and GAN fingerprint detection."""

import numpy as np
from PIL import Image

from app.core.logging import get_logger

logger = get_logger(__name__)


class FrequencyDetector:
    """Detects AI-generated images through frequency domain analysis.

    Uses FFT to identify GAN fingerprints and spectral artifacts that
    are characteristic of synthetically generated images.
    """

    def analyze(self, image: Image.Image) -> dict:
        """Analyze image in frequency domain for AI generation artifacts.

        Returns:
            dict with keys:
                - score: float 0-1 (higher = more likely AI)
                - signals: list of string explanations
        """
        signals = []
        score_components = []

        img_array = np.array(image.convert("L").resize((256, 256)), dtype=np.float64)

        # FFT analysis
        fft_score, fft_signals = self._analyze_fft_spectrum(img_array)
        signals.extend(fft_signals)
        score_components.append(fft_score)

        # GAN fingerprint analysis
        gan_score, gan_signals = self._detect_gan_fingerprint(img_array)
        signals.extend(gan_signals)
        score_components.append(gan_score)

        # High-frequency energy analysis
        hf_score, hf_signals = self._analyze_high_frequency_energy(img_array)
        signals.extend(hf_signals)
        score_components.append(hf_score)

        final_score = sum(score_components) / max(len(score_components), 1)
        final_score = min(max(final_score, 0.0), 1.0)

        logger.info("frequency_analysis_complete", score=final_score, signal_count=len(signals))
        return {"score": final_score, "signals": signals}

    def _analyze_fft_spectrum(self, img_array: np.ndarray) -> tuple[float, list[str]]:
        """Analyze the FFT power spectrum for anomalies."""
        signals = []

        # Compute 2D FFT
        f_transform = np.fft.fft2(img_array)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.log1p(np.abs(f_shift))

        # Analyze radial power distribution
        center = np.array(magnitude_spectrum.shape) // 2
        y, x = np.ogrid[: magnitude_spectrum.shape[0], : magnitude_spectrum.shape[1]]
        r = np.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2).astype(int)

        max_r = min(center)
        radial_profile = np.zeros(max_r)
        for i in range(max_r):
            mask = r == i
            if mask.any():
                radial_profile[i] = magnitude_spectrum[mask].mean()

        # Check for abnormal spectral drop-off (GAN artifact)
        if len(radial_profile) > 10:
            mid = len(radial_profile) // 2
            low_freq_energy = radial_profile[:mid].mean()
            high_freq_energy = radial_profile[mid:].mean()

            if low_freq_energy > 0:
                ratio = high_freq_energy / low_freq_energy
            else:
                ratio = 0.0

            # AI-generated images often have steeper spectral drop-off
            if ratio < 0.15:
                signals.append("Abnormal spectral drop-off detected (possible GAN artifact)")
                return 0.7, signals
            elif ratio < 0.25:
                signals.append("Moderate spectral irregularity detected")
                return 0.4, signals

        return 0.2, signals

    def _detect_gan_fingerprint(self, img_array: np.ndarray) -> tuple[float, list[str]]:
        """Detect periodic GAN fingerprint patterns in frequency domain."""
        signals = []

        f_transform = np.fft.fft2(img_array)
        f_shift = np.fft.fftshift(f_transform)
        magnitude = np.abs(f_shift)

        # Remove DC component
        center = np.array(magnitude.shape) // 2
        magnitude[center[0] - 2 : center[0] + 3, center[1] - 2 : center[1] + 3] = 0

        # Look for periodic peaks (GAN fingerprints appear as regular peaks)
        threshold = np.mean(magnitude) + 4 * np.std(magnitude)
        peaks = magnitude > threshold
        peak_count = np.sum(peaks)

        # Analyze peak distribution for periodicity
        peak_ratio = peak_count / magnitude.size

        if peak_ratio > 0.005:
            signals.append(
                f"GAN fingerprint pattern: {peak_count} periodic spectral peaks detected"
            )
            return 0.8, signals
        elif peak_ratio > 0.002:
            signals.append("Mild periodic spectral pattern detected")
            return 0.4, signals

        return 0.15, signals

    def _analyze_high_frequency_energy(self, img_array: np.ndarray) -> tuple[float, list[str]]:
        """Analyze high-frequency energy distribution for AI artifacts."""
        signals = []

        f_transform = np.fft.fft2(img_array)
        f_shift = np.fft.fftshift(f_transform)
        magnitude = np.abs(f_shift)

        h, w = magnitude.shape
        center_h, center_w = h // 2, w // 2

        # Define high-frequency region (outer 25% of spectrum)
        mask = np.zeros_like(magnitude, dtype=bool)
        y, x = np.ogrid[:h, :w]
        r = np.sqrt((x - center_w) ** 2 + (y - center_h) ** 2)
        outer_radius = min(center_h, center_w)
        inner_radius = outer_radius * 0.75
        mask = r > inner_radius

        total_energy = magnitude.sum()
        high_freq_energy = magnitude[mask].sum()

        if total_energy > 0:
            hf_ratio = high_freq_energy / total_energy
        else:
            hf_ratio = 0.0

        # AI-generated images typically have less high-frequency detail
        if hf_ratio < 0.02:
            signals.append("Unusually low high-frequency energy (typical of AI generation)")
            return 0.6, signals
        elif hf_ratio < 0.05:
            signals.append("Moderate high-frequency energy deficit")
            return 0.35, signals

        return 0.1, signals
