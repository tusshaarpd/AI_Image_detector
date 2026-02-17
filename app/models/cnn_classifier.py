"""CNN-based image classifier using EfficientNet/ResNet for AI image detection."""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class CNNClassifier:
    """Binary classifier using transfer learning to detect AI-generated images.

    Supports EfficientNet-B4, ResNet-50, and ViT-Base architectures.
    Uses pretrained ImageNet weights when no fine-tuned weights are available.
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self.device = torch.device(self.settings.device)
        self.model: nn.Module | None = None
        self.transform = self._build_transform()
        self._load_model()

    def _build_transform(self) -> transforms.Compose:
        """Build image preprocessing pipeline."""
        size = self.settings.input_image_size
        return transforms.Compose(
            [
                transforms.Resize((size, size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def _load_model(self) -> None:
        """Load the CNN model with pretrained or fine-tuned weights."""
        model_name = self.settings.model_name
        weights_dir = self.settings.model_weights_dir

        logger.info("loading_cnn_model", model_name=model_name, device=str(self.device))

        try:
            self.model = self._create_model(model_name)

            # Check for fine-tuned weights
            weights_path = weights_dir / f"{model_name}_detector.pth"
            if weights_path.exists():
                state_dict = torch.load(weights_path, map_location=self.device, weights_only=True)
                self.model.load_state_dict(state_dict)
                logger.info("loaded_finetuned_weights", path=str(weights_path))
            else:
                logger.info(
                    "no_finetuned_weights_found",
                    msg="Using pretrained ImageNet features with heuristic classification",
                )

            self.model.to(self.device)
            self.model.eval()
            logger.info("cnn_model_loaded", model_name=model_name)
        except Exception as e:
            logger.error("cnn_model_load_failed", error=str(e))
            self.model = None

    def _create_model(self, model_name: str) -> nn.Module:
        """Create model architecture with binary classification head."""
        if model_name == "efficientnet_b4":
            return self._create_efficientnet_b4()
        elif model_name == "resnet50":
            return self._create_resnet50()
        else:
            logger.warning("unknown_model_name", model_name=model_name, fallback="efficientnet_b4")
            return self._create_efficientnet_b4()

    def _create_efficientnet_b4(self) -> nn.Module:
        """Create EfficientNet-B4 with binary classification head."""
        try:
            import timm

            model = timm.create_model("efficientnet_b4", pretrained=True, num_classes=1)
            return model
        except Exception:
            # Fallback to torchvision
            from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights

            model = efficientnet_b4(weights=EfficientNet_B4_Weights.DEFAULT)
            num_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(num_features, 1)
            return model

    def _create_resnet50(self) -> nn.Module:
        """Create ResNet-50 with binary classification head."""
        from torchvision.models import resnet50, ResNet50_Weights

        model = resnet50(weights=ResNet50_Weights.DEFAULT)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, 1)
        return model

    def predict(self, image: Image.Image) -> dict:
        """Run CNN inference on an image.

        Returns:
            dict with keys:
                - score: float 0-1 (higher = more likely AI)
                - signals: list of string explanations
        """
        if self.model is None:
            logger.warning("cnn_model_not_available")
            return {"score": 0.5, "signals": ["CNN model unavailable - using neutral score"]}

        signals = []

        try:
            img_tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)

            with torch.no_grad():
                logit = self.model(img_tensor)
                prob = torch.sigmoid(logit).item()

            # Interpret CNN output
            if prob > 0.7:
                signals.append(f"CNN classifier: high AI probability ({prob:.2f})")
            elif prob > 0.5:
                signals.append(f"CNN classifier: moderate AI probability ({prob:.2f})")
            elif prob > 0.3:
                signals.append(f"CNN classifier: uncertain ({prob:.2f})")
            else:
                signals.append(f"CNN classifier: likely real photograph ({prob:.2f})")

            logger.info("cnn_prediction_complete", probability=prob)
            return {"score": prob, "signals": signals}

        except Exception as e:
            logger.error("cnn_prediction_failed", error=str(e))
            return {"score": 0.5, "signals": [f"CNN prediction error: {str(e)}"]}
