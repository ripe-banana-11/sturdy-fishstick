from __future__ import annotations

import time
from typing import Iterable
import numpy as np
import torch
from PIL import Image

from transformers import AutoModelForImageSegmentation
from torchvision import transforms

from config.settings import BackgroundRemovalConfig
from logger_config import logger
from modules.background_removal.rmbg_manager import BackgroundRemovalService


class BiRefNetBackgroundRemovalService(BackgroundRemovalService):
    """
    BiRefNet background removal service using AutoModelForImageSegmentation.
    Supports ZhengPeng7/BiRefNet and similar models.
    """

    def _initialize_model_and_transforms(self) -> tuple[AutoModelForImageSegmentation, transforms.Compose]:
        """
        Initialize model and transforms for BiRefNet.
        """
        model = None
        transform = transforms.Compose(
            [
                transforms.Resize(self.settings.input_image_size),
                transforms.ToTensor(),
            ]
        )
        return model, transform

    def _load_model(self) -> AutoModelForImageSegmentation:
        """
        Load the BiRefNet background removal model.
        """
        model = AutoModelForImageSegmentation.from_pretrained(
            self.settings.model_id,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        ).to(self.device)
        return model

    def _remove_background(self, image: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Remove the background from the image using BiRefNet.
        
        Returns:
            tuple: (rgb_tensor, mask_tensor) where rgb_tensor is (C, H, W) and mask is (H, W)
        """
        # Convert to RGB if needed
        rgb_image = image.convert('RGB')
        
        # Apply transforms: PIL.Image (H, W, C) -> Tensor (C, H, W)
        rgb_tensor = self.transforms(rgb_image).to(self.device)
        
        # Normalize for model input and add batch dimension: (C, H, W) -> (1, C, H, W)
        input_tensor = self.normalize(rgb_tensor).unsqueeze(0)
        
        with torch.no_grad():
            # Get mask from model: (1, 1, H, W)
            preds = self.model(input_tensor)[-1].sigmoid()
            # Remove batch and channel dimensions, quantize: (1, 1, H, W) -> (H, W)
            mask = preds[0].squeeze().mul_(255).int().div(255).float()
        
        return rgb_tensor, mask
