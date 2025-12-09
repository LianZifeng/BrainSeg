"""
Code for B-CLIP block
"""

import torch.nn as nn
from imageencoder import ImageEncoder
from textencoder import load_text_encoder
from typing import Optional
import numpy as np
import torch

class BCLIP(nn.Module):
    def __init__(
        self,
        in_channels: int = 6,
        pretrained_pth: Optional[str] = None,
        init_logit_scale: float = np.log(1 / 0.07),
        init_logit_bias: Optional[float] = None,
        use_v2=False
        ):
        super().__init__()
        self.pretrained_pth = pretrained_pth
        self.image_encoder = ImageEncoder(in_channels=in_channels, use_v2=use_v2)
        if self.pretrained_pth is not None:
            model_dict = torch.load(self.pretrained_pth, map_location='cuda')
            self.image_encoder.load_from(model_dict)
        _, self.text_encoder = load_text_encoder()
        self.logit_scale = nn.Parameter(torch.ones([]) * init_logit_scale)
        if init_logit_bias is not None:
            self.logit_bias = nn.Parameter(torch.ones([]) * init_logit_bias)
        else:
            self.logit_bias = None

    def forward(self, image, text):
        image_features = self.image_encoder(image)
        text_features, _ = self.text_encoder(text)
        logit_scale = self.logit_scale.exp()
        if self.logit_bias is not None:
            return image_features, text_features, logit_scale, self.logit_bias
        return image_features, text_features, logit_scale
