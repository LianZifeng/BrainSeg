"""
Code for B-CLIP text encoder
"""

import open_clip
import pytorch_lightning as pl
from peft import get_peft_model, LoraConfig
import torch.nn.functional as F

class TextEncoder(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def freeze(self):
        super().freeze()

    def freeze_body(self) -> None:
        for para in self.model.base_model.model.transformer.parameters():
            para.requires_grad = False

    def freeze_head(self) -> None:
        for para in self.model.base_model.model.proj.parameters():
            para.requires_grad = False

    def forward(self, text_token):
        feature, token_feature = self.model(text_token)
        feature = F.normalize(feature, dim=-1)
        return feature, token_feature

def load_text_encoder():
    model, preprocess = open_clip.create_model_from_pretrained(
        'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224',
        pretrained='/your/path/to/BiomedCLIP/open_clip_pytorch_model.bin',
        cache_dir='/your/path/to/BiomedCLIP')
    tokenizer = open_clip.get_tokenizer(
        model_name='hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224',
        cache_dir='/your/path/to/BiomedCLIP')

    text_encoder = model.text

    target_modules = [
                         f'transformer.encoder.layer.{i}.attention.self.{j}' for i in range(12) for j in ['query', 'key', 'value']
                     ] + [
                         f'transformer.encoder.layer.{i}.attention.output.dense' for i in range(12)
                     ] + [
                         f'transformer.encoder.layer.{i}.intermediate.dense' for i in range(12)
                     ] + [
                         f'transformer.encoder.layer.{i}.output.dense' for i in range(12)
                     ] + [
                         'proj.0', 'proj.2'
                     ]
    lora_config = LoraConfig(r=8, lora_alpha=32, lora_dropout=0.1, target_modules=target_modules)
    lora_model = get_peft_model(text_encoder, lora_config)

    return tokenizer, TextEncoder(lora_model)