"""
Model.py — Neural network architectures. 

All model classes share the same forward interface:
    forward(img, geo) -> logits (B, num_classes)

    img : float32 Tensor (B, 3, 224, 224)
    geo : depends on encoder family
        continuous  -> float32 (B, geo_dim)
        discrete    -> int64   (B, 2)   where col-0 = idx1, col-1 = idx2

Models:
    ContinuousGeoModel       — wrap / raw / sh, early fusion
    ContinuousLateFusionModel — wrap / raw / sh, late fusion
    DiscreteEarlyFusionModel — hex / geo_label, early fusion
    DiscreteLateFusionModel  — hex / geo_label, late fusion

All expose .param_groups(backbone_lr, head_lr) for AdamW with split LRs.

Factory:
  build_model(encoder_type, fusion_type, num_classes, *, geo_dim, geo_encoder)
"""

import torch
import torch.nn as nn
from torchvision import models

from scripts.server.Encoder import CONTINUOUS_ENCODERS, DISCRETE_ENCODERS

IMG_FEAT_DIM = 2048  # ResNet-50 penultimate layer


def _build_resnet50() -> nn.Module:
    backbone = models.resnet50(pretrained=True)
    backbone.fc = nn.Identity()
    return backbone


def _build_classifier(in_dim: int, num_classes: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_dim, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(512, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Linear(256, num_classes),
    )


class ContinuousGeoModel(nn.Module):
    """
    ResNet-50 + small MLP for continuous geo vectors, early fusion.

    geo input: float32 (B, geo_dim)
    """

    def __init__(self, num_classes: int, geo_dim: int, geo_hidden: int = 64):
        super().__init__()
        self.backbone = _build_resnet50()
        self.location_encoder = nn.Sequential(
            nn.Linear(geo_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, geo_hidden),
            nn.ReLU(inplace=True),
        )
        self.classifier = _build_classifier(IMG_FEAT_DIM + geo_hidden, num_classes)

    def forward(self, img: torch.Tensor, geo: torch.Tensor) -> torch.Tensor:
        img_feat = self.backbone(img)
        geo_feat = self.location_encoder(geo)
        return self.classifier(torch.cat([img_feat, geo_feat], dim=1))

    def param_groups(self, backbone_lr: float = 3e-5, head_lr: float = 1e-3) -> list:
        return [
            {'params': self.backbone.parameters(),         'lr': backbone_lr},
            {'params': self.location_encoder.parameters(), 'lr': head_lr},
            {'params': self.classifier.parameters(),       'lr': head_lr},
        ]


class ContinuousLateFusionModel(nn.Module):
    """
    ResNet-50 + small MLP for continuous geo vectors, late fusion.
    Each modality gets its own classification head; logits are weight-averaged.

    geo input: float32 (B, geo_dim)
    """

    def __init__(self, num_classes: int, geo_dim: int, geo_hidden: int = 64,
                 img_weight: float = 1.0, geo_weight: float = 1.0):
        super().__init__()
        self.backbone = _build_resnet50()
        self.location_encoder = nn.Sequential(
            nn.Linear(geo_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, geo_hidden),
            nn.ReLU(inplace=True),
        )
        self.img_head   = _build_classifier(IMG_FEAT_DIM, num_classes)
        self.geo_head   = _build_classifier(geo_hidden,   num_classes)
        self.img_weight = img_weight
        self.geo_weight = geo_weight

    def forward(self, img: torch.Tensor, geo: torch.Tensor) -> torch.Tensor:
        img_feat  = self.backbone(img)
        geo_feat  = self.location_encoder(geo)
        logit_img = self.img_head(img_feat)
        logit_geo = self.geo_head(geo_feat)
        w = self.img_weight + self.geo_weight
        return (self.img_weight * logit_img + self.geo_weight * logit_geo) / w

    def param_groups(self, backbone_lr: float = 3e-5, head_lr: float = 1e-3) -> list:
        return [
            {'params': self.backbone.parameters(),         'lr': backbone_lr},
            {'params': self.location_encoder.parameters(), 'lr': head_lr},
            {'params': self.img_head.parameters(),         'lr': head_lr},
            {'params': self.geo_head.parameters(),         'lr': head_lr},
        ]


class DiscreteEarlyFusionModel(nn.Module):
    """
    ResNet-50 + learnable geo embedding, early fusion.

    geo input: int64 (B, 2) — col-0 is idx1, col-1 is idx2.
    """

    def __init__(self, num_classes: int, geo_encoder: nn.Module):
        super().__init__()
        self.backbone    = _build_resnet50()
        self.geo_encoder = geo_encoder
        self.classifier  = _build_classifier(IMG_FEAT_DIM + geo_encoder.out_dim, num_classes)

    def forward(self, img: torch.Tensor, geo: torch.Tensor) -> torch.Tensor:
        img_feat = self.backbone(img)
        geo_feat = self.geo_encoder(geo[:, 0], geo[:, 1])
        return self.classifier(torch.cat([img_feat, geo_feat], dim=1))

    def param_groups(self, backbone_lr: float = 3e-5, head_lr: float = 1e-3) -> list:
        return [
            {'params': self.backbone.parameters(), 'lr': backbone_lr},
            *self.geo_encoder.param_groups(lr=head_lr),
            {'params': self.classifier.parameters(), 'lr': head_lr},
        ]

class DiscreteLateFusionModel(nn.Module):
    """
    ResNet-50 + learnable geo embedding, late fusion.
    Each modality gets its own classification head; logits are weight-averaged.

    geo input: int64 (B, 2) — col-0 is idx1, col-1 is idx2.
    """

    def __init__(self, num_classes: int, geo_encoder: nn.Module,
                 img_weight: float = 1.0, geo_weight: float = 1.0):
        super().__init__()
        self.backbone    = _build_resnet50()
        self.geo_encoder = geo_encoder
        self.img_head    = _build_classifier(IMG_FEAT_DIM,          num_classes)
        self.geo_head    = _build_classifier(geo_encoder.out_dim,   num_classes)
        self.img_weight  = img_weight
        self.geo_weight  = geo_weight

    def forward(self, img: torch.Tensor, geo: torch.Tensor) -> torch.Tensor:
        img_feat  = self.backbone(img)
        geo_feat  = self.geo_encoder(geo[:, 0], geo[:, 1])
        logit_img = self.img_head(img_feat)
        logit_geo = self.geo_head(geo_feat)
        w = self.img_weight + self.geo_weight
        return (self.img_weight * logit_img + self.geo_weight * logit_geo) / w

    def param_groups(self, backbone_lr: float = 3e-5, head_lr: float = 1e-3) -> list:
        return [
            {'params': self.backbone.parameters(), 'lr': backbone_lr},
            *self.geo_encoder.param_groups(lr=head_lr),
            {'params': self.img_head.parameters(), 'lr': head_lr},
            {'params': self.geo_head.parameters(), 'lr': head_lr},
        ]

class DiscreteBaseline(nn.Module):
    """
    ResNet-50
    single modality: Img
    """

    def __init__(self, num_classes: int, geo_encoder: nn.Module,
                 img_weight: float = 1.0, geo_weight: float = 0.0):
        super().__init__()
        self.backbone    = _build_resnet50()
        self.geo_encoder = geo_encoder
        self.img_head    = _build_classifier(IMG_FEAT_DIM,          num_classes)
        self.geo_head    = _build_classifier(geo_encoder.out_dim,   num_classes)
        self.img_weight  = img_weight
        self.geo_weight  = geo_weight

    def forward(self, img: torch.Tensor, geo: torch.Tensor) -> torch.Tensor:
        img_feat  = self.backbone(img)
        geo_feat  = self.geo_encoder(geo[:, 0], geo[:, 1])
        logit_img = self.img_head(img_feat)
        logit_geo = self.geo_head(geo_feat)
        return (self.img_weight * logit_img) / self.img_weight

    def param_groups(self, backbone_lr: float = 3e-5, head_lr: float = 1e-3) -> list:
        return [
            {'params': self.backbone.parameters(), 'lr': backbone_lr},
            {'params': self.img_head.parameters(), 'lr': head_lr},
        ]



def build_model(encoder_type: str, fusion_type: str, num_classes: int,
                geo_dim: int = None, geo_encoder: nn.Module = None) -> nn.Module:
    """
    Return the right model for a given encoder x fusion combination.

    Args:
        encoder_type : 'wrap', 'raw', 'sh', 'hex', or 'geo_label'
        fusion_type  : 'early' or 'late'
        num_classes  : number of output species classes
        geo_dim      : output dim of a continuous encoder (required for wrap/raw/sh)
        geo_encoder  : HexGridEncoder or GeoLabelEncoder instance (required for hex/geo_label)
    """
    if encoder_type in CONTINUOUS_ENCODERS:
        if geo_dim is None:
            raise ValueError("geo_dim is required for continuous encoders")
        if fusion_type == 'late':
            return ContinuousLateFusionModel(num_classes, geo_dim)
        return ContinuousGeoModel(num_classes, geo_dim)

    if encoder_type in DISCRETE_ENCODERS:
        if geo_encoder is None:
            raise ValueError("geo_encoder is required for discrete encoders")
        if fusion_type == 'early':
            return DiscreteEarlyFusionModel(num_classes, geo_encoder)
        elif fusion_type == 'late':
            return DiscreteLateFusionModel(num_classes, geo_encoder)
        else:
            raise ValueError(f"Unknown fusion_type '{fusion_type}'. Use 'early' or 'late'.")
    if encoder_type == "None":
        if fusion_type == 'early':
            return DiscreteBaseline(num_classes, geo_encoder)
        elif fusion_type == 'late':
            return DiscreteBaseline(num_classes, geo_encoder)

    raise ValueError(f"Unknown encoder_type '{encoder_type}'")