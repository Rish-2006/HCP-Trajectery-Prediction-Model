import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models import resnet18, ResNet18_Weights


class ImageEncoder(nn.Module):
    """
    Encodes a single nuScenes camera keyframe (e.g. CAM_FRONT) into a
    fixed-size embedding, meant to be fused into MTRMotionTransformer
    alongside the existing agent-track / map-polyline tokens.

    Backbone: torchvision's resnet18 with the final classification layer
    removed, followed by a linear projection to out_dim (default 256, to
    match MTRMotionTransformer's d_model so it can be fused directly).

    pretrained=True loads ImageNet weights (recommended — this is a small
    dataset by CNN-training standards, so starting from ImageNet features
    instead of random init matters a lot). Requires an internet connection
    the first time (downloads ~45MB of weights, cached after that).

    freeze_backbone=True keeps the ResNet weights fixed and only trains the
    projection head — a reasonable starting point given how little labeled
    data this project has relative to what a full ResNet fine-tune wants.
    Set to False once there's enough data/training budget to fine-tune the
    whole backbone.
    """

    def __init__(self, out_dim: int = 256, pretrained: bool = True, freeze_backbone: bool = True):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)
        # Keep every layer up to (and including) the global average pool;
        # drop resnet's original 1000-class classification head.
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])  # -> (B, 512, 1, 1)
        self.backbone_out_dim = 512
        self.proj = nn.Linear(self.backbone_out_dim, out_dim)

        self._backbone_frozen = freeze_backbone
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            # requires_grad=False alone does NOT stop BatchNorm layers from
            # updating their running_mean/running_var buffers, or from using
            # noisy per-batch statistics instead of those running stats,
            # whenever the outer model is in .train() mode (which it is for
            # the whole rest of training). With batch_size as small as 2,
            # per-batch BatchNorm statistics are extremely unstable — so the
            # "frozen" backbone was still silently drifting and behaving
            # differently at train time vs. eval time, even though its
            # weights never changed. Force every BatchNorm submodule into
            # permanent eval mode (always use the stable pretrained running
            # stats, never per-batch stats, never update the buffers) so
            # this component behaves identically during training and
            # evaluation, matching what "frozen" is actually supposed to mean.
            for m in self.backbone.modules():
                if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                    m.eval()

    def train(self, mode: bool = True):
        """Overridden so that calling .train() on the outer model (as the
        training loop does every epoch) can never pull the frozen backbone's
        BatchNorm layers back into training-mode behaviour."""
        super().train(mode)
        if self._backbone_frozen:
            for m in self.backbone.modules():
                if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                    m.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, 224, 224) already preprocessed via preprocess_image().
        Returns: (B, out_dim)"""
        feats = self.backbone(x)      # (B, 512, 1, 1)
        feats = feats.flatten(1)      # (B, 512)
        return self.proj(feats)       # (B, out_dim)


# ImageNet normalization stats — required because the backbone starts from
# ImageNet-pretrained weights, which were trained on images normalized this
# way. Skipping this would badly hurt the pretrained features' usefulness.
_IMAGE_TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def preprocess_image(pil_image) -> torch.Tensor:
    """Takes a PIL RGB image (e.g. straight from load_camera_image() in
    nuscenes_parser.py) and returns a normalized (3, 224, 224) tensor ready
    to be batched and passed into ImageEncoder."""
    return _IMAGE_TRANSFORM(pil_image)


if __name__ == "__main__":
    # Standalone smoke test: random image tensor through the encoder,
    # confirming output shape and that gradients reach the trainable params
    # (the projection head; backbone stays frozen by default).
    encoder = ImageEncoder(out_dim=256, pretrained=False, freeze_backbone=True)
    dummy_batch = torch.randn(4, 3, 224, 224)
    out = encoder(dummy_batch)
    print(f"Output shape: {out.shape}")  # expect (4, 256)

    loss = out.sum()
    loss.backward()
    proj_grad_norm = encoder.proj.weight.grad.norm().item()
    backbone_grad_is_none = all(p.grad is None for p in encoder.backbone.parameters())
    print(f"proj.weight grad norm: {proj_grad_norm:.4f} (should be > 0)")
    print(f"backbone frozen (grads None): {backbone_grad_is_none} (should be True)")