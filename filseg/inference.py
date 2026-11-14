"""Inference with the team's k07 EfficientNetV2-S U-Net checkpoints."""

import numpy as np
import torch
from PIL import Image

SIZE = 2048
ENCODER = "tu-tf_efficientnetv2_s.in21k_ft_in1k"
D4 = ((0, False), (0, True), (2, False), (2, True), (1, False), (1, True), (3, False), (3, True))
Y, X = np.ogrid[:SIZE, :SIZE]
DISK = (Y - (SIZE - 1) / 2) ** 2 + (X - (SIZE - 1) / 2) ** 2 <= 950 ** 2


def read_jpeg(path):
    """Load a full-resolution observation as grayscale uint8."""
    with Image.open(path) as image:
        pixels = np.asarray(image.convert("L"), dtype=np.uint8)
    if pixels.shape != (SIZE, SIZE):
        raise ValueError(f"{path}: expected a {SIZE}×{SIZE} image, got {pixels.shape}.")
    return pixels


def normalize(image):
    """Use the same disk mean and standard deviation as k07 test inference."""
    values = image[DISK].astype(np.float32) / 255
    return (image.astype(np.float32) / 255 - values.mean()) / max(values.std(), 1e-3)


def load_model(checkpoint, device, *, encoder=ENCODER, channels_last=False):
    """Load a local U-Net state dict without downloading encoder weights."""
    import segmentation_models_pytorch as smp

    model = smp.Unet(encoder_name=encoder, encoder_weights=None, in_channels=1, classes=1)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model = model.to(device)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    return model.eval()


@torch.no_grad()
def predict_map(model, normalized, device, batch_size=4, *, channels_last=False):
    """Average inverse-aligned D4 probabilities, then quantize as in k07."""
    image = torch.from_numpy(normalized).to(device)[None, None]
    total = torch.zeros(normalized.shape, device=device)
    for start in range(0, len(D4), batch_size):
        transforms = D4[start:start + batch_size]
        batch = []
        for turns, flip in transforms:
            member = image.flip(3) if flip else image
            batch.append(torch.rot90(member, turns, (2, 3)) if turns else member)
        inputs = torch.cat(batch)
        if channels_last:
            inputs = inputs.contiguous(memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            probabilities = torch.sigmoid(model(inputs).float())
        for probability, (turns, flip) in zip(probabilities[:, 0], transforms):
            aligned = torch.rot90(probability, -turns, (0, 1)) if turns else probability
            total += aligned.flip(1) if flip else aligned
    return np.clip(np.round((total / len(D4)).cpu().numpy() * 255), 0, 255).astype(np.uint8)
