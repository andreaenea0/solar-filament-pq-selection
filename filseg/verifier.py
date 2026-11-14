"""Verifier candidate crops and offline ResNet-18 inference."""

import cv2
import numpy as np
import torch
from pycocotools import mask as mu

CROP_SIZE = 96


def own_crops(image, probability, rles):
    """Build crops4 rows in tree order with the own map in both map channels."""
    return candidate_crops(image, (probability, probability), rles)


def candidate_crops(image, maps, rles):
    """Build crops4 rows with public/own or own/own channels."""
    height, width = image.shape
    crops = np.empty((len(rles), 4, CROP_SIZE, CROP_SIZE), dtype=np.uint8)
    for i, rle in enumerate(rles):
        x, y, w, h = mu.toBbox(rle).astype(int)
        side = int(max(w, h) * 1.5) + 16
        x0 = max(x + w // 2 - side // 2, 0)
        y0 = max(y + h // 2 - side // 2, 0)
        region = np.s_[y0:min(y0 + side, height), x0:min(x0 + side, width)]
        channels = (image[region], maps[0][region], maps[1][region], mu.decode(rle)[region] * 255)
        for channel, pixels in enumerate(channels):
            crops[i, channel] = cv2.resize(
                np.ascontiguousarray(pixels), (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_AREA
            )
    return crops


def load_verifier(checkpoint, device, *, architecture="resnet18.a1_in1k"):
    """Load a local verifier with the original channels-last layout."""
    import timm

    model = timm.create_model(architecture, pretrained=False, in_chans=4, num_classes=1)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    return model.to(device, memory_format=torch.channels_last).eval()


@torch.inference_mode()
def verifier_gains(models, crops, device):
    """Average raw verifier predictions over all eight transforms and all models."""
    if not len(crops):
        return np.empty(0, dtype=np.float32)
    out = []
    for start in range(0, len(crops), 1024):
        xs = torch.from_numpy(crops[start:start + 1024]).to(device).float() / 255
        seed_scores = []
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            for model in models:
                scores = [
                    model(torch.rot90(xs.flip(3) if flip else xs, turns, (2, 3))).squeeze(1).float()
                    for turns in range(4) for flip in (False, True)
                ]
                seed_scores.append(torch.stack(scores).mean(0).cpu().numpy())
        out.append(np.mean(seed_scores, axis=0))
    return np.concatenate(out)
