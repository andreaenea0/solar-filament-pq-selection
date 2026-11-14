"""Local test-time replay of the public k02 U-Net ensemble."""

import json
from pathlib import Path

import numpy as np

from .inference import load_model, predict_map


def public_checkpoints(receipt):
    """Read exactly five folds for each public encoder family."""
    base = receipt.resolve().parent
    families = ({}, {})
    files = set()
    for record in json.loads(receipt.read_text(encoding="utf-8"))["ensemble"]["models"]:
        encoder = record["encoder"]
        if "resnet34" in encoder:
            family = families[0]
        elif "efficientnetv2_s" in encoder:
            family = families[1]
        else:
            raise ValueError(f"Unexpected public encoder: {encoder}.")
        fold = int(record["fold"])
        checkpoint = (base / Path(record["file"])).resolve()
        if not checkpoint.is_relative_to(base) or fold in family:
            raise ValueError(f"Invalid public checkpoint or repeated fold: {record['file']}.")
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Public checkpoint missing: {checkpoint}")
        if checkpoint in files:
            raise ValueError(f"Repeated public checkpoint: {checkpoint}.")
        files.add(checkpoint)
        family[fold] = (checkpoint, encoder)
    if any(set(family) != set(range(5)) for family in families):
        raise ValueError("Public weights must contain folds 0 through 4 in both families.")
    return [[family[fold] for fold in range(5)] for family in families]


def load_public_models(receipt, device):
    """Load the public U-Nets in receipt family and fold order."""
    return [[load_model(path, device, encoder=encoder, channels_last=True) for path, encoder in family]
            for family in public_checkpoints(receipt)]


def public_map(models, normalized, device, batch_size):
    """Round D4 maps per fold, then round the mean in each family and across families."""
    channels = []
    for family in models:
        total = np.zeros(normalized.shape, dtype=np.float32)
        for model in family:
            total += predict_map(model, normalized, device, batch_size, channels_last=True)
        channels.append((total / len(family)).round())
    return np.mean(channels, axis=0).round().astype(np.uint8)
