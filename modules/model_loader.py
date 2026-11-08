"""Build and load the trained PowerLine_AI Faster R-CNN model."""
from pathlib import Path
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from config import CLASS_NAMES, DEVICE, MODEL_PATH, NUM_CLASSES


def build_model():
    """Construct the same Faster R-CNN ResNet-50 FPN architecture."""
    model = fasterrcnn_resnet50_fpn(
        weights=None,
        weights_backbone=None,
    )

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features,
        NUM_CLASSES,
    )
    return model


def load_model(checkpoint_path=None, device=DEVICE):
    """
    Load model weights strictly.

    Supports both a training checkpoint containing 'model_state_dict'
    and a raw model state_dict.
    """
    path = Path(checkpoint_path or MODEL_PATH).expanduser()

    if not path.is_file():
        raise FileNotFoundError(
            "PowerLine_AI checkpoint was not found.\n"
            f"Configured path: {path}\n"
            "Place checkpoint_epoch_10.pth in the project's models/ folder "
            "or set POWERLINE_MODEL_PATH to its full path."
        )

    model = build_model()

    try:
        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )
    except TypeError:
        # Compatibility with older PyTorch versions that do not accept
        # weights_only. Use only with a checkpoint you trust.
        checkpoint = torch.load(path, map_location="cpu")

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    if not isinstance(state_dict, dict):
        raise TypeError("Checkpoint does not contain a valid state dictionary.")

    # strict=True ensures no missing or unexpected model parameters.
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    metadata = {}
    if isinstance(checkpoint, dict):
        for key in ("epoch", "loss"):
            if key in checkpoint:
                value = checkpoint[key]
                metadata[key] = value.item() if hasattr(value, "item") else value

    return model, {
        "checkpoint_path": str(path.resolve()),
        "device": str(device),
        "num_classes": NUM_CLASSES,
        "class_names": CLASS_NAMES.copy(),
        "metadata": metadata,
    }
