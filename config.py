"""Central configuration for PowerLine_AI."""
import os
from pathlib import Path
import torch

BASE_DIR = Path(__file__).resolve().parent

# Preferred portable location: PowerLine_AI/models/checkpoint_epoch_10.pth
# Override with POWERLINE_MODEL_PATH when the checkpoint is stored elsewhere.
DEFAULT_MODEL_PATH = BASE_DIR / "models" / "checkpoint_epoch_10.pth"
LEGACY_MODEL_PATH = Path(r"D:\python\DLproject_checkpoints\checkpoint_epoch_10.pth")

MODEL_PATH = Path(
    os.getenv("POWERLINE_MODEL_PATH", str(DEFAULT_MODEL_PATH))
).expanduser()

# Temporary backward-compatible fallback for the original Windows setup.
if not MODEL_PATH.is_file() and LEGACY_MODEL_PATH.is_file():
    MODEL_PATH = LEGACY_MODEL_PATH

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = [
    "Background",
    "Yoke",
    "Yoke Suspension",
    "Spacer",
    "Stockbridge Damper",
    "Lightning Rod Shackle",
    "Lightning Rod Suspension",
    "Polymer Insulator",
    "Glass Insulator",
    "Tower ID Plate",
    "Vari-Grip",
    "Polymer Insulator Lower Shackle",
    "Polymer Insulator Upper Shackle",
    "Polymer Insulator Tower Shackle",
    "Glass Insulator Big Shackle",
    "Glass Insulator Small Shackle",
    "Glass Insulator Tower Shackle",
    "Spiral Damper",
    "Sphere",
]

NUM_CLASSES = len(CLASS_NAMES)
DEFAULT_CONFIDENCE_THRESHOLD = 0.50

if NUM_CLASSES != 19:
    raise RuntimeError(
        f"Expected 19 classes (background + 18 components), got {NUM_CLASSES}"
    )
