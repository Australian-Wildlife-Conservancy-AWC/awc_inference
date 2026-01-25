"""AWC Helpers - Wildlife detection and classification inference tools."""

from .awc_inference import (
    DetectAndClassify,
    SpeciesClasInference,
    format_md_detections,
    load_classification_model,
)
from .math_utils import crop_image, pil_to_tensor

__version__ = "0.1.0"

__all__ = [
    "DetectAndClassify",
    "SpeciesClasInference",
    "format_md_detections",
    "load_classification_model",
    "crop_image",
    "pil_to_tensor",
]
