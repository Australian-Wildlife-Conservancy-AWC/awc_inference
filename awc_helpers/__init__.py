"""AWC Helpers - Wildlife detection and classification inference tools."""

from .awc_inference import (
    DetectAndClassify,
    SpeciesClasInference,
    format_md_detections,
    load_classification_model,
)
from .format_utils import (
    output_csv,
    output_timelapse_json,
    truncate_float,
    truncate_float_array,
)
from .math_utils import crop_image, pil_to_tensor

__version__ = "0.1.2"

__all__ = [
    "DetectAndClassify",
    "SpeciesClasInference",
    "format_md_detections",
    "load_classification_model",
    "crop_image",
    "pil_to_tensor",
    "output_csv",
    "output_timelapse_json",
    "truncate_float",
    "truncate_float_array",
]
