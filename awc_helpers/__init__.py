"""AWC Helpers - Wildlife detection and classification inference tools."""

from importlib.metadata import version

from .awc_inference import (
    DetectAndClassify,
    SpeciesClasInference,
    format_md_detections,
    load_classification_model,
    AWCResult
)
from .format_utils import (
    output_csv,
    output_timelapse_json,
    truncate_float,
    truncate_float_array,
    get_all_image_paths,
    visualize_detections,
    get_time_identifier
)
from .math_utils import crop_image, pil_to_tensor

__version__ = version("awc_helpers")

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
    "get_all_image_paths",
    "visualize_detections",
    "AWCResult",
    "get_time_identifier"
]
