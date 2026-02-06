# AWC Helpers - Technical Documentation

This document provides detailed technical documentation for the main classes and functions in the `awc_helpers` library.

## Table of Contents

- [DetectAndClassify](#detectandclassify)
  - [Initialization](#detectandclassify-initialization)
  - [predict()](#detectandclassifypredict)
- [SpeciesClasInference](#speciesclasinference)
  - [Initialization](#speciesclasinference-initialization)
  - [predict_batch()](#speciesclasinferencepredict_batch)
- [Utility Functions](#utility-functions)
  - [get_all_image_paths()](#get_all_image_paths)
  - [visualize_detections()](#visualize_detections)

---

## DetectAndClassify

End-to-end wildlife detection and classification pipeline that combines MegaDetector for animal detection with a species classifier.

### DetectAndClassify Initialization

```python
DetectAndClassify(
    detector_path: str,
    classifier_path: str,
    label_names: List[str],
    classifier_base: str = 'tf_efficientnet_b5.ns_jft_in1k',
    detection_threshold: float = 0.1,
    clas_threshold: float = 0.5,
    resize_size: int = 300,
    force_cpu: bool = False,
    skip_clas_errors: bool = True
)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `detector_path` | `str` | *required* | Path to the MegaDetector model weights (.pt file) |
| `classifier_path` | `str` | *required* | Path to the species classifier weights (.pth file) |
| `label_names` | `List[str]` | *required* | List of species class names matching the classifier output |
| `classifier_base` | `str` | `'tf_efficientnet_b5.ns_jft_in1k'` | Name of the base timm model architecture |
| `detection_threshold` | `float` | `0.1` | Minimum confidence threshold for animal detections |
| `clas_threshold` | `float` | `0.5` | Minimum confidence threshold for classification predictions |
| `resize_size` | `int` | `300` | Target image size for classification model input |
| `force_cpu` | `bool` | `False` | If True, use CPU even if CUDA is available |
| `skip_clas_errors` | `bool` | `True` | If True, skip classification errors instead of raising exceptions |

#### Example

```python
from awc_helpers import DetectAndClassify

pipeline = DetectAndClassify(
    detector_path="models/md_v1000.0.0-redwood.pt",
    classifier_path="models/awc-135-v1.pth",
    label_names=["kangaroo", "wallaby", "wombat", "echidna"],
    detection_threshold=0.1,
    clas_threshold=0.5,
)
```

---

### DetectAndClassify.predict()

Run detection and classification on input images. Processes images through MegaDetector to find animals, then classifies each detected animal using the species classifier.

```python
predict(
    inp: Union[str, Image.Image, List[Union[str, Image.Image]]],
    identifier: Union[str, List[str], None] = None,
    clas_bs: int = 4,
    topn: int = 1,
    output_name: str = None,
    show_progress: bool = False
) -> List[Tuple]
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `inp` | `str`, `PIL.Image`, or `List` | *required* | Single image or list of images. Can be file paths or PIL Image objects |
| `identifier` | `str`, `List[str]`, or `None` | `None` | Optional identifier(s) for tracking results. If None, uses file paths or timestamps |
| `clas_bs` | `int` | `4` | Batch size for classification inference |
| `topn` | `int` | `1` | Number of top classification predictions to return per detection |
| `output_name` | `str` | `None` | If provided, saves results to CSV and JSON files instead of returning |
| `show_progress` | `bool` | `False` | If True, display tqdm progress bars |

#### Returns

`List[Tuple]` - List of result tuples, one per detected animal.

Each tuple has the format:
```
(identifier, bbox_conf, bbox, label1, prob1, label2, prob2, ...)
```

| Field | Type | Description |
|-------|------|-------------|
| `identifier` | `str` | Image path or custom identifier |
| `bbox_conf` | `float` | Detection confidence score |
| `bbox` | `Tuple[float, float, float, float]` | Normalized bounding box (x, y, width, height) |
| `labelN` | `str` | Classification label (up to `topn` pairs) |
| `probN` | `float` | Classification confidence |

#### Example

```python
from awc_helpers import DetectAndClassify

pipeline = DetectAndClassify(
    detector_path="models/md_v1000.0.0-redwood.pt",
    classifier_path="models/awc-135-v1.pth",
    label_names=["kangaroo", "wallaby", "wombat"]
)

# Single image
results = pipeline.predict("path/to/image.jpg")

# Multiple images with progress bar
results = pipeline.predict(
    inp=["image1.jpg", "image2.jpg", "image3.jpg"],
    clas_bs=8,
    topn=2,
    show_progress=True
)

# Save results to file
pipeline.predict(
    inp=["image1.jpg", "image2.jpg"],
    output_name="results"  # Creates results.csv and results.json
)
```

#### Example Output

```python
[
    ('path/to/image1.jpg', 0.9234, (0.123, 0.456, 0.234, 0.345), 'kangaroo', 0.8721),
    ('path/to/image1.jpg', 0.8567, (0.567, 0.234, 0.189, 0.278), 'wallaby', 0.7234),
    ('path/to/image2.jpg', 0.9012, (0.234, 0.345, 0.156, 0.198), 'wombat', 0.9156),
]
```

---

## SpeciesClasInference

Species classification inference engine for wildlife images. Use this class when you already have bounding boxes from MegaDetector and only need to run classification.

### SpeciesClasInference Initialization

```python
SpeciesClasInference(
    classifier_path: str,
    classifier_base: str,
    label_names: List[str] = None,
    prob_round: int = 4,
    clas_threshold: float = 0.5,
    resize_size: int = 300,
    force_cpu: bool = False,
    use_fp16: bool = False,
    skip_errors: bool = True
)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `classifier_path` | `str` | *required* | Path to the fine-tuned model weights file |
| `classifier_base` | `str` | *required* | Name of the base timm model architecture |
| `label_names` | `List[str]` | `None` | List of class label names for output mapping |
| `prob_round` | `int` | `4` | Number of decimal places to round probabilities |
| `clas_threshold` | `float` | `0.5` | Minimum confidence threshold for including predictions |
| `resize_size` | `int` | `300` | Target image size for model input |
| `force_cpu` | `bool` | `False` | If True, use CPU even if CUDA is available |
| `use_fp16` | `bool` | `False` | If True, use FP16 mixed precision (CUDA only) |
| `skip_errors` | `bool` | `True` | If True, skip images that fail processing |

#### Example

```python
from awc_helpers import SpeciesClasInference

classifier = SpeciesClasInference(
    classifier_path="models/awc-135-v1.pth",
    classifier_base="tf_efficientnet_b5.ns_jft_in1k",
    label_names=["kangaroo", "wallaby", "wombat"],
    clas_threshold=0.5,
    use_fp16=True,
)
```

---

### SpeciesClasInference.predict_batch()

Run classification inference on a batch of pre-cropped inputs.

```python
predict_batch(
    inputs: List[Union[
        Tuple[str, float, Tuple[float, float, float, float]],
        Tuple[Image.Image, str, float, Tuple[float, float, float, float]]
    ]],
    pred_topn: int = 1,
    batch_size: int = 1,
    show_progress: bool = False
) -> List[Tuple]
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `inputs` | `List[Tuple]` | *required* | List of input tuples (see format below) |
| `pred_topn` | `int` | `1` | Number of top predictions to return per image |
| `batch_size` | `int` | `1` | Number of images to process at once |
| `show_progress` | `bool` | `False` | If True, display a tqdm progress bar |

**Input tuple formats:**
- From file: `(img_path, bbox_confidence, bbox)`
- From PIL Image: `(PIL.Image, identifier, bbox_confidence, bbox)`

Where `bbox` is a normalized bounding box tuple `(x, y, width, height)` with values in range [0, 1].

#### Returns

`List[Tuple]` - List of result tuples with the same format as `DetectAndClassify.predict()`.

#### Example

```python
from awc_helpers import SpeciesClasInference

classifier = SpeciesClasInference(
    classifier_path="models/awc-135-v1.pth",
    classifier_base="tf_efficientnet_b5.ns_jft_in1k",
    label_names=["kangaroo", "wallaby", "wombat"],
)

# Input: list of (image_path, bbox_confidence, bbox) tuples
inputs = [
    ("image1.jpg", 0.95, (0.1, 0.2, 0.3, 0.4)),
    ("image2.jpg", 0.87, (0.2, 0.3, 0.25, 0.35)),
]

results = classifier.predict_batch(
    inputs=inputs,
    pred_topn=2,
    batch_size=16,
    show_progress=True
)
```

#### Example Output

```python
[
    ('image1.jpg', 0.95, (0.1, 0.2, 0.3, 0.4), 'kangaroo', 0.8234, 'wallaby', 0.1205),
    ('image2.jpg', 0.87, (0.2, 0.3, 0.25, 0.35), 'wombat', 0.9123),
]
```

---

## Utility Functions

### get_all_image_paths()

Recursively get all image file paths in a directory.

```python
from awc_helpers.format_utils import get_all_image_paths

get_all_image_paths(directory: str) -> List[str]
```

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `directory` | `str` | The directory to search for image files |

#### Returns

`List[str]` - List of absolute paths to image files (`.jpg`, `.jpeg`, `.png`) found within the directory and subdirectories.

#### Example

```python
from awc_helpers.format_utils import get_all_image_paths

image_paths = get_all_image_paths("/data/camera_trap_images")
print(f"Found {len(image_paths)} images")

# Use with DetectAndClassify
results = pipeline.predict(image_paths, show_progress=True)
```

#### Example Output

```python
[
    '/data/camera_trap_images/site1/IMG_0001.jpg',
    '/data/camera_trap_images/site1/IMG_0002.jpg',
    '/data/camera_trap_images/site2/photo_001.png',
    '/data/camera_trap_images/site2/subfolder/image.jpeg',
]
```

---

### visualize_detections()

Plot detection and classification results on images.

```python
from awc_helpers.format_utils import visualize_detections

visualize_detections(
    clas_results: List[Tuple],
    plot_type: str = 'full',
    common_name: bool = True,
    return_fig: bool = False,
    font_size: int = 10,
    fig_size: Tuple[int, int] = (12, 8)
) -> Optional[matplotlib.figure.Figure]
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `clas_results` | `List[Tuple]` | *required* | Result tuples from `predict()`. Must all belong to the same image |
| `plot_type` | `str` | `'full'` | `'full'` for full image with bboxes, `'crop'` for cropped detections grid |
| `common_name` | `bool` | `True` | If True and label contains `\|`, show only the part after `\|` |
| `return_fig` | `bool` | `False` | If True, return the matplotlib figure instead of displaying |
| `font_size` | `int` | `10` | Font size for labels and titles |
| `fig_size` | `Tuple[int, int]` | `(12, 8)` | Figure size in inches |

#### Returns

- If `return_fig=False`: `None` (displays the plot)
- If `return_fig=True`: `matplotlib.figure.Figure` object

#### Example

```python
from awc_helpers import DetectAndClassify
from awc_helpers.format_utils import visualize_detections

pipeline = DetectAndClassify(
    detector_path="models/md_v1000.0.0-redwood.pt",
    classifier_path="models/awc-135-v1.pth",
    label_names=["kangaroo", "wallaby", "wombat"],
)

# Get results for a single image
results = pipeline.predict("wildlife_photo.jpg", topn=1)

# Visualize with bounding boxes overlaid on full image
visualize_detections(results, plot_type='full', fig_size=(16, 12))

# Visualize as a grid of cropped detections
visualize_detections(results, plot_type='crop')

# Get figure for saving
fig = visualize_detections(results, plot_type='full', return_fig=True)
fig.savefig("detection_results.png", dpi=150, bbox_inches='tight')
```

#### Example Output

**`plot_type='full'`**: Displays the original image with green bounding boxes and labels overlaid on each detection.

![Full image visualization](images/vis_full.png)

**`plot_type='crop'`**: Displays a grid of cropped animal images, each labeled with the species and confidence.

![Cropped detections grid](images/vis_crop.png)

---

## Quick Reference

```python
from awc_helpers import DetectAndClassify
from awc_helpers.format_utils import get_all_image_paths, visualize_detections

# Initialize pipeline
pipeline = DetectAndClassify(
    detector_path="models/md_v1000.0.0-redwood.pt",
    classifier_path="models/awc-135-v1.pth",
    label_names=["species_a", "species_b", "species_c"],
)

# Get all images from a folder
images = get_all_image_paths("/path/to/images")

# Run inference
results = pipeline.predict(images, show_progress=True)

# Visualize a single image's results
single_image_results = [r for r in results if r[0] == images[0]]
visualize_detections(single_image_results, plot_type='full')

# Save results to files
pipeline.predict(images, output_name="output")  # Creates output.csv and output.json
```
