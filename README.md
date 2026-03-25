# AWC Helpers

Wildlife detection and species classification inference tools combining MegaDetector with custom species classifiers.

## Installation

### 1. Install PyTorch

**Windows (with CUDA GPU):**
```bash
pip install "torch<=2.9.1" "torchvision<=0.24.1" --index-url https://download.pytorch.org/whl/cu128
```

**Linux / Mac / Windows with CPU:**
```bash
pip install "torch<=2.9.1" "torchvision<=0.24.1"
```

### 2. Install AWC Helpers

**From PyPI:**
```bash
pip install awc-helpers
```

## Usage

```python
from awc_helpers import DetectAndClassify

# Initialize the pipeline
pipeline = DetectAndClassify(
    detector_path="models/md_v1000.0.0-redwood.pt",
    classifier_path="models/awc-135-v1.pth",
    label_names=["species_a", "species_b", "species_c"],
    detection_threshold=0.1,
    clas_threshold=0.5,
)

# Run inference on image paths
results = pipeline.predict(
    inp=["path/to/image1.jpg", "path/to/image2.jpg"],
    clas_bs=4
)

for result in results:
    print(result)
# print example:
# AWCResult(identifier='"path/to/image1.jpg"', bbox=(0.1, 0.2, 0.3, 0.4), bbox_label='animal', bbox_conf=0.95, labels_probs=(('kangaroo', 0.87),))

```

## Documenntation
Refer to [this](DOCUMENTATION.md) for more details

## License

This project is licensed under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](LICENSE) (CC BY-NC-SA 4.0).

**Non-commercial use only. Derivative works must use the same license.**
