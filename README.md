# AWC Helpers

Wildlife detection and species classification inference tools combining MegaDetector with custom species classifiers.

## Installation

### 1. Install PyTorch

**Windows (with CUDA GPU):**
```bash
pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
```

**Linux / Mac / CPU:**
```bash
pip install torch==2.9.1
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
    detector_path="path/to/megadetector.pt",
    classifier_path="path/to/species_classifier.pth",
    label_names=["species_a", "species_b", "species_c"],
    detection_threshold=0.1,
    clas_threshold=0.5,
)

# Run inference on image paths
results = pipeline.predict(
    inp=["path/to/image1.jpg", "path/to/image2.jpg"],
    clas_bs=4
)

# Results format: [(image_path, bbox_confidence, bbox, label, label_confidence), ...]
for result in results:
    print(result)
# print example:
# ("path/to/image1.jpg",
#  0.804,
#  (0.2246, 0.5885, 0.0678, 0.1022),
#  'Acanthagenys rufogularis | Spiny-cheeked Honeyeater',
#  0.9948)
```

## License

This project is licensed under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](LICENSE) (CC BY-NC-SA 4.0).

**Non-commercial use only. Derivative works must use the same license.**
