# AWC Helpers

Wildlife detection and species classification inference tools combining MegaDetector with custom species classifiers.

## Installation

```bash
pip install git+https://github.com/awc-admin/awc_inference.git
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
    inp=["image1.jpg", "image2.jpg"],
    clas_bs=4
)

# Results format: [(identifier, bbox, label, confidence), ...]
for result in results:
    print(result)
```

## License

This project is licensed under the [Creative Commons Attribution-NonCommercial 4.0 International License](LICENSE) (CC BY-NC 4.0).

**Non-commercial use only.**
