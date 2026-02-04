import csv
import json
import math
from datetime import datetime
from typing import List, Tuple, Dict, Any
from collections import OrderedDict


def truncate_float(x: float, precision: int = 3) -> float:
    """
    Truncates the fractional portion of a floating-point value to a specific number of
    floating-point digits.
    Source: https://github.com/agentmorris/MegaDetector/blob/main/megadetector/utils/ct_utils.py

    Args:
        x (float): scalar to truncate
        precision (int, optional): the number of significant digits to preserve, should be >= 1

    Returns:
        float: truncated version of [x]
    """
    return math.floor(x * (10 ** precision)) / (10 ** precision)


def truncate_float_array(arr: List[float], precision: int = 4) -> List[float]:
    return [truncate_float(x, precision) for x in arr]


def output_timelapse_json(clas_results: List[Tuple], json_name: str, label_names: List[str]):
    """
    Convert classification results to timelapse JSON format.
    
    Args:
        clas_results: List of result tuples, one per detected animal. Each tuple contains:
            (identifier, bbox_conf, bbox, label1, prob1, label2, prob2, ...) where the
            number of label/prob pairs depends on pred_topn and clas_threshold.
        json_name: Output JSON file name.
        label_names: List of all label names.
    """
    if not json_name.endswith('.json'):
        json_name += '.json'
    
    # Group detections by file using OrderedDict to preserve order
    images_dict: Dict[str, List[Dict[str, Any]]] = OrderedDict()
    
    for result in clas_results:
        identifier = result[0]
        bbox_conf = result[1]
        bbox = result[2]
        
        # Initialize file entry if not exists
        if identifier not in images_dict:
            images_dict[identifier] = []
        
        # If bbox is None or empty, this image has no detections
        if bbox is None or bbox_conf is None:
            continue
        
        # Build detection object
        detection = {
            "category": "1",  # Always "1" for animal
            "conf": truncate_float(bbox_conf, precision=3),
            "bbox": truncate_float_array(list(bbox), precision=4)
        }
        
        clas2idx = {name: str(i + 1) for i, name in enumerate(label_names)}

        classifications = []
        for i in range(3, len(result), 2):
            if i + 1 < len(result):
                label_str = result[i]
                prob = result[i + 1]
                if label_str is not None and prob is not None:
                    classifications.append([clas2idx[label_str], truncate_float(prob, precision=3)])
        
        if classifications:
            detection["classifications"] = classifications
        
        images_dict[identifier].append(detection)
    
    # Build images list
    images = []
    for file_path, detections in images_dict.items():
        images.append({
            "file": file_path,
            "detections": detections
        })
    
    idx2clas = {str(i + 1): name for i, name in enumerate(label_names)}
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # Build output structure
    output = {
        "images": images,
        "detection_categories": {
            "1": "animal",
            "2": "person",
            "3": "vehicle"
        },
        "info": {
            "detection_completion_time": current_time,
            "format_version": "1.4",
            "detector": "md_v1000.0.0-redwood.pt",
            "detector_metadata": {
            "megadetector_version": "1000-redwood"
            },
            "python_library": "awc-helpers"
        },
        "classification_categories": idx2clas
    }
    
    # Write to file
    with open(json_name, 'w') as f:
        json.dump(output, f, indent=1)

def output_csv(clas_results: List[Tuple],csv_name: str):
    """
    Convert classification results to CSV format.
    Args:
        clas_results: List of result tuples, one per detected animal. Each tuple contains:
            (identifier, bbox_conf, bbox, label1, prob1, label2, prob2, ...) where the
            number of label/prob pairs depends on pred_topn and clas_threshold.
        csv_name: Output CSV file name.
    """
    if not csv_name.endswith('.csv'):
        csv_name += '.csv'

    # Determine the maximum number of label/prob pairs
    max_pairs = 0
    for result in clas_results:
        num_pairs = (len(result) - 3) // 2
        if num_pairs > max_pairs:
            max_pairs = num_pairs

    # Create CSV header
    header = ['Image Path', 'Bounding Box Confidence', 'Bounding Box Normalized']
    for i in range(1, max_pairs + 1):
        header.append(f'Label {i}')
        header.append(f'Confidence {i}')

    # Write to CSV
    with open(csv_name, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(header)
        for result in clas_results:
            row = list(result)
            # Pad the row with empty strings if necessary
            while len(row) < 3 + 2 * max_pairs:
                row.append('')
            writer.writerow(row)