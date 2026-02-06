import csv
import json
import math
import os
from datetime import datetime
from typing import List, Tuple, Dict, Any
from collections import OrderedDict
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
from importlib.metadata import version
from .math_utils import bbox_to_pixels, crop_image

def get_all_image_paths(directory):
    """
    Recursively gets all image file paths in a given directory.

    Args:
    directory (str): The directory to search for image files.

    Returns:
    list: A list of paths to image files found within the directory and its subdirectories.
    """
    image_extensions = {'.jpg', '.jpeg', '.png'}
    image_paths = []

    def scan_directory(dir_path):
        with os.scandir(dir_path) as it:
            for entry in it:
                if entry.is_file() and any(entry.name.lower().endswith(ext) for ext in image_extensions):
                    image_paths.append(entry.path)
                elif entry.is_dir():
                    scan_directory(entry.path)

    scan_directory(directory)
    return image_paths

def visualize_detections(clas_results: List[Tuple],
                         plot_type: str = 'full',
                         common_name: bool = True,
                         return_fig: bool = False,
                         font_size: int = 10,
                         fig_size: Tuple[int, int] = (12, 8)):
    """
    Plot the detections and classifications 
    
    Args:
        clas_results: List of result tuples, one per detected animal. Each tuple contains:
            (identifier, bbox_conf, bbox, label1, prob1, label2, prob2, ...) where the
            number of label/prob pairs depends on pred_topn and clas_threshold. 
            The result(s) only belong to one image (with the same path)
        plot_type: 'full' to plot full image with green bboxes and label+prob, 'crop' to plot a series of cropped animals.
        common_name: Whether to show common shorter names (True) or full names (False) 
        return_fig: Whether to return the matplotlib figure object.
        font_size: Font size for labels and titles.
        fig_size: Size of the figure when plotting.
    """

    
    if not clas_results:
        print("No detections to visualize.")
        return None if return_fig else None
    if isinstance(clas_results,tuple):
        clas_results = [clas_results]

    # for multiple entry, check whether they belong to the same image
    if len(set(result[0] for result in clas_results)) > 1:
        raise ValueError("All results must belong to the same image for visualization.")
    
    img_path = clas_results[0][0]
    img = Image.open(img_path)
    img_w, img_h = img.size
    
    def _get_label_text(result):
        """Extract label and prob text from result tuple."""
        if len(result) <= 3:
            return None
        # get the first label/prob pair only
        label = result[3]
        prob = result[4]
        if common_name and '|' in label:
            label = label.split('|')[-1].strip()
        return f"{label} ({prob:.2f})"
    
    if plot_type == 'full':
        fig, ax = plt.subplots(1, 1, figsize=fig_size)
        ax.imshow(img)
        ax.axis('off')
        
        for result in clas_results:
            bbox_norm = result[2]
            if bbox_norm is None:
                continue
            
            x, y, w, h = bbox_to_pixels(bbox_norm, img_w, img_h)
            
            rect = Rectangle((x, y), w, h, linewidth=2, edgecolor='lime', facecolor='none')
            ax.add_patch(rect)
            
            # Add label if available
            label_text = _get_label_text(result)
            if label_text:
                ax.text(x, y - 5, label_text, color='lime', fontsize=font_size,
                        fontweight='normal', verticalalignment='bottom',
                        bbox=dict(boxstyle='square,pad=0.1', facecolor='black', edgecolor='none'))
        
        ax.set_title(img_path, fontsize=font_size)
        plt.tight_layout()
        
    elif plot_type == 'crop':
        n_crops = len(clas_results)
        if n_crops == 0:
            print("No detections to crop.")
            return None
        
        # Calculate grid dimensions
        n_cols = min(4, n_crops)
        n_rows = math.ceil(n_crops / n_cols)
        
        # Each square crop gets a cell_size x cell_size subplot
        cell_size = 4  # inches per subplot
        fig, axes = plt.subplots(n_rows, n_cols, 
                                 figsize=(cell_size * n_cols, cell_size * n_rows),
                                 squeeze=False)
        
        for idx, result in enumerate(clas_results):
            row, col = idx // n_cols, idx % n_cols
            ax = axes[row, col]
            
            bbox_norm = result[2]
            if bbox_norm is None:
                ax.axis('off')
                continue
            
            crop = crop_image(img, bbox_norm, square_crop=True)
            
            ax.imshow(crop)
            ax.axis('off')
            
            label_text = _get_label_text(result)
            title = label_text if label_text else f"conf: {result[1]:.2f}"
            ax.set_title(title, fontsize=font_size)
        
        # Hide unused subplots
        for idx in range(n_crops, n_rows * n_cols):
            row, col = idx // n_cols, idx % n_cols
            axes[row, col].axis('off')
        
        plt.suptitle(img_path, fontsize=font_size)
        plt.tight_layout()
    
    else:
        raise ValueError(f"Unknown plot_type: {plot_type}. Use 'full' or 'crop'.")
    
    if return_fig:
        return fig
    
    plt.show()
    plt.close(fig)


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

    try:
        ver = version("awc_helpers")
    except Exception:
        ver = "unknown"
        
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
            "python_library": f"awc-helpers-{ver}"
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
            (img_id, bbox_conf, bbox, label1, prob1, label2, prob2, ...) where the
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