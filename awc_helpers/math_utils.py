from PIL import Image, ImageOps
from typing import Sequence
import torch
import numpy as np

def crop_image(img: Image.Image, bbox_norm: Sequence[float], square_crop: bool = True) -> Image.Image:
    """
    Crop image based on normalized bounding box (MegaDetector format).
    
    Args:
        img: PIL Image to crop
        bbox_norm: [x_min, y_min, width, height] all normalized 0-1
        square_crop: Whether to make the crop square with padding
        
    Returns:
        Cropped PIL Image
    """
    img_w, img_h = img.size
    xmin = int(bbox_norm[0] * img_w)
    ymin = int(bbox_norm[1] * img_h)
    box_w = int(bbox_norm[2] * img_w)
    box_h = int(bbox_norm[3] * img_h)
    
    if square_crop:
        box_size = max(box_w, box_h)
        xmin = max(0, min(xmin - int((box_size - box_w) / 2), img_w - box_w))
        ymin = max(0, min(ymin - int((box_size - box_h) / 2), img_h - box_h))
        box_w = min(img_w, box_size)
        box_h = min(img_h, box_size)

    if box_w == 0 or box_h == 0:
        raise ValueError(f'Invalid crop dimensions (w={box_w}, h={box_h})')

    crop = img.crop(box=[xmin, ymin, xmin + box_w, ymin + box_h])

    if square_crop and (box_w != box_h):
        crop = ImageOps.pad(crop, size=(box_size, box_size), color=0)

    return crop


def pil_to_tensor(img: Image.Image,resize_size=300) -> torch.Tensor:
    """
    Convert PIL image to normalized tensor ready for model input.
    - Use SMALLER ratio so crop fits within image
    - Center crop to target aspect ratio
    - Resize crop to target size
    
    Args:
        img: PIL Image (already cropped and in RGB)
        
    Returns:
        Tensor of shape (1, 3, H, W)
    """
    target_w = target_h = resize_size
    w, h = img.size  # PIL: (width, height)
    
    # Fastai ResizeMethod.Crop uses SMALLER ratio
    # This ensures crop fits within the image bounds
    ratio_w = w / target_w
    ratio_h = h / target_h
    m = min(ratio_w, ratio_h)
    
    # Crop size that when resized will give target size
    cp_w = int(m * target_w)
    cp_h = int(m * target_h)
    
    # Center crop position (pcts = 0.5, 0.5 for validation)
    left = (w - cp_w) // 2
    top = (h - cp_h) // 2
    
    # Crop
    img = img.crop((left, top, left + cp_w, top + cp_h))
    
    # Resize to target
    img = img.resize((target_w, target_h), Image.BILINEAR)
    
    # To tensor: HWC uint8 -> NCHW float32 [0, 1]
    img_array = np.asarray(img, dtype=np.float32) / 255.0
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
    
    return img_tensor