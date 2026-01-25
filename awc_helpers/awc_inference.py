
"""Wildlife species detection and classification inference module.

This module provides classes and functions for running inference pipelines
that combine MegaDetector-based animal detection with species classification
using fine-tuned image classification models.

Classes:
    SpeciesClasInference: Run species classification on pre-detected animal crops.
    DetectAndClassify: End-to-end pipeline combining detection and classification.

Functions:
    format_md_detections: Format MegaDetector outputs for classification input.
    load_classification_model: Load a timm-based classification model.
"""

from zoneinfo import ZoneInfo
import datetime
import timm
import torch
import numpy as np
from pathlib import Path
from megadetector.detection import run_detector
from typing import List, Tuple, Union
from PIL import Image
from .math_utils import crop_image, pil_to_tensor
import logging

logger = logging.getLogger(__name__)



def format_md_detections(md_result: dict,
                         filter_category: str = 'animal',
                         for_clas: bool = True) -> List:
    """
    Format MegaDetector outputs for classification input or other uses.

    Args:
        md_result: Dictionary containing MegaDetector detection results with keys
            'file', 'detections', and optionally 'PIL' for in-memory images.
        filter_category: Category to filter detections by (e.g., 'animal', 'person').
            If None or empty, all detections are included.
        for_clas: If True, format output for classification pipeline input.
            Otherwise, format output with full detection metadata.

    Returns:
        List of formatted detection results. Format depends on `for_clas`:
        - If for_clas=True: List of (source, bbox) or (PIL, file, bbox) tuples
        - If for_clas=False: List of [file, category, bbox, confidence] lists
    """
    md_animal_id = next((k for k, v in run_detector.DEFAULT_DETECTOR_LABEL_MAP.items() if v == filter_category), None)
    results=[]
    img_file = md_result['file']
    if 'detections' in md_result and md_result['detections'] is not None and len(md_result['detections'])>0:
        for i,_d in enumerate(md_result['detections']):
            if not filter_category or _d['category'] == md_animal_id:
                if for_clas:
                    if 'PIL' in md_result:
                        results.append((md_result['PIL'], img_file, tuple(_d['bbox'])))
                    else:
                        results.append((Path(img_file).as_posix(),tuple(_d['bbox'])))
                else:
                    results.append([Path(img_file).as_posix(),_d['category'],tuple(_d['bbox']),_d['conf']])
    return results

def load_classification_model(
    finetuned_model: str = None,
    classification_model: str = 'tf_efficientnet_b5.ns_jft_in1k',
    label_info: Union[List[str], int] = None
):
    """
    Load a timm-based image classification model.

    Creates a classification model using the timm library, optionally loading
    fine-tuned weights from a checkpoint file.

    Args:
        finetuned_model: Path to fine-tuned model weights (.pth file).
            If None, loads pretrained ImageNet weights.
        classification_model: Name of the timm model architecture.
            Hyphens are automatically converted to underscores.
        label_info: Either a list of class label names, or an integer
            specifying the number of output classes.

    Returns:
        torch.nn.Module: The loaded classification model.

    Raises:
        FileNotFoundError: If finetuned_model path does not exist.
    """
    # Convert model name format for timm
    timm_model_name = classification_model.replace('-', '_')
    num_classes = label_info if isinstance(label_info, int) else len(label_info)
    if finetuned_model is not None:
        # Create model with timm (without pretrained weights)
        model = timm.create_model(timm_model_name, pretrained=False, num_classes=num_classes)
        # Load fine-tuned weights
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        state_dict = torch.load(finetuned_model, map_location=device)
        ret = model.load_state_dict(state_dict, strict=False)
        if len(ret.missing_keys):
            logger.warning(f'Missing weights: {ret.missing_keys}')
        if len(ret.unexpected_keys):
            logger.warning(f'Unexpected weights: {ret.unexpected_keys}')
        logger.info(f'Loaded finetuned timm classification model: {Path(finetuned_model).name} with {num_classes} classes')
    else:
        model = timm.create_model(timm_model_name, pretrained=True, num_classes=num_classes)
        logger.info(f'Loaded pretrained timm classification model: {timm_model_name} with {num_classes} classes')
    return model

class SpeciesClasInference:
    """
    Species classification inference engine for wildlife images.

    This class handles loading a classification model and running inference
    on cropped animal detections. It supports batch processing, GPU acceleration,
    and mixed-precision inference.

    Attributes:
        device: PyTorch device (cuda or cpu) for model inference.
        model: The loaded classification model.
        label_names: List of class label names.
        clas_threshold: Minimum confidence threshold for predictions.
        pred_topn: Number of top predictions to return per image.
        prob_round: Decimal places to round probabilities.
        use_fp16: Whether to use FP16 mixed precision inference.
        resize_size: Target size for image resizing before inference.
        skip_errors: Whether to skip images that fail to process.

    Example:
        >>> classifier = SpeciesClasInference(
        ...     classifier_path='model.pth',
        ...     classifier_base='tf_efficientnet_b5.ns_jft_in1k',
        ...     label_names=['cat', 'dog', 'bird']
        ... )
        >>> results = classifier.predict_batch([(image_path, bbox)])
    """

    def __init__(self,
                 classifier_path: str,
                 classifier_base: str,
                 label_names: List[str] = None,
                 pred_topn: int = 1,
                 prob_round: int = 4,
                 clas_threshold: float = 0.5,
                 resize_size: int = 300,
                 force_cpu: bool = False,
                 use_fp16: bool = False,
                 skip_errors: bool = True):
        """
        Initialize the species classification inference engine.
        """
        if torch.cuda.is_available() and not force_cpu:
            self.device = torch.device('cuda')
            logger.info(f"\tGPU Device: {torch.cuda.get_device_name()}")
        else:
            self.device = torch.device('cpu')

        self.model = load_classification_model(finetuned_model=classifier_path,
                                               classification_model=classifier_base,
                                               label_info=label_names)
        self.label_names = label_names
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.clas_threshold = clas_threshold
        self.pred_topn=pred_topn
        self.prob_round=prob_round
        self.use_fp16=use_fp16 and self.device.type=='cuda'
        self.resize_size=resize_size
        self.skip_errors=skip_errors

    def _prepare_crop(
        self,
        source: Union[str, Image.Image],
        bbox_norm: Tuple[float, float, float, float]
    ) -> Image.Image:
        """
        Load (if path) and crop image to bounding box.
        
        Args:
            source: Image path string OR PIL Image
            bbox_norm: Normalized bounding box (x_min, y_min, width, height)
            
        Returns:
            Cropped RGB PIL Image
        """
        if isinstance(source, str):
            with Image.open(source) as img:
                img.load()
                img = img.convert('RGB') if img.mode != 'RGB' else img
                return crop_image(img, bbox_norm, square_crop=True)
        else:
            img = source.convert('RGB') if source.mode != 'RGB' else source
            return crop_image(img, bbox_norm, square_crop=True)
    
    def _predict(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Run classification model on input tensor.
        
        Args:
            input_tensor: Tensor of shape (B, 3, H, W)
            
        Returns:
            Tuple of 
            - Probabilities tensor of shape (B, num_classes)
            - Indices tensor of shape (B, num_classes)
        """
        with torch.no_grad():
            if self.use_fp16:
                with torch.amp.autocast('cuda'):
                    logits = self.model(input_tensor)
            else:
                logits = self.model(input_tensor)
            
            # Softmax in fp32 for numerical stability
            probs = torch.nn.functional.softmax(logits.float(), dim=1)

            top_probs, top_indices = torch.topk(probs, k=self.pred_topn, dim=1)
            return (top_probs.cpu().numpy().round(self.prob_round),
                    top_indices.cpu().numpy())    

    def _format_output(self,
                       identifier: Union[str, None],
                       bbox_norm: Tuple[float, float, float, float],
                       top_probs: np.ndarray,
                       top_indices: np.ndarray):
        """
        Build output result tuple from predictions.
        
        Args:
            identifier: Image path or custom ID
            bbox_norm: Bounding box used
            top_probs: Top-k probabilities (1D array)
            top_indices: Top-k class indices (1D array)
            
        Returns:
            Tuple: (identifier, bbox, label1, prob1, label2, prob2, ...)
        """
        result = [identifier, bbox_norm]
        for k in range(len(top_indices)):
            label = self.label_names[top_indices[k]]
            prob = round(float(top_probs[k]), self.prob_round)
            if prob >= self.clas_threshold:
                result.extend([label, prob])
            
        return tuple(result)

    def predict_batch(
        self,
        inputs: List[Union[Tuple[str, Tuple[float, float, float, float]], Tuple[Image.Image, str, Tuple[float, float, float, float]]]],
        batch_size: int = 16,
    ) -> List[Tuple]:
        """
        Run inference on a batch of inputs.
        
        Args:
            inputs: List of (source, bbox_norm) tuples where source is path, or (PIL Image, id, bbox_norm) tuples for streaming 
            pred_topn: Number of top predictions to return
            prob_round: Decimal places to round probabilities
            batch_size: Number of images to process at once
            
        Returns:
            List of result tuples
        """
        results = []
        
        for batch_start in range(0, len(inputs), batch_size):
            batch_inputs = inputs[batch_start:batch_start + batch_size]
            
            # Preprocess batch
            batch_tensors = []
            batch_metadata = []  # (identifier, bbox)
            
            for *sources, bbox in batch_inputs:
                try:
                    identifier = sources[0] if len(sources)==1 else sources[1]
                    cropped = self._prepare_crop(sources[0], bbox)
                    tensor = pil_to_tensor(cropped,resize_size=self.resize_size).to(self.device)
                    batch_tensors.append(tensor)
                    batch_metadata.append((identifier, bbox))
                except Exception as e:
                    if self.skip_errors:
                        logger.warning(f"Failed to process {identifier}: {e}")
                        continue
                    raise
            
            if not batch_tensors:
                continue
            
            # Stack and run inference
            batch_tensor = torch.cat(batch_tensors, dim=0)
            top_probs, top_indices = self._predict(batch_tensor)
            
            # Build results
            for i, (identifier, bbox) in enumerate(batch_metadata):
                result = self._format_output(
                    identifier, bbox,
                    top_probs[i], top_indices[i],
                )
                results.append(result)
        
        return results
    

class DetectAndClassify:
    """
    End-to-end wildlife detection and classification pipeline.

    Combines MegaDetector for animal detection with a species classifier
    to provide a complete inference pipeline from raw images to species
    predictions.

    Attributes:
        md_detector: MegaDetector model instance for animal detection.
        clas_inference: SpeciesClasInference instance for classification.
        detection_threshold: Minimum confidence for detection filtering.

    Example:
        >>> pipeline = DetectAndClassify(
        ...     detector_path='md_v5a.0.0.pt',
        ...     classifier_path='species_model.pth',
        ...     label_names=['kangaroo', 'wallaby', 'wombat']
        ... )
        >>> results = pipeline.predict('wildlife_image.jpg')
    """

    def __init__(self,
                 detector_path: str,
                 classifier_path: str,
                 label_names: List[str],
                 classifier_base: str = 'tf_efficientnet_b5.ns_jft_in1k',
                 detection_threshold: float = 0.1,
                 clas_threshold: float = 0.5,
                 pred_topn: int = 1,
                 resize_size: int = 300,
                 force_cpu: bool = False,
                 skip_clas_errors: bool = True):
        """
        Initialize the detection and classification pipeline.

        Args:
            detector_path: Path to the MegaDetector model weights.
            classifier_path: Path to the species classifier weights.
            label_names: List of species class names.
            classifier_base: Name of the base timm model architecture.
            detection_threshold: Minimum confidence for animal detections.
            clas_threshold: Minimum confidence for classification predictions.
            pred_topn: Number of top classification predictions to return.
            resize_size: Target image size for classification model input.
            force_cpu: If True, use CPU even if CUDA is available.
            skip_clas_errors: If True, skip classification errors instead of raising.
        """
        self.md_detector = run_detector.load_detector(str(detector_path),
                                                      force_cpu=force_cpu)
        self.clas_inference = SpeciesClasInference(classifier_path=classifier_path,
                                                   classifier_base=classifier_base,
                                                   clas_threshold=clas_threshold,
                                                   label_names=label_names,
                                                   pred_topn=pred_topn,
                                                   resize_size=resize_size,
                                                   force_cpu=force_cpu,
                                                   skip_errors=skip_clas_errors)
        self.detection_threshold = detection_threshold
    
    def _validate_input(
        self,
        inp: Union[str, Image.Image, List[Union[str, Image.Image]]],
        identifier: Union[str, List[str], None]
    ) -> Tuple[List, List]:
        """
        Validate and normalize input images and identifiers.

        Args:
            inp: Single image or list of images (paths or PIL Images).
            identifier: Optional identifier(s) for the images. If None,
                uses file paths for string inputs or timestamps for PIL images.

        Returns:
            Tuple of (normalized_inputs, normalized_identifiers)

        Raises:
            AssertionError: If identifier list length doesn't match input list length.
        """
        if not isinstance(inp, (list, tuple)):
            inp = [inp]
        elif len(inp)==0:
            return [],[]
        if identifier is None:
            if isinstance(inp[0], str):
                identifier = inp
            else:
                # identifier based on date+time in human readable format, utc time
                now = datetime.datetime.now(ZoneInfo("Australia/Perth"))
                now_str = now.strftime("%Y%m%d_%H%M%S") + f"_{now.microsecond // 1000:03d}"
                identifier = [now_str] if len(inp) == 1 else [f'{now_str}_{i+1}' for i in range(len(inp))]

        elif not isinstance(identifier, (list, tuple)):
            identifier = [identifier]
        
        assert len(identifier) == len(inp), "Length of identifier list (containing e.g. image names) must match length of input list."
        return inp, identifier
    
    def predict(
        self,
        inp: Union[str, Image.Image, List[Union[str, Image.Image]]],
        identifier: Union[str, List[str], None] = None,
        clas_bs: int = 4
    ) -> List[Tuple]:
        """
        Run detection and classification on input images.

        Processes images through the MegaDetector to find animals, then
        classifies each detected animal using the species classifier.

        Args:
            inp: Single image or list of images. Can be file paths (str)
                or PIL Image objects.
            identifier: Optional identifier(s) for tracking results back to
                source images. If None, uses file paths or timestamps.
            clas_bs: Batch size for classification inference.

        Returns:
            List of result tuples, one per detected animal. Each tuple contains:
            (identifier, bbox, label1, prob1, label2, prob2, ...) where the
            number of label/prob pairs depends on pred_topn and clas_threshold.
        """
        inp, identifier = self._validate_input(inp, identifier)
        if len(inp) == 0:
            return []
        
        md_results=[]
        for item,id in zip(inp, identifier):
            img = item
            if isinstance(item,str):
                img = Image.open(item)
            try:
                md_result = self.md_detector.generate_detections_one_image(img,id,
                                                    detection_threshold=self.detection_threshold)
                if not isinstance(item,str):
                    md_result['PIL'] = img
                md_results.extend(format_md_detections(md_result))
            finally:
                if isinstance(item,str):
                    img.close()
                    
        clas_results = self.clas_inference.predict_batch(md_results, batch_size=clas_bs)
        return clas_results
