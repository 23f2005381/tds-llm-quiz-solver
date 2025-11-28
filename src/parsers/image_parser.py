# =============================================================================
# FILE: src/parsers/image_parser.py
# =============================================================================
"""
Image parser with OCR support using Tesseract and EasyOCR.
Handles scanned documents, screenshots, and image-based text extraction.
References: [web:269][web:272][web:279]
"""

import pytesseract  # Tesseract OCR[web:269]
from PIL import Image
from pathlib import Path
from typing import Dict, Any, Optional, List
import structlog

# Optional: EasyOCR for better accuracy on certain images
# Use importlib to import dynamically so static analyzers won't error if easyocr is not installed.
try:
    import importlib
    easyocr = importlib.import_module('easyocr')
    EASYOCR_AVAILABLE = True
except Exception:
    easyocr = None
    EASYOCR_AVAILABLE = False

logger = structlog.get_logger()


class ImageParser:
    """
    Parse images with OCR to extract text content.
    Supports both Tesseract and EasyOCR for comparison.
    """
    
    # Initialize EasyOCR reader (lazy loading)
    _easyocr_reader = None
    
    @staticmethod
    def parse_file(
        filepath: Path,
        method: str = 'tesseract',
        languages: List[str] = ['en']
    ) -> Dict[str, Any]:
        """
        Parse image file and extract text using OCR[web:269][web:272].
        
        Args:
            filepath: Path to image file
            method: 'tesseract' or 'easyocr' (if available)
            languages: List of language codes (e.g., ['en', 'es'])
            
        Returns:
            Dict containing extracted text and metadata
        """
        try:
            # Open image
            image = Image.open(filepath)
            
            # Convert to RGB if needed
            if image.mode not in ('RGB', 'L'):
                image = image.convert('RGB')
            
            # Choose OCR method[web:272]
            if method == 'easyocr' and EASYOCR_AVAILABLE:
                result = ImageParser._parse_with_easyocr(image, languages)
            else:
                result = ImageParser._parse_with_tesseract(image, languages)
            
            # Add image metadata
            result.update({
                'image_size': image.size,
                'image_format': image.format,
                'image_mode': image.mode,
                'filepath': str(filepath)
            })
            
            logger.info(
                "image_ocr_completed",
                filepath=str(filepath),
                method=method,
                text_length=len(result.get('text', ''))
            )
            
            return result
            
        except Exception as e:
            logger.error("image_parse_error", filepath=str(filepath), error=str(e))
            return {
                'error': str(e),
                'type': 'image',
                'filepath': str(filepath)
            }
    
    @staticmethod
    def _parse_with_tesseract(image: Image.Image, languages: List[str]) -> Dict[str, Any]:
        """
        Parse image using Tesseract OCR[web:269].
        Tesseract performs best on printed text and documents.
        
        Args:
            image: PIL Image object
            languages: List of language codes
            
        Returns:
            Dict with extracted text and confidence
        """
        lang_string = '+'.join(languages)
        
        # Basic text extraction[web:269]
        text = pytesseract.image_to_string(image, lang=lang_string)
        
        # Detailed data with bounding boxes and confidence[web:269]
        data = pytesseract.image_to_data(
            image,
            lang=lang_string,
            output_type=pytesseract.Output.DICT
        )
        
        # Calculate average confidence
        confidences = [int(conf) for conf in data['conf'] if conf != '-1']
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        # Extract words with positions
        words = []
        for i in range(len(data['text'])):
            if int(data['conf'][i]) > 0:  # Filter out low confidence
                words.append({
                    'text': data['text'][i],
                    'confidence': int(data['conf'][i]),
                    'bbox': {
                        'x': data['left'][i],
                        'y': data['top'][i],
                        'width': data['width'][i],
                        'height': data['height'][i]
                    }
                })
        
        return {
            'type': 'image_ocr',
            'method': 'tesseract',
            'text': text.strip(),
            'average_confidence': round(avg_confidence, 2),
            'words': words,
            'word_count': len([w for w in words if w['text'].strip()])
        }
    
    @staticmethod
    def _parse_with_easyocr(image: Image.Image, languages: List[str]) -> Dict[str, Any]:
        """
        Parse image using EasyOCR[web:272][web:279].
        EasyOCR performs better on complex layouts and handwriting.
        
        Args:
            image: PIL Image object
            languages: List of language codes
            
        Returns:
            Dict with extracted text and confidence
        """
        # Initialize reader if not already done
        if ImageParser._easyocr_reader is None:
            ImageParser._easyocr_reader = easyocr.Reader(languages, gpu=False)
        
        # Perform OCR
        results = ImageParser._easyocr_reader.readtext(image, detail=1)
        
        # Extract text and metadata
        all_text = []
        words = []
        confidences = []
        
        for (bbox, text, confidence) in results:
            all_text.append(text)
            confidences.append(confidence)
            
            words.append({
                'text': text,
                'confidence': round(confidence * 100, 2),  # Convert to percentage
                'bbox': {
                    'points': bbox  # 4-point polygon
                }
            })
        
        full_text = ' '.join(all_text)
        avg_confidence = (sum(confidences) / len(confidences) * 100) if confidences else 0
        
        return {
            'type': 'image_ocr',
            'method': 'easyocr',
            'text': full_text.strip(),
            'average_confidence': round(avg_confidence, 2),
            'words': words,
            'word_count': len(words)
        }
    
    @staticmethod
    def preprocess_image(image: Image.Image) -> Image.Image:
        """
        Preprocess image for better OCR results[web:269].
        Applies grayscale conversion and optional contrast enhancement.
        
        Args:
            image: PIL Image object
            
        Returns:
            Preprocessed image
        """
        from PIL import ImageEnhance
        
        # Convert to grayscale
        if image.mode != 'L':
            image = image.convert('L')
        
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(2.0)
        
        return image
