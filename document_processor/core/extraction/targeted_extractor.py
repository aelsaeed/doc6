"""
Base targeted extraction module using doctr for locating keywords and LayoutLMv3 for extracting specific fields
"""
import os
import logging
import traceback
from typing import Dict, List, Tuple, Any, Optional
import re
import torch
import numpy as np
import fitz  # PyMuPDF
from PIL import Image, ImageDraw, ImageFont
from collections import Counter

# Import doctr
try:
    from doctr.io import DocumentFile
    from doctr.models import ocr_predictor
    DOCTR_AVAILABLE = True
except ImportError:
    DOCTR_AVAILABLE = False

# Import LayoutLMv3
try:
    from transformers import LayoutLMv3TokenizerFast, LayoutLMv3ForTokenClassification
    LAYOUTLMV3_AVAILABLE = True
except ImportError:
    LAYOUTLMV3_AVAILABLE = False

from document_processor.utils.custom_exceptions import (
    FileTypeError, FileReadError, TextExtractionError, EmptyTextError
)
from document_processor.utils.gpu_utils import check_gpu_availability
from document_processor.utils.validation import validate_file
from document_processor.core.extraction.base_extractor import BaseDocumentExtractor
from document_processor.core.extraction.document_extractors import DocumentExtractorFactory

logger = logging.getLogger(__name__)

class BaseDocumentExtractor:
    """Base class for all document type extractors"""
    
    def __init__(self, doctr_model=None):
        """
        Initialize the base document extractor
        
        Args:
            doctr_model: Pre-initialized doctr model (optional)
        """
        self.doctr_model = doctr_model
        self.device = check_gpu_availability()
    
    def extract_fields(self, words, coordinates, combined_text):
        """
        Must be implemented by subclasses
        
        Args:
            words (List[str]): List of words extracted from the document
            coordinates (List): List of word coordinates
            combined_text (str): Full text of the document
            
        Returns:
            Dict[str, str]: Extracted fields
        """
        raise NotImplementedError("Subclasses must implement extract_fields")
        
    def get_field_schema(self):
        """
        Return the expected fields for this document type
        
        Returns:
            List[str]: List of field names
        """
        raise NotImplementedError("Subclasses must implement get_field_schema")
    
    def create_word_map(self, words, coordinates):
        """
        Create a mapping of words with their positions
        
        Args:
            words (List[str]): List of words
            coordinates (List): List of coordinates for each word
            
        Returns:
            List[Dict]: List of word info dictionaries
        """
        word_map = []
        for i, (word, bbox) in enumerate(zip(words, coordinates)):
            # Calculate center point
            (x0, y0), (x1, y1) = bbox
            center_x = (x0 + x1) / 2
            center_y = (y0 + y1) / 2
            word_map.append({
                "word": word,
                "bbox": bbox,
                "center": (center_x, center_y),
                "index": i
            })
        return word_map


class TargetedExtractor:
    """
    Extractor that scans for specific keywords using doctr and extracts values using the appropriate document extractor
    """
    
    def __init__(self):
        """Initialize targeted extractor"""
        # Initialize device
        self.device = check_gpu_availability()
        logger.info(f"Targeted extractor initialized with device: {self.device}")
        
        # Initialize doctr if available
        self.doctr_model = None
        if DOCTR_AVAILABLE:
            try:
                logger.info("Initializing doctr OCR model")
                self.doctr_model = ocr_predictor(
                    det_arch='db_resnet50',
                    reco_arch='crnn_vgg16_bn',
                    pretrained=True
                )
                self.doctr_model.det_predictor.model = self.doctr_model.det_predictor.model.to(self.device)
                self.doctr_model.reco_predictor.model = self.doctr_model.reco_predictor.model.to(self.device)
                logger.info("doctr OCR model initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize doctr OCR model: {str(e)}")
                self.doctr_model = None
        else:
            logger.warning("doctr not available. Install with 'pip install python-doctr'")
        
        # Initialize LayoutLMv3 if available
        self.tokenizer = None
        self.layoutlmv3_model = None
        if LAYOUTLMV3_AVAILABLE:
            try:
                logger.info("Initializing LayoutLMv3 model")
                self.tokenizer = LayoutLMv3TokenizerFast.from_pretrained('microsoft/layoutlmv3-base')
                self.layoutlmv3_model = LayoutLMv3ForTokenClassification.from_pretrained('microsoft/layoutlmv3-base')
                self.layoutlmv3_model.to(self.device)
                logger.info("LayoutLMv3 model initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize LayoutLMv3 model: {str(e)}")
                self.layoutlmv3_model = None
        else:
            logger.warning("LayoutLMv3 not available. Install with 'pip install transformers'")
    
    def extract_doctr_data(self, file_path: str) -> Dict[str, Any]:
        """
        Extract text and coordinates using doctr for visualization
        
        Args:
            file_path (str): Path to document file
            
        Returns:
            Dict[str, Any]: Dictionary with extracted text, words and coordinates
        """
        try:
            # Validate file
            result, error = validate_file(file_path)
            if not result:
                raise error
            
            # Check if we have the required model
            if self.doctr_model is None:
                logger.warning("doctr model not available, using basic extraction")
                return self._extract_using_basic(file_path)
            
            # Load document based on file type
            ext = os.path.splitext(file_path)[1].lower()
            if ext == '.pdf':
                doc = DocumentFile.from_pdf(file_path)
            else:
                doc = DocumentFile.from_images(file_path)
            
            # Process the document with doctr
            logger.info(f"Processing document: {file_path}")
            result = self.doctr_model(doc)
            
            # Extract all words and their coordinates
            all_words = []
            all_coords = []
            words_with_coords = []
            combined_text = ""
            
            for page_idx, page in enumerate(result.pages):
                for block in page.blocks:
                    for line in block.lines:
                        for word in line.words:
                            all_words.append(word.value)
                            all_coords.append(word.geometry)
                            words_with_coords.append({
                                "text": word.value,
                                "coords": word.geometry,
                                "page": page_idx + 1
                            })
                            combined_text += word.value + " "
            
            # Create visualization if words were extracted
            debug_path = None
            if all_words and (ext == '.pdf' or ext in ['.jpg', '.jpeg', '.png']):
                # If PDF, convert first page to image if it doesn't exist
                if ext == '.pdf':
                    image_path = file_path.replace('.pdf', '_page_1.jpg')
                    if not os.path.exists(image_path):
                        with fitz.open(file_path) as pdf:
                            if len(pdf) > 0:
                                page = pdf[0]
                                pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
                                pix.save(image_path)
                else:
                    image_path = file_path
                
                # Create debug visualization
                debug_path = self.visualize_bboxes(image_path, all_coords, all_words)
                logger.info(f"Created bounding box visualization: {debug_path}")
            
            return {
                "text": combined_text,
                "words": all_words,
                "coords": all_coords,
                "words_with_coords": words_with_coords,
                "visualization_path": debug_path
            }
            
        except Exception as e:
            logger.error(f"Error in doctr extraction for {file_path}: {str(e)}")
            logger.error(traceback.format_exc())
            return {"text": "", "words": [], "coords": [], "words_with_coords": []}
    
    def extract_fields(self, file_path: str, doc_type: str) -> Dict[str, Any]:
        """
        Extract specific fields based on document type
        
        Args:
            file_path (str): Path to document file
            doc_type (str): Document type
            
        Returns:
            Dict[str, Any]: Dictionary with extracted fields
        """
        # Extract doctr data (words and coordinates)
        doctr_result = self.extract_doctr_data(file_path)
        
        # Get the appropriate extractor for this document type
        extractor = DocumentExtractorFactory.get_extractor(doc_type, self.doctr_model)
        
        # Extract fields using the specialized extractor
        if doctr_result.get("words"):
            extracted_fields = extractor.extract_fields(
                doctr_result.get("words", []),
                doctr_result.get("coords", []),
                doctr_result.get("text", "")
            )
        else:
            extracted_fields = {}
        
        # Ensure all expected fields are present, even if not found
        field_schema = extractor.get_field_schema()
        for field in field_schema:
            if field not in extracted_fields:
                extracted_fields[field] = "Not found"
        
        # Add visualization path if available
        if "visualization_path" in doctr_result:
            extracted_fields["_debug_visualization"] = doctr_result["visualization_path"]
        
        return {
            "extracted_fields": extracted_fields,
            "document_type": doc_type
        }
    
    def _extract_using_basic(self, file_path: str, doc_type: str = None) -> Dict[str, Any]:
        """
        Fallback extraction using regex patterns when doctr is not available
        
        Args:
            file_path (str): Path to document file
            doc_type (str, optional): Document type for targeted extraction
            
        Returns:
            Dict[str, Any]: Dictionary with extracted fields
        """
        try:
            import fitz  # PyMuPDF
            text = ""
            with fitz.open(file_path) as doc:
                for page in doc:
                    text += page.get_text()
            
            if not text.strip():
                raise EmptyTextError(file_path)
            
            # Log full text for debugging
            logger.debug(f"Basic extraction text: {text[:200]}...")
            
            # Get the appropriate extractor if doc_type is provided
            if doc_type:
                extractor = DocumentExtractorFactory.get_extractor(doc_type, None)
                field_schema = extractor.get_field_schema()
                
                # Use a simplistic approach for extraction
                extracted_fields = {}
                for field in field_schema:
                    # Try to extract using regex patterns
                    value = self._simple_regex_extract(text, field)
                    if value:
                        extracted_fields[field] = value
                    else:
                        extracted_fields[field] = "Not found"
            else:
                extracted_fields = {}
            
            return {
                "extracted_fields": extracted_fields,
                "document_type": doc_type,
                "text": text
            }
                
        except Exception as e:
            logger.error(f"Error in basic extraction for {file_path}: {str(e)}")
            raise TextExtractionError(file_path, f"Basic extraction failed: {str(e)}")
    
    def _simple_regex_extract(self, text, field_name):
        """
        Simple regex extraction for basic mode
        
        Args:
            text (str): Full text
            field_name (str): Field to extract
            
        Returns:
            str: Extracted value or None
        """
        patterns = {
            "employee_name": r'(?:employee|employee\'s)[^:]*?:\s*([A-Za-z\s.]+)',
            "employee_ssn": r'(?:SSN|social security)[^:]*?:\s*(\d{3}-\d{2}-\d{4})',
            "employer_name": r'(?:employer|employer\'s)[^:]*?:\s*([A-Za-z\s.]+)',
            "employer_ein": r'(?:ein|employer identification)[^:]*?:\s*(\d{2}-\d{7})',
            "wages": r'(?:wages|box 1)[^:]*?:\s*\$?(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
            "tax_year": r'(?:tax year|year)[^:]*?:\s*(20\d{2})'
        }
        
        pattern = patterns.get(field_name)
        if pattern:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return None
    
    def visualize_bboxes(self, image_path, bboxes, words):
        """
        Visualize bounding boxes on the image for debugging
        
        Args:
            image_path (str): Path to the image
            bboxes (List): List of bounding boxes
            words (List[str]): List of words corresponding to bounding boxes
            
        Returns:
            str: Path to the debug image
        """
        try:
            img = Image.open(image_path)
            draw = ImageDraw.Draw(img)
            
            # Get image dimensions
            width, height = img.size
            
            # Draw bounding boxes and words
            for bbox, word in zip(bboxes, words):
                (x0, y0), (x1, y1) = bbox
                # Convert normalized coordinates to pixel coordinates
                x0, y0 = int(x0 * width), int(y0 * height)
                x1, y1 = int(x1 * width), int(y1 * height)
                
                # Draw rectangle
                draw.rectangle([x0, y0, x1, y1], outline="red", width=2)
                
                # Draw word above rectangle
                draw.text((x0, y0-10), word, fill="red")
            
            # Save visualization
            debug_path = os.path.splitext(image_path)[0] + "_debug" + os.path.splitext(image_path)[1]
            img.save(debug_path)
            logger.info(f"Saved bounding box visualization to {debug_path}")
            
            return debug_path
        except Exception as e:
            logger.error(f"Error creating bounding box visualization: {str(e)}")
            return None