"""
Base extractor class for document processing
"""
import logging
from typing import Dict, List, Any

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