"""
Document processor service that coordinates the document processing pipeline
"""
import os
import logging
import traceback
import threading
import shutil
from pathlib import Path
import fitz  # PyMuPDF
from PIL import Image

from document_processor.core.classification.classifier import DocumentClassifier
from document_processor.core.extraction.text_extractor import DocumentExtractorFactory, extract_document_content
from document_processor.core.extraction.document_extractors.w2_extractor import W2Extractor  # Import the improved W2 extractor
from document_processor.core.information.financial_extractor import FinancialEntityExtractor
from document_processor.utils.custom_exceptions import ProcessingError, DocumentNotSupportedError
from document_processor.utils.file_utils import get_file_extension, is_valid_document
from document_processor.core.processing_modes import ProcessingMode
from document_processor.utils.validation import validate_file

logger = logging.getLogger(__name__)

class ProcessorService:
    """
    Service for coordinating the document processing pipeline,
    including classification, text extraction, and entity extraction.
    """
    
    def __init__(self, config):
        """
        Initialize the processor service
        
        Args:
            config: Application configuration
        """
        self.config = config
        self.upload_folder = config.get('UPLOAD_FOLDER', 'uploads')
        self.models_folder = config.get('MODELS_FOLDER', 'models')
        self.static_folder = config.get('STATIC_FOLDER', 'static')
        self.documents_folder = os.path.join(self.static_folder, 'documents')
        
        # Initialize components
        self.classifier = DocumentClassifier()
        self.text_extractor = DocumentExtractorFactory  # This is a factory class, not an instance
        self.financial_extractor = FinancialEntityExtractor()
        self.w2_extractor = W2Extractor()  # Initialize the improved W2 extractor
        
        # Ensure required folders exist
        os.makedirs(self.static_folder, exist_ok=True)
        os.makedirs(self.documents_folder, exist_ok=True)
        
        logger.info("Document processor service initialized")
    
    def process_document(self, file_path, mode=ProcessingMode.OPTIMAL, doc_type=None):
        """
        Process a document through the pipeline
        
        Args:
            file_path (str): Path to the document
            mode (ProcessingMode): Processing mode to use (now always OPTIMAL)
            doc_type (str, optional): Document type (not required, will be determined automatically)
            
        Returns:
            dict: Processing results including classification, extracted text, and entities
        """
        try:
            logger.info(f"Processing document {file_path} using {mode.value} mode")
            
            # Check if file exists
            if not os.path.exists(file_path):
                raise DocumentNotSupportedError(f"File does not exist: {file_path}")
            
            # Validate document
            result, error = validate_file(file_path)
            if not result:
                raise error
            
            # Copy the document to the static folder for direct access
            document_path = self._copy_document_to_static(file_path)
            
            # First, extract the text content from the document
            extraction_result = extract_document_content(file_path)
            text = extraction_result.get('text', '')
            
            # Classify the document if doc_type is not provided
            if doc_type is None:
                doc_type, confidence = self.classifier.classify(text)
            else:
                confidence = 1.0  # If doc_type is provided, assume full confidence
            
            # Get words and coordinates for field extraction
            words, coordinates = self._extract_words_and_coordinates(file_path)
            
            # Select the appropriate extractor based on document type
            extractor = self._get_extractor_for_doc_type(doc_type)
            
            # Extract fields using the appropriate extractor
            # Pass the image path to enable visualization
            extracted_fields = extractor.extract_fields(
                words, 
                coordinates, 
                text,
                image_path=file_path
            )
            
            # Process entity extraction
            entities = self.financial_extractor.extract_entities(text)
            
            # Combine the results
            result = {
                "file_path": file_path,
                "file_name": os.path.basename(file_path),
                "doc_type": doc_type,
                "classification_confidence": confidence,
                "text": text,
                "entities": entities,
                "extracted_fields": extracted_fields,
                "processing_mode": mode.value
            }
            
            # Add tables if available
            if 'tables' in extraction_result:
                result["tables"] = extraction_result['tables']
            
            # Add visualization path if available from extractor
            if "_visualization_path" in extracted_fields:
                result["visualization_path"] = extracted_fields.pop("_visualization_path")
                # Convert to relative path for web display if needed
                if result["visualization_path"].startswith(self.static_folder):
                    result["visualization_path"] = os.path.relpath(
                        result["visualization_path"], 
                        self.static_folder
                    )
            
            # Add document path to the result
            result["document_path"] = document_path
            
            return result
                
        except Exception as e:
            logger.error(f"Error processing document {file_path}: {str(e)}")
            logger.error(traceback.format_exc())
            raise ProcessingError(f"Document processing failed: {str(e)}")
    
    def _get_extractor_for_doc_type(self, doc_type):
        """
        Get the appropriate extractor for a document type
        
        Args:
            doc_type (str): Document type
            
        Returns:
            BaseDocumentExtractor: The appropriate extractor
        """
        # Use specific extractors for supported document types
        if doc_type and "W2" in doc_type:
            logger.info(f"Using specialized W2 extractor for document type {doc_type}")
            return self.w2_extractor
        
        # Default to generic extractor for other document types
        # You could add more specialized extractors for different document types here
        return self.text_extractor
    
    def _extract_words_and_coordinates(self, file_path):
        """
        Extract words and their coordinates from a document
        
        Args:
            file_path (str): Path to the document
            
        Returns:
            tuple: (List of words, List of coordinates)
        """
        words = []
        coordinates = []
        
        try:
            ext = get_file_extension(file_path).lower()
            
            if ext == 'pdf':
                # Use PyMuPDF to extract words and coordinates from PDF
                doc = fitz.open(file_path)
                for page in doc:
                    word_list = page.get_text("words")
                    for word_info in word_list:
                        # word_info format: (x0, y0, x1, y1, word, block_no, line_no, word_no)
                        word = word_info[4]
                        bbox = ((word_info[0], word_info[1]), (word_info[2], word_info[3]))
                        words.append(word)
                        coordinates.append(bbox)
                doc.close()
            else:
                # For non-PDF files, use OCR or other methods
                # This is a simplified placeholder - in a real implementation, 
                # you would integrate with an OCR library
                logger.warning(f"Word coordinate extraction not fully implemented for {ext} files")
                # Return empty lists if not implemented
                from document_processor.core.extraction.text_extractor import extract_document_content
                extraction_result = extract_document_content(file_path)
                text = extraction_result.get('text', '')
                # Simple fallback: split text into words
                words = text.split()
                # Create dummy coordinates
                for i, word in enumerate(words):
                    # Create a simple left-to-right layout with dummy coordinates
                    x0 = i * 10
                    y0 = 10
                    x1 = x0 + len(word) * 5
                    y1 = y0 + 10
                    coordinates.append(((x0, y0), (x1, y1)))
        
        except Exception as e:
            logger.error(f"Error extracting words and coordinates: {str(e)}")
            logger.error(traceback.format_exc())
        
        return words, coordinates
    
    def _copy_document_to_static(self, file_path):
        """
        Copy the document to the static folder for direct access
        
        Args:
            file_path (str): Path to the original document
            
        Returns:
            str: Path to the copied document (relative to static folder)
        """
        try:
            # Create a unique filename for the document
            original_filename = os.path.basename(file_path)
            # Generate a unique filename to avoid overwriting existing files
            base_name, extension = os.path.splitext(original_filename)
            unique_filename = f"{base_name}_{os.path.getmtime(file_path):.0f}{extension}"
            
            # Path in documents subfolder
            document_path = os.path.join('documents', unique_filename)
            full_document_path = os.path.join(self.static_folder, document_path)
            
            # Copy the file
            shutil.copy2(file_path, full_document_path)
            logger.info(f"Copied document to {full_document_path}")
            
            return document_path
            
        except Exception as e:
            logger.error(f"Error copying document {file_path}: {str(e)}")
            logger.error(traceback.format_exc())
            return None
    
    def reprocess_document(self, session_id):
        """
        Reprocess a document with a different mode or updated models
        
        Args:
            session_id (str): Session ID of the previously processed document
            
        Returns:
            bool: True if reprocessing was successful, False otherwise
        """
        # This is a placeholder for reprocessing functionality
        # In a real implementation, you would retrieve the document from storage,
        # possibly with a different processing mode or updated models
        logger.warning("Document reprocessing is not implemented")
        return False
