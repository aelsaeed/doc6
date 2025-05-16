"""
Registry of document type extractors
"""
import re
from document_processor.core.extraction.document_extractors.w2_extractor import W2Extractor
from document_processor.core.extraction.base_extractor import BaseDocumentExtractor

class GenericDocumentExtractor(BaseDocumentExtractor):
    """Generic document extractor for unknown document types"""
    
    def get_field_schema(self):
        return [
            "name", "address", "date", "amount", "account_number", "reference"
        ]
    
    def extract_fields(self, words, coordinates, combined_text):
        """
        Generic extraction for unknown document types
        
        Args:
            words (List[str]): List of words
            coordinates (List): List of coordinates
            combined_text (str): Full text
            
        Returns:
            Dict[str, str]: Extracted fields
        """
        # Create a word map
        word_map = self.create_word_map(words, coordinates)
        
        # Simple extraction of common fields
        fields = {}
        
        # Look for names (capital words)
        for i, word in enumerate(words):
            if word[0].isupper() and len(word) > 1:
                if i+1 < len(words) and words[i+1][0].isupper() and len(words[i+1]) > 1:
                    fields["name"] = f"{word} {words[i+1]}"
                    break
        
        # Look for dates
        date_pattern = r'(\d{1,2}/\d{1,2}/\d{2,4}|\d{1,2}-\d{1,2}-\d{2,4}|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4})'
        date_matches = re.findall(date_pattern, combined_text)
        if date_matches:
            fields["date"] = date_matches[0]
        
        # Look for money amounts
        amount_pattern = r'\$(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)'
        amount_matches = re.findall(amount_pattern, combined_text)
        if amount_matches:
            fields["amount"] = amount_matches[0]
        
        # Look for account numbers
        account_pattern = r'(?:Account|Acct|A/C)(?:\.|\s|#|:)?\s*(\d+)'
        account_match = re.search(account_pattern, combined_text, re.IGNORECASE)
        if account_match:
            fields["account_number"] = account_match.group(1)
        
        return fields


class DocumentExtractorFactory:
    """Factory to create the right extractor for each document type"""
    
    @staticmethod
    def get_extractor(doc_type, doctr_model=None):
        """
        Get the appropriate extractor for a document type
        
        Args:
            doc_type (str): Document type name
            doctr_model: OCR model to use
            
        Returns:
            BaseDocumentExtractor: The appropriate extractor
        """
        extractors = {
            "W2 (Form W-2)": W2Extractor,
            # Add more document types as they are implemented
        }
        
        extractor_class = extractors.get(doc_type)
        if extractor_class:
            return extractor_class(doctr_model)
        else:
            # Fall back to a generic extractor
            return GenericDocumentExtractor(doctr_model)