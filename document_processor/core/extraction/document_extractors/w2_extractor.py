"""
W-2 form specific extractor using template-based approach with visualization
"""
import re
import logging
from typing import Dict, List, Any
import os
from PIL import Image

from document_processor.core.extraction.base_extractor import BaseDocumentExtractor

# Import the template mapping definitions
from document_processor.core.extraction.w2_template import (
    W2_FIELD_DEFINITIONS, W2_FORM_REGIONS,
    extract_w2_field, process_w2_form, validate_w2_extraction, create_visualization
)

logger = logging.getLogger(__name__)

class W2Extractor(BaseDocumentExtractor):
    """Specialized extractor for W-2 tax forms using template-based approach"""
    
    def get_field_schema(self):
        """Return the list of fields for W-2 forms"""
        return [
            "employee_name", "employee_ssn", "employer_name", "employer_ein",
            "wages", "federal_tax", "social_security_wages", "social_security_tax",
            "medicare_wages", "medicare_tax", "state_wages", "state_tax", "tax_year",
            "control_number"  # Added control number as it appears on the form
        ]
    
    def extract_fields(self, words, coordinates, combined_text, image_path=None):
        """
        Extract fields from W-2 form using template-based approach
        
        Args:
            words (List[str]): List of words from the document
            coordinates (List): Corresponding bounding box coordinates
            combined_text (str): Full text of the document
            image_path (str, optional): Path to the original image for visualization
            
        Returns:
            Dict[str, str]: Extracted fields
        """
        # Convert words and coordinates to the text blocks format expected by the template processor
        text_blocks = self._prepare_text_blocks(words, coordinates)
        
        # Process with template-based extraction, including visualization
        template_results = process_w2_form(text_blocks, image_path)
        
        # Create a word map for fallback methods
        word_map = self.create_word_map(words, coordinates)
        
        # Use hybrid approach - template-based extraction with fallbacks
        extracted_fields = {}
        
        # List of extraction methods to try for each field
        field_extraction_methods = {
            "employee_name": [
                lambda: template_results.get("employee_name"),
                lambda: self._extract_employee_name(words, word_map, combined_text)
            ],
            "employee_ssn": [
                lambda: template_results.get("employee_ssn"),
                lambda: self._extract_ssn(words, word_map, combined_text)
            ],
            "employer_name": [
                lambda: template_results.get("employer_name"),
                lambda: self._extract_employer_name(words, word_map, combined_text)
            ],
            "employer_ein": [
                lambda: template_results.get("employer_ein"),
                lambda: self._extract_employer_ein(words, word_map, combined_text)
            ],
            "control_number": [
                lambda: template_results.get("control_number"),
                lambda: self._extract_control_number(words, word_map, combined_text)
            ],
            "wages": [
                lambda: template_results.get("wages"),
                lambda: self._extract_box_value(words, word_map, combined_text, "1")
            ],
            "federal_tax": [
                lambda: template_results.get("federal_tax"),
                lambda: self._extract_box_value(words, word_map, combined_text, "2")
            ],
            "social_security_wages": [
                lambda: template_results.get("social_security_wages"),
                lambda: self._extract_box_value(words, word_map, combined_text, "3")
            ],
            "social_security_tax": [
                lambda: template_results.get("social_security_tax"),
                lambda: self._extract_box_value(words, word_map, combined_text, "4")
            ],
            "medicare_wages": [
                lambda: template_results.get("medicare_wages"),
                lambda: self._extract_box_value(words, word_map, combined_text, "5")
            ],
            "medicare_tax": [
                lambda: template_results.get("medicare_tax"),
                lambda: self._extract_box_value(words, word_map, combined_text, "6")
            ],
            "tax_year": [
                lambda: template_results.get("tax_year"),
                lambda: self._extract_tax_year(words, word_map, combined_text)
            ]
        }
        
        # Try each method in order until we get a result
        for field, methods in field_extraction_methods.items():
            for method in methods:
                value = method()
                if value:
                    extracted_fields[field] = value
                    break
        
        # Special case handling for potentially problematic fields
        
        # Check if control number is getting mixed with employee name
        if "employee_name" in extracted_fields and "control_number" in extracted_fields:
            if extracted_fields["control_number"] in extracted_fields["employee_name"]:
                # Fix employee name by removing control number
                extracted_fields["employee_name"] = extracted_fields["employee_name"].replace(
                    extracted_fields["control_number"], "").strip()
        
        # Make sure the federal tax is not the same as wages (common error)
        if "federal_tax" in extracted_fields and "wages" in extracted_fields:
            if extracted_fields["federal_tax"] == extracted_fields["wages"]:
                # Try to find the correct federal tax again with stricter context
                federal_tax = self._extract_box_value_with_context(
                    words, word_map, combined_text, "2", "Federal"
                )
                if federal_tax:
                    extracted_fields["federal_tax"] = federal_tax
        
        # Validate the extraction and log any issues
        validation_results = validate_w2_extraction(extracted_fields)
        for field, result in validation_results.items():
            if not result["is_valid"]:
                logger.warning(f"Validation issue with {field}: {result['message']}")
        
        # Add visualization path if available
        if "_visualization_path" in template_results:
            extracted_fields["_visualization_path"] = template_results["_visualization_path"]
        
        return extracted_fields
    
    def _prepare_text_blocks(self, words, coordinates):
        """
        Convert words and coordinates to text blocks format for template processing
        
        Args:
            words (List[str]): List of words
            coordinates (List): Corresponding coordinates
            
        Returns:
            List[Dict]: List of text blocks with text and relative coordinates
        """
        # Find the document bounds
        min_x, min_y = float('inf'), float('inf')
        max_x, max_y = 0, 0
        
        for (x0, y0), (x1, y1) in coordinates:
            min_x = min(min_x, x0)
            min_y = min(min_y, y0)
            max_x = max(max_x, x1)
            max_y = max(max_y, y1)
        
        # Calculate document width and height
        doc_width = max_x - min_x
        doc_height = max_y - min_y
        
        # Create text blocks with relative positions
        text_blocks = []
        for word, ((x0, y0), (x1, y1)) in zip(words, coordinates):
            # Calculate center point with relative coordinates (0-1 range)
            center_x = ((x0 + x1) / 2 - min_x) / doc_width
            center_y = ((y0 + y1) / 2 - min_y) / doc_height
            
            text_blocks.append({
                "text": word,
                "x": center_x,
                "y": center_y,
                "width": (x1 - x0) / doc_width,
                "height": (y1 - y0) / doc_height,
                "abs_coords": ((x0, y0), (x1, y1))
            })
        
        return text_blocks
    
    # Additional helper methods
    
    def _extract_control_number(self, words, word_map, combined_text):
        """Extract control number from W-2 form"""
        # Look for control number in box d
        control_pattern = r"(?:d\s+)?Control\s+number.*?([A-Z0-9]+)"
        control_match = re.search(control_pattern, combined_text, re.IGNORECASE | re.DOTALL)
        if control_match:
            return control_match.group(1).strip()
        
        # Look for alphanumeric patterns near "Control number" text
        for i, word in enumerate(words):
            if "control" in word.lower() and i+2 < len(words):
                # Look ahead for alphanumeric strings that could be control numbers
                for j in range(i+1, min(i+6, len(words))):
                    if re.match(r'^[A-Z0-9]+$', words[j]):
                        return words[j]
        
        return None
    
    def _extract_box_value_with_context(self, words, word_map, combined_text, box_number, context_word):
        """
        Extract box value with specific context requirement
        
        Args:
            words (List[str]): List of words
            word_map (List[Dict]): Word map with spatial information
            combined_text (str): Full text
            box_number (str): Box number to extract
            context_word (str): Context word that must be nearby
            
        Returns:
            str: Extracted value or None
        """
        # First look for pattern with context
        pattern = fr'(?:{context_word}).*?(?:box|Box)?\s*{box_number}\b[^\d]*?(\$?\d{{1,3}}(?:,\d{{3}})*(?:\.\d{{1,2}})?)'
        context_match = re.search(pattern, combined_text, re.IGNORECASE | re.DOTALL)
        if context_match:
            return context_match.group(1).replace('$', '').strip()
        
        # Then try finding based on spatial positioning with context
        context_positions = []
        box_positions = []
        
        # Find context word and box number positions
        for info in word_map:
            if context_word.lower() in info["word"].lower():
                context_positions.append(info["center"])
            if info["word"] == box_number:
                box_positions.append(info["center"])
        
        # Look for values near both a context position and box position
        if context_positions and box_positions:
            for info in word_map:
                if re.match(r'\$?\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?', info["word"]):
                    # Calculate distance to nearest context and box
                    min_context_dist = min(
                        abs(info["center"][0] - cp[0]) + abs(info["center"][1] - cp[1])
                        for cp in context_positions
                    )
                    min_box_dist = min(
                        abs(info["center"][0] - bp[0]) + abs(info["center"][1] - bp[1])
                        for bp in box_positions
                    )
                    
                    # If reasonably close to both context and box
                    if min_context_dist < 0.3 and min_box_dist < 0.2:
                        return info["word"].replace('$', '').strip()
        
        return None
    
    # Keep previous methods as fallbacks
    
    def _extract_employee_name(self, words, word_map, combined_text):
        """Extract employee name from W-2 form"""
        # First try to find employee's first/last name fields
        name_pattern = r"Employee's\s+first\s+name.*?([A-Z][a-z]+\s+[A-Z](?:\s|\.|[a-z]*)\s+[A-Z][a-zA-Z]+)"
        name_match = re.search(name_pattern, combined_text, re.IGNORECASE | re.DOTALL)
        if name_match:
            return name_match.group(1).strip()

        # Look for the control number first to exclude it
        control_num = None
        for i, word in enumerate(words):
            if "control" in word.lower() and i+2 < len(words):
                for j in range(i+1, min(i+6, len(words))):
                    if re.match(r'^[A-Z0-9]+$', words[j]):
                        control_num = words[j]
                        break
                if control_num:
                    break

        # Strategy: Look for Jane/John pattern
        for i in range(len(words) - 2):
            if re.match(r'(?:Jane|John)', words[i], re.IGNORECASE) and i+2 < len(words):
                if len(words[i+1]) == 1 and words[i+1].isupper() and words[i+2].isupper():
                    name = f"{words[i]} {words[i+1]} {words[i+2]}"
                    # Make sure control number is not in the name
                    if control_num and control_num in name:
                        name = name.replace(control_num, "").strip()
                    return name
        
        # Strategy: Look for patterns in box e
        employee_box_pattern = r"(?:e\s+)?Employee's\s+(?:first\s+)?name.*?([A-Z][a-zA-Z]*\s+(?:[A-Z]\.?\s+)?[A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)?)"
        box_match = re.search(employee_box_pattern, combined_text, re.IGNORECASE | re.DOTALL)
        if box_match:
            name = box_match.group(1).strip()
            # Make sure control number is not in the name
            if control_num and control_num in name:
                name = name.replace(control_num, "").strip()
            return name
        
        return None
    
    def _extract_ssn(self, words, word_map, combined_text):
        """Extract Social Security Number from W-2 form"""
        # Look for SSN in the format XXX-XX-XXXX
        ssn_pattern = r'(\d{3}-\d{2}-\d{4})'
        ssn_matches = re.findall(ssn_pattern, combined_text)
        
        if ssn_matches:
            # Look for SSN near the word "social security" or "SSN" or in box a
            ssn_context_pattern = r'(?:a\s+|Employee.*?SSN|Employee.*?social security).*?(\d{3}-\d{2}-\d{4})'
            context_match = re.search(ssn_context_pattern, combined_text, re.IGNORECASE | re.DOTALL)
            if context_match:
                return context_match.group(1)
            
            # If no contextual match, return the first SSN found
            return ssn_matches[0]
        
        # Try finding a 9-digit number that might be an SSN
        ssn_no_hyphens_pattern = r'(?<!\d)(\d{9})(?!\d)'
        no_hyphen_matches = re.findall(ssn_no_hyphens_pattern, combined_text)
        for match in no_hyphen_matches:
            # Skip obvious non-SSNs
            if match in ['000000000', '999999999']:
                continue
            return f"{match[:3]}-{match[3:5]}-{match[5:]}"
        
        return None
    
    def _extract_employer_name(self, words, word_map, combined_text):
        """Extract employer name from W-2 form"""
        # Strategy 1: Look for "The Big Company" pattern
        for i, word in enumerate(words):
            if word.lower() == "the" and i+2 < len(words):
                if words[i+1][0].isupper() and words[i+2][0].isupper():
                    company_name = f"{word} {words[i+1]} {words[i+2]}"
                    return company_name.strip()
        
        # Strategy 2: Look for employer name in box c
        employer_box_pattern = r"(?:c\s+)?Employer's\s+name.*?([A-Z][A-Za-z\s&\.,]+)(?:d|e|address|$|\n)"
        box_match = re.search(employer_box_pattern, combined_text, re.IGNORECASE | re.DOTALL)
        if box_match:
            # Clean up the extracted name
            name = box_match.group(1).strip()
            # Remove any address or extraneous information
            name_lines = name.split('\n')
            return name_lines[0].strip()
        
        # Strategy 3: Look for company indicators
        company_indicators = ["Inc", "LLC", "Corp", "Company", "Co", "Ltd"]
        for i, word in enumerate(words):
            if any(indicator in word for indicator in company_indicators):
                # Extract company name by looking backward from the indicator
                company_parts = []
                for j in range(i, max(-1, i-5), -1):
                    if j < 0:
                        break
                    if words[j][0].isupper():
                        company_parts.insert(0, words[j])
                    else:
                        break
                if company_parts:
                    return ' '.join(company_parts + [word]).strip()
        
        return None
    
    def _extract_employer_ein(self, words, word_map, combined_text):
        """Extract employer EIN from W-2 form"""
        # Look for EIN in box b using format XX-XXXXXXX
        ein_box_pattern = r"(?:b\s+)?(?:Employer(?:'s)?\s+(?:identification|ID)\s+number|EIN).*?(\d{2}-\d{7})"
        box_match = re.search(ein_box_pattern, combined_text, re.IGNORECASE | re.DOTALL)
        if box_match:
            return box_match.group(1)
        
        # General pattern for EIN
        ein_pattern = r'(\d{2}-\d{7})'
        ein_matches = re.findall(ein_pattern, combined_text)
        if ein_matches:
            return ein_matches[0]
        
        # Try finding a 9-digit number that might be an EIN (not an SSN)
        ein_no_hyphen_pattern = r'(?<!\d)(\d{9})(?!\d)'
        no_hyphen_matches = re.findall(ein_no_hyphen_pattern, combined_text)
        for match in no_hyphen_matches:
            # Skip if it looks like an SSN (appears near "SSN" or "social security")
            nearby_text = combined_text[max(0, combined_text.find(match)-30):min(len(combined_text), combined_text.find(match)+30)]
            if "SSN" in nearby_text or "social security" in nearby_text.lower():
                continue
            # Format as EIN
            return f"{match[:2]}-{match[2:]}"
        
        return None
    
    def _extract_box_value(self, words, word_map, combined_text, box_number):
        """
        Extract the value from a specific box on the W-2 form
        
        Args:
            words (List[str]): List of words
            word_map (List[Dict]): Word map with spatial information
            combined_text (str): Full text
            box_number (str): Box number to extract (e.g., "1", "2")
            
        Returns:
            str: Extracted value or None
        """
        # Strategy 1: Look for box followed by dollar amount
        box_value_pattern = fr'(?:Box|box)?\s*{box_number}\b[^\d]*?(\$?\d{{1,3}}(?:,\d{{3}})*(?:\.\d{{1,2}})?)'
        box_match = re.search(box_value_pattern, combined_text)
        if box_match:
            return box_match.group(1).replace('$', '').strip()
        
        # Strategy 2: Find direct numeric values near box indicator
        box_coords = []
        for i, word_info in enumerate(word_map):
            # Find locations of the box number
            if word_info["word"] == box_number:
                box_coords.append(word_info["center"])
        
        if box_coords:
            # For each box position, look for nearby numbers
            for box_pos in box_coords:
                # Find words to the right, sorted by proximity
                nearby_words = sorted(
                    [w for w in word_map if w["center"][0] > box_pos[0]],
                    key=lambda w: abs(w["center"][1] - box_pos[1]) + 0.5 * (w["center"][0] - box_pos[0])
                )
                
                # Check several nearest words for numeric values
                for word_info in nearby_words[:7]:
                    if re.match(r'\$?\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?', word_info["word"]):
                        return word_info["word"].replace('$', '').strip()
        
        # Strategy 3: Look for specific field labels based on box number
        box_label_patterns = {
            "1": [r'Wages,?\s+tips,?\s+other.*?(\$?\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?)'],
            "2": [r'Federal\s+(?:income)?\s*tax\s+withheld.*?(\$?\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?)'],
            "3": [r'Social\s+security\s+wages.*?(\$?\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?)'],
            "4": [r'Social\s+security\s+tax.*?(\$?\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?)'],
            "5": [r'Medicare\s+wages.*?(\$?\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?)'],
            "6": [r'Medicare\s+tax.*?(\$?\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?)'],
            "16": [r'State\s+wages.*?(\$?\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?)'],
            "17": [r'State\s+(?:income)?\s*tax.*?(\$?\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?)']
        }
        
        if box_number in box_label_patterns:
            for pattern in box_label_patterns[box_number]:
                label_match = re.search(pattern, combined_text, re.IGNORECASE | re.DOTALL)
                if label_match:
                    return label_match.group(1).replace('$', '').strip()
        
        return None
    
    def _extract_tax_year(self, words, word_map, combined_text):
        """Extract tax year from W-2 form"""
        # Look for 4-digit years
        year_pattern = r'(20\d{2})'
        year_matches = re.findall(year_pattern, combined_text)
        
        # Strategy 1: Look for year with context like "Tax Year" or "For tax year"
        year_context_patterns = [
            r'(?:Tax|tax)\s+(?:Year|year)[^\d]*?(20\d{2})',
            r'(?:For|for)\s+(?:Tax|tax)\s+(?:Year|year)[^\d]*?(20\d{2})',
            r'(?:W-2|W2).*?(20\d{2})',
            r'(?:Department of the Treasury).*?(20\d{2})'
        ]
        
        for pattern in year_context_patterns:
            context_match = re.search(pattern, combined_text)
            if context_match:
                return context_match.group(1)
        
        # Strategy 2: Look at the bottom of the form where the year typically appears
        for word_info in word_map:
            if re.match(r'20\d{2}', word_info["word"]) and word_info["center"][1] > 0.7:
                return word_info["word"]
        
        # Strategy 3: If form has a clear year in the header/title area
        # This often appears in isolation or in a prominent position
        if year_matches:
            # If there's only one year mentioned, it's likely the tax year
            if len(set(year_matches)) == 1:
                return year_matches[0]
            
            # Otherwise, take the most common year mentioned
            year_counts = {}
            for year in year_matches:
                year_counts[year] = year_counts.get(year, 0) + 1
            return max(year_counts.items(), key=lambda x: x[1])[0]
        
        return None
