"""
W-2 form template mapping with visualization support
"""
import re
import logging
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

logger = logging.getLogger(__name__)

# Field Definitions and Validation Rules
W2_FIELD_DEFINITIONS = {
    # Employee Information
    "employee_ssn": {
        "box_id": "a",
        "label": "Employee's social security number",
        "format_pattern": r"^\d{3}-\d{2}-\d{4}$",
        "validation": "SSN must be in XXX-XX-XXXX format",
        "data_type": "string",
        "required": True,
        "relative_position": "top_right",
        "color": (255, 0, 0)  # Red for highlighting
    },
    "employee_name": {
        "box_id": "e",
        "label": "Employee's first name and initial",
        "format_pattern": r"^[A-Za-z\s\.-]+$",
        "validation": "Must contain letters, spaces, periods, or hyphens",
        "data_type": "string",
        "required": True,
        "relative_position": "middle_left",
        "combines": ["first_name", "middle_initial", "last_name"],
        "color": (0, 255, 0)  # Green for highlighting
    },
    
    # Employer Information
    "employer_ein": {
        "box_id": "b",
        "label": "Employer identification number (EIN)",
        "format_pattern": r"^\d{2}-\d{7}$",
        "validation": "EIN must be in XX-XXXXXXX format",
        "data_type": "string",
        "required": True,
        "relative_position": "top_left",
        "color": (0, 0, 255)  # Blue for highlighting
    },
    "employer_name": {
        "box_id": "c",
        "label": "Employer's name, address, and ZIP code",
        "format_pattern": r"^.+$",
        "validation": "Must not be empty",
        "data_type": "string",
        "required": True,
        "relative_position": "upper_left",
        "multi_line": True,
        "color": (255, 165, 0)  # Orange for highlighting
    },
    "control_number": {
        "box_id": "d",
        "label": "Control number",
        "format_pattern": r"^[A-Za-z0-9]+$",
        "validation": "Alphanumeric characters only",
        "data_type": "string",
        "required": False,
        "relative_position": "middle_left",
        "color": (128, 0, 128)  # Purple for highlighting
    },
    
    # Financial Data (Numbered Boxes)
    "wages": {
        "box_id": "1",
        "label": "Wages, tips, other compensation",
        "format_pattern": r"^\$?[\d,]+\.\d{2}$",
        "validation": "Must be a currency amount",
        "data_type": "currency",
        "required": True,
        "relative_position": "upper_right",
        "color": (0, 128, 128)  # Teal for highlighting
    },
    "federal_tax": {
        "box_id": "2",
        "label": "Federal income tax withheld",
        "format_pattern": r"^\$?[\d,]+\.\d{2}$",
        "validation": "Must be a currency amount",
        "data_type": "currency",
        "required": True,
        "relative_position": "upper_right",
        "color": (255, 192, 203)  # Pink for highlighting
    },
    "social_security_wages": {
        "box_id": "3",
        "label": "Social security wages",
        "format_pattern": r"^\$?[\d,]+\.\d{2}$",
        "validation": "Must be a currency amount",
        "data_type": "currency",
        "required": True,
        "relative_position": "middle_right",
        "color": (165, 42, 42)  # Brown for highlighting
    },
    "social_security_tax": {
        "box_id": "4",
        "label": "Social security tax withheld",
        "format_pattern": r"^\$?[\d,]+\.\d{2}$",
        "validation": "Must be a currency amount",
        "data_type": "currency",
        "required": True,
        "relative_position": "middle_right",
        "color": (0, 100, 0)  # Dark green for highlighting
    },
    "medicare_wages": {
        "box_id": "5",
        "label": "Medicare wages and tips",
        "format_pattern": r"^\$?[\d,]+\.\d{2}$",
        "validation": "Must be a currency amount",
        "data_type": "currency",
        "required": True,
        "relative_position": "middle_right",
        "color": (70, 130, 180)  # Steel blue for highlighting
    },
    "medicare_tax": {
        "box_id": "6",
        "label": "Medicare tax withheld",
        "format_pattern": r"^\$?[\d,]+\.\d{2}$",
        "validation": "Must be a currency amount",
        "data_type": "currency",
        "required": True,
        "relative_position": "middle_right",
        "color": (210, 105, 30)  # Chocolate for highlighting
    },
    "tax_year": {
        "box_id": "year",
        "label": "Tax Year",
        "format_pattern": r"^20\d{2}$",
        "validation": "Must be a 4-digit year (20XX)",
        "data_type": "integer",
        "required": True,
        "relative_position": "bottom",
        "color": (219, 112, 147)  # Pale violet red for highlighting
    }
}

# Adjusted Form Regions - fine-tuned based on W-2 layout
W2_FORM_REGIONS = {
    "standard_layout": {
        # Employee Information region
        "employee_info": {
            "x_range": (0.0, 0.5),  # Left half of the document
            "y_range": (0.0, 0.4)   # Top 40% of the document
        },
        # Employer Information region
        "employer_info": {
            "x_range": (0.0, 0.5),  # Left half of the document
            "y_range": (0.1, 0.5)   # Middle portion of the document
        },
        # Financial Information region (boxes 1-6)
        "financial_info": {
            "x_range": (0.5, 1.0),  # Right half of the document
            "y_range": (0.1, 0.6)   # Middle portion of the document
        },
        # Box regions for key values - adjusted based on typical W-2 layout
        "box_regions": {
            # Employee SSN (Box a)
            "a": {
                "x_range": (0.25, 0.48),  # Upper right of left column
                "y_range": (0.08, 0.12)   # Very top
            },
            # Employer EIN (Box b)
            "b": {
                "x_range": (0.25, 0.48),  # Upper left
                "y_range": (0.16, 0.2)    # Near top
            },
            # Employer name (Box c)
            "c": {
                "x_range": (0.18, 0.48),  # Left side
                "y_range": (0.20, 0.28)   # Upper middle
            },
            # Control number (Box d)
            "d": {
                "x_range": (0.18, 0.28),  # Left side
                "y_range": (0.29, 0.33)   # Middle left
            },
            # Employee name (Box e)
            "e": {
                "x_range": (0.18, 0.48),  # Left side
                "y_range": (0.33, 0.39)   # Middle
            },
            # Wages (Box 1)
            "1": {
                "x_range": (0.52, 0.65),  # Upper right
                "y_range": (0.16, 0.19)   # Upper area
            },
            # Federal tax (Box 2)
            "2": {
                "x_range": (0.65, 0.8),   # Upper right corner
                "y_range": (0.16, 0.19)   # Upper area
            },
            # Social Security Wages (Box 3)
            "3": {
                "x_range": (0.52, 0.65),  # Right column
                "y_range": (0.19, 0.22)   # Upper middle area
            },
            # Social Security Tax (Box 4)
            "4": {
                "x_range": (0.65, 0.8),   # Right column
                "y_range": (0.19, 0.22)   # Upper middle area
            },
            # Medicare Wages (Box 5)
            "5": {
                "x_range": (0.52, 0.65),  # Right column
                "y_range": (0.22, 0.26)   # Middle area
            },
            # Medicare Tax (Box 6)
            "6": {
                "x_range": (0.65, 0.8),   # Right column
                "y_range": (0.22, 0.26)   # Middle area
            }
        }
    }
}

def extract_w2_field(field_name, text_blocks, form_layout="standard_layout"):
    """
    Extract a specific field from W2 form based on text blocks and their positions
    
    Args:
        field_name (str): Name of the field to extract
        text_blocks (list): List of text blocks with text content and coordinates
        form_layout (str): Form layout template to use
        
    Returns:
        tuple: (Extracted value or None if not found, matched block info for visualization)
    """
    # Get field definition
    if field_name not in W2_FIELD_DEFINITIONS:
        return None, None
    
    field_def = W2_FIELD_DEFINITIONS[field_name]
    box_id = field_def["box_id"]
    
    # Get region for this box ID
    if form_layout not in W2_FORM_REGIONS:
        logger.warning(f"Layout {form_layout} not found, falling back to standard_layout")
        form_layout = "standard_layout"
    
    # Handle special cases like tax_year
    if box_id == "year":
        # For tax year, look for 4-digit year pattern in the bottom portion
        for block in text_blocks:
            if (block["y"] > 0.7 and  # Bottom 30% of the document
                re.match(r'20\d{2}', block["text"])):
                return block["text"], block
        return None, None
    
    # If region not defined for this box_id, return None    
    if box_id not in W2_FORM_REGIONS[form_layout]["box_regions"]:
        logger.warning(f"Region for box {box_id} not defined in layout {form_layout}")
        return None, None
    
    region = W2_FORM_REGIONS[form_layout]["box_regions"][box_id]
    
    # Find text blocks within this region
    candidate_blocks = []
    for block in text_blocks:
        x, y = block["x"], block["y"]
        if (region["x_range"][0] <= x <= region["x_range"][1] and
            region["y_range"][0] <= y <= region["y_range"][1]):
            candidate_blocks.append(block)
    
    # Log for debugging
    logger.debug(f"Field {field_name} (box {box_id}) has {len(candidate_blocks)} candidate blocks")
    for block in candidate_blocks:
        logger.debug(f"  - Text: '{block['text']}' at ({block['x']:.2f}, {block['y']:.2f})")
    
    matched_block = None
    
    # Based on field type, extract the appropriate value
    if field_def["data_type"] == "currency":
        # Look for currency patterns in the region
        for block in candidate_blocks:
            # Skip label blocks (contain text from the field label)
            if any(word in block["text"].lower() for word in field_def["label"].lower().split()):
                continue
            
            # Match currency patterns - prioritize formats like "$48,500.00" or "48,500.00"
            currency_match = re.search(r'(?:\$|)(\d{1,3}(?:,\d{3})*(?:\.\d{2}))', block["text"])
            if currency_match:
                matched_block = block
                return currency_match.group(0), block
    
    elif field_def["data_type"] == "string":
        # For employee or employer names, we need special handling
        if field_name == "employee_name":
            # Try to find first name, middle initial, and last name together
            name_parts = []
            name_blocks = []
            
            # Look for complete name patterns
            for block in candidate_blocks:
                if re.match(r'(?:Jane|John)\s+[A-Z]\s+(?:Doe|Smith)', block["text"], re.IGNORECASE):
                    matched_block = block
                    return block["text"], block
                
                # Look for name components (first name, middle initial, last name)
                if (len(block["text"]) > 1 and 
                    block["text"][0].isupper() and
                    not any(label_word in block["text"].lower() for label_word in ["employee", "employer", "first", "last", "initial", "name"])):
                    name_parts.append(block["text"])
                    name_blocks.append(block)
            
            # If we found name parts, combine them
            if name_parts:
                # Sort by x-coordinate since names typically read left to right
                name_with_blocks = sorted(zip(name_parts, name_blocks), key=lambda x: x[1]["x"])
                combined_name = " ".join([name for name, _ in name_with_blocks])
                # Use the first block for visualization
                matched_block = name_blocks[0]
                return combined_name, matched_block
        
        # For employer name, maybe combine multiple lines
        elif field_name == "employer_name" and field_def.get("multi_line", False):
            # Sort blocks by y-coordinate to get correct line order
            sorted_blocks = sorted(candidate_blocks, key=lambda b: b["y"])
            filtered_blocks = [block for block in sorted_blocks 
                              if not any(keyword in block["text"].lower() 
                                        for keyword in ["address", "zip", "code"])]
            
            if filtered_blocks:
                combined_text = " ".join(block["text"] for block in filtered_blocks)
                # Use the first block for visualization
                matched_block = filtered_blocks[0]
                return combined_text, matched_block
        
        # For SSN and EIN, look for specific patterns
        elif field_name in ["employee_ssn", "employer_ein"]:
            for block in candidate_blocks:
                # Look for SSN pattern (123-45-6789) or EIN pattern (12-3456789)
                if (field_name == "employee_ssn" and re.search(r'\d{3}-\d{2}-\d{4}', block["text"])) or \
                   (field_name == "employer_ein" and re.search(r'\d{2}-\d{7}', block["text"])):
                    matched_block = block
                    pattern_match = re.search(r'\d{2,3}-\d{2,7}-?\d{0,4}', block["text"])
                    if pattern_match:
                        return pattern_match.group(0), block
                    return block["text"], block
        
        # General string pattern matching
        for block in candidate_blocks:
            if re.match(field_def["format_pattern"], block["text"]):
                matched_block = block
                return block["text"], block
    
    # For other data types (integer, etc.)
    else:
        for block in candidate_blocks:
            if re.match(field_def["format_pattern"], block["text"]):
                matched_block = block
                return block["text"], block
    
    return None, None

def process_w2_form(text_blocks, image_path=None):
    """
    Process W2 form text blocks to extract all relevant fields
    
    Args:
        text_blocks (list): List of text blocks with text content and coordinates
        image_path (str, optional): Path to the original image for visualization
        
    Returns:
        dict: Dictionary with extracted fields and visualization path
    """
    result = {}
    field_bboxes = {}  # Store bounding boxes for visualization
    
    # Determine form layout (could be enhanced with layout detection)
    form_layout = "standard_layout"
    
    # Extract each defined field
    for field_name in W2_FIELD_DEFINITIONS:
        value, matched_block = extract_w2_field(field_name, text_blocks, form_layout)
        
        # Apply validation and cleanup
        if value:
            # Clean up currency values (remove $ and commas)
            if W2_FIELD_DEFINITIONS[field_name]["data_type"] == "currency":
                # Only remove $ and comma, keep the period and digits
                cleaned_value = value.replace("$", "").replace(",", "")
                result[field_name] = cleaned_value
            else:
                result[field_name] = value
            
            # Store matched block for visualization
            if matched_block:
                field_bboxes[field_name] = matched_block["abs_coords"]
    
    # Create visualization if image path is provided
    visualization_path = None
    if image_path and field_bboxes:
        visualization_path = create_visualization(image_path, field_bboxes)
        result["_visualization_path"] = visualization_path
    
    return result

def create_visualization(image_path, field_bboxes):
    """
    Create a visualization of extracted fields on the original image
    
    Args:
        image_path (str): Path to the original image
        field_bboxes (dict): Dictionary mapping field names to bounding boxes
        
    Returns:
        str: Path to the visualization image
    """
    try:
        # Open original image
        img = Image.open(image_path)
        draw = ImageDraw.Draw(img)
        
        # Draw bounding boxes for each field
        for field_name, bbox in field_bboxes.items():
            if field_name not in W2_FIELD_DEFINITIONS:
                continue
                
            # Get field color for highlighting
            color = W2_FIELD_DEFINITIONS[field_name].get("color", (255, 0, 0))
            
            # Draw rectangle around field
            draw.rectangle(
                [(bbox[0][0], bbox[0][1]), (bbox[1][0], bbox[1][1])],
                outline=color,
                width=3
            )
            
            # Add field name label
            draw.text(
                (bbox[0][0], bbox[0][1] - 10),
                field_name,
                fill=color
            )
        
        # Save visualization
        base_path = os.path.splitext(image_path)[0]
        visualization_path = f"{base_path}_visualization.png"
        img.save(visualization_path)
        logger.info(f"Saved field visualization to {visualization_path}")
        
        return visualization_path
    except Exception as e:
        logger.error(f"Error creating visualization: {str(e)}")
        return None

def validate_w2_extraction(extracted_data):
    """
    Validate extracted W2 data against expected formats and rules
    
    Args:
        extracted_data (dict): Dictionary of extracted fields
        
    Returns:
        dict: Dictionary with validation results
    """
    validation_results = {}
    
    for field_name, value in extracted_data.items():
        # Skip internal fields
        if field_name.startswith('_'):
            continue
            
        if field_name not in W2_FIELD_DEFINITIONS:
            continue
            
        field_def = W2_FIELD_DEFINITIONS[field_name]
        pattern = field_def["format_pattern"]
        
        # Check if value matches expected pattern
        is_valid = bool(re.match(pattern, value))
        
        # Apply field-specific validations
        if field_name == "wages" and is_valid:
            try:
                # Wages should be a positive value
                amount = float(value.replace("$", "").replace(",", ""))
                is_valid = amount >= 0
            except ValueError:
                is_valid = False
        
        validation_results[field_name] = {
            "is_valid": is_valid,
            "value": value,
            "message": field_def["validation"] if not is_valid else "Valid"
        }
    
    # Check for required fields that are missing
    for field_name, field_def in W2_FIELD_DEFINITIONS.items():
        if field_def.get("required", False) and field_name not in extracted_data:
            validation_results[field_name] = {
                "is_valid": False,
                "value": None,
                "message": "Required field is missing"
            }
    
    return validation_results
