"""
Defines processing modes for the document processor
"""
from enum import Enum

class ProcessingMode(Enum):
    """Enum for document processing modes"""
    OPTIMAL = "optimal"  # Uses doctr for initial extraction and LayoutLMv3 for targeted field extraction