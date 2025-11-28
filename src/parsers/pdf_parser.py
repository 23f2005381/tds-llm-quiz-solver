# src/parsers/pdf_parser.py
import fitz  # PyMuPDF is imported as "fitz"
import structlog
from typing import Optional, List, Dict, Any
import pandas as pd

logger = structlog.get_logger()

class PDFParser:
    """Fast PDF text and table extraction using PyMuPDF"""
    
    @staticmethod
    def extract_text(content: bytes, page_number: Optional[int] = None) -> str:
        """
        Extract text from PDF.
        
        Args:
            content: PDF file as bytes
            page_number: Specific page to extract (1-indexed), None for all pages
            
        Returns:
            Extracted text content
        """
        doc = fitz.open(stream=content, filetype="pdf")
        
        if page_number:
            if page_number < 1 or page_number > len(doc):
                raise ValueError(f"Invalid page number: {page_number}")
            text = doc[page_number - 1].get_text()
        else:
            text = "\n".join([page.get_text() for page in doc])
        
        doc.close()
        logger.info("PDF text extracted", total_pages=len(doc), text_length=len(text))
        return text
    
    @staticmethod
    def extract_tables(content: bytes, page_number: Optional[int] = None) -> List[pd.DataFrame]:
        """
        Extract tables from PDF using PyMuPDF's table detection.
        
        Args:
            content: PDF file as bytes
            page_number: Specific page to extract
            
        Returns:
            List of DataFrames containing extracted tables
        """
        doc = fitz.open(stream=content, filetype="pdf")
        tables = []
        
        pages_to_process = [doc[page_number - 1]] if page_number else doc
        
        for page in pages_to_process:
            # Find tables on page
            page_tables = page.find_tables()
            for table in page_tables:
                # Convert to pandas DataFrame
                df = pd.DataFrame(table.extract())
                # Use first row as header
                df.columns = df.iloc[0]
                df = df[1:]
                tables.append(df)
        
        doc.close()
        logger.info("PDF tables extracted", table_count=len(tables))
        return tables
