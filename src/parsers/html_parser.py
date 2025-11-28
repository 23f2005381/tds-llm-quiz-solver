# src/parsers/html_parser.py
from bs4 import BeautifulSoup
import pandas as pd
import structlog
from typing import Dict, Any, List

logger = structlog.get_logger()

class HTMLParser:
    """HTML content and table extraction"""
    
    @staticmethod
    def parse(content: str) -> Dict[str, Any]:
        """
        Parse HTML and extract tables and text.
        
        Args:
            content: HTML content string
            
        Returns:
            Dict with text, tables, and metadata
        """
        soup = BeautifulSoup(content, 'lxml')
        
        # Extract tables
        tables = []
        for table in soup.find_all('table'):
            try:
                df = pd.read_html(str(table))[0]
                tables.append(df)
            except Exception as e:
                logger.warning("Table parse failed", error=str(e))
        
        # Extract clean text
        text = soup.get_text(separator='\n', strip=True)
        
        return {
            'text': text,
            'tables': tables,
            'table_count': len(tables)
        }
