# I want to understand the different  techniques for data preprocessing that are to implemented on the data extracted from these different sources and if that comprehensive data preprocessing has been performed on the data in these code blocks on the data or need to preformed separately if not implemented what  are the implementations and how to implement them gracefully data preprocessing was considered one of the most difficult work as I am unaware of what kind of extracted data will be there extracted by the scraper 
# sourcing from an api 
# =============================================================================
# FILE: src/parsers/csv_parser.py
# =============================================================================
"""
Advanced CSV Parser with automatic encoding detection, dialect detection,
and robust error handling for messy CSV files.
References: [web:230][web:242][web:245]
"""

import csv
import chardet  # For encoding detection[web:242]
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, List
import structlog
from io import StringIO

logger = structlog.get_logger()


class CSVParser:
    """
    Robust CSV parser with automatic encoding and dialect detection.
    Handles messy CSV files with inconsistent formatting.
    """
    
    @staticmethod
    def parse_file(filepath: Path, encoding: Optional[str] = None) -> Dict[str, Any]:
        """
        Parse CSV file with automatic encoding detection.
        
        Args:
            filepath: Path to CSV file
            encoding: Optional explicit encoding (auto-detected if None)
            
        Returns:
            Dict containing parsed data, metadata, and statistics
        """
        try:
            # Step 1: Detect encoding if not provided[web:242]
            if not encoding:
                encoding = CSVParser._detect_encoding(filepath)
                logger.info("csv_encoding_detected", filepath=str(filepath), encoding=encoding)
            
            # Step 2: Read raw content
            with open(filepath, 'r', encoding=encoding, errors='replace') as f:
                content = f.read()
            
            # Step 3: Detect CSV dialect (delimiter, quote char, etc.)[web:230]
            dialect = CSVParser._detect_dialect(content)
            
            # Step 4: Parse with pandas using detected parameters
            df = pd.read_csv(
                StringIO(content),
                delimiter=dialect.delimiter,
                quotechar=dialect.quotechar,
                skipinitialspace=True,
                encoding=encoding,
                on_bad_lines='warn'  # Skip malformed rows with warning
            )
            
            # Step 5: Data cleaning and type inference
            df = CSVParser._clean_dataframe(df)
            
            # Step 6: Generate summary statistics
            summary = CSVParser._generate_summary(df)
            
            logger.info(
                "csv_parsed_successfully",
                rows=len(df),
                columns=len(df.columns),
                encoding=encoding,
                delimiter=repr(dialect.delimiter)
            )
            
            return {
                'type': 'csv',
                'encoding': encoding,
                'delimiter': dialect.delimiter,
                'shape': df.shape,
                'columns': df.columns.tolist(),
                'dtypes': df.dtypes.astype(str).to_dict(),
                'data': df.to_dict(orient='records'),
                'dataframe': df,  # For direct pandas operations
                'summary': summary,
                'has_nulls': df.isnull().any().any(),
                'null_counts': df.isnull().sum().to_dict()
            }
            
        except Exception as e:
            logger.error("csv_parse_error", filepath=str(filepath), error=str(e))
            return {
                'error': str(e),
                'type': 'csv',
                'filepath': str(filepath)
            }
    
    @staticmethod
    def _detect_encoding(filepath: Path) -> str:
        """
        Detect file encoding using chardet library[web:242].
        
        Args:
            filepath: Path to file
            
        Returns:
            Detected encoding string (e.g., 'utf-8', 'ISO-8859-1')
        """
        with open(filepath, 'rb') as f:
            raw_data = f.read()
        
        result = chardet.detect(raw_data)
        encoding = result['encoding']
        confidence = result['confidence']
        
        # Fallback to utf-8 if confidence is too low
        if confidence < 0.7:
            logger.warning(
                "low_encoding_confidence",
                detected=encoding,
                confidence=confidence,
                fallback='utf-8'
            )
            encoding = 'utf-8'
        
        return encoding
    
    @staticmethod
    def _detect_dialect(content: str) -> csv.Dialect:
        """
        Detect CSV dialect (delimiter, quote char, etc.)[web:230][web:245].
        
        Args:
            content: CSV file content as string
            
        Returns:
            csv.Dialect object with detected parameters
        """
        try:
            # Use first 10KB for dialect detection
            sample = content[:10000]
            dialect = csv.Sniffer().sniff(sample)
            logger.info(
                "csv_dialect_detected",
                delimiter=repr(dialect.delimiter),
                quotechar=repr(dialect.quotechar)
            )
            return dialect
        except Exception as e:
            logger.warning("dialect_detection_failed", error=str(e), using_default=True)
            # Return default dialect
            class DefaultDialect(csv.excel):
                delimiter = ','
                quotechar = '"'
            return DefaultDialect()
    
    @staticmethod
    def _clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean dataframe: strip whitespace, handle missing values, infer types.
        
        Args:
            df: Input dataframe
            
        Returns:
            Cleaned dataframe
        """
        # Strip whitespace from string columns
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].str.strip() if df[col].dtype == 'object' else df[col]
        
        # Attempt to convert columns to appropriate types
        for col in df.columns:
            try:
                # Try converting to numeric
                df[col] = pd.to_numeric(df[col], errors='ignore')
            except Exception:
                pass
            
            try:
                # Try converting to datetime
                if df[col].dtype == 'object':
                    df[col] = pd.to_datetime(df[col], errors='ignore')
            except Exception:
                pass
        
        return df
    
    @staticmethod
    def _generate_summary(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate comprehensive summary statistics.
        
        Args:
            df: Input dataframe
            
        Returns:
            Dict containing summary statistics
        """
        summary = {}
        
        # Numeric columns summary
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            summary['numeric'] = df[numeric_cols].describe().to_dict()
        
        # Categorical columns summary
        categorical_cols = df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            summary['categorical'] = {
                col: {
                    'unique_count': df[col].nunique(),
                    'top_values': df[col].value_counts().head(5).to_dict()
                }
                for col in categorical_cols
            }
        
        return summary
    
    @staticmethod
    def parse_tsv(filepath: Path) -> Dict[str, Any]:
        """
        Parse TSV (Tab-Separated Values) file.
        
        Args:
            filepath: Path to TSV file
            
        Returns:
            Parsed data dictionary
        """
        # TSV is just CSV with tab delimiter
        result = CSVParser.parse_file(filepath)
        result['type'] = 'tsv'
        return result
