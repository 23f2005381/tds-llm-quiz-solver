# =============================================================================
# FILE: src/parsers/json_parser.py
# =============================================================================
"""
Advanced JSON parser with streaming support for large files.
Handles standard JSON, JSON Lines (JSONL), and nested structures.
References: [web:244][web:247][web:252][web:254]
"""

import json
import ijson  # Streaming JSON parser[web:244]
from pathlib import Path
from typing import Dict, Any, List, Optional, Iterator
import structlog

logger = structlog.get_logger()


class JSONParser:
    """
    Parse JSON files with streaming support for large files.
    Supports JSON, JSONL (JSON Lines), and partial JSON extraction.
    """
    
    @staticmethod
    def parse_file(filepath: Path, streaming: bool = False, max_size_mb: int = 10) -> Dict[str, Any]:
        """
        Parse JSON file with automatic streaming for large files[web:244][web:254].
        
        Args:
            filepath: Path to JSON file
            streaming: Force streaming mode
            max_size_mb: Threshold for automatic streaming (MB)
            
        Returns:
            Dict containing parsed JSON data
        """
        try:
            file_size_mb = filepath.stat().st_size / (1024 * 1024)
            
            # Auto-enable streaming for large files[web:254]
            if file_size_mb > max_size_mb or streaming:
                logger.info(
                    "using_streaming_parser",
                    filepath=str(filepath),
                    size_mb=round(file_size_mb, 2)
                )
                return JSONParser._parse_streaming(filepath)
            else:
                return JSONParser._parse_standard(filepath)
                
        except Exception as e:
            logger.error("json_parse_error", filepath=str(filepath), error=str(e))
            return {
                'error': str(e),
                'type': 'json',
                'filepath': str(filepath)
            }
    
    @staticmethod
    def _parse_standard(filepath: Path) -> Dict[str, Any]:
        """
        Standard JSON parsing (loads entire file into memory).
        
        Args:
            filepath: Path to JSON file
            
        Returns:
            Parsed JSON data
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info("json_parsed_standard", filepath=str(filepath))
        
        return {
            'type': 'json',
            'data': data,
            'is_array': isinstance(data, list),
            'is_object': isinstance(data, dict),
            'item_count': len(data) if isinstance(data, (list, dict)) else None
        }
    
    @staticmethod
    def _parse_streaming(filepath: Path) -> Dict[str, Any]:
        """
        Streaming JSON parsing for large files using ijson[web:244][web:247].
        
        Args:
            filepath: Path to JSON file
            
        Returns:
            Parsed JSON data (limited to prevent memory issues)
        """
        items = []
        item_count = 0
        max_items = 10000  # Limit items to prevent memory overflow
        
        with open(filepath, 'rb') as f:
            # Parse top-level items
            parser = ijson.items(f, 'item')  # Assumes array of items
            
            for item in parser:
                if item_count < max_items:
                    items.append(item)
                item_count += 1
                
                if item_count >= max_items:
                    logger.warning(
                        "truncated_large_json",
                        total_items=item_count,
                        loaded_items=max_items
                    )
                    break
        
        logger.info(
            "json_parsed_streaming",
            filepath=str(filepath),
            total_items=item_count,
            loaded_items=len(items)
        )
        
        return {
            'type': 'json',
            'data': items,
            'is_array': True,
            'item_count': item_count,
            'truncated': item_count > max_items,
            'streaming': True
        }
    
    @staticmethod
    def parse_jsonl(filepath: Path) -> Dict[str, Any]:
        """
        Parse JSON Lines format (newline-delimited JSON)[web:254][web:256].
        
        Args:
            filepath: Path to JSONL file
            
        Returns:
            Dict containing list of parsed JSON objects
        """
        try:
            data = []
            line_count = 0
            error_count = 0
            
            with open(filepath, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        obj = json.loads(line)
                        data.append(obj)
                        line_count += 1
                    except json.JSONDecodeError as e:
                        error_count += 1
                        logger.warning(
                            "jsonl_line_parse_error",
                            line_number=line_num,
                            error=str(e)
                        )
            
            logger.info(
                "jsonl_parsed",
                filepath=str(filepath),
                lines=line_count,
                errors=error_count
            )
            
            return {
                'type': 'jsonl',
                'data': data,
                'line_count': line_count,
                'error_count': error_count
            }
            
        except Exception as e:
            logger.error("jsonl_parse_error", filepath=str(filepath), error=str(e))
            return {
                'error': str(e),
                'type': 'jsonl',
                'filepath': str(filepath)
            }
    
    @staticmethod
    def extract_nested_field(filepath: Path, json_path: str) -> List[Any]:
        """
        Extract specific nested fields using JSON path[web:247].
        Example: extract all ages - json_path='item.age'
        
        Args:
            filepath: Path to JSON file
            json_path: Dot-notation path to extract
            
        Returns:
            List of extracted values
        """
        try:
            values = []
            
            with open(filepath, 'rb') as f:
                # Parse using path
                for value in ijson.items(f, json_path):
                    values.append(value)
            
            logger.info(
                "nested_field_extracted",
                path=json_path,
                count=len(values)
            )
            
            return values
            
        except Exception as e:
            logger.error("nested_extraction_error", path=json_path, error=str(e))
            return []
