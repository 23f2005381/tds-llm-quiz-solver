# =============================================================================
# FILE: src/parsers/xml_parser.py
# =============================================================================
"""
Advanced XML parser with namespace handling, XPath queries, and streaming.
Supports both ElementTree and lxml for performance optimization.
References: [web:270][web:273][web:280]
"""

import xml.etree.ElementTree as ET
from lxml import etree
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import structlog
import pandas as pd

logger = structlog.get_logger()


class XMLParser:
    """
    Parse XML files with advanced features like namespace handling,
    XPath queries, and iterative parsing for large files.
    """
    
    @staticmethod
    def parse_file(
        filepath: Path,
        streaming: bool = False,
        max_size_mb: int = 50
    ) -> Dict[str, Any]:
        """
        Parse XML file with automatic streaming for large files[web:280].
        
        Args:
            filepath: Path to XML file
            streaming: Force streaming mode
            max_size_mb: Threshold for automatic streaming
            
        Returns:
            Dict containing parsed XML data and metadata
        """
        try:
            file_size_mb = filepath.stat().st_size / (1024 * 1024)
            
            # Use streaming for large files[web:280]
            if file_size_mb > max_size_mb or streaming:
                logger.info(
                    "using_xml_streaming",
                    filepath=str(filepath),
                    size_mb=round(file_size_mb, 2)
                )
                return XMLParser._parse_streaming(filepath)
            else:
                return XMLParser._parse_standard(filepath)
                
        except Exception as e:
            logger.error("xml_parse_error", filepath=str(filepath), error=str(e))
            return {
                'error': str(e),
                'type': 'xml',
                'filepath': str(filepath)
            }
    
    @staticmethod
    def _parse_standard(filepath: Path) -> Dict[str, Any]:
        """
        Standard XML parsing using lxml with security features[web:270][web:273].
        
        Args:
            filepath: Path to XML file
            
        Returns:
            Parsed XML data
        """
        # Secure parser configuration[web:273]
        parser = etree.XMLParser(
            encoding="UTF-8",
            resolve_entities=False,  # Disable entity expansion (security)
            no_network=True,         # Disable network access (security)
            strip_cdata=False,
            recover=True,            # Attempt to recover from errors
            ns_clean=True            # Clean namespaces
        )
        
        tree = etree.parse(str(filepath), parser)
        root = tree.getroot()
        
        # Extract namespaces
        namespaces = XMLParser._extract_namespaces(root)
        
        # Convert to nested dictionary
        data = XMLParser._element_to_dict(root)
        
        # Try to extract tables if present
        tables = XMLParser._extract_tables(root, namespaces)
        
        logger.info(
            "xml_parsed_standard",
            root_tag=root.tag,
            namespace_count=len(namespaces),
            table_count=len(tables)
        )
        
        return {
            'type': 'xml',
            'root_tag': XMLParser._clean_tag(root.tag),
            'namespaces': namespaces,
            'data': data,
            'tables': tables,
            'table_count': len(tables)
        }
    
    @staticmethod
    def _parse_streaming(filepath: Path) -> Dict[str, Any]:
        """
        Memory-efficient iterative XML parsing[web:280].
        
        Args:
            filepath: Path to XML file
            
        Returns:
            Parsed XML data (may be truncated)
        """
        items = []
        item_count = 0
        max_items = 10000
        root_tag = None
        
        # Iterative parsing[web:280]
        context = etree.iterparse(
            str(filepath),
            events=('start', 'end'),
            tag=None
        )
        
        for event, elem in context:
            if event == 'start' and root_tag is None:
                root_tag = XMLParser._clean_tag(elem.tag)
            
            if event == 'end':
                # Process element
                if item_count < max_items:
                    items.append({
                        'tag': XMLParser._clean_tag(elem.tag),
                        'text': elem.text.strip() if elem.text else None,
                        'attributes': dict(elem.attrib)
                    })
                
                item_count += 1
                
                # Clear element to free memory[web:280]
                elem.clear()
                while elem.getprevious() is not None:
                    del elem.getparent()[0]
        
        del context
        
        logger.info(
            "xml_parsed_streaming",
            total_elements=item_count,
            loaded_elements=len(items)
        )
        
        return {
            'type': 'xml',
            'root_tag': root_tag,
            'data': items,
            'element_count': item_count,
            'truncated': item_count > max_items,
            'streaming': True
        }
    
    @staticmethod
    def _extract_namespaces(root: etree.Element) -> Dict[str, str]:
        """Extract all namespaces from XML document"""
        return dict(root.nsmap) if hasattr(root, 'nsmap') else {}
    
    @staticmethod
    def _clean_tag(tag: str) -> str:
        """Remove namespace prefix from tag"""
        return tag.split('}')[-1] if '}' in tag else tag
    
    @staticmethod
    def _element_to_dict(element: etree.Element) -> Union[Dict, str, List]:
        """
        Recursively convert XML element to nested dictionary.
        
        Args:
            element: XML element
            
        Returns:
            Nested dict representation
        """
        result = {}
        
        # Add attributes
        if element.attrib:
            result['@attributes'] = dict(element.attrib)
        
        # Add text content
        if element.text and element.text.strip():
            if len(element) == 0:  # Leaf node
                return element.text.strip()
            result['#text'] = element.text.strip()
        
        # Process children
        children_dict = {}
        for child in element:
            child_tag = XMLParser._clean_tag(child.tag)
            child_data = XMLParser._element_to_dict(child)
            
            if child_tag in children_dict:
                # Multiple children with same tag -> list
                if not isinstance(children_dict[child_tag], list):
                    children_dict[child_tag] = [children_dict[child_tag]]
                children_dict[child_tag].append(child_data)
            else:
                children_dict[child_tag] = child_data
        
        result.update(children_dict)
        
        return result if len(result) > 0 else None
    
    @staticmethod
    def _extract_tables(root: etree.Element, namespaces: Dict) -> List[pd.DataFrame]:
        """
        Extract tabular data from XML if structured as rows/records.
        
        Args:
            root: Root XML element
            namespaces: Namespace mapping
            
        Returns:
            List of pandas DataFrames
        """
        tables = []
        
        # Find repeating elements (potential tables)
        for parent in root.iter():
            children = list(parent)
            if len(children) < 2:
                continue
            
            # Check if children have same tag
            tags = [XMLParser._clean_tag(child.tag) for child in children]
            if len(set(tags)) == 1 and len(children) >= 2:
                try:
                    records = []
                    for child in children:
                        record = {}
                        for field in child:
                            field_tag = XMLParser._clean_tag(field.tag)
                            record[field_tag] = field.text or ''
                        
                        # Include attributes
                        if child.attrib:
                            for key, val in child.attrib.items():
                                record[f'@{key}'] = val
                        
                        records.append(record)
                    
                    if records and len(records[0]) > 0:
                        df = pd.DataFrame(records)
                        tables.append(df)
                        logger.debug(f"Extracted table with {len(df)} rows")
                except Exception as e:
                    logger.debug(f"Failed to extract table: {e}")
        
        return tables
    
    @staticmethod
    def xpath_query(filepath: Path, xpath: str) -> List[Any]:
        """
        Execute XPath query on XML document[web:270].
        
        Args:
            filepath: Path to XML file
            xpath: XPath query string
            
        Returns:
            List of matching elements/values
        """
        try:
            parser = etree.XMLParser(resolve_entities=False, no_network=True)
            tree = etree.parse(str(filepath), parser)
            results = tree.xpath(xpath)
            
            logger.info(f"XPath query '{xpath}' returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error("xpath_query_error", xpath=xpath, error=str(e))
            return []
