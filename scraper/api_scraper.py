# =============================================================================
# FILE: src/scrapers/api_scraper.py
# =============================================================================
"""
Discover and scrape API endpoints by analyzing network traffic,
HAR files, and XHR/Fetch requests.
References: [web:298][web:301]
"""

from playwright.async_api import Page, Route
import json
from typing import List, Dict, Any, Optional
import structlog
import re

logger = structlog.get_logger()


class APIEndpointScraper:
    """
    Discover hidden API endpoints and scrape data directly from APIs
    instead of parsing HTML. More efficient and reliable.
    """
    
    def __init__(self):
        self.discovered_endpoints: List[Dict[str, Any]] = []
        self.api_responses: Dict[str, Any] = {}
    
    async def discover_api_endpoints(
        self,
        page: Page,
        url: str,
        patterns: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Discover API endpoints by intercepting network requests[web:298][web:301].
        
        Args:
            page: Playwright page object
            url: Page URL to analyze
            patterns: Optional regex patterns to filter APIs (e.g., ['/api/', '/v1/'])
            
        Returns:
            List of discovered API endpoints with metadata
        """
        self.discovered_endpoints = []
        patterns = patterns or ['/api/', '/v1/', '/v2/', '/graphql', '.json']
        
        # Intercept all requests
        async def handle_request(route: Route):
            request = route.request
            
            # Check if URL matches API patterns
            is_api = any(pattern in request.url for pattern in patterns)
            
            if is_api:
                endpoint_info = {
                    'url': request.url,
                    'method': request.method,
                    'headers': request.headers,
                    'post_data': request.post_data if request.method == 'POST' else None,
                    'resource_type': request.resource_type
                }
                
                self.discovered_endpoints.append(endpoint_info)
                
                logger.info(
                    "api_endpoint_discovered",
                    method=request.method,
                    url=request.url
                )
            
            # Continue the request
            await route.continue_()
        
        # Enable request interception
        await page.route('**/*', handle_request)
        
        # Navigate to page
        await page.goto(url, wait_until='networkidle')
        
        # Remove route after collection
        await page.unroute('**/*')
        
        logger.info(
            "api_discovery_complete",
            total_endpoints=len(self.discovered_endpoints)
        )
        
        return self.discovered_endpoints
    
    async def capture_api_responses(
        self,
        page: Page,
        url: str,
        api_patterns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Capture actual API response data[web:298].
        
        Args:
            page: Playwright page object
            url: Page URL
            api_patterns: Patterns to match API endpoints
            
        Returns:
            Dict mapping API URLs to response data
        """
        self.api_responses = {}
        api_patterns = api_patterns or ['/api/', '.json']
        
        # Listen for responses
        async def handle_response(response):
            if any(pattern in response.url for pattern in api_patterns):
                try:
                    # Capture JSON responses
                    if 'application/json' in response.headers.get('content-type', ''):
                        data = await response.json()
                        self.api_responses[response.url] = {
                            'status': response.status,
                            'headers': response.headers,
                            'data': data
                        }
                        logger.info(
                            "api_response_captured",
                            url=response.url,
                            status=response.status
                        )
                except Exception as e:
                    logger.warning(f"Failed to parse response: {e}")
        
        page.on('response', handle_response)
        
        # Navigate to trigger API calls
        await page.goto(url, wait_until='networkidle')
        
        # Remove listener
        page.remove_listener('response', handle_response)
        
        logger.info(
            "api_responses_captured",
            total_responses=len(self.api_responses)
        )
        
        return self.api_responses
    
    @staticmethod
    async def export_har_file(page: Page, filepath: str):
        """
        Export HAR (HTTP Archive) file for offline analysis[web:298][web:301].
        
        Args:
            page: Playwright page object
            filepath: Path to save HAR file
        """
        # Note: HAR export requires browser context setup
        logger.info("har_export_requested", filepath=filepath)
        # Implementation note: HAR recording must be started at browser context creation
        # See Playwright documentation for browser.new_context(record_har_path=filepath)
    
    @staticmethod
    def parse_graphql_queries(endpoint_data: List[Dict]) -> List[Dict[str, Any]]:
        """
        Extract GraphQL queries from discovered endpoints.
        
        Args:
            endpoint_data: List of endpoint dicts
            
        Returns:
            List of parsed GraphQL queries
        """
        graphql_queries = []
        
        for endpoint in endpoint_data:
            if 'graphql' in endpoint['url'].lower():
                post_data = endpoint.get('post_data')
                if post_data:
                    try:
                        data = json.loads(post_data)
                        graphql_queries.append({
                            'url': endpoint['url'],
                            'query': data.get('query'),
                            'variables': data.get('variables'),
                            'operation_name': data.get('operationName')
                        })
                    except json.JSONDecodeError:
                        pass
        
        return graphql_queries
