# =============================================================================
# FILE: src/services/data_fetcher.py (UPGRADED VERSION)
# =============================================================================
import aiohttp
import aiofiles
from pathlib import Path
import structlog
from typing import Optional, Dict, Any
import hashlib
import random
from tenacity import retry, stop_after_attempt, wait_exponential
from ..core.config import settings

logger = structlog.get_logger()

class DataFetcher:
    """Advanced HTTP client with User-Agent rotation and anti-blocking"""
    
    # Realistic User-Agent pool for rotation[web:188][web:191]
    USER_AGENTS = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:132.0) Gecko/20100101 Firefox/132.0',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.1 Safari/605.1.15',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36'
    ]
    
    def __init__(self):
        self.temp_dir = Path(settings.TEMP_DIR)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.session: Optional[aiohttp.ClientSession] = None
    
    def _get_random_headers(self, custom_headers: Optional[Dict] = None) -> Dict[str, str]:
        """Generate realistic request headers with rotation[web:188][web:191]"""
        headers = {
            'User-Agent': random.choice(self.USER_AGENTS),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
            'Cache-Control': 'max-age=0'
        }
        
        if custom_headers:
            headers.update(custom_headers)
        
        return headers
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def download_file(
        self, 
        url: str,
        custom_headers: Optional[Dict] = None
    ) -> Path:
        """Download file with retry logic and anti-blocking headers"""
        logger.info("downloading_file", url=url)
        
        # Generate unique filename
        url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
        ext = Path(url).suffix or ".bin"
        filename = f"download_{url_hash}{ext}"
        filepath = self.temp_dir / filename
        
        # Skip if already downloaded
        if filepath.exists():
            logger.info("file_already_cached", filepath=str(filepath))
            return filepath
        
        headers = self._get_random_headers(custom_headers)
        
        timeout = aiohttp.ClientTimeout(total=120, connect=30)
        
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url, headers=headers) as response:
                status = response.status
                if status >= 400:
                    text_preview = (await response.text())[:200]
                    logger.error(
                        "download_http_error",
                        url=url,
                        status=status,
                        preview=text_preview,
                    )
                    raise aiohttp.ClientResponseError(
                        response.request_info,
                        response.history,
                        status=status,
                        message=f"HTTP {status} while downloading",
                        headers=response.headers,
                    )
                response.raise_for_status()
                max_mb = settings.MAX_FILE_SIZE_MB
                
                # Check file size
                content_length = response.headers.get('Content-Length')
                if content_length:
                    size_mb = int(content_length) / (1024 * 1024)
                    if size_mb > settings.MAX_FILE_SIZE_MB:
                        raise ValueError(f"File too large: {size_mb:.1f}MB (limit {max_mb}MB)")
                    logger.info(f"Downloading {size_mb:.2f}MB file",size_mb=size_mb)
                else:
                        logger.info("Downloading file_without_length_header")
                written_bytes = 0
                hard_limit_bytes = max_mb * 1024 * 1024

                # Download with progress
                async with aiofiles.open(filepath, 'wb') as f:
                    async for chunk in response.content.iter_chunked(8192):
                        written_bytes += len(chunk)
                        if written_bytes > hard_limit_bytes:
                            logger.error(
                                "download_size_limit_exceeded",
                                url=url,
                                bytes_written=written_bytes,
                            )
                            raise ValueError("File exceeded maximum allowed size")
                        await f.write(chunk)
    
        logger.info("file_downloaded", filepath=str(filepath))
        return filepath
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def fetch_api_data(
        self,
        url: str,
        method: str = 'GET',
        headers: Optional[Dict] = None,
        params: Optional[Dict] = None,
        json_data: Optional[Dict] = None
    ) -> Any:
        """Fetch data from API with custom headers support"""
        logger.info("fetching_api_data", url=url, method=method)
        
        request_headers = self._get_random_headers(headers)
        
        timeout = aiohttp.ClientTimeout(total=60)
        
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.request(
                method,
                url,
                headers=request_headers,
                params=params,
                json=json_data
            ) as response:
                    status = response.status
                    if status >= 400:
                        text_preview = (await response.text())[:200]
                        logger.error(
                            "api_http_error",
                            url=url,
                            status=status,
                            preview=text_preview,
                        )
                        raise aiohttp.ClientResponseError(
                            response.request_info,
                            response.history,
                            status=status,
                            message=f"HTTP {status} while calling API",
                            headers=response.headers,
                        )
                    response.raise_for_status()

                    content_type = response.headers.get('Content-Type', '')
                    
                    if 'application/json' in content_type:
                        data = await response.json()
                        logger.info("api_data_fetched", url=url, method=method, type="json")
                        return data
                    else:
                        text = await response.text()
                        logger.info("api_data_fetched", url=url, method=method, type="text")
                        return text