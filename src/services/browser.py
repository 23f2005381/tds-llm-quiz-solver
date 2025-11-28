# =============================================================================
# FILE: src/services/browser.py (UPGRADED VERSION)
# =============================================================================
from playwright.async_api import async_playwright, Browser, Page, Playwright
import structlog
from typing import Optional, Dict, Any, List
import base64
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential

logger = structlog.get_logger()

class BrowserService:
    """Advanced headless browser automation with anti-detection"""
    
    def __init__(self):
        self.playwright: Optional[Playwright] = None
        self.browser: Optional[Browser] = None
        self.context = None
    
    async def __aenter__(self):
        """Context manager entry - initialize browser with stealth mode"""
        self.playwright = await async_playwright().start()
        
        # Launch with stealth settings
        self.browser = await self.playwright.chromium.launch(
            headless=True,
            args=[
                '--no-sandbox',
                '--disable-setuid-sandbox',
                '--disable-dev-shm-usage',
                '--disable-blink-features=AutomationControlled',
                '--disable-web-security',
                '--disable-features=IsolateOrigins,site-per-process'
            ]
        )
        
        # Create context with realistic fingerprint
        self.context = await self.browser.new_context(
            viewport={'width': 1920, 'height': 1080},
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
            locale='en-US',
            timezone_id='America/New_York',
            extra_http_headers={
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Encoding': 'gzip, deflate, br',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1'
            }
        )
        
        logger.info("Browser service started with stealth mode")
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup"""
        if self.context:
            await self.context.close()
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()
        logger.info("Browser service stopped")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def extract_quiz_content(self, url: str) -> Dict[str, Any]:
        """
        Extract quiz content with intelligent detection.
        Handles base64-encoded content, dynamic loading, and multiple selectors.
        """
        page = await self.context.new_page()
        
        try:
            # Navigate with extended timeout
            await page.goto(url, wait_until='networkidle', timeout=60000)
            
            # Wait for dynamic content to load
            await asyncio.sleep(2)
            
            # Try multiple selector strategies
            initial_content = await self._extract_with_fallback(page)
            
            # Execute any embedded JavaScript (base64 decoding, etc.)
            await self._execute_page_scripts(page)
            await asyncio.sleep(1)  # Wait for script execution
            
            # Re-extract after script execution
            final_content = await self._extract_with_fallback(page)
            
            # Extract all links
            links = await page.evaluate('''() => {
                const links = Array.from(document.querySelectorAll('a[href]'));
                return links.map(link => ({
                    href: link.href,
                    text: link.textContent.trim()
                }));
            }''')
            
            # Extract all images
            images = await page.evaluate('''() => {
                const imgs = Array.from(document.querySelectorAll('img[src]'));
                return imgs.map(img => ({
                    src: img.src,
                    alt: img.alt || ''
                }));
            }''')
            
            # Extract any code blocks or pre tags
            code_blocks = await page.evaluate('''() => {
                const codes = Array.from(document.querySelectorAll('pre, code'));
                return codes.map(code => code.textContent);
            }''')
            question_candidates = await page.evaluate(
                """() => {
                    const selectors = ['h1', 'h2', 'h3', 'p', 'li', 'label', '.question', '#question'];
                    const texts = [];
                    selectors.forEach(sel => {
                        document.querySelectorAll(sel).forEach(el => {
                            const t = el.textContent.trim();
                            if (t && t.length > 10) texts.push(t);
                        });
                    });
                    return texts.slice(0, 20);
                }"""
            )
            logger.info(
                "Quiz content extracted",
                url=url,
                links_count=len(links),
                images_count=len(images),
                code_blocks=len(code_blocks),
                question_candidates=len(question_candidates),

            )
            
            return {
                'html': final_content['html'],
                'text': final_content['text'],
                'links': links,
                'images': images,
                'code_blocks': code_blocks,
                'question_candidates': question_candidates,
                'url': url,
                'title': await page.title()
            }
            
        finally:
            await page.close()
    
    async def _extract_with_fallback(self, page: Page) -> Dict[str, str]:
        """Try multiple selectors to extract content"""
        selectors = ['#result', '#quiz', '#content', 'main', 'body']
        
        for selector in selectors:
            try:
                element = await page.query_selector(selector)
                if element:
                    html = await element.inner_html()
                    text = await element.inner_text()
                    if text.strip():  # Only return if has content
                        logger.info(f"Content found with selector: {selector}")
                        return {'html': html, 'text': text}
            except Exception as e:
                logger.debug(f"Selector {selector} failed: {e}")
                continue
        
        # Fallback to full page
        html = await page.content()
        text = await page.evaluate("() => document.body.innerText")
        return {'html': html, 'text': text}
    
    async def _execute_page_scripts(self, page: Page):
        """Execute any embedded scripts (e.g., base64 decoding from lecture example)"""
        try:
            # Execute all script tags that modify innerHTML
            await page.evaluate('''() => {
                const scripts = document.querySelectorAll('script');
                scripts.forEach(script => {
                    try {
                        if (script.textContent && !script.src) {
                            eval(script.textContent);
                        }
                    } catch (e) {
                        console.log('Script execution error:', e);
                    }
                });
            }''')
        except Exception as e:
            logger.warning(f"Script execution warning: {e}")
    
    async def download_file(self, url: str) -> bytes:
        """Download binary file from URL with retry logic"""
        page = await self.context.new_page()
        
        try:
            response = await page.goto(url, wait_until='networkidle', timeout=60000)
            if response.status >= 400:
                raise ValueError(f"HTTP {response.status} error downloading {url}")
            
            content = await response.body()
            logger.info("File downloaded", url=url, size_bytes=len(content))
            return content
        finally:
            await page.close()
    
    async def scrape_table_from_page(self, url: str) -> List[List[str]]:
        """Extract HTML tables from page"""
        page = await self.context.new_page()
        
        try:
            await page.goto(url, wait_until='networkidle', timeout=60000)
            
            tables = await page.evaluate('''() => {
                const tables = Array.from(document.querySelectorAll('table'));
                return tables.map(table => {
                    const rows = Array.from(table.querySelectorAll('tr'));
                    return rows.map(row => {
                        const cells = Array.from(row.querySelectorAll('td, th'));
                        return cells.map(cell => cell.textContent.trim());
                    });
                });
            }''')
            
            logger.info(f"Extracted {len(tables)} tables from {url}")
            return tables
        finally:
            await page.close()
