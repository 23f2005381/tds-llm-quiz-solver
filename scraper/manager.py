# =============================================================================
# FILE: src/scrapers/dynamic_scraper.py
# =============================================================================
"""
Advanced scraper for dynamic content: infinite scroll, lazy loading,
and AJAX-loaded content using Playwright.
References: [web:297][web:300][web:307][web:309]
"""

from playwright.async_api import Page, Browser
import asyncio
from typing import List, Dict, Any, Optional
import structlog

logger = structlog.get_logger()


class DynamicContentScraper:
    """
    Scrape dynamically loaded content including infinite scroll,
    lazy loading, and AJAX updates using Intersection Observer pattern.
    """
    
    @staticmethod
    async def scrape_infinite_scroll(
        page: Page,
        scroll_selector: str = 'body',
        scroll_pause_time: float = 2.0,
        max_scrolls: int = 10
    ) -> List[str]:
        """
        Scrape infinite scroll pages using scroll simulation[web:307][web:309].
        
        Args:
            page: Playwright page object
            scroll_selector: Element to scroll (default: body)
            scroll_pause_time: Wait time between scrolls
            max_scrolls: Maximum number of scrolls
            
        Returns:
            List of collected data/HTML
        """
        collected_data = []
        last_height = 0
        scroll_count = 0
        
        while scroll_count < max_scrolls:
            # Get current scroll height
            current_height = await page.evaluate('document.body.scrollHeight')
            
            # Scroll to bottom
            await page.evaluate(f'''
                document.querySelector("{scroll_selector}").scrollTo({{
                    top: document.body.scrollHeight,
                    behavior: 'smooth'
                }});
            ''')
            
            # Wait for content to load
            await asyncio.sleep(scroll_pause_time)
            
            # Check if new content loaded
            new_height = await page.evaluate('document.body.scrollHeight')
            
            if new_height == last_height:
                logger.info("infinite_scroll_completed", scrolls=scroll_count)
                break
            
            # Collect visible content
            content = await page.content()
            collected_data.append(content)
            
            last_height = new_height
            scroll_count += 1
            
            logger.debug(f"Scroll {scroll_count}: height={new_height}")
        
        logger.info(
            "infinite_scroll_finished",
            total_scrolls=scroll_count,
            data_count=len(collected_data)
        )
        
        return collected_data
    
    @staticmethod
    async def scrape_with_intersection_observer(
        page: Page,
        trigger_selector: str,
        content_selector: str,
        max_iterations: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Use Intersection Observer API for lazy loading[web:297][web:300][web:307].
        More efficient than scrolling for detecting when elements appear.
        
        Args:
            page: Playwright page object
            trigger_selector: Element that triggers loading (e.g., footer)
            content_selector: Elements to collect (e.g., .item)
            max_iterations: Max load iterations
            
        Returns:
            List of collected element data
        """
        collected_items = []
        iteration = 0
        
        # Inject Intersection Observer script[web:297][web:307]
        await page.evaluate(f'''
            window.loadCount = 0;
            window.isLoading = false;
            
            const observer = new IntersectionObserver((entries) => {{
                entries.forEach(entry => {{
                    if (entry.isIntersecting && !window.isLoading) {{
                        window.isLoading = true;
                        window.loadCount++;
                        
                        // Trigger custom event for detection
                        window.dispatchEvent(new CustomEvent('contentLoaded'));
                    }}
                }});
            }}, {{
                root: null,
                threshold: 0.5,
                rootMargin: '0px'
            }});
            
            const trigger = document.querySelector("{trigger_selector}");
            if (trigger) {{
                observer.observe(trigger);
            }}
        ''')
        
        while iteration < max_iterations:
            # Wait for content loaded event
            try:
                await page.wait_for_event(
                    'contentLoaded',
                    timeout=5000
                )
            except Exception:
                logger.info("No more content to load")
                break
            
            # Wait for new content to render
            await asyncio.sleep(1)
            
            # Collect new items
            items = await page.eval_on_selector_all(
                content_selector,
                '''elements => elements.map(el => ({
                    html: el.outerHTML,
                    text: el.textContent.trim(),
                    attributes: Object.fromEntries(
                        Array.from(el.attributes).map(attr => [attr.name, attr.value])
                    )
                }))'''
            )
            
            collected_items.extend(items)
            iteration += 1
            
            # Reset loading flag
            await page.evaluate('window.isLoading = false')
            
            logger.debug(f"Iteration {iteration}: collected {len(items)} items")
        
        logger.info(
            "intersection_observer_scraping_complete",
            iterations=iteration,
            total_items=len(collected_items)
        )
        
        return collected_items
    
    @staticmethod
    async def scrape_ajax_content(
        page: Page,
        wait_for_selector: Optional[str] = None,
        wait_for_network_idle: bool = True,
        timeout: int = 30000
    ) -> str:
        """
        Wait for AJAX content to fully load before scraping.
        
        Args:
            page: Playwright page object
            wait_for_selector: Optional selector to wait for
            wait_for_network_idle: Wait for network to be idle
            timeout: Maximum wait time (ms)
            
        Returns:
            Page HTML after AJAX completion
        """
        try:
            if wait_for_selector:
                await page.wait_for_selector(
                    wait_for_selector,
                    state='visible',
                    timeout=timeout
                )
            
            if wait_for_network_idle:
                await page.wait_for_load_state('networkidle', timeout=timeout)
            
            # Additional wait for animations
            await asyncio.sleep(1)
            
            html = await page.content()
            logger.info("ajax_content_loaded", html_length=len(html))
            
            return html
            
        except Exception as e:
            logger.error("ajax_wait_error", error=str(e))
            # Return current content anyway
            return await page.content()
    
    @staticmethod
    async def detect_lazy_images(page: Page) -> List[str]:
        """
        Detect and trigger lazy-loaded images.
        
        Args:
            page: Playwright page object
            
        Returns:
            List of image URLs after lazy loading
        """
        # Scroll through page to trigger lazy loading
        await page.evaluate('''
            async () => {
                const scrollHeight = document.body.scrollHeight;
                const viewportHeight = window.innerHeight;
                const scrollSteps = Math.ceil(scrollHeight / viewportHeight);
                
                for (let i = 0; i < scrollSteps; i++) {
                    window.scrollTo(0, i * viewportHeight);
                    await new Promise(resolve => setTimeout(resolve, 500));
                }
                
                window.scrollTo(0, 0);
            }
        ''')
        
        # Collect all image sources
        image_urls = await page.evaluate('''
            () => {
                const images = Array.from(document.querySelectorAll('img'));
                return images.map(img => img.src || img.dataset.src || '').filter(Boolean);
            }
        ''')
        
        logger.info(f"Detected {len(image_urls)} images after lazy loading")
        return image_urls
