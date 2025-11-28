# =============================================================================
# FILE: debug_page.py (in project root)
# =============================================================================
"""
Debug script to inspect what's on a quiz page.
Run this separately from the API server.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.services.browser import BrowserService


async def inspect_page(url: str):
    """Inspect what content is on a quiz page"""
    
    print(f"\n{'='*60}")
    print(f"INSPECTING: {url}")
    print(f"{'='*60}\n")
    
    async with BrowserService() as browser:
        content = await browser.extract_quiz_content(url)
        
        print("=== PAGE METADATA ===")
        print(f"Title: {content.get('title')}")
        print(f"URL: {url}")
        
        print("\n=== PAGE TEXT (first 2000 chars) ===")
        print(content['text'][:2000])
        print("...")
        
        print(f"\n=== LINKS FOUND: {len(content.get('links', []))} ===")
        for i, link in enumerate(content.get('links', [])[:20], 1):
            href = link.get('href', 'no href')
            text = link.get('text', 'no text').strip()[:50]
            print(f"{i:2}. [{text}] -> {href}")
        
        print(f"\n=== IMAGES: {len(content.get('images', []))} ===")
        for i, img in enumerate(content.get('images', [])[:10], 1):
            print(f"{i}. {img}")
        
        print(f"\n=== CODE BLOCKS: {len(content.get('code_blocks', []))} ===")
        for i, block in enumerate(content.get('code_blocks', []), 1):
            print(f"\nBlock {i}:")
            print(block[:500])
            if len(block) > 500:
                print("...")
        
        print("\n" + "="*60)
        print("INSPECTION COMPLETE")
        print("="*60)


if __name__ == "__main__":
    # URL to inspect
    quiz_url = "https://exam.sanand.workers.dev/tds-2025-09-ga8"
    
    # Or pass as command line argument
    if len(sys.argv) > 1:
        quiz_url = sys.argv[1]
    
    asyncio.run(inspect_page(quiz_url))
