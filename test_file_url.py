import asyncio
import tempfile
import os
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, BrowserConfig, CacheMode

async def test_file_url_handling():
    """Test if file:// URL handling is implemented in our version of Crawl4AI"""
    print("\n=== Testing file:// URL handling ===")
    
    # Create a temporary HTML file
    with tempfile.NamedTemporaryFile(suffix='.html', mode='w', delete=False) as f:
        f.write("""
        <html>
            <head><title>Test File URL</title></head>
            <body>
                <h1>This is a local HTML file</h1>
                <p>Testing file:// URL handling in Crawl4AI</p>
            </body>
        </html>
        """)
        temp_file_path = f.name
    
    try:
        # Configure browser and crawler
        browser_config = BrowserConfig(headless=True, verbose=True)
        crawler_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS)
        
        # Convert to file:// URL format
        file_url = f"file://{temp_file_path}"
        print(f"Attempting to crawl: {file_url}")
        
        # Crawl the local file
        async with AsyncWebCrawler(config=browser_config) as crawler:
            result = await crawler.arun(file_url, crawler_config)
            
            if hasattr(result, 'success'):
                print(f"Crawl success: {result.success}")
                if result.success:
                    print("Markdown content:")
                    if hasattr(result, 'markdown'):
                        print(result.markdown)
                    elif hasattr(result, 'raw_markdown'):
                        print(result.raw_markdown)
                    else:
                        print("No markdown attribute found in result")
                else:
                    print(f"Error: {getattr(result, 'error_message', 'Unknown error')}")
            else:
                print(f"Result type: {type(result)}")
                if isinstance(result, dict):
                    print(f"Result keys: {result.keys()}")
    finally:
        # Clean up the temporary file
        os.unlink(temp_file_path)

async def test_raw_html_handling():
    """Test if raw: URL handling is implemented in our version of Crawl4AI"""
    print("\n=== Testing raw: URL handling ===")
    
    # Create a simple HTML string
    raw_html = """
    <html>
        <head><title>Test Raw HTML</title></head>
        <body>
            <h1>This is raw HTML content</h1>
            <p>Testing raw: URL handling in Crawl4AI</p>
        </body>
    </html>
    """
    
    # Configure browser and crawler
    browser_config = BrowserConfig(headless=True, verbose=True)
    crawler_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS)
    
    # Format with raw: prefix
    raw_url = f"raw:{raw_html}"
    print(f"Attempting to crawl raw HTML")
    
    # Crawl the raw HTML
    async with AsyncWebCrawler(config=browser_config) as crawler:
        result = await crawler.arun(raw_url, crawler_config)
        
        if hasattr(result, 'success'):
            print(f"Crawl success: {result.success}")
            if result.success:
                print("Markdown content:")
                if hasattr(result, 'markdown'):
                    print(result.markdown)
                elif hasattr(result, 'raw_markdown'):
                    print(result.raw_markdown)
                else:
                    print("No markdown attribute found in result")
            else:
                print(f"Error: {getattr(result, 'error_message', 'Unknown error')}")
        else:
            print(f"Result type: {type(result)}")
            if isinstance(result, dict):
                print(f"Result keys: {result.keys()}")

async def main():
    await test_file_url_handling()
    await test_raw_html_handling()

if __name__ == "__main__":
    asyncio.run(main()) 