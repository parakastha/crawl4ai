from crawl4ai import (AsyncWebCrawler, CrawlerRunConfig, BrowserConfig, 
                   DefaultMarkdownGenerator, CacheMode, LXMLWebScrapingStrategy)
import asyncio

async def test():
    async with AsyncWebCrawler(config=BrowserConfig(headless=True)) as crawler:
        config = CrawlerRunConfig(
            cache_mode=CacheMode.ENABLED, 
            markdown_generator=DefaultMarkdownGenerator(),
            scraping_strategy=LXMLWebScrapingStrategy()
        )
        
        result = await crawler.arun('https://docs.crawl4ai.com/', config)
        print(f'Result type: {type(result)}')
        
        # Check if it has __dict__ attribute
        if hasattr(result, '__dict__'):
            print(f'Result __dict__: {result.__dict__}')
        else:
            print('Result has no __dict__ attribute')
        
        # Check if it's a callable
        if callable(result):
            print('Result is callable')
        else:
            print('Result is not callable')
        
        # Try accessing raw_markdown attribute
        try:
            print(f'Has raw_markdown: {hasattr(result, "raw_markdown")}')
            if hasattr(result, "raw_markdown"):
                print(f'raw_markdown length: {len(result.raw_markdown)}')
        except Exception as e:
            print(f'Error accessing raw_markdown: {e}')
        
        # Try accessing as dictionary
        try:
            if hasattr(result, 'get'):
                print(f'Has get method: {hasattr(result, "get")}')
                print(f'raw_markdown from get: {result.get("raw_markdown")}')
        except Exception as e:
            print(f'Error accessing as dictionary: {e}')

if __name__ == "__main__":
    asyncio.run(test()) 