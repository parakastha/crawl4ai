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
        
        # Check for various markdown attributes
        try:
            print(f'Has markdown: {hasattr(result, "markdown")}')
            if hasattr(result, "markdown"):
                print(f'markdown length: {len(result.markdown)}')
                
            print(f'Has raw_markdown: {hasattr(result, "raw_markdown")}')
            if hasattr(result, "raw_markdown"):
                print(f'raw_markdown length: {len(result.raw_markdown)}')
                
            print(f'Has processed_markdown: {hasattr(result, "processed_markdown")}')
            if hasattr(result, "processed_markdown"):
                print(f'processed_markdown length: {len(result.processed_markdown)}')
                
            print(f'Has filtered_markdown: {hasattr(result, "filtered_markdown")}')
            if hasattr(result, "filtered_markdown"):
                print(f'filtered_markdown length: {len(result.filtered_markdown)}')
                
            print(f'Has fit_markdown: {hasattr(result, "fit_markdown")}')
            if hasattr(result, "fit_markdown"):
                print(f'fit_markdown length: {len(result.fit_markdown)}')
        except Exception as e:
            print(f'Error checking markdown attributes: {e}')
        
        # Try accessing as dictionary
        try:
            if hasattr(result, 'get'):
                print(f'Has get method: {hasattr(result, "get")}')
                for key in ["markdown", "raw_markdown", "processed_markdown", "filtered_markdown", "fit_markdown"]:
                    value = result.get(key)
                    if value:
                        print(f'{key} from get: {len(value)} chars')
        except Exception as e:
            print(f'Error accessing as dictionary: {e}')

if __name__ == "__main__":
    asyncio.run(test()) 