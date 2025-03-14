import asyncio
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, BFSDeepCrawlStrategy, BrowserConfig, DefaultMarkdownGenerator

async def main():
    # Set up browser config
    browser_config = BrowserConfig(headless=True)
    
    # Set up deep crawl strategy
    deep_crawl_strategy = BFSDeepCrawlStrategy(
        max_depth=2, 
        max_pages=5,
        include_external=False
    )
    
    # Set up crawler run config
    crawler_run_config = CrawlerRunConfig(
        markdown_generator=DefaultMarkdownGenerator(),
        deep_crawl_strategy=deep_crawl_strategy
    )
    
    # Execute the crawl
    async with AsyncWebCrawler(config=browser_config) as crawler:
        crawl_result = await crawler.arun("https://docs.crawl4ai.com/", crawler_run_config)
        
        # Print crawl result type and attributes
        print("Crawl result type:", type(crawl_result))
        print("Is list?", isinstance(crawl_result, list))
        print("Has __iter__?", hasattr(crawl_result, '__iter__'))
        print("Has __len__?", hasattr(crawl_result, '__len__'))
        
        # Try to use it as a list
        if hasattr(crawl_result, '__len__') and len(crawl_result) > 0:
            print(f"Result has {len(crawl_result)} items")
            # Print the first item's attributes
            first_item = crawl_result[0]
            print("First item type:", type(first_item))
            print("First item attributes:", dir(first_item))
            
            # Check for url and markdown attributes
            if hasattr(first_item, 'url'):
                print("First item URL:", first_item.url)
            if hasattr(first_item, 'markdown'):
                print("First item markdown length:", len(first_item.markdown))
                # Print the first few lines of markdown
                print("First item markdown preview:", first_item.markdown[:200] + "...")
        else:
            print("Result cannot be used as a list")
        
        # Check if it's a container with _results attribute
        if hasattr(crawl_result, '_results'):
            print(f"Found {len(crawl_result._results)} pages in _results attribute")

if __name__ == "__main__":
    asyncio.run(main()) 