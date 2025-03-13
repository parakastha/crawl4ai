#!/usr/bin/env python3
"""
Example script demonstrating how to use the crawl agent programmatically.
"""

import asyncio
import json
import logging
from crawl_agent import CrawlConfig, crawl_url

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def main():
    """Run an example crawl."""
    # Example 1: Basic crawl
    logger.info("Running Example 1: Basic crawl")
    config = CrawlConfig(
        url="https://example.com",
        headless=True,
        content_filter_type="Pruning",
        threshold=0.48
    )
    
    result = await crawl_url(config)
    if result.get("success"):
        logger.info(f"Basic crawl successful! Content saved to {result['fit_file_path']}")
    else:
        logger.error(f"Basic crawl failed: {result.get('error')}")
    
    # Example 2: Deep crawl with BFS strategy
    logger.info("\nRunning Example 2: Deep crawl with BFS strategy")
    deep_config = CrawlConfig(
        url="https://example.com",
        headless=True,
        content_filter_type="Pruning",
        threshold=0.48,
        enable_deep_crawl=True,
        crawl_strategy="BFS (Breadth-First)",
        max_depth=2,
        max_pages=5
    )
    
    deep_result = await crawl_url(deep_config)
    if deep_result.get("success"):
        logger.info(f"Deep crawl successful! Content saved to {deep_result['fit_file_path']}")
        logger.info(f"Total pages crawled: {deep_result.get('total_pages', 0)}")
    else:
        logger.error(f"Deep crawl failed: {deep_result.get('error')}")
    
    # Example 3: Using BM25 content filter
    logger.info("\nRunning Example 3: Using BM25 content filter")
    bm25_config = CrawlConfig(
        url="https://example.com",
        headless=True,
        content_filter_type="BM25",
        user_query="example information technology",
        bm25_threshold=1.0
    )
    
    bm25_result = await crawl_url(bm25_config)
    if bm25_result.get("success"):
        logger.info(f"BM25 filter crawl successful! Content saved to {bm25_result['fit_file_path']}")
    else:
        logger.error(f"BM25 filter crawl failed: {bm25_result.get('error')}")

if __name__ == "__main__":
    asyncio.run(main())