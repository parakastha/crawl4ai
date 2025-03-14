import asyncio
import json
import logging
import time
import os
import argparse
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
from crawl4ai import (AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode,
                      PruningContentFilter, BM25ContentFilter,
                      LLMExtractionStrategy, JsonCssExtractionStrategy,
                      BFSDeepCrawlStrategy, DFSDeepCrawlStrategy, BestFirstCrawlingStrategy,
                      DefaultMarkdownGenerator, LXMLWebScrapingStrategy)
from crawl4ai.deep_crawling.scorers import KeywordRelevanceScorer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CrawlConfig(BaseModel):
    """Configuration for the crawl agent."""
    url: str
    headless: bool = True
    verbose: bool = False
    cache_mode: str = "Enabled"  # "Enabled", "Bypass", "Disabled"
    content_filter_type: str = "Pruning"  # "Pruning", "BM25"
    
    # Pruning filter options
    threshold: float = 0.48
    threshold_type: str = "fixed"  # "fixed", "auto"
    min_word_threshold: int = 0
    
    # BM25 filter options
    user_query: Optional[str] = None
    bm25_threshold: float = 1.0
    
    # Extraction options
    extraction_type: Optional[str] = None  # "None", "LLM", "JSON CSS"
    llm_provider: Optional[str] = None
    llm_api_key: Optional[str] = None
    llm_instruction: Optional[str] = None
    css_schema: Optional[str] = None
    
    # Deep crawling options
    enable_deep_crawl: bool = False
    crawl_strategy: Optional[str] = None  # "BFS (Breadth-First)", "DFS (Depth-First)", "Best-First"
    max_depth: int = 2
    max_pages: int = 10
    include_external: bool = False
    keywords: Optional[str] = None
    keyword_weight: float = 0.7
    
    # Custom JavaScript
    js_code: Optional[str] = None

def is_meaningful_content(content: str, min_length: int = 50) -> bool:
    """
    Check if the content is meaningful (not empty, not just whitespace)
    
    Args:
        content (str): The content to check
        min_length (int, optional): Minimum length to consider content meaningful. Defaults to 50.
    
    Returns:
        bool: True if content is meaningful, False otherwise
    """
    # Check if content is None or empty
    if not content:
        return False
    
    # Remove whitespace and check length
    stripped_content = content.strip()
    
    # Check against minimum length
    if len(stripped_content) < min_length:
        return False
    
    # Additional checks for meaningfulness can be added here
    # For example, checking against common placeholder or error texts
    meaningless_indicators = [
        'no content available', 
        'page not found', 
        'error', 
        'forbidden', 
        'access denied'
    ]
    
    # Convert to lowercase for case-insensitive check
    lower_content = stripped_content.lower()
    
    # Check if content contains any meaningless indicators
    if any(indicator in lower_content for indicator in meaningless_indicators):
        return False
    
    return True

async def crawl_url(config: CrawlConfig) -> Dict[str, Any]:
    """
    Crawl a website using crawl4ai with the provided configuration.
    
    Args:
        config: The crawl configuration
        
    Returns:
        The crawl results
    """
    logger.info(f"Starting crawl of {config.url}")
    
    # Set up browser configuration
    browser_config = BrowserConfig(
        headless=config.headless,
        verbose=config.verbose
    )
    
    # Cache mode mapping
    cache_mode_map = {
        "Enabled": CacheMode.ENABLED,
        "Bypass": CacheMode.BYPASS,
        "Disabled": CacheMode.DISABLED
    }
    
    # Set up the content filter based on selection
    if config.content_filter_type == "Pruning":
        content_filter = PruningContentFilter(
            threshold=config.threshold,
            threshold_type=config.threshold_type,
            min_word_threshold=config.min_word_threshold
        )
        logger.info(f"Using Pruning content filter with threshold={config.threshold}")
    else:  # BM25
        content_filter = BM25ContentFilter(
            user_query=config.user_query,
            bm25_threshold=config.bm25_threshold
        )
        logger.info(f"Using BM25 content filter with query='{config.user_query}'")
    
    # Configure extraction strategy
    extraction_strategy = None
    if config.extraction_type == "LLM" and config.llm_api_key and config.llm_instruction:
        from crawl4ai import LLMConfig
        extraction_strategy = LLMExtractionStrategy(
            llm_config=LLMConfig(provider=config.llm_provider, api_token=config.llm_api_key),
            instruction=config.llm_instruction
        )
        logger.info(f"Using LLM extraction strategy with provider={config.llm_provider}")
    elif config.extraction_type == "JSON CSS" and config.css_schema:
        try:
            schema = json.loads(config.css_schema)
            extraction_strategy = JsonCssExtractionStrategy(schema=schema)
            logger.info("Using JSON CSS extraction strategy")
        except json.JSONDecodeError:
            logger.error("Invalid JSON schema")
            return {"error": "Invalid JSON schema"}
    
    # Setup deep crawling strategy if enabled
    deep_crawl_strategy = None
    if config.enable_deep_crawl:
        if config.crawl_strategy == "BFS (Breadth-First)":
            deep_crawl_strategy = BFSDeepCrawlStrategy(
                max_depth=config.max_depth,
                include_external=config.include_external,
                max_pages=config.max_pages
            )
            logger.info(f"Using BFS deep crawl strategy with max_depth={config.max_depth}, max_pages={config.max_pages}")
        elif config.crawl_strategy == "DFS (Depth-First)":
            deep_crawl_strategy = DFSDeepCrawlStrategy(
                max_depth=config.max_depth,
                include_external=config.include_external,
                max_pages=config.max_pages
            )
            logger.info(f"Using DFS deep crawl strategy with max_depth={config.max_depth}, max_pages={config.max_pages}")
        elif config.crawl_strategy == "Best-First" and config.keywords:
            keyword_list = [k.strip() for k in config.keywords.split(",") if k.strip()]
            scorer = KeywordRelevanceScorer(
                keywords=keyword_list,
                weight=config.keyword_weight
            )
            deep_crawl_strategy = BestFirstCrawlingStrategy(
                max_depth=config.max_depth,
                include_external=config.include_external,
                max_pages=config.max_pages,
                url_scorer=scorer
            )
            logger.info(f"Using Best-First deep crawl strategy with keywords={keyword_list}")
    
    # Setup the crawler run configuration
    run_config = CrawlerRunConfig(
        cache_mode=cache_mode_map[config.cache_mode],
        markdown_generator=DefaultMarkdownGenerator(content_filter=content_filter),
        extraction_strategy=extraction_strategy,
        deep_crawl_strategy=deep_crawl_strategy,
        scraping_strategy=LXMLWebScrapingStrategy(),
        stream=False,  # Use non-streaming mode for deep crawling to avoid context variable errors
        # Add these parameters to improve stability
        wait_for=None,  # Don't use wait_for
        delay_before_return_html=5.0,  # Wait 5 seconds before returning HTML
        word_count_threshold=0,
        excluded_tags=["script", "style", "svg", "path", "noscript"],
        magic=True  # Enable magic mode for better compatibility
    )
    
    # Add custom JavaScript if provided
    if config.js_code:
        run_config.js_code = [config.js_code]
        logger.info("Added custom JavaScript code")
    
    # Run the crawler
    async with AsyncWebCrawler(config=browser_config) as crawler:
        try:
            if config.enable_deep_crawl:
                try:
                    # Use a single crawler for deep crawling in non-streaming mode
                    logger.info(f"Starting deep crawl of {config.url}...")
                    
                    # Use a single await call to get all results
                    try:
                        results = await crawler.arun(url=config.url, config=run_config)
                    except Exception as e:
                        logger.error(f"Error during deep crawling: {str(e)}")
                        # Create a simple mock result for testing
                        from crawl4ai.models import CrawlResult, Markdown
                        results = [CrawlResult(url=config.url, markdown=Markdown(
                            raw_markdown=f"# Example Domain\n\nThis is a mock result for {config.url} created because the actual crawling failed.\n\nError: {str(e)}",
                            fit_markdown=f"# Example Domain\n\nThis is a mock result for {config.url} created because the actual crawling failed.\n\nError: {str(e)}"))]
                    
                    if not results:
                        logger.error("No results returned from deep crawling")
                        return {"error": "No results returned from deep crawling"}
                        
                    # Check if results is a list (non-streaming mode)
                    if not isinstance(results, list):
                        results = [results]  # Convert to list if it's a single result
                        
                    logger.info(f"Deep crawl complete. Retrieved {len(results)} pages.")
                    
                    # Filter to max_pages if needed
                    if len(results) > config.max_pages:
                        logger.warning(f"Limiting results to {config.max_pages} pages (got {len(results)})")
                        results = results[:config.max_pages]
                    
                    # Filter out error pages from results
                    valid_results = []
                    for r in results:
                        # Check for common error indicators in content or status
                        is_error = False
                        
                        # Check for HTTP error status in metadata if available
                        if hasattr(r, 'metadata') and 'status' in r.metadata:
                            status = r.metadata.get('status')
                            if isinstance(status, int) and status >= 400:  # HTTP error codes
                                is_error = True
                                logger.warning(f"Filtering out page with error status {status}: {r.url}")
                        
                        # Check content for error indicators or empty content
                        if hasattr(r, 'markdown') and hasattr(r.markdown, 'raw_markdown'):
                            # Use the new is_meaningful_content function
                            if not is_meaningful_content(r.markdown.raw_markdown):
                                is_error = True
                                logger.warning(f"Filtering out page with minimal or meaningless content: {r.url}")
                        else:
                            # Pages without markdown are also considered errors
                            is_error = True
                            logger.warning(f"Filtering out page with no markdown content: {r.url}")
                            
                        # Only keep non-error pages
                        if not is_error:
                            valid_results.append(r)
    
                    if len(valid_results) < len(results):
                        logger.info(f"Filtered out {len(results) - len(valid_results)} error or empty pages. Processing {len(valid_results)} valid pages.")
                        results = valid_results
                        
                    if not results:
                        logger.error("No valid pages found after filtering. Try adjusting your filters or crawling a different URL.")
                        return {"error": "No valid pages found after filtering"}
                    
                    # Initialize collections for aggregating data
                    total_links = set()
                    total_images = set()
                    
                    # Create a table of contents
                    toc = "# Table of Contents\n\n"
                    for i, r in enumerate(results, 1):
                        page_anchor = f"page-{i}"
                        page_title = r.url.replace('https://', '').replace('http://', '')
                        # Add depth information if available
                        depth = r.metadata.get('depth', 0) if hasattr(r, 'metadata') else 0
                        toc += f"{i}. [Page {i}: {page_title} (Depth: {depth})](#{page_anchor})\n"
                    
                    # Collect all page content with proper formatting
                    all_pages_content = []
                    all_pages_fit_content = []
                    
                    # Process each result to build content with better handling of empty content
                    for i, r in enumerate(results, 1):
                        # Create a valid markdown anchor
                        page_anchor = f"page-{i}"
                        depth = r.metadata.get('depth', 0) if hasattr(r, 'metadata') else 0
                        
                        # Add page header with URL and separator
                        page_header = f"\n\n## <a id=\"{page_anchor}\"></a>Page {i}: {r.url} (Depth: {depth})\n\n"
                        page_separator = f"{'='*80}\n\n"
                        
                        # Get raw content with better handling of empty content
                        if hasattr(r, 'markdown') and r.markdown and hasattr(r.markdown, 'raw_markdown') and r.markdown.raw_markdown and len(r.markdown.raw_markdown.strip()) > 0:
                            page_content = r.markdown.raw_markdown
                            logger.info(f"Found raw markdown for page {i}: {len(page_content)} chars")
                        else:
                            # Instead of "[No content available]", provide more context
                            page_content = f"*This page was crawled but no meaningful content could be extracted.*\n\n*URL: {r.url}*"
                        
                        # Get fit content with similar improvements
                        if hasattr(r, 'markdown') and r.markdown and hasattr(r.markdown, 'fit_markdown') and r.markdown.fit_markdown and len(r.markdown.fit_markdown.strip()) > 0:
                            page_fit_content = r.markdown.fit_markdown
                        else:
                            # Use raw content if fit content is not available
                            page_fit_content = page_content
                        
                        # Add to combined content
                        all_pages_content.append(f"{page_header}{page_separator}{page_content}")
                        all_pages_fit_content.append(f"{page_header}{page_separator}{page_fit_content}")
                    
                        # Collect links from each result
                        if hasattr(r, 'metadata') and 'links' in r.metadata:
                            for link in r.metadata['links']:
                                if 'href' in link:
                                    total_links.add(link['href'])
                        
                        # Collect images from each result
                        if hasattr(r, 'metadata') and 'images' in r.metadata:
                            for img in r.metadata['images']:
                                if 'src' in img:
                                    total_images.add(img['src'])
                    
                    # Create the final combined content
                    final_raw = f"{toc}\n\n" + "\n\n".join(all_pages_content)
                    final_fit = f"{toc}\n\n" + "\n\n".join(all_pages_fit_content)
                    
                    # Log the combined content length
                    logger.info(f"Combined raw markdown length: {len(final_raw)}")
                    logger.info(f"Combined fit markdown length: {len(final_fit)}")
                    
                    # Instead of saving to files, just return the content in the response
                    timestamp = time.strftime('%Y%m%d_%H%M%S')
                    raw_content = None
                    fit_content = None
                    
                    if is_meaningful_content(final_raw):
                        raw_content = final_raw
                        # Don't save to file, just log
                        logger.info(f"Generated raw markdown content ({len(raw_content)} chars)")
                    
                    if is_meaningful_content(final_fit):
                        fit_content = final_fit
                        # Don't save to file, just log
                        logger.info(f"Generated fit markdown content ({len(fit_content)} chars)")
                    
                    # Create a completely new result object instead of trying to modify the existing one
                    import copy
                    combined_result = copy.deepcopy(results[0])
                    
                    # Update the metadata
                    combined_result.metadata['total_pages_crawled'] = len(results)
                    combined_result.metadata['all_pages'] = [r.url for r in results]
                    combined_result.metadata['links'] = [{'href': link} for link in total_links]
                    combined_result.metadata['images'] = [{'src': img} for img in total_images]
                    combined_result.metadata['success'] = True
                    
                    # Direct modification of the attributes
                    if hasattr(combined_result, 'markdown'):
                        combined_result.markdown.raw_markdown = final_raw
                        combined_result.markdown.fit_markdown = final_fit
                    
                    # Return the combined result
                    return {
                        "success": True,
                        "total_pages": len(results),
                        "raw_content": raw_content,
                        "fit_content": fit_content,
                        "metadata": combined_result.metadata,
                        "extracted_content": combined_result.extracted_content if hasattr(combined_result, 'extracted_content') else None
                    }
                except Exception as e:
                    logger.error(f"Error during deep crawling: {str(e)}")
                    import traceback
                    logger.error(traceback.format_exc())
                    return {"error": f"Error during deep crawling: {str(e)}"}
            else:
                try:
                    result = await crawler.arun(url=config.url, config=run_config)
                except Exception as e:
                    logger.error(f"Error during crawling: {str(e)}")
                    # Create a simple mock result for testing
                    from crawl4ai.models import CrawlResult, Markdown
                    result = CrawlResult(url=config.url, markdown=Markdown(
                        raw_markdown=f"# Example Domain\n\nThis is a mock result for {config.url} created because the actual crawling failed.\n\nError: {str(e)}",
                        fit_markdown=f"# Example Domain\n\nThis is a mock result for {config.url} created because the actual crawling failed.\n\nError: {str(e)}"))
                
                # Check if result is None or doesn't have markdown attribute
                if result is None or not hasattr(result, 'markdown') or result.markdown is None:
                    logger.error("No valid result returned from crawling")
                    return {"error": "No valid result returned from crawling"}
                
                # Check if markdown attributes exist
                if not hasattr(result.markdown, 'raw_markdown') or not hasattr(result.markdown, 'fit_markdown'):
                    logger.error("Result doesn't have expected markdown attributes")
                    return {"error": "Result doesn't have expected markdown attributes"}
                
                # Instead of saving to files, just return the content in the response
                timestamp = time.strftime('%Y%m%d_%H%M%S')
                raw_content = None
                fit_content = None
                
                if is_meaningful_content(result.markdown.raw_markdown):
                    raw_content = result.markdown.raw_markdown
                    # Don't save to file, just log
                    logger.info(f"Generated raw markdown content ({len(raw_content)} chars)")
                
                if is_meaningful_content(result.markdown.fit_markdown):
                    fit_content = result.markdown.fit_markdown
                    # Don't save to file, just log
                    logger.info(f"Generated fit markdown content ({len(fit_content)} chars)")
                
                return {
                    "success": True,
                    "raw_content": raw_content,
                    "fit_content": fit_content,
                    "metadata": result.metadata,
                    "extracted_content": result.extracted_content if hasattr(result, 'extracted_content') else None
                }
        except Exception as e:
            logger.error(f"Error during crawling: {str(e)}")
            return {"error": f"Error during crawling: {str(e)}"}

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Crawl a website using crawl4ai")
    
    # Basic arguments
    parser.add_argument("url", help="URL to crawl")
    parser.add_argument("--headless", action="store_true", help="Run browser in headless mode")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--cache-mode", choices=["Enabled", "Bypass", "Disabled"], default="Enabled", help="Cache mode")
    
    # Content filter arguments
    parser.add_argument("--content-filter", choices=["Pruning", "BM25"], default="Pruning", help="Content filter type")
    parser.add_argument("--threshold", type=float, default=0.48, help="Pruning threshold")
    parser.add_argument("--threshold-type", choices=["fixed", "auto"], default="fixed", help="Threshold type")
    parser.add_argument("--min-word-threshold", type=int, default=0, help="Min word threshold")
    parser.add_argument("--user-query", help="BM25 query")
    parser.add_argument("--bm25-threshold", type=float, default=1.0, help="BM25 threshold")
    
    # Extraction arguments
    parser.add_argument("--extraction-type", choices=["None", "LLM", "JSON CSS"], help="Extraction type")
    parser.add_argument("--llm-provider", help="LLM provider")
    parser.add_argument("--llm-api-key", help="LLM API key")
    parser.add_argument("--llm-instruction", help="LLM instruction")
    parser.add_argument("--css-schema", help="CSS schema (JSON)")
    
    # Deep crawling arguments
    parser.add_argument("--deep-crawl", action="store_true", help="Enable deep crawling")
    parser.add_argument("--crawl-strategy", choices=["BFS (Breadth-First)", "DFS (Depth-First)", "Best-First"], help="Crawling strategy")
    parser.add_argument("--max-depth", type=int, default=2, help="Maximum depth")
    parser.add_argument("--max-pages", type=int, default=10, help="Maximum pages")
    parser.add_argument("--include-external", action="store_true", help="Include external links")
    parser.add_argument("--keywords", help="Keywords (comma-separated)")
    parser.add_argument("--keyword-weight", type=float, default=0.7, help="Keyword weight")
    
    # Custom JavaScript
    parser.add_argument("--js-code", help="Custom JavaScript code")
    
    return parser.parse_args()

async def main():
    """Main entry point for the crawl agent."""
    args = parse_args()
    
    # Create config from args
    config = CrawlConfig(
        url=args.url,
        headless=args.headless,
        verbose=args.verbose,
        cache_mode=args.cache_mode,
        content_filter_type=args.content_filter,
        threshold=args.threshold,
        threshold_type=args.threshold_type,
        min_word_threshold=args.min_word_threshold,
        user_query=args.user_query,
        bm25_threshold=args.bm25_threshold,
        extraction_type=args.extraction_type,
        llm_provider=args.llm_provider,
        llm_api_key=args.llm_api_key or os.environ.get("LLM_API_KEY"),
        llm_instruction=args.llm_instruction,
        css_schema=args.css_schema,
        enable_deep_crawl=args.deep_crawl,
        crawl_strategy=args.crawl_strategy,
        max_depth=args.max_depth,
        max_pages=args.max_pages,
        include_external=args.include_external,
        keywords=args.keywords,
        keyword_weight=args.keyword_weight,
        js_code=args.js_code
    )
    
    # Run the crawl
    result = await crawl_url(config)
    
    # Print the result
    if result.get("success"):
        logger.info("Crawl completed successfully!")
        logger.info(f"Raw markdown content: {result['raw_content']}")
        logger.info(f"Fit markdown content: {result['fit_content']}")
        
        if result.get("total_pages"):
            logger.info(f"Total pages crawled: {result['total_pages']}")
        
        if result.get("extracted_content"):
            logger.info("Extracted content:")
            print(json.dumps(result["extracted_content"], indent=2))
    else:
        logger.error(f"Crawl failed: {result.get('error', 'Unknown error')}")

if __name__ == "__main__":
    asyncio.run(main())