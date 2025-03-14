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
    cache_mode: str = "ENABLED"  # "ENABLED", "BYPASS", "DISABLED", "READ_ONLY", "WRITE_ONLY"
    content_filter_type: str = "Pruning"  # "Pruning", "BM25"
    
    # Proxy settings
    use_proxy: bool = False
    proxy_server: Optional[str] = None
    proxy_username: Optional[str] = None
    proxy_password: Optional[str] = None
    
    # Pruning filter options
    threshold: float = 0.48
    threshold_type: str = "fixed"  # "fixed", "auto"
    min_word_threshold: int = 0
    
    # BM25 filter options
    user_query: str = ""
    bm25_threshold: float = 0.1
    
    # AI agent options
    use_ai_agent: bool = False
    analyze_website: bool = False
    enhance_content: bool = False
    store_results: bool = True
    ai_question: str = ""
    
    # Deep crawl options
    deep_crawl: bool = False
    deep_crawl_strategy: str = "BFS"  # "BFS", "DFS", "Best-First"
    max_depth: int = 2
    max_pages: int = 10
    follow_external_links: bool = False
    
    # Extraction strategy
    extraction_strategy: str = "Basic"  # "Basic", "LLM", "JSON CSS"
    css_selector: str = ""
    
    # JavaScript execution
    js_code: str = ""
    delay_before_return_html: int = 0
    wait_for: str = ""  # CSS selector or XPath
    
    # Output options
    remove_overlay_elements: bool = False
    save_raw_markdown: bool = False
    magic: bool = False  # Magic mode for better extraction
    word_count_threshold: int = 0
    excluded_tags: List[str] = []

def is_meaningful_content(content: str, min_length: int = 50, is_deep_crawl: bool = False) -> bool:
    """
    Check if the content is meaningful (not empty, not just whitespace)
    
    Args:
        content (str): The content to check
        min_length (int, optional): Minimum length to consider content meaningful. Defaults to 50.
        is_deep_crawl (bool, optional): Whether this is checking for deep crawl content. Deep crawl uses lower thresholds.
    
    Returns:
        bool: True if content is meaningful, False otherwise
    """
    # Check if content is None or empty
    if not content:
        return False
    
    # Remove whitespace and check length
    stripped_content = content.strip()
    
    # Use lower threshold for deep crawling content
    effective_min_length = 20 if is_deep_crawl else min_length
    
    # Check against minimum length
    if len(stripped_content) < effective_min_length:
        return False
    
    # Additional checks for meaningfulness can be added here
    # For example, checking against common placeholder or error texts
    meaningless_indicators = [
        'no content available', 
        'page not found', 
        '404', 
        'forbidden', 
        'access denied'
    ]
    
    # For deep crawl, only check for critical error indicators
    if is_deep_crawl:
        meaningless_indicators = ['page not found', '404 not found', 'access denied 403']
    
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
    
    # Add proxy configuration if enabled
    if config.use_proxy and config.proxy_server:
        proxy_config = {
            "server": config.proxy_server
        }
        
        # Add authentication if provided
        if config.proxy_username and config.proxy_password:
            proxy_config["username"] = config.proxy_username
            proxy_config["password"] = config.proxy_password
        
        # Log proxy usage (without credentials)
        if config.proxy_username:
            logger.info(f"Using proxy server with authentication: {config.proxy_server}")
        else:
            logger.info(f"Using proxy server: {config.proxy_server}")
            
        # Set proxy in browser config
        browser_config.proxy = proxy_config
    
    # Initialize AI agent if enabled
    if config.use_ai_agent:
        # Wrap AI agent initialization in try/except to ensure basic crawling still works
        try:
            from ai_agent import ai_agent
            has_ai_agent = True
            logger.info("AI agent loaded successfully")
        except Exception as e:
            logger.error(f"Error initializing AI agent module: {e}")
            has_ai_agent = False
            # Disable AI features if agent can't be initialized
            config.enhance_content = False
            config.analyze_website = False
            config.ai_question = ""
    else:
        has_ai_agent = False
    
    # Analyze website to determine optimal crawling strategy
    strategy = None
    if config.analyze_website and has_ai_agent:
        logger.info(f"Analyzing website {config.url} with AI agent...")
        try:
            strategy = await ai_agent.analyze_website(config.url)
            if strategy:
                logger.info(f"AI agent recommended strategy: {strategy.model_dump()}")
                # Apply AI agent recommendations
                config.deep_crawl = True
                config.deep_crawl_strategy = strategy.strategy_type
                config.max_depth = strategy.max_depth
                config.max_pages = strategy.max_pages
                config.content_filter_type = strategy.content_filter_type
                config.threshold = strategy.threshold
                # If keywords are provided, use best-first strategy
                if strategy.focus_keywords:
                    config.deep_crawl_strategy = "Best-First"
                    logger.info(f"Using focus keywords: {strategy.focus_keywords}")
        except Exception as e:
            logger.error(f"Error analyzing website with AI agent: {e}")
            # Fall back to default parameters if analysis fails
            strategy = None

    # Provide a default strategy if AI failed to generate one
    if config.analyze_website and not strategy:
        logger.warning("AI website analysis failed, using default strategy")
        # Set default values as a fallback
        strategy_dict = {
            "max_depth": 2,
            "max_pages": 10,
            "strategy_type": "bfs",
            "content_filter_type": "Pruning",
            "threshold": 0.48,
            "focus_keywords": []
        }
        logger.info(f"AI agent recommended strategy: {strategy_dict}")
    
    # Configure content filter
    content_filter = None
    if config.content_filter_type == "Pruning":
        content_filter = PruningContentFilter(
            threshold=config.threshold,
            threshold_type=config.threshold_type,
            min_word_threshold=config.min_word_threshold
        )
        logger.info(f"Using Pruning content filter with threshold={config.threshold}")
    elif config.content_filter_type == "BM25" and config.user_query:
        content_filter = BM25ContentFilter(
            query=config.user_query,
            threshold=config.bm25_threshold
        )
        logger.info(f"Using BM25 content filter with query='{config.user_query}'")
    
    # Configure extraction strategy
    extraction_strategy = None
    if config.extraction_strategy == "LLM":
        # Use LLM extraction if available
        pass
    elif config.extraction_strategy == "JSON CSS" and config.css_selector:
        try:
            schema = json.loads(config.css_selector)
            extraction_strategy = JsonCssExtractionStrategy(schema=schema)
            logger.info("Using JSON CSS extraction strategy")
        except Exception as e:
            logger.error(f"Error parsing CSS schema: {e}")
    
    # Setup deep crawling strategy if enabled
    deep_crawl_strategy = None
    if config.deep_crawl:
        if config.deep_crawl_strategy == "BFS":
            deep_crawl_strategy = BFSDeepCrawlStrategy(
                max_depth=config.max_depth,
                include_external=config.follow_external_links,
                max_pages=config.max_pages
            )
            logger.info(f"Using BFS deep crawl strategy with max_depth={config.max_depth}, max_pages={config.max_pages}")
        elif config.deep_crawl_strategy == "DFS":
            deep_crawl_strategy = DFSDeepCrawlStrategy(
                max_depth=config.max_depth,
                include_external=config.follow_external_links,
                max_pages=config.max_pages
            )
            logger.info(f"Using DFS deep crawl strategy with max_depth={config.max_depth}, max_pages={config.max_pages}")
        elif config.deep_crawl_strategy == "Best-First" and config.use_ai_agent and has_ai_agent:
            # Use AI agent to score URLs if available
            class AILinkScorer:
                def __init__(self, agent, query):
                    self.agent = agent
                    self.query = query
                
                async def score(self, page_url, links):
                    scored_links = self.agent.analyze_link_relevance(page_url, links, self.query)
                    return scored_links
            
            deep_crawl_strategy = BestFirstCrawlingStrategy(
                max_depth=config.max_depth,
                include_external=config.follow_external_links,
                max_pages=config.max_pages,
                url_scorer=AILinkScorer(ai_agent, config.user_query)
            )
            logger.info(f"Using AI-powered Best-First deep crawl strategy")
        # Fallback to regular BFS if Best-First is selected but AI agent is not available
        elif config.deep_crawl_strategy == "Best-First":
            logger.warning("Best-First strategy selected but AI agent not available. Using BFS instead.")
            deep_crawl_strategy = BFSDeepCrawlStrategy(
                max_depth=config.max_depth,
                include_external=config.follow_external_links,
                max_pages=config.max_pages
            )
            logger.info(f"Using BFS deep crawl strategy with max_depth={config.max_depth}, max_pages={config.max_pages}")
    
    # Map string cache modes to CacheMode enum values
    cache_mode_map = {
        "ENABLED": CacheMode.ENABLED,
        "DISABLED": CacheMode.DISABLED,
        "BYPASS": CacheMode.BYPASS,
        "READ_ONLY": CacheMode.READ_ONLY,
        "WRITE_ONLY": CacheMode.WRITE_ONLY
    }
    
    # Get the appropriate CacheMode enum value, defaulting to ENABLED
    cache_mode_enum = cache_mode_map.get(config.cache_mode, CacheMode.ENABLED)
    
    # Setup the crawler run configuration
    crawler_run_config = CrawlerRunConfig(
        cache_mode=cache_mode_enum,
        markdown_generator=DefaultMarkdownGenerator(),
        extraction_strategy=extraction_strategy,
        deep_crawl_strategy=deep_crawl_strategy,
        scraping_strategy=LXMLWebScrapingStrategy(),
        stream=False,  # Use non-streaming mode for deep crawling to avoid context variable errors
        wait_for=config.wait_for,
        delay_before_return_html=config.delay_before_return_html,
        word_count_threshold=config.word_count_threshold,
        excluded_tags=config.excluded_tags,
        magic=config.magic,
        remove_overlay_elements=config.remove_overlay_elements
    )
    
    # Set content filter if needed - set it as an attribute after initialization
    if content_filter:
        setattr(crawler_run_config, 'content_filter', content_filter)
    
    # Results container
    result = {
        "url": config.url,
        "crawl_time": time.time(),
        "status": "failed",
        "raw_content": "",
        "fit_content": "",
        "stats": {},
        "ai_enhanced_content": "",
        "ai_answer": ""
    }
    
    # Execute the crawl
    async with AsyncWebCrawler(config=browser_config) as crawler:
        try:
            if config.deep_crawl:
                try:
                    # Use a single crawler for deep crawling in non-streaming mode
                    logger.info(f"Starting deep crawl of {config.url}...")
                    crawl_results = await crawler.arun(config.url, crawler_run_config)
                    
                    if crawl_results:
                        logger.info(f"Deep crawl complete. Retrieved results.")
                        
                        # Combine all the raw markdown from successful pages
                        combined_raw = ""
                        combined_fit = ""
                        
                        # Check if result is a single CrawlResult or a list
                        if isinstance(crawl_results, list):
                            pages = crawl_results
                        else:
                            # If it's a single result object, check if it's a container
                            if hasattr(crawl_results, '_results'):
                                # It's a CrawlResultContainer
                                pages = crawl_results._results
                            else:
                                # Single result
                                pages = [crawl_results]
                        
                        logger.info(f"Processing {len(pages)} pages of results.")
                        
                        for idx, page in enumerate(pages, 1):
                            # Handle both dictionary-like objects and CrawlResult objects
                            try:
                                # Try accessing as CrawlResult object
                                url = getattr(page, "url", config.url if idx == 1 else "unknown")
                                # Try to get markdown content (newer version) or raw_markdown (older version)
                                raw_markdown = getattr(page, "markdown", None)
                                if raw_markdown is None:
                                    raw_markdown = getattr(page, "raw_markdown", "")
                                
                                # For the processed/fit content, try different possible attribute names
                                fit_markdown = getattr(page, "fit_markdown", None)
                                if fit_markdown is None:
                                    fit_markdown = getattr(page, "filtered_markdown", None)
                                if fit_markdown is None:
                                    fit_markdown = getattr(page, "processed_markdown", None)
                                if fit_markdown is None:
                                    # If no specialized attribute is found, use the raw markdown
                                    fit_markdown = raw_markdown
                            except AttributeError:
                                # Fall back to dictionary access if needed
                                url = page.get("url", config.url if idx == 1 else "unknown")
                                raw_markdown = page.get("markdown", "") or page.get("raw_markdown", "")
                                fit_markdown = page.get("fit_markdown", "") or page.get("filtered_markdown", "") or page.get("processed_markdown", "") or raw_markdown
                            
                            if raw_markdown:
                                logger.info(f"Found raw markdown for page {idx}: {len(raw_markdown)} chars")
                                combined_raw += f"\n\n## Page: {url}\n\n{raw_markdown}"
                            
                            if fit_markdown:
                                combined_fit += f"\n\n## Page: {url}\n\n{fit_markdown}"
                        
                        if combined_raw:
                            logger.info(f"Combined raw markdown length: {len(combined_raw)}")
                            result["raw_content"] = combined_raw
                        
                        if combined_fit:
                            logger.info(f"Combined fit markdown length: {len(combined_fit)}")
                            result["fit_content"] = combined_fit
                        
                        # Generate stats
                        result["stats"] = {
                            "pages_crawled": len(pages),
                            "successful_pages": sum(1 for p in pages if (hasattr(p, "markdown") and p.markdown) or 
                                                   (hasattr(p, "raw_markdown") and p.raw_markdown) or 
                                                   (hasattr(p, "get") and (p.get("markdown") or p.get("raw_markdown")))),
                            "total_content_length": len(combined_raw)
                        }
                        
                        # Use AI agent for content enhancement if enabled
                        if config.use_ai_agent and config.enhance_content and combined_raw and has_ai_agent:
                            try:
                                logger.info("Enhancing content with AI agent...")
                                enhanced_content = await ai_agent.enhance_content(combined_raw, config.user_query)
                                if enhanced_content:
                                    result["ai_enhanced_content"] = enhanced_content
                                    logger.info(f"AI enhanced content length: {len(enhanced_content)}")
                                else:
                                    # If enhance_content returns None or empty, use the raw content
                                    result["ai_enhanced_content"] = combined_raw
                                    logger.warning("AI enhancement returned empty result, using raw content")
                            except Exception as e:
                                logger.error(f"Error enhancing content with AI agent: {e}")
                                # Use raw content as a fallback
                                result["ai_enhanced_content"] = combined_raw
                                logger.info(f"Using raw content as fallback for AI enhancement ({len(combined_raw)} chars)")
                        
                        # Use AI agent to answer question if provided
                        if config.use_ai_agent and config.ai_question and combined_raw and has_ai_agent:
                            try:
                                logger.info(f"Answering question with AI agent: {config.ai_question}")
                                answer = await ai_agent.answer_question(config.ai_question, combined_raw)
                                if answer:
                                    result["ai_answer"] = answer
                                    logger.info(f"AI answer length: {len(answer)}")
                                else:
                                    # If answer_question returns None or empty, provide a generic message
                                    result["ai_answer"] = "Unable to generate answer. Please try again with a different question."
                                    logger.warning("AI answer returned empty result")
                            except Exception as e:
                                logger.error(f"Error answering question with AI agent: {e}")
                                result["ai_answer"] = f"An error occurred while generating the answer: {str(e)}"
                                logger.info("Set error message as AI answer due to exception")
                        
                        # Store results in AI agent memory if enabled
                        if config.use_ai_agent and config.store_results and combined_raw and has_ai_agent:
                            try:
                                ai_agent.store_crawl_results(
                                    config.url,
                                    combined_raw,
                                    {
                                        "crawl_time": result["crawl_time"],
                                        "pages_crawled": len(pages),
                                        "user_query": config.user_query
                                    }
                                )
                                logger.info("Stored crawl results in AI agent memory")
                            except Exception as e:
                                logger.error(f"Error storing crawl results in AI agent memory: {e}")
                                # Continue execution even if storage fails
                        
                        result["status"] = "success"
                        logger.info(f"Generated raw markdown content ({len(combined_raw)} chars)")
                        logger.info(f"Generated fit markdown content ({len(combined_fit)} chars)")
                    else:
                        logger.warning("No pages retrieved from deep crawl")
                
                except Exception as e:
                    logger.error(f"Error during deep crawl: {e}")
            else:
                # Single page crawl
                crawler_result = await crawler.arun(config.url, crawler_run_config)
                
                # Handle different types of results (CrawlResultContainer, CrawlResult, dict)
                if crawler_result:
                    raw_content = ""
                    fit_content = ""
                    
                    # Check if it's a container with _results
                    if hasattr(crawler_result, '_results') and crawler_result._results:
                        # Use the first result in the container
                        page = crawler_result._results[0]
                        try:
                            # Try to access as a CrawlResult object
                            raw_content = getattr(page, "markdown", None)
                            if raw_content is None:
                                raw_content = getattr(page, "raw_markdown", "")
                            
                            # For the processed/fit content, try different possible attribute names
                            fit_content = getattr(page, "fit_markdown", None)
                            if fit_content is None:
                                fit_content = getattr(page, "filtered_markdown", None)
                            if fit_content is None:
                                fit_content = getattr(page, "processed_markdown", None)
                            if fit_content is None:
                                # If no specialized attribute is found, use the raw markdown
                                fit_content = raw_content
                        except AttributeError:
                            logger.warning(f"Unexpected result structure in container")
                    else:
                        # Try to access as a direct CrawlResult object
                        try:
                            raw_content = getattr(crawler_result, "markdown", None)
                            if raw_content is None:
                                raw_content = getattr(crawler_result, "raw_markdown", "")
                            
                            # For the processed/fit content, try different possible attribute names
                            fit_content = getattr(crawler_result, "fit_markdown", None)
                            if fit_content is None:
                                fit_content = getattr(crawler_result, "filtered_markdown", None)
                            if fit_content is None:
                                fit_content = getattr(crawler_result, "processed_markdown", None)
                            if fit_content is None:
                                # If no specialized attribute is found, use the raw markdown
                                fit_content = raw_content
                        except AttributeError:
                            # Fall back to dictionary access
                            if isinstance(crawler_result, dict):
                                raw_content = crawler_result.get("markdown", "") or crawler_result.get("raw_markdown", "")
                                fit_content = crawler_result.get("fit_markdown", "") or crawler_result.get("filtered_markdown", "") or crawler_result.get("processed_markdown", "") or raw_content
                            else:
                                logger.warning(f"Unexpected result type: {type(crawler_result)}")
                    
                    if raw_content:
                        result["raw_content"] = raw_content
                        logger.info(f"Generated raw markdown content ({len(raw_content)} chars)")
                    
                    if fit_content:
                        result["fit_content"] = fit_content
                    
                    # Add stats to the result
                    result["stats"] = {
                        "pages_crawled": 1,
                        "successful_pages": 1 if raw_content else 0,
                        "total_content_length": len(raw_content)
                    }
                    
                    # Use AI agent for content enhancement if enabled
                    if config.use_ai_agent and config.enhance_content and raw_content and has_ai_agent:
                        try:
                            logger.info("Enhancing content with AI agent...")
                            enhanced_content = await ai_agent.enhance_content(raw_content, config.user_query)
                            if enhanced_content:
                                result["ai_enhanced_content"] = enhanced_content
                                logger.info(f"AI enhanced content length: {len(enhanced_content)}")
                            else:
                                # If enhance_content returns None or empty, use the raw content
                                result["ai_enhanced_content"] = raw_content
                                logger.warning("AI enhancement returned empty result, using raw content")
                        except Exception as e:
                            logger.error(f"Error enhancing content with AI agent: {e}")
                            # Use raw content as a fallback
                            result["ai_enhanced_content"] = raw_content
                            logger.info(f"Using raw content as fallback for AI enhancement ({len(raw_content)} chars)")
                        
                        # Use AI agent to answer question if provided
                        if config.use_ai_agent and config.ai_question and raw_content and has_ai_agent:
                            try:
                                logger.info(f"Answering question with AI agent: {config.ai_question}")
                                answer = await ai_agent.answer_question(config.ai_question, raw_content)
                                if answer:
                                    result["ai_answer"] = answer
                                    logger.info(f"AI answer length: {len(answer)}")
                                else:
                                    # If answer_question returns None or empty, provide a generic message
                                    result["ai_answer"] = "Unable to generate answer. Please try again with a different question."
                                    logger.warning("AI answer returned empty result")
                            except Exception as e:
                                logger.error(f"Error answering question with AI agent: {e}")
                                result["ai_answer"] = f"An error occurred while generating the answer: {str(e)}"
                                logger.info("Set error message as AI answer due to exception")
                        
                        # Store results in AI agent memory if enabled
                        if config.use_ai_agent and config.store_results and raw_content and has_ai_agent:
                            try:
                                ai_agent.store_crawl_results(
                                    config.url,
                                    raw_content,
                                    {
                                        "crawl_time": result["crawl_time"],
                                        "user_query": config.user_query
                                    }
                                )
                                logger.info("Stored crawl results in AI agent memory")
                            except Exception as e:
                                logger.error(f"Error storing crawl results in AI agent memory: {e}")
                                # Continue execution even if storage fails
                        
                        result["status"] = "success"
                    else:
                        logger.warning("No content retrieved from URL")
            
            # Save raw markdown to file if requested
            if config.save_raw_markdown and result["raw_content"]:
                filename = f"crawl4ai_raw_{time.strftime('%Y%m%d_%H%M%S')}.md"
                with open(filename, "w", encoding="utf-8") as f:
                    f.write(result["raw_content"])
                logger.info(f"Saved raw markdown to {filename}")
        
        except Exception as e:
            logger.error(f"Error during crawl: {e}")
            result["error"] = str(e)
    
    return result

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Crawl a website using crawl4ai")
    
    # Basic arguments
    parser.add_argument("url", help="URL to crawl")
    parser.add_argument("--headless", action="store_true", help="Run browser in headless mode")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--cache-mode", choices=["ENABLED", "BYPASS", "DISABLED", "READ_ONLY", "WRITE_ONLY"], default="ENABLED", help="Cache mode")
    
    # Content filter arguments
    parser.add_argument("--content-filter", choices=["Pruning", "BM25"], default="Pruning", help="Content filter type")
    parser.add_argument("--threshold", type=float, default=0.48, help="Pruning threshold")
    parser.add_argument("--threshold-type", choices=["fixed", "auto"], default="fixed", help="Threshold type")
    parser.add_argument("--min-word-threshold", type=int, default=0, help="Min word threshold")
    parser.add_argument("--user-query", help="BM25 query")
    parser.add_argument("--bm25-threshold", type=float, default=0.1, help="BM25 threshold")
    
    # AI agent arguments
    parser.add_argument("--use-ai-agent", action="store_true", help="Enable AI agent")
    parser.add_argument("--analyze-website", action="store_true", help="Analyze website")
    parser.add_argument("--enhance-content", action="store_true", help="Enhance content")
    parser.add_argument("--store-results", action="store_true", help="Store results")
    parser.add_argument("--ai-question", help="AI question")
    
    # Deep crawling arguments
    parser.add_argument("--deep-crawl", action="store_true", help="Enable deep crawling")
    parser.add_argument("--crawl-strategy", choices=["BFS", "DFS", "Best-First"], help="Crawling strategy")
    parser.add_argument("--max-depth", type=int, default=2, help="Maximum depth")
    parser.add_argument("--max-pages", type=int, default=10, help="Maximum pages")
    parser.add_argument("--follow-external-links", action="store_true", help="Follow external links")
    
    # Extraction strategy
    parser.add_argument("--extraction-strategy", choices=["Basic", "LLM", "JSON CSS"], help="Extraction strategy")
    parser.add_argument("--css-selector", help="CSS selector")
    
    # JavaScript execution
    parser.add_argument("--js-code", help="Custom JavaScript code")
    parser.add_argument("--delay-before-return-html", type=int, default=0, help="Delay before returning HTML")
    parser.add_argument("--wait-for", help="CSS selector or XPath to wait for")
    
    # Output options
    parser.add_argument("--remove-overlay-elements", action="store_true", help="Remove overlay elements")
    parser.add_argument("--save-raw-markdown", action="store_true", help="Save raw markdown")
    parser.add_argument("--magic", action="store_true", help="Enable magic mode")
    parser.add_argument("--word-count-threshold", type=int, default=0, help="Word count threshold")
    parser.add_argument("--excluded-tags", nargs="+", help="Excluded tags")
    
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
        extraction_strategy=args.extraction_strategy,
        css_selector=args.css_selector,
        deep_crawl=args.deep_crawl,
        deep_crawl_strategy=args.crawl_strategy,
        max_depth=args.max_depth,
        max_pages=args.max_pages,
        follow_external_links=args.follow_external_links,
        use_ai_agent=args.use_ai_agent,
        analyze_website=args.analyze_website,
        enhance_content=args.enhance_content,
        store_results=args.store_results,
        ai_question=args.ai_question,
        js_code=args.js_code,
        delay_before_return_html=args.delay_before_return_html,
        wait_for=args.wait_for,
        remove_overlay_elements=args.remove_overlay_elements,
        save_raw_markdown=args.save_raw_markdown,
        magic=args.magic,
        word_count_threshold=args.word_count_threshold,
        excluded_tags=args.excluded_tags
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