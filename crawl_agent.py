import asyncio
import json
import logging
import time
import os
import argparse
from typing import Dict, Any, List, Optional, Tuple, Union
from pydantic import BaseModel
from crawl4ai import (AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode,
                      PruningContentFilter, BM25ContentFilter,
                      LLMExtractionStrategy, JsonCssExtractionStrategy, JsonXPathExtractionStrategy,
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
    
    # SSL Certificate settings
    ignore_https_errors: bool = False  # Ignore HTTPS errors (invalid certificates)
    cert_file: Optional[str] = None  # Path to custom certificate file
    
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
    openai_api_key: Optional[str] = None
    
    # Deep crawl options
    deep_crawl: bool = False
    deep_crawl_strategy: str = "BFS"  # "BFS", "DFS", "Best-First"
    max_depth: int = 2
    max_pages: int = 10
    follow_external_links: bool = False
    
    # Extraction strategy
    extraction_strategy: str = "Basic"  # "Basic", "LLM", "JSON CSS"
    css_selector: str = ""
    
    # URL filtering options
    exclude_external_links: bool = False
    exclude_social_media_links: bool = False
    exclude_social_media_domains: List[str] = []
    exclude_domains: List[str] = []
    
    # Media filtering options
    exclude_external_images: bool = False
    
    # File downloading options
    download_pdfs: bool = False
    download_images: bool = False
    download_documents: bool = False
    download_dir: str = "downloads"
    max_file_size_mb: int = 10
    
    # Lazy loading options
    enable_lazy_loading: bool = False
    lazy_load_max_scrolls: int = 10
    lazy_load_scroll_step: int = 800
    lazy_load_wait_time: float = 1.0
    
    # Multi-URL crawling
    urls: List[str] = []
    batch_crawl: bool = False
    parallel_crawl: bool = False
    max_concurrent: int = 5
    
    # Content chunking options
    enable_chunking: bool = False
    chunking_strategy: str = "semantic"  # "fixed", "sentence", "semantic"
    chunk_size: int = 4000
    chunk_overlap: int = 200
    
    # Content clustering options
    enable_clustering: bool = False
    clustering_strategy: str = "kmeans"  # "kmeans", "hierarchical" 
    n_clusters: int = 5
    
    # Page Interaction options
    # JavaScript execution
    js_code: str = ""
    
    # JavaScript execution
    js_only: bool = False  # Whether to run JS in the existing page without navigation
    session_id: Optional[str] = None  # Session ID for multi-step interactions
    page_timeout: int = 60000  # Page load timeout in milliseconds
    delay_before_return_html: int = 0  # Delay in seconds after page load
    wait_for: str = ""  # CSS selector or JS expression to wait for (prefix with "css:" or "js:")
    
    # Lazy Loading options
    enable_lazy_loading: bool = False  # Automatically scroll and wait for lazy content
    lazy_load_scroll_step: int = 800  # Pixels to scroll each step 
    lazy_load_max_scrolls: int = 5  # Maximum number of scroll operations
    lazy_load_wait_time: int = 1000  # Milliseconds to wait between scrolls
    
    # Authentication & Hooks
    auth_hook_js: str = ""  # Custom JavaScript for authentication
    pre_request_hook_js: str = ""  # JavaScript to run before page load
    post_request_hook_js: str = ""  # JavaScript to run after page load but before extraction
    
    # Advanced interaction options
    simulate_user: bool = False  # Simulate human-like behavior
    override_navigator: bool = False  # Override navigator properties to avoid bot detection
    mean_delay: float = 1.0  # Mean delay between requests in seconds (for multi-URL crawling)
    max_range: float = 0.5  # Max variance of delay between requests
    
    # Multi-step interaction
    multi_step_enabled: bool = False  # Enable multi-step interaction
    multi_step_js_actions: List[str] = []  # List of JS actions to perform sequentially
    multi_step_wait_conditions: List[str] = []  # List of wait conditions for each step
    multi_step_delays: List[int] = []  # Delays after each step in seconds
    
    # Output options
    remove_overlay_elements: bool = False
    save_raw_markdown: bool = False
    magic: bool = False  # Magic mode for anti-bot detection
    
    # Content selection options
    word_count_threshold: int = 0
    excluded_tags: List[str] = ["script", "style", "svg", "noscript"]
    process_iframes: bool = False  # Merge iframe content into the final output
    
    # Link filtering options
    exclude_external_links: bool = False  # Skip links pointing to external domains
    exclude_social_media_links: bool = False  # Skip links pointing to social media sites
    exclude_domains: List[str] = []  # Custom list of domains to exclude
    exclude_social_media_domains: List[str] = []  # Custom list of social media domains to exclude
    
    # Media filtering options
    exclude_external_images: bool = False  # Don't include images from external domains
    
    # File downloading options
    download_pdf: bool = False  # Download PDF files found during crawling
    download_docs: bool = False  # Download document files (doc, docx, xls, xlsx, etc.)
    download_path: str = "./downloads"  # Path to save downloaded files
    
    # Multi-URL crawling options
    crawl_in_parallel: bool = False  # Crawl multiple URLs in parallel
    max_concurrent_crawls: int = 3  # Maximum number of concurrent crawls
    
    # Additional HTML filtering
    keep_data_attributes: bool = False  # Keep data-* attributes in HTML
    keep_attrs: List[str] = []  # List of specific attributes to keep
    remove_forms: bool = False  # Remove form elements from output

def is_meaningful_content(content: str, min_length: int = 50, is_deep_crawl: bool = False) -> bool:
    """
    Check if the content is meaningful enough to process.
    
    Args:
        content: The content to check
        min_length: Minimum length of content to be considered meaningful
        is_deep_crawl: Whether this is part of a deep crawl
        
    Returns:
        Whether the content is meaningful
    """
    if not content:
        return False
    
    # For deep crawls, we might want to keep shorter content
    if is_deep_crawl:
        min_length = min_length // 2
    
    # Check length
    if len(content) < min_length:
        return False
    
    # Check if it's mostly whitespace
    content_no_whitespace = content.strip()
    if not content_no_whitespace:
        return False
    
    return True


def get_content_from_result(result: Any) -> Optional[str]:
    """
    Extract the raw content from a crawler result.
    
    Args:
        result: The result from the crawler
        
    Returns:
        The raw content or None if extraction failed
    """
    # Handle CrawlResultContainer
    if hasattr(result, '_results'):
        # It's a CrawlResultContainer, try to get markdown from first result
        if result._results and len(result._results) > 0:
            first_result = result._results[0]
            # Try markdown, then raw_markdown, then html
            if hasattr(first_result, 'markdown'):
                return first_result.markdown
            elif hasattr(first_result, 'raw_markdown'):
                return first_result.raw_markdown
            elif hasattr(first_result, 'html'):
                return f"HTML content (no markdown): {len(first_result.html)} chars"
    
    # Handle CrawlResult directly
    if hasattr(result, 'markdown'):
        return result.markdown
    elif hasattr(result, 'raw_markdown'):
        return result.raw_markdown
    
    # Try dict-like access patterns
    try:
        if isinstance(result, dict) or hasattr(result, 'get'):
            # Try different attribute names that might contain the content
            for attr in ['markdown', 'raw_markdown', 'raw_content', 'content']:
                content = result.get(attr)
                if content:
                    return content
    except (AttributeError, TypeError):
        pass
    
    # Last resort: check if the result itself is a string
    if isinstance(result, str):
        return result
    
    # Couldn't extract content
    logger.warning(f"Couldn't extract content from result of type {type(result)}")
    return None


def get_fit_content_from_result(result: Any) -> Optional[str]:
    """
    Extract the fit content from a crawler result.
    
    Args:
        result: The result from the crawler
        
    Returns:
        The fit content or None if extraction failed
    """
    # Handle CrawlResultContainer
    if hasattr(result, '_results'):
        # It's a CrawlResultContainer, try to get fit_markdown from first result
        if result._results and len(result._results) > 0:
            first_result = result._results[0]
            # Try fit_markdown
            if hasattr(first_result, 'fit_markdown'):
                return first_result.fit_markdown
    
    # Handle CrawlResult directly
    if hasattr(result, 'fit_markdown'):
        return result.fit_markdown
    
    # Try dict-like access patterns
    try:
        if isinstance(result, dict) or hasattr(result, 'get'):
            # Try different attribute names that might contain the content
            for attr in ['fit_markdown', 'fit_content']:
                content = result.get(attr)
                if content:
                    return content
    except (AttributeError, TypeError):
        pass
    
    # No fit content found, return None
    return None


async def crawl_url(config: CrawlConfig) -> Dict[str, Any]:
    """
    Crawl a website using crawl4ai with the provided configuration.
    
    Args:
        config: The crawl configuration
        
    Returns:
        The crawl results
    """
    # Start timing the crawl
    start_time = time.time()
    
    # Check if we're in multi-URL mode
    if config.urls and len(config.urls) > 0:
        return await crawl_multiple_urls(config)
        
    logger.info(f"Starting crawl of {config.url}")
    
    # Set up browser configuration
    browser_config = BrowserConfig(
        headless=config.headless,
        verbose=config.verbose,
        ignore_https_errors=config.ignore_https_errors
    )
    
    # Add SSL certificate settings if provided
    if config.cert_file:
        if os.path.exists(config.cert_file):
            browser_config.cert_file = config.cert_file
            logger.info(f"Using custom SSL certificate: {config.cert_file}")
        else:
            logger.warning(f"Certificate file not found: {config.cert_file}")
    
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
    
    # Setup extraction strategy
    extraction_strategy = None
    
    # Setup LLM extraction if selected
    if config.extraction_strategy == "LLM" and config.openai_api_key:
        try:
            os.environ["OPENAI_API_KEY"] = config.openai_api_key
            extraction_strategy = LLMExtractionStrategy()
            logger.info("Using LLM extraction strategy")
        except Exception as e:
            logger.error(f"Error setting up LLM extraction: {e}")
    
    # Setup JSON CSS extraction if selected
    elif config.extraction_strategy == "CSS Selectors" and config.css_selector:
        try:
            schema = json.loads(config.css_selector)
            extraction_strategy = JsonCssExtractionStrategy(schema=schema)
            logger.info("Using JSON CSS extraction strategy")
        except Exception as e:
            logger.error(f"Error parsing CSS schema: {e}")
    
    # Setup JSON XPath extraction if selected
    elif config.extraction_strategy == "XPath Selectors" and config.css_selector:
        try:
            schema = json.loads(config.css_selector)
            extraction_strategy = JsonXPathExtractionStrategy(schema=schema)
            logger.info("Using JSON XPath extraction strategy")
        except Exception as e:
            logger.error(f"Error parsing XPath schema: {e}")
    
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
    
    # Map string cache modes to CacheMode enum values - make case insensitive
    cache_mode_map = {
        "ENABLED": CacheMode.ENABLED,
        "DISABLED": CacheMode.DISABLED,
        "BYPASS": CacheMode.BYPASS,
        "READ_ONLY": CacheMode.READ_ONLY,
        "WRITE_ONLY": CacheMode.WRITE_ONLY,
        # Add lowercase variants for robustness
        "enabled": CacheMode.ENABLED,
        "disabled": CacheMode.DISABLED,
        "bypass": CacheMode.BYPASS,
        "read_only": CacheMode.READ_ONLY,
        "write_only": CacheMode.WRITE_ONLY
    }
    
    # Check what values are actually in the CacheMode enum
    valid_cache_modes = [mode.name for mode in CacheMode]
    logger.info(f"Valid cache modes: {valid_cache_modes}")
    
    # Get the appropriate CacheMode enum value
    if hasattr(CacheMode, config.cache_mode):
        # Direct enum access if name matches
        cache_mode_enum = getattr(CacheMode, config.cache_mode)
        logger.info(f"Using cache mode: {cache_mode_enum}")
    elif config.cache_mode.upper() in cache_mode_map:
        # Fallback to our mapping
        cache_mode_enum = cache_mode_map[config.cache_mode.upper()]
        logger.info(f"Using mapped cache mode: {cache_mode_enum}")
    else:
        # Default to ENABLED
        logger.warning(f"Unrecognized cache mode '{config.cache_mode}', using ENABLED instead")
        cache_mode_enum = CacheMode.ENABLED
    
    # Setup the crawler run configuration
    crawler_run_config = CrawlerRunConfig(
        cache_mode=cache_mode_enum,
        markdown_generator=DefaultMarkdownGenerator(),
        extraction_strategy=extraction_strategy,
        deep_crawl_strategy=deep_crawl_strategy,
        scraping_strategy=LXMLWebScrapingStrategy(),
        stream=False,  # Use non-streaming mode for deep crawling to avoid context variable errors
        
        # Basic interaction parameters
        js_code=config.js_code,
        wait_for=config.wait_for,
        delay_before_return_html=config.delay_before_return_html,
        page_timeout=config.page_timeout,
        
        # Advanced interaction parameters
        magic=config.magic,
        remove_overlay_elements=config.remove_overlay_elements,
        process_iframes=config.process_iframes,
        simulate_user=config.simulate_user,
        override_navigator=config.override_navigator,
        
        # Content selection options
        word_count_threshold=config.word_count_threshold,
        excluded_tags=config.excluded_tags,
        
        # Link filtering options
        exclude_external_links=config.exclude_external_links,
        exclude_social_media_links=config.exclude_social_media_links,
        exclude_domains=config.exclude_domains,
        exclude_social_media_domains=config.exclude_social_media_domains,
        
        # Media filtering options
        exclude_external_images=config.exclude_external_images,
        
        # Additional HTML filtering
        keep_data_attributes=config.keep_data_attributes,
        keep_attrs=config.keep_attrs,
        remove_forms=config.remove_forms,
        
        # Session parameters for multi-step interaction
        js_only=config.js_only if not config.multi_step_enabled else False,  # Only use in first step if multi-step enabled
        session_id=None if not config.multi_step_enabled else config.session_id,
        
        # Multi-URL batch processing parameters
        mean_delay=config.mean_delay,
        max_range=config.max_range
    )
    
    # Apply custom hooks and lazy loading settings via JavaScript if enabled
    # (since these aren't directly supported by CrawlerRunConfig)
    if config.pre_request_hook_js:
        existing_js = crawler_run_config.js_code or ""
        crawler_run_config.js_code = config.pre_request_hook_js + ";\n" + existing_js
    
    if config.post_request_hook_js:
        existing_js = crawler_run_config.js_code or ""
        crawler_run_config.js_code = existing_js + ";\n" + config.post_request_hook_js
    
    if config.auth_hook_js:
        existing_js = crawler_run_config.js_code or ""
        crawler_run_config.js_code = config.auth_hook_js + ";\n" + existing_js
    
    # Implement lazy loading via JavaScript if enabled
    if config.enable_lazy_loading:
        lazy_loading_js = f"""
        // Auto-scroll for lazy loading
        async function autoScroll() {{
            for (let i = 0; i < {config.lazy_load_max_scrolls}; i++) {{
                window.scrollBy(0, {config.lazy_load_scroll_step});
                // Wait for content to load
                await new Promise(resolve => setTimeout(resolve, {config.lazy_load_wait_time}));
            }}
        }}
        await autoScroll();
        """
        existing_js = crawler_run_config.js_code or ""
        crawler_run_config.js_code = existing_js + ";\n" + lazy_loading_js
    
    # Implement file downloading through JavaScript if enabled
    if config.download_pdf or config.download_images or config.download_docs:
        # Create a directory for downloads if it doesn't exist
        os.makedirs(config.download_path, exist_ok=True)
        
        # Initialize an empty list to track downloaded files
        # This will be added to the result dictionary later
        downloaded_files = []
        
        # Get file URLs based on type
        download_js = f"""
        // Get file URLs to download
        const downloadUrls = [];
        
        // Helper function to check file extension
        function hasExtension(url, extensions) {{
            const urlLower = url.toLowerCase();
            return extensions.some(ext => urlLower.endsWith('.' + ext));
        }}
        
        // Collect links based on file type
        document.querySelectorAll('a[href]').forEach(link => {{
            const url = link.href;
            
            if ({str(config.download_pdf).lower()} && hasExtension(url, ['pdf'])) {{
                downloadUrls.push({{url: url, type: 'pdf'}});
            }}
            
            if ({str(config.download_images).lower()} && hasExtension(url, ['jpg', 'jpeg', 'png', 'gif', 'webp'])) {{
                downloadUrls.push({{url: url, type: 'image'}});
            }}
            
            if ({str(config.download_docs).lower()} && hasExtension(url, ['doc', 'docx', 'xls', 'xlsx', 'ppt', 'pptx', 'txt', 'csv'])) {{
                downloadUrls.push({{url: url, type: 'document'}});
            }}
        }});
        
        // Also check for image tags
        if ({str(config.download_images).lower()}) {{
            document.querySelectorAll('img[src]').forEach(img => {{
                if (img.src && (img.src.startsWith('http://') || img.src.startsWith('https://'))) {{
                    downloadUrls.push({{url: img.src, type: 'image'}});
                }}
            }});
        }}
        
        // Return the URLs to be downloaded separately
        return JSON.stringify(downloadUrls);
        """
        
        # Store the download JavaScript for later use, but don't add it to crawler_run_config.js_code
        # because we'll need to handle the downloads manually after the page is crawled
        setattr(crawler_run_config, '_download_js', download_js)
    
    # Set content filter if needed - set it as an attribute after initialization
    if content_filter:
        setattr(crawler_run_config, 'content_filter', content_filter)
    
    # Initialize variables for the crawl result
    html_content = ""
    markdown_content = ""
    extraction_result = {}
    link_count = 0
    image_count = 0
    excluded_links = []
    excluded_images = []
    downloaded_files = []  # Re-initialize this here to ensure it's available in all code paths
    
    # Execute the crawl using AsyncWebCrawler
    try:
        async with AsyncWebCrawler(config=browser_config) as crawler:
            # Run the crawler with our configuration
            crawl_result = await crawler.arun(config.url, crawler_run_config)
            
            # Extract data from the crawl result
            if hasattr(crawl_result, 'html'):
                html_content = crawl_result.html
            
            if hasattr(crawl_result, 'markdown'):
                markdown_content = crawl_result.markdown
                logger.info(f"Generated raw markdown content ({len(markdown_content)} chars)")
            else:
                logger.warning("No content retrieved from URL")
            
            # Try to get fit content if available
            fit_content = get_fit_content_from_result(crawl_result)
            if fit_content:
                logger.info(f"Found fit content ({len(fit_content)} chars)")
            else:
                # If no fit content found, use raw markdown as fallback
                fit_content = markdown_content
                logger.info("Using raw markdown as fit content")
            
            # Get extraction results if available
            if hasattr(crawl_result, 'extraction'):
                extraction_result = crawl_result.extraction
            
            # Get statistics if available
            if hasattr(crawl_result, 'stats'):
                stats = crawl_result.stats
                link_count = stats.get('link_count', 0)
                image_count = stats.get('image_count', 0)
                excluded_links = stats.get('filtered_links', [])
                excluded_images = stats.get('filtered_images', [])
    except Exception as e:
        logger.error(f"Error during crawling: {e}")
        # Return a result with error information
        return {
            "url": config.url,
            "title": config.url,
            "status": "failed",
            "error": str(e),
            "content": "",
            "markdown": "",
            "raw_content": "",  # Add raw_content for app.py compatibility
            "fit_content": "",  # Add fit_content for app.py compatibility
            "extraction": {},
            "stats": {
                "crawl_time": time.time() - start_time,
                "link_count": 0,
                "image_count": 0,
                "filtered_links": [],
                "filtered_images": [],
                "downloaded_files": []
            }
        }
    
    # Process the result for output
    result = {
        "url": config.url,
        "title": config.url,  # Use URL as fallback since title() method is not available
        "status": "success",  # Add success status here
        "content": html_content,
        "markdown": markdown_content,
        "raw_content": markdown_content,  # Add raw_content for app.py compatibility
        "fit_content": fit_content,
        "extraction": extraction_result,
        "stats": {
            "crawl_time": time.time() - start_time,
            "link_count": link_count,
            "image_count": image_count,
            "filtered_links": excluded_links,
            "filtered_images": excluded_images,
            "downloaded_files": downloaded_files
        }
    }
    
    # Try to extract title from the HTML content if available
    if html_content:
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html_content, 'html.parser')
            title_tag = soup.find('title')
            if title_tag and title_tag.text:
                result["title"] = title_tag.text.strip()
        except Exception as e:
            logger.warning(f"Could not extract title from HTML: {e}")

    # Note: The result dictionary was already populated earlier, no need to redefine it

    # Apply chunking if enabled
    if config.enable_chunking:
        try:
            from text_chunking import get_chunker
            
            # Get the appropriate chunker
            chunker = get_chunker(
                chunker_type=config.chunking_strategy,
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap,
                max_chunk_size=config.chunk_size,
                min_chunk_size=config.chunk_size // 10
            )
            
            # Chunk the content
            if html_content:
                html_chunks = chunker.chunk_html(html_content)
                result["html_chunks"] = html_chunks
                logger.info(f"Created {len(html_chunks)} HTML chunks using {config.chunking_strategy} strategy")
            
            if markdown_content:
                text_chunks = chunker.chunk_text(markdown_content)
                result["markdown_chunks"] = text_chunks
                logger.info(f"Created {len(text_chunks)} markdown chunks using {config.chunking_strategy} strategy")
                
        except Exception as e:
            logger.error(f"Error during text chunking: {e}")
            # Don't interrupt the flow if chunking fails
    
    # Apply clustering if enabled
    if config.enable_clustering and markdown_content:
        try:
            from clustering_strategies import get_clustering_strategy
            
            # Get the appropriate clustering strategy
            clustering = get_clustering_strategy(
                strategy_name=config.clustering_strategy,
                n_clusters=config.n_clusters
            )
            
            # Use chunks if available, otherwise use the full content
            if config.enable_chunking and "markdown_chunks" in result and result["markdown_chunks"]:
                texts = result["markdown_chunks"]
            else:
                # Create simple paragraph-based chunks for clustering
                texts = [p for p in markdown_content.split("\n\n") if p.strip()]
            
            # Apply clustering
            if texts:
                cluster_labels, cluster_model = clustering.cluster(texts)
                cluster_summary = clustering.get_cluster_summary(texts, cluster_labels)
                
                # Get keywords for each cluster
                cluster_keywords = {}
                for cluster_idx in range(max(cluster_labels) + 1 if cluster_labels else 0):
                    keywords = clustering.get_cluster_keywords(cluster_model, cluster_idx)
                    cluster_keywords[cluster_idx] = keywords
                
                # Add clustering results to output
                result["clustering"] = {
                    "strategy": config.clustering_strategy,
                    "n_clusters": config.n_clusters,
                    "labels": cluster_labels,
                    "summary": cluster_summary,
                    "keywords": cluster_keywords
                }
                
                logger.info(f"Applied {config.clustering_strategy} clustering, created {len(cluster_summary)} clusters")
            
        except Exception as e:
            logger.error(f"Error during clustering: {e}")
            # Don't interrupt the flow if clustering fails
    
    return result

async def crawl_multiple_urls(config: CrawlConfig) -> Dict[str, Any]:
    """
    Crawl multiple websites using the provided configuration.
    
    Args:
        config: The crawl configuration with urls list
        
    Returns:
        Combined crawl results
    """
    # Ensure we have URLs to crawl
    if not config.urls:
        logger.error("No URLs provided for multi-URL crawling")
        return {
            "status": "failed",
            "error": "No URLs provided for multi-URL crawling"
        }
    
    logger.info(f"Starting multi-URL crawl of {len(config.urls)} URLs")
    
    # Results container
    multi_result = {
        "urls": config.urls.copy(),
        "crawl_time": time.time(),
        "status": "success",
        "results": [],
        "stats": {
            "total_urls": len(config.urls),
            "successful_urls": 0,
            "failed_urls": 0,
            "total_content_length": 0,
            "total_pages_crawled": 0
        }
    }
    
    if config.crawl_in_parallel and len(config.urls) > 1:
        # Parallel crawling
        logger.info(f"Crawling {len(config.urls)} URLs in parallel with max {config.max_concurrent_crawls} concurrent tasks")
        
        # Create a semaphore to limit concurrency
        semaphore = asyncio.Semaphore(config.max_concurrent_crawls)
        
        async def crawl_with_semaphore(url):
            async with semaphore:
                # Create a copy of the config with the current URL
                url_config = config.copy()
                url_config.url = url
                url_config.urls = []  # Clear the URLs list to avoid recursion
                
                # Add random delay between requests
                if config.mean_delay > 0:
                    import random
                    delay = config.mean_delay + random.uniform(-config.max_range, config.max_range)
                    delay = max(0.1, delay)  # Ensure minimum delay of 0.1s
                    logger.info(f"Waiting {delay:.1f}s before crawling {url}")
                    await asyncio.sleep(delay)
                
                return await crawl_url(url_config)
        
        # Create tasks for all URLs
        tasks = [crawl_with_semaphore(url) for url in config.urls]
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for i, result in enumerate(results):
            url = config.urls[i]
            
            if isinstance(result, Exception):
                logger.error(f"Error crawling {url}: {str(result)}")
                multi_result["results"].append({
                    "url": url,
                    "status": "failed",
                    "error": str(result)
                })
                multi_result["stats"]["failed_urls"] += 1
            else:
                multi_result["results"].append(result)
                
                if result.get("status") == "success":
                    multi_result["stats"]["successful_urls"] += 1
                    multi_result["stats"]["total_content_length"] += len(result.get("raw_content", ""))
                    
                    # Add pages crawled if available
                    if "stats" in result and "pages_crawled" in result["stats"]:
                        multi_result["stats"]["total_pages_crawled"] += result["stats"]["pages_crawled"]
                else:
                    multi_result["stats"]["failed_urls"] += 1
    else:
        # Sequential crawling
        logger.info(f"Crawling {len(config.urls)} URLs sequentially")
        
        for url in config.urls:
            # Create a copy of the config with the current URL
            url_config = config.copy()
            url_config.url = url
            url_config.urls = []  # Clear the URLs list to avoid recursion
            
            # Add random delay between requests
            if config.mean_delay > 0 and multi_result["stats"]["successful_urls"] + multi_result["stats"]["failed_urls"] > 0:
                import random
                delay = config.mean_delay + random.uniform(-config.max_range, config.max_range)
                delay = max(0.1, delay)  # Ensure minimum delay of 0.1s
                logger.info(f"Waiting {delay:.1f}s before crawling {url}")
                await asyncio.sleep(delay)
            
            try:
                result = await crawl_url(url_config)
                multi_result["results"].append(result)
                
                if result.get("status") == "success":
                    multi_result["stats"]["successful_urls"] += 1
                    multi_result["stats"]["total_content_length"] += len(result.get("raw_content", ""))
                    
                    # Add pages crawled if available
                    if "stats" in result and "pages_crawled" in result["stats"]:
                        multi_result["stats"]["total_pages_crawled"] += result["stats"]["pages_crawled"]
                else:
                    multi_result["stats"]["failed_urls"] += 1
            except Exception as e:
                logger.error(f"Error crawling {url}: {str(e)}")
                multi_result["results"].append({
                    "url": url,
                    "status": "failed",
                    "error": str(e)
                })
                multi_result["stats"]["failed_urls"] += 1
    
    # Set overall status
    if multi_result["stats"]["successful_urls"] == 0:
        multi_result["status"] = "failed"
    elif multi_result["stats"]["failed_urls"] > 0:
        multi_result["status"] = "partial"
    
    # Create combined content if needed
    if config.use_ai_agent and config.enhance_content:
        # Combine all successful raw content
        combined_raw = ""
        for result in multi_result["results"]:
            if result.get("status") == "success" and result.get("raw_content"):
                combined_raw += f"\n## {result['url']}\n\n{result['raw_content']}\n\n"
        
        if combined_raw:
            multi_result["combined_raw_content"] = combined_raw
            
            # Try to enhance the combined content
            if has_ai_agent:
                try:
                    multi_result["combined_enhanced_content"] = await ai_agent.enhance_content(combined_raw)
                except Exception as e:
                    logger.error(f"Error enhancing combined content: {str(e)}")
    
    logger.info(f"Multi-URL crawl complete. Successful: {multi_result['stats']['successful_urls']}, Failed: {multi_result['stats']['failed_urls']}")
    return multi_result

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
    parser.add_argument("--extraction-strategy", choices=["Basic", "LLM", "CSS Selectors", "XPath Selectors"], help="Extraction strategy")
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
    if result.get("status") == "success":
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