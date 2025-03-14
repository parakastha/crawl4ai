import streamlit as st
import asyncio
import json
import time
import os
from dotenv import load_dotenv
from crawl_agent import crawl_url, CrawlConfig

# Load environment variables
load_dotenv()

st.set_page_config(page_title="Crawl4AI", page_icon="🕸️", layout="wide")

# Custom CSS
st.markdown("""
<style>
.feature-card {
    background-color: #f0f2f6;
    border-radius: 10px;
    padding: 15px;
    margin-bottom: 15px;
}
.feature-title {
    font-weight: bold;
        color: #4A4FE8;
}
.result-section {
    background-color: #f8f9fa;
    border-radius: 5px;
    padding: 15px;
    margin-top: 15px;
    }
</style>
""", unsafe_allow_html=True)

# Title and Introduction
st.title("🕸️ Crawl4AI: Advanced Web Crawling")
st.markdown("A powerful tool to extract and analyze web content with flexible configuration options.")

# Features section
with st.expander("✨ Features", expanded=False):
    features = [
        {
            "title": "Browser Configuration",
            "description": "Control browser settings, including headless mode and cache options."
        },
        {
            "title": "Proxy Support",
            "description": "Connect through proxy servers to access geo-restricted content or avoid IP blocking. Supports authentication for private proxies."
        },
        {
            "title": "SSL Certificate Options",
            "description": "Ignore HTTPS errors or provide custom certificate files for secure connections."
        },
        {
            "title": "Content Filtering",
            "description": "Filter content using Pruning or BM25 algorithms with adjustable thresholds."
        },
        {
            "title": "Deep Crawling",
            "description": "Crawl multiple pages with BFS, DFS, or Best-First strategies. Control depth and max pages."
        },
        {
            "title": "Extraction Strategies",
            "description": "Extract content using Basic, LLM, or JSON CSS strategies."
        },
        {
            "title": "AI Agent",
            "description": "Enable AI capabilities for website analysis, content enhancement, and answering questions about the crawled content."
        },
        {
            "title": "JavaScript Execution",
            "description": "Run custom JavaScript on the page before extraction."
        },
        {
            "title": "Lazy Loading Support",
            "description": "Automatically scroll and wait for content that loads dynamically as you scroll."
        },
        {
            "title": "File Downloading",
            "description": "Download PDFs, images, and documents found during crawling with size limits and filtering."
        },
        {
            "title": "Multi-URL Crawling",
            "description": "Crawl multiple URLs in batch mode, with parallel processing and configurable delays."
        },
        {
            "title": "Authentication & Hooks",
            "description": "Execute custom JavaScript at different stages to handle logins and other custom interactions."
        },
        {
            "title": "Statistics & Analysis",
            "description": "Get detailed statistics about the crawl, including page counts and content metrics."
        }
    ]
    
    cols = st.columns(2)
    for i, feature in enumerate(features):
        with cols[i % 2]:
            st.markdown(f"""
            <div class="feature-card">
                <div class="feature-title">{feature['title']}</div>
                <div>{feature['description']}</div>
            </div>
            """, unsafe_allow_html=True)

# Sidebar for Configuration
with st.sidebar:
    st.header("Configuration")
    
    with st.expander("🌐 URL", expanded=True):
        url = st.text_input("URL to Crawl", value="https://docs.crawl4ai.com/")
    
    with st.expander("🌐 Browser Settings", expanded=False):
        headless = st.checkbox("Headless Mode", value=True)
        verbose = st.checkbox("Verbose Output", value=False)
        # Ensure we're using the correct case for cache mode values to match the CacheMode enum
        cache_mode_options = ["ENABLED", "BYPASS", "DISABLED", "READ_ONLY", "WRITE_ONLY"]
        cache_mode = st.selectbox("Cache Mode", cache_mode_options, index=0)
        
        # SSL certificate settings
        st.markdown("### SSL Certificate Settings")
        ignore_https_errors = st.checkbox("Ignore HTTPS Errors", value=False, 
                                       help="Ignore HTTPS errors like invalid certificates")
        cert_file = st.text_input("Custom Certificate File Path", value="",
                               help="Path to custom certificate file (leave empty for default)")
        
        # Proxy settings
        st.markdown("### Proxy Settings")
        use_proxy = st.checkbox("Use Proxy", value=False)
        proxy_server = None
        proxy_username = None
        proxy_password = None
        
        if use_proxy:
            proxy_server = st.text_input("Proxy Server (e.g., http://proxy.example.com:8080)")
            use_auth = st.checkbox("Proxy Authentication")
            if use_auth:
                proxy_username = st.text_input("Proxy Username")
                proxy_password = st.text_input("Proxy Password", type="password")
            
            st.info("Using a proxy can help bypass geo-restrictions and avoid IP blocking, but may slow down crawling.")
            
            if proxy_server:
                st.success(f"Proxy configured: {proxy_server}")
                if proxy_username:
                    st.success(f"Proxy authentication enabled for user: {proxy_username}")
    
    with st.expander("🔍 Content Filtering", expanded=False):
        filter_type = st.selectbox("Filtering Method", ["Pruning", "BM25"], index=0)
        
        if filter_type == "Pruning":
            threshold = st.slider("Pruning Threshold", 0.0, 1.0, 0.48, 0.01)
            min_word_threshold = st.number_input("Minimum Word Count", 0, 1000, 0)
        else:  # BM25
            user_query = st.text_input("BM25 Query (keywords)")
            bm25_threshold = st.slider("BM25 Threshold", 0.0, 1.0, 0.1, 0.01)
    
    with st.expander("🤖 AI Agent", expanded=False):
        use_ai_agent = st.checkbox("Enable AI Agent", value=False, help="Use AI to enhance crawling capabilities")
        
        if use_ai_agent:
            analyze_website = st.checkbox("Analyze Website", value=True, 
                                         help="Automatically determine the best crawling strategy based on website structure")
            enhance_content = st.checkbox("Enhance Content", value=True,
                                        help="Improve and organize the extracted content")
            store_results = st.checkbox("Remember Results", value=True,
                                      help="Store crawl results for future reference")
            ai_question = st.text_input("Ask a Question About the Content", 
                                       help="The AI will answer based on the crawled content")
            
            # Check if OpenAI API key is set
            if not os.getenv("OPENAI_API_KEY"):
                st.warning("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable to enable full AI capabilities.")
                st.text_input("OpenAI API Key (temporary)", type="password", key="temp_api_key")
                if st.session_state.get("temp_api_key"):
                    os.environ["OPENAI_API_KEY"] = st.session_state.temp_api_key
                    st.success("API key set for this session.")
    
    with st.expander("🕸️ Deep Crawling", expanded=False):
        deep_crawl = st.checkbox("Enable Deep Crawling", value=False)
        
        if deep_crawl:
            deep_crawl_strategy = st.selectbox("Crawling Strategy", ["BFS", "DFS", "Best-First"], index=0)
            max_depth = st.slider("Maximum Depth", 1, 10, 2)
            max_pages = st.slider("Maximum Pages", 1, 100, 10)
            follow_external_links = st.checkbox("Follow External Links", value=False)
    
    with st.expander("📱 Page Interaction", expanded=False):
        st.markdown("### Basic Interaction")
        js_code = st.text_area("Custom JavaScript (runs before extraction)", 
                            help="JavaScript code to execute on the page, like clicking buttons or scrolling")
        wait_for = st.text_input("Wait for Element/Condition", 
                               help="CSS selector (prefix with 'css:') or JS condition (prefix with 'js:') to wait for")
        delay_before_return_html = st.slider("Delay Before Return HTML (seconds)", 0, 30, 0,
                                          help="Wait this many seconds after page load/JS execution before capturing HTML")
        page_timeout = st.number_input("Page Timeout (milliseconds)", min_value=5000, max_value=300000, value=60000, step=5000,
                                     help="Maximum time to wait for page to load")
        
        st.markdown("### Lazy Loading")
        enable_lazy_loading = st.checkbox("Enable Lazy Loading", value=False,
                                       help="Automatically scroll and wait for lazy-loaded content")
        if enable_lazy_loading:
            lazy_load_cols = st.columns(3)
            with lazy_load_cols[0]:
                lazy_load_scroll_step = st.number_input("Scroll Step (pixels)", value=800, min_value=100, max_value=5000,
                                                     help="Pixels to scroll each step")
            with lazy_load_cols[1]:
                lazy_load_max_scrolls = st.number_input("Max Scrolls", value=5, min_value=1, max_value=50,
                                                     help="Maximum number of scroll operations")
            with lazy_load_cols[2]:
                lazy_load_wait_time = st.number_input("Wait Time (ms)", value=1000, min_value=100, max_value=10000,
                                                   help="Milliseconds to wait between scrolls")
            
            st.info("Lazy loading is useful for pages that load content as you scroll. " +
                   "The crawler will perform multiple scrolls with waits in between to allow content to load.")
        
        st.markdown("### Authentication & Hooks")
        auth_hook_js = st.text_area("Authentication Hook JS", "",
                                 help="JavaScript to handle authentication on the page")
        pre_request_hook_js = st.text_area("Pre-Request Hook JS", "",
                                        help="JavaScript to run before page navigation")
        post_request_hook_js = st.text_area("Post-Request Hook JS", "",
                                         help="JavaScript to run after page loads but before extraction")
        
        if auth_hook_js or pre_request_hook_js or post_request_hook_js:
            st.info("Hooks allow you to execute custom JavaScript at different stages of the crawling process. " +
                   "The Authentication Hook is specifically for handling logins and other authentication flows.")
        
        st.markdown("### Multi-Step Interaction")
        multi_step_enabled = st.checkbox("Enable Multi-Step Interaction", value=False,
                                      help="Perform a sequence of interactions with the page")
        
        if multi_step_enabled:
            session_id = st.text_input("Session ID", value="session1",
                                    help="Identifier for the browser session to reuse across steps")
            js_only = st.checkbox("JS Only Mode", value=True,
                               help="Just run JavaScript in existing page without re-navigation")
            
            st.markdown("#### Step Configuration")
            num_steps = st.number_input("Number of Steps", min_value=1, max_value=10, value=2)
            
            multi_step_js_actions = []
            multi_step_wait_conditions = []
            multi_step_delays = []
            
            for i in range(num_steps):
                st.markdown(f"##### Step {i+1}")
                cols = st.columns(3)
                with cols[0]:
                    action = st.text_area(f"JS Action {i+1}", height=100, 
                                     placeholder="e.g., document.querySelector('button.load-more').click();",
                                     key=f"action_{i}")
                    multi_step_js_actions.append(action)
                
                with cols[1]:
                    condition = st.text_input(f"Wait Condition {i+1}", 
                                          placeholder="e.g., css:.new-content or js:()=>document.querySelectorAll('.item').length>20",
                                          key=f"condition_{i}")
                    multi_step_wait_conditions.append(condition)
                
                with cols[2]:
                    delay = st.number_input(f"Delay (s) {i+1}", min_value=0, max_value=30, value=1, key=f"delay_{i}")
                    multi_step_delays.append(delay)
        
        st.markdown("### Advanced Interaction")
        simulate_user = st.checkbox("Simulate User", value=False, 
                                 help="Simulate human-like behavior to avoid bot detection")
        override_navigator = st.checkbox("Override Navigator", value=False,
                                      help="Override navigator properties to avoid bot detection")
        process_iframes = st.checkbox("Process Iframes", value=False,
                                   help="Extract content from iframes embedded in the page")
        
        # Move Multi-URL Crawling Settings out of the expander
        magic = st.checkbox("Magic Mode", value=False, 
                         help="Enable advanced anti-bot detection features")
        remove_overlay_elements = st.checkbox("Remove Overlay Elements", value=False,
                                           help="Automatically remove modals and overlays")
    
    # Multi-URL Crawling Settings as a separate section outside any other expander
    with st.expander("🔄 Multi-URL Crawling", expanded=False):
        st.markdown("### Batch Crawling")
        urls_text = st.text_area("URLs to Crawl (one per line)", "",
                              help="Enter multiple URLs to crawl in batch (leave empty to use the single URL field)")
        
        # Extract URLs from the text area
        urls = [url.strip() for url in urls_text.split("\n") if url.strip()]
        
        if urls:
            st.success(f"Configured {len(urls)} URLs for batch crawling")
            
            crawl_in_parallel = st.checkbox("Crawl in Parallel", value=True,
                                        help="Process multiple URLs concurrently (faster)")
            
            if crawl_in_parallel:
                max_concurrent_crawls = st.slider("Max Concurrent Crawls", 1, 10, 3,
                                              help="Maximum number of URLs to process simultaneously")
            
            st.markdown("### Delay Settings")
            mean_delay = st.slider("Mean Delay Between Requests (seconds)", 0.1, 10.0, 1.0, 0.1,
                               help="Average delay between requests in multi-URL crawling")
            max_range = st.slider("Max Delay Variance (seconds)", 0.0, 5.0, 0.5, 0.1,
                              help="Maximum random variance added to delay between requests")
            
            st.info("Adding random delays between requests helps avoid getting blocked by websites. " +
                   f"Actual delay will be between {max(0.1, mean_delay-max_range):.1f}s and {mean_delay+max_range:.1f}s.")
        else:
            st.info("Enter multiple URLs above to enable batch crawling. Leave empty to use the single URL field.")
            mean_delay = 1.0
            max_range = 0.5

    with st.expander("🔧 Advanced Settings", expanded=False):
        # Extraction strategy selector
        extraction_strategy = st.selectbox(
            "Extraction Strategy",
            ["Auto", "CSS Selectors", "XPath Selectors", "LLM"],
            help="Choose extraction strategy for webpage content. Auto: default extraction, CSS/XPath: extract using selectors, LLM: AI-powered extraction"
        )

        if extraction_strategy == "CSS Selectors":
            css_selectors_schema = st.text_area(
                "CSS Selectors Schema (JSON)",
                value=json.dumps({
                    "title": "h1",
                    "content": "article p",
                    "author": ".author-name"
                }, indent=2),
                height=200,
                help="JSON schema mapping field names to CSS selectors"
            )
        elif extraction_strategy == "XPath Selectors":
            xpath_selectors_schema = st.text_area(
                "XPath Selectors Schema (JSON)",
                value=json.dumps({
                    "title": "//h1",
                    "content": "//article//p",
                    "author": "//*[contains(@class, 'author-name')]"
                }, indent=2),
            height=200,
                help="JSON schema mapping field names to XPath expressions"
            )
        elif extraction_strategy == "LLM":
            st.warning("LLM extraction requires an OpenAI API key in the settings tab.")
        
        save_raw_markdown = st.checkbox("Save Raw Markdown to File", value=False)
        
        st.markdown("### Content Selection Options")
        word_count_threshold = st.number_input("Word Count Threshold", 0, 1000, 0,
                                           help="Minimum words per block to include in output")
        excluded_tags = st.multiselect("Excluded Tags", 
                                    ["script", "style", "svg", "path", "noscript", "header", "footer", "nav", "form"], 
                                    default=["script", "style", "svg", "noscript"],
                                    help="HTML tags to exclude from extraction")
        
        st.markdown("### Link Filtering")
        exclude_external_links = st.checkbox("Exclude External Links", value=False,
                                          help="Remove links pointing to external domains")
        exclude_social_media_links = st.checkbox("Exclude Social Media Links", value=False,
                                             help="Remove links pointing to social media platforms")
        
        col1, col2 = st.columns(2)
        with col1:
            exclude_domains_text = st.text_area("Domains to Exclude (one per line)", "",
                                           help="Specific domains to exclude from links")
            exclude_domains = [domain.strip() for domain in exclude_domains_text.split("\n") if domain.strip()]
        
        with col2:
            exclude_social_domains_text = st.text_area("Social Media Domains to Exclude (one per line)", "",
                                                  help="Additional social media domains to exclude")
            exclude_social_media_domains = [domain.strip() for domain in exclude_social_domains_text.split("\n") if domain.strip()]
        
        st.markdown("### Media Filtering")
        exclude_external_images = st.checkbox("Exclude External Images", value=False,
                                          help="Don't include images from external domains")
        
        st.markdown("### Additional HTML Filtering")
        col1, col2 = st.columns(2)
        with col1:
            keep_data_attributes = st.checkbox("Keep Data Attributes", value=False,
                                           help="Keep data-* attributes in HTML output")
            remove_forms = st.checkbox("Remove Forms", value=False,
                                    help="Remove form elements from output")
        
        with col2:
            keep_attrs_text = st.text_area("Attributes to Keep (one per line)", "",
                                      help="Specific HTML attributes to keep in the output")
            keep_attrs = [attr.strip() for attr in keep_attrs_text.split("\n") if attr.strip()]

    with st.expander("📄 File Downloading", expanded=False):
        st.markdown("### File Download Settings")
        download_pdf = st.checkbox("Download PDF Files", value=False,
                               help="Download PDF files found during crawling")
        download_images = st.checkbox("Download Images", value=False,
                                  help="Download images found during crawling")
        download_docs = st.checkbox("Download Documents", value=False,
                                help="Download document files (doc, docx, xls, xlsx, etc.)")
        
        if download_pdf or download_images or download_docs:
            download_path = st.text_input("Download Path", value="./downloads",
                                      help="Directory where downloaded files will be saved")
            max_file_size_mb = st.slider("Maximum File Size (MB)", 1, 100, 10,
                                      help="Maximum size for downloaded files in megabytes")
            
            st.info("Files will be saved to the specified folder with their original filenames. " +
                   "The crawler will skip files larger than the maximum size.")

    # Add a divider for better organization
    st.divider()
    
    # Content Processing Options
    st.subheader("🧩 Content Processing")
    
    # Add chunking options
    enable_chunking = st.checkbox(
        "Enable Text Chunking", 
        value=False,
        help="Split content into manageable chunks for better processing"
    )
    
    if enable_chunking:
        chunking_col1, chunking_col2 = st.columns(2)
        
        with chunking_col1:
            chunking_strategy = st.selectbox(
                "Chunking Strategy",
                ["semantic", "sentence", "fixed", "regex", "sliding_window", "overlapping_window"],
                help="Method to split content: semantic (preserves meaning), sentence (splits by sentences), fixed (equal size chunks), regex (pattern-based), sliding_window (overlapping with step size), overlapping_window (specified overlap)"
            )
        
        with chunking_col2:
            if chunking_strategy in ["semantic", "sentence", "fixed"]:
                chunk_size = st.number_input(
                    "Max Chunk Size (chars)",
                    min_value=1000,
                    max_value=10000,
                    value=4000,
                    step=500,
                    help="Maximum size of each chunk in characters"
                )
                
                chunk_overlap = st.number_input(
                    "Chunk Overlap (chars)",
                    min_value=0,
                    max_value=1000,
                    value=200,
                    step=50,
                    help="Overlap between chunks in characters"
                )
            elif chunking_strategy == "regex":
                regex_pattern = st.text_area(
                    "Regex Pattern",
                    value=r"\n\n",
                    help="Regular expression pattern to split the text (e.g., \\n\\n splits on double newlines)"
                )
            elif chunking_strategy == "sliding_window":
                window_size = st.number_input(
                    "Window Size (words)",
                    min_value=50,
                    max_value=1000,
                    value=100,
                    step=10,
                    help="Size of each window in words"
                )
                
                step_size = st.number_input(
                    "Step Size (words)",
                    min_value=10,
                    max_value=500,
                    value=50,
                    step=10,
                    help="Number of words to slide the window for each chunk"
                )
            elif chunking_strategy == "overlapping_window":
                window_size = st.number_input(
                    "Window Size (words)",
                    min_value=50,
                    max_value=1000,
                    value=500,
                    step=50,
                    help="Size of each window in words"
                )
                
                overlap = st.number_input(
                    "Overlap (words)",
                    min_value=10,
                    max_value=500,
                    value=50,
                    step=10,
                    help="Number of words to overlap between chunks"
                )
    
    # Add clustering options
    enable_clustering = st.checkbox(
        "Enable Content Clustering", 
        value=False,
        help="Group similar content together using machine learning"
    )
    
    if enable_clustering:
        clustering_col1, clustering_col2 = st.columns(2)
        
        with clustering_col1:
            clustering_strategy = st.selectbox(
                "Clustering Algorithm",
                ["kmeans", "hierarchical"],
                help="Method to cluster content: K-means (centroid-based), Hierarchical (tree-based)"
            )
        
        with clustering_col2:
            n_clusters = st.number_input(
                "Number of Clusters",
                min_value=2,
                max_value=20,
                value=5,
                step=1,
                help="Target number of content clusters to create"
            )

# Main content area with crawl button
crawl_button = st.button("🕸️ Start Crawling")

if crawl_button:
    with st.spinner("Crawling website..."):
        # Create config based on selected options
        config = CrawlConfig(
            url=url,
            headless=headless,
            verbose=verbose,
            cache_mode=cache_mode,
            content_filter_type=filter_type,
            
            # SSL certificate settings
            ignore_https_errors=ignore_https_errors,
            cert_file=cert_file if cert_file else None,
            
            # Proxy settings
            use_proxy=use_proxy,
            proxy_server=proxy_server if use_proxy else None,
            proxy_username=proxy_username if use_proxy and proxy_username else None,
            proxy_password=proxy_password if use_proxy and proxy_password else None,
            
            # Content filter settings
            threshold=threshold if filter_type == "Pruning" else 0.48,
            min_word_threshold=min_word_threshold if filter_type == "Pruning" else 0,
            user_query=user_query if filter_type == "BM25" else "",
            bm25_threshold=bm25_threshold if filter_type == "BM25" else 0.1,
            
            # AI agent settings
            use_ai_agent=use_ai_agent,
            analyze_website=analyze_website if use_ai_agent else False,
            enhance_content=enhance_content if use_ai_agent else False,
            store_results=store_results if use_ai_agent else False,
            ai_question=ai_question if use_ai_agent else "",
            
            # Deep crawling settings
            deep_crawl=deep_crawl,
            deep_crawl_strategy=deep_crawl_strategy if deep_crawl else "BFS",
            max_depth=max_depth if deep_crawl else 2,
            max_pages=max_pages if deep_crawl else 10,
            follow_external_links=follow_external_links if deep_crawl else False,
            
            # Page Interaction settings
            js_code=js_code,
            wait_for=wait_for,
            delay_before_return_html=delay_before_return_html,
            page_timeout=page_timeout,
            
            # Lazy loading settings
            enable_lazy_loading=enable_lazy_loading if 'enable_lazy_loading' in locals() else False,
            lazy_load_scroll_step=lazy_load_scroll_step if 'lazy_load_scroll_step' in locals() and enable_lazy_loading else 800,
            lazy_load_max_scrolls=lazy_load_max_scrolls if 'lazy_load_max_scrolls' in locals() and enable_lazy_loading else 5,
            lazy_load_wait_time=lazy_load_wait_time if 'lazy_load_wait_time' in locals() and enable_lazy_loading else 1000,
            
            # Authentication & hooks
            auth_hook_js=auth_hook_js if 'auth_hook_js' in locals() else "",
            pre_request_hook_js=pre_request_hook_js if 'pre_request_hook_js' in locals() else "",
            post_request_hook_js=post_request_hook_js if 'post_request_hook_js' in locals() else "",
            
            # Multi-step interaction settings
            multi_step_enabled=multi_step_enabled if 'multi_step_enabled' in locals() else False,
            session_id=session_id if 'session_id' in locals() and multi_step_enabled else None,
            js_only=js_only if 'js_only' in locals() and multi_step_enabled else False,
            multi_step_js_actions=multi_step_js_actions if 'multi_step_js_actions' in locals() and multi_step_enabled else [],
            multi_step_wait_conditions=multi_step_wait_conditions if 'multi_step_wait_conditions' in locals() and multi_step_enabled else [],
            multi_step_delays=multi_step_delays if 'multi_step_delays' in locals() and multi_step_enabled else [],
            
            # Advanced interaction settings
            simulate_user=simulate_user if 'simulate_user' in locals() else False,
            override_navigator=override_navigator if 'override_navigator' in locals() else False,
            process_iframes=process_iframes if 'process_iframes' in locals() else False,
            
            # Multi-URL crawling settings
            urls=urls if 'urls' in locals() and urls else [],
            crawl_in_parallel=crawl_in_parallel if 'crawl_in_parallel' in locals() and 'urls' in locals() and urls else False,
            max_concurrent_crawls=max_concurrent_crawls if 'max_concurrent_crawls' in locals() and 'crawl_in_parallel' in locals() and crawl_in_parallel else 3,
            mean_delay=mean_delay if 'mean_delay' in locals() else 1.0,
            max_range=max_range if 'max_range' in locals() else 0.5,
            
            # File downloading settings
            download_pdf=download_pdf if 'download_pdf' in locals() else False,
            download_images=download_images if 'download_images' in locals() else False,
            download_docs=download_docs if 'download_docs' in locals() else False,
            download_path=download_path if 'download_path' in locals() and ('download_pdf' in locals() or 'download_images' in locals() or 'download_docs' in locals()) else "./downloads",
            max_file_size_mb=max_file_size_mb if 'max_file_size_mb' in locals() and ('download_pdf' in locals() or 'download_images' in locals() or 'download_docs' in locals()) else 10,
            
            # Advanced settings
            extraction_strategy=extraction_strategy,
            css_selector=css_selectors_schema if extraction_strategy == "CSS Selectors" else xpath_selectors_schema if extraction_strategy == "XPath Selectors" else "",
            magic=magic,
            remove_overlay_elements=remove_overlay_elements,
            save_raw_markdown=save_raw_markdown,
            word_count_threshold=word_count_threshold,
            excluded_tags=excluded_tags,
            exclude_external_links=exclude_external_links,
            exclude_social_media_links=exclude_social_media_links,
            exclude_domains=exclude_domains,
            exclude_social_media_domains=exclude_social_media_domains,
            exclude_external_images=exclude_external_images,
            keep_data_attributes=keep_data_attributes,
            remove_forms=remove_forms,
            keep_attrs=keep_attrs,
            
            # Chunking options
            enable_chunking=enable_chunking,
            chunking_strategy=chunking_strategy if enable_chunking else "semantic",
            
            # Default chunking parameters
            chunk_size=chunk_size if enable_chunking and chunking_strategy in ["semantic", "sentence", "fixed"] else 4000,
            chunk_overlap=chunk_overlap if enable_chunking and chunking_strategy in ["semantic", "sentence", "fixed"] else 200,
            
            # Regex chunking parameters
            regex_pattern=regex_pattern if enable_chunking and chunking_strategy == "regex" else r"\n\n",
            
            # Sliding window parameters
            window_size=window_size if enable_chunking and chunking_strategy in ["sliding_window", "overlapping_window"] else 500,
            step_size=step_size if enable_chunking and chunking_strategy == "sliding_window" else 50,
            
            # Overlapping window parameters
            overlap=overlap if enable_chunking and chunking_strategy == "overlapping_window" else 50,
            
            # Clustering options
            enable_clustering=enable_clustering,
            clustering_strategy=clustering_strategy if enable_clustering else "kmeans",
            n_clusters=n_clusters if enable_clustering else 5,
            
            # OpenAI API key for LLM extraction
            openai_api_key=os.getenv("OPENAI_API_KEY") if extraction_strategy == "LLM" else None
        )
        
        # Run the crawl
        result = asyncio.run(crawl_url(config))
        
        # Store the result in session state so it's not lost when switching tabs
        st.session_state.crawl_result = result
        
        if result.get("status") == "success" or result.get("raw_content"):
            # Stats tab - use both status and raw_content to determine success
            # This ensures we show content even if AI features failed
            tabs = st.tabs(["Results", "Raw Markdown", "Processed Markdown", "Stats", "AI Enhanced Content", "AI Answer"])
            
            with tabs[0]:
                st.success(f"Successfully crawled {url}")
                
                # Check if it's a multi-URL result
                if "urls" in result and "results" in result:
                    # Display multi-URL crawl results
                    st.markdown("""
                    <div class="result-section">
                        <h3>Multi-URL Crawl Results</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Stats about the multi-URL crawl
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("URLs Crawled", result["stats"].get("total_urls", 0))
                    with col2:
                        st.metric("Successful URLs", result["stats"].get("successful_urls", 0))
                    with col3:
                        st.metric("Failed URLs", result["stats"].get("failed_urls", 0))
                    
                    # Overall content stats
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Total Pages Crawled", result["stats"].get("total_pages_crawled", 0))
                    with col2:
                        st.metric("Total Content Length", f"{result['stats'].get('total_content_length', 0)/1000:.1f}K chars")
                    
                    # URL-specific results
                    st.markdown("### Results by URL")
                    
                    # Create a simple list of URL results without using expanders
                    for i, url_result in enumerate(result["results"]):
                        url = url_result.get("url", f"URL {i+1}")
                        status = url_result.get("status", "unknown")
                        
                        # Use color coding for success/failure
                        if status == "success":
                            st.success(f"{url} - Success")
                            # Display basic stats about this URL result if available
                            if "stats" in url_result:
                                st.text(f"Content length: {len(url_result.get('raw_content', ''))}")
                        else:
                            st.error(f"{url} - Failed: {url_result.get('error', 'Unknown error')}")
                    
                    # Download buttons for combined content
                    if "combined_raw_content" in result:
                        st.download_button(
                            label="Download Combined Raw Content",
                            data=result["combined_raw_content"],
                            file_name=f"crawl4ai_combined_raw_{time.strftime('%Y%m%d_%H%M%S')}.md",
                            mime="text/markdown"
                        )
                    
                    if "combined_enhanced_content" in result:
                        st.download_button(
                            label="Download Combined AI-Enhanced Content",
                            data=result["combined_enhanced_content"],
                            file_name=f"crawl4ai_combined_enhanced_{time.strftime('%Y%m%d_%H%M%S')}.md",
                            mime="text/markdown"
                        )
                else:
                    # Display stats in a card for single URL crawl
                    st.markdown("""
                    <div class="result-section">
                        <h3>Crawl Results</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Stats about the crawl
                    if "stats" in result:
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Pages Crawled", result["stats"].get("pages_crawled", 1))
                        with col2:
                            st.metric("Successful Pages", result["stats"].get("successful_pages", 1))
                        with col3:
                            st.metric("Content Length", f"{result['stats'].get('total_content_length', len(result.get('raw_content', '')))/1000:.1f}K chars")
                    
                    # Download buttons for the markdown files
                    if result.get("raw_content"):
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.download_button(
                                label="Download Raw Markdown",
                                data=result["raw_content"],
                                file_name=f"crawl4ai_raw_{time.strftime('%Y%m%d_%H%M%S')}.md",
                                mime="text/markdown"
                            )
                    
                    with col2:
                        if result.get("fit_content"):
                            st.download_button(
                                label="Download Processed Markdown",
                                data=result["fit_content"],
                                file_name=f"crawl4ai_processed_{time.strftime('%Y%m%d_%H%M%S')}.md",
                                mime="text/markdown"
                            )
                        
                        with col3:
                            if result.get("ai_enhanced_content"):
                                st.download_button(
                                    label="Download AI Enhanced Content",
                                    data=result["ai_enhanced_content"],
                                    file_name=f"crawl4ai_ai_enhanced_{time.strftime('%Y%m%d_%H%M%S')}.md",
                                    mime="text/markdown"
                                )
            
            # Raw markdown tab
            with tabs[1]:
                if result.get("raw_content"):
                    st.markdown("""
                    <div class="result-section">
                        <h3>Raw Markdown Content</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.code(result["raw_content"], language="markdown")
                else:
                    st.warning("No raw markdown content was generated. Try adjusting the crawl parameters.")
            
            # Processed markdown tab
            with tabs[2]:
                if result.get("fit_content") and len(result["fit_content"]) > 0:
                    st.markdown("""
                    <div class="result-section">
                        <h3>Processed Markdown Content</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.code(result["fit_content"], language="markdown")
                else:
                    # If no processed content exists, show the raw content instead
                    st.warning("No processed markdown was generated. Showing raw markdown instead.")
                    if result.get("raw_content"):
                        st.code(result["raw_content"], language="markdown")
                    else:
                        st.error("No content was generated. Try adjusting the crawl parameters.")
            
            with tabs[3]:
                st.markdown("### Crawl Statistics")
                
                # Detailed stats
                if "stats" in result:
                    st.json(result["stats"])
                else:
                    st.info("No detailed statistics available.")
            
            # AI enhanced content tab
            with tabs[4]:
                if result.get("ai_enhanced_content") and result["ai_enhanced_content"] != result.get("raw_content"):
                    st.markdown("""
                    <div class="result-section">
                        <h3>AI Enhanced Content</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(result["ai_enhanced_content"])
                else:
                    st.warning("AI enhancement was not applied or failed. Check your AI agent settings and API key.")
                    if result.get("raw_content"):
                        st.markdown("### Raw content:")
                        st.markdown(result["raw_content"][:2000] + "..." if len(result.get("raw_content", "")) > 2000 else result.get("raw_content", ""))
            
            # AI answer tab
            with tabs[5]:
                if result.get("ai_answer") and not result["ai_answer"].startswith("Error") and not result["ai_answer"].startswith("An error occurred"):
                    st.markdown("""
                    <div class="result-section">
                        <h3>AI Answer</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if ai_question:
                        st.markdown(f"**Question:** {ai_question}")
                    st.markdown(f"**Answer:** {result['ai_answer']}")
                else:
                    if "ai_answer" in result and (result["ai_answer"].startswith("Error") or result["ai_answer"].startswith("An error occurred")):
                        st.error(f"AI answer error: {result['ai_answer']}")
                    else:
                        st.warning("No AI answer was generated. Make sure you've enabled the AI agent and asked a question.")
                    
                    # New question input for retrying
                    new_question = st.text_input("Ask a new question about the content")
                    if st.button("Get Answer") and new_question and result.get("raw_content"):
                        with st.spinner("Generating answer..."):
                            # Set up a new config with just the question
                            question_config = CrawlConfig(
                                url=url,
                                use_ai_agent=True,
                                ai_question=new_question
                            )
                            
                            # Create a simplified result with just the content and question
                            from crawl_agent import AIAgent
                            try:
                                from ai_agent import ai_agent
                                answer = asyncio.run(ai_agent.answer_question(new_question, result["raw_content"]))
                                st.markdown(f"**Question:** {new_question}")
                                st.markdown(f"**Answer:** {answer}")
                            except Exception as e:
                                st.error(f"Error generating answer: {str(e)}")
        else:
            st.error(f"Failed to crawl {url}: {result.get('error', 'Unknown error')}")
            if "error" in result:
                st.code(result["error"])

# Add a feature to view last crawl result
if not crawl_button and "crawl_result" in st.session_state:
    if st.button("View Last Crawl Result"):
        result = st.session_state.crawl_result
        # Then reuse the code from above to display the results

# Display crawl results
if "result" in st.session_state and st.session_state["result"]:
    result = st.session_state["result"]
    
    # Create tabs for different views of the data
    result_tabs = st.tabs(["📝 Content", "🌐 Links", "🖼️ Media", "📊 Stats", "🧩 Chunks", "🔍 Clusters"])
    
    with result_tabs[0]:  # Content tab
        if "markdown" in result:
            st.markdown(result["markdown"])
        elif "content" in result:
            st.text(result["content"])
        else:
            st.warning("No content extracted from the webpage.")
    
    with result_tabs[1]:  # Links tab
        if "links" in result and result["links"]:
            st.subheader("📋 Extracted Links")
            
            # Create a DataFrame for better display
            links_data = []
            for link in result["links"]:
                link_type = "External" if link.get("is_external", False) else "Internal"
                links_data.append({
                    "URL": link.get("url", ""),
                    "Text": link.get("text", ""),
                    "Type": link_type
                })
            
            if links_data:
                st.dataframe(links_data, use_container_width=True)
            else:
                st.info("No links found.")
        else:
            st.info("No links extracted.")
    
    with result_tabs[2]:  # Media tab
        if "images" in result and result["images"]:
            st.subheader("🖼️ Images")
            
            # Create a DataFrame for better display
            images_data = []
            for img in result["images"]:
                img_type = "External" if img.get("is_external", False) else "Internal"
                images_data.append({
                    "URL": img.get("src", ""),
                    "Alt Text": img.get("alt", ""),
                    "Type": img_type
                })
            
            if images_data:
                st.dataframe(images_data, use_container_width=True)
            else:
                st.info("No images found.")
        else:
            st.info("No images extracted.")
    
    with result_tabs[3]:  # Stats tab
        if "stats" in result:
            stats = result["stats"]
            
            st.subheader("📊 Crawl Statistics")
            
            # Create columns for a cleaner layout
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Crawl Time", f"{stats.get('crawl_time', 0):.2f} seconds")
                st.metric("Links Found", stats.get("link_count", 0))
                st.metric("Images Found", stats.get("image_count", 0))
            
            with col2:
                st.metric("Filtered Links", len(stats.get("filtered_links", [])))
                st.metric("Filtered Images", stats.get("filtered_images", 0))
                st.metric("Downloaded Files", len(stats.get("downloaded_files", [])))
            
            # List downloaded files if any
            if "downloaded_files" in stats and stats["downloaded_files"]:
                st.subheader("📥 Downloaded Files")
                for file_path in stats["downloaded_files"]:
                    st.text(file_path)
            else:
                st.info("No statistics available.")
    
    with result_tabs[4]:  # Chunks tab
        if "markdown_chunks" in result and result["markdown_chunks"]:
            chunks = result["markdown_chunks"]
            st.subheader(f"🧩 Content Chunks ({len(chunks)})")
            
            for i, chunk in enumerate(chunks):
                with st.expander(f"Chunk {i+1} ({len(chunk)} chars)"):
                    st.markdown(chunk)
        
        elif "html_chunks" in result and result["html_chunks"]:
            chunks = result["html_chunks"]
            st.subheader(f"🧩 HTML Chunks ({len(chunks)})")
            
            for i, chunk in enumerate(chunks):
                with st.expander(f"Chunk {i+1} ({len(chunk)} chars)"):
                    st.code(chunk[:1000] + "..." if len(chunk) > 1000 else chunk, language="html")
        else:
            st.info("No chunks available. Enable chunking in advanced settings to see content chunks.")
    
    with result_tabs[5]:  # Clusters tab
        if "clustering" in result and result["clustering"]:
            clustering = result["clustering"]
            
            st.subheader(f"🔍 Content Clusters ({len(clustering['summary'])})")
            
            # Display cluster information
            for cluster_idx, summary in clustering["summary"].items():
                # Get keywords for this cluster
                keywords = clustering["keywords"].get(cluster_idx, [])
                keywords_str = ", ".join(keywords) if keywords else "No keywords found"
                
                with st.expander(f"Cluster {cluster_idx+1} ({summary['count']} items) - Keywords: {keywords_str}"):
                    for i, sample in enumerate(summary["samples"]):
                        st.markdown(f"**Sample {i+1}:**")
                        st.markdown(sample[:500] + "..." if len(sample) > 500 else sample)
                        st.divider()
        else:
            st.info("No clustering results available. Enable clustering in advanced settings to group similar content.")