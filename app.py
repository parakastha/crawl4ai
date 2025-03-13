import streamlit as st
import asyncio
import pandas as pd
import time
import os
import json
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from crawl4ai.extraction_strategy import LLMExtractionStrategy, JsonCssExtractionStrategy
from crawl4ai.content_filter_strategy import PruningContentFilter, BM25ContentFilter
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from crawl4ai.deep_crawling import BFSDeepCrawlStrategy, DFSDeepCrawlStrategy, BestFirstCrawlingStrategy
from crawl4ai.deep_crawling.filters import FilterChain, URLPatternFilter, DomainFilter, ContentTypeFilter
from crawl4ai.deep_crawling.scorers import KeywordRelevanceScorer
from crawl4ai.content_scraping_strategy import LXMLWebScrapingStrategy

# Set page config
st.set_page_config(
    page_title="Crawl4AI Web Interface",
    page_icon="🕸️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #4A4FE8;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #888;
        margin-top: 0;
    }
    .success-msg {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #D1F0E0;
        color: #0F5132;
    }
    .card {
        padding: 1.5rem;
        border-radius: 0.5rem;
        background-color: #F8F9FA;
        margin-bottom: 1rem;
    }
    .url-input {
        padding: 0.5rem;
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# Header and description
st.markdown('<h1 class="main-header">Crawl4AI Web Interface</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">A powerful web interface for the Crawl4AI library</p>', unsafe_allow_html=True)

# Sidebar for configuration
with st.sidebar:
    st.title("Configuration")
    
    # Basic configuration options
    st.header("Basic Settings")
    
    headless = st.toggle("Headless Mode", value=True, help="Run browser in headless mode")
    verbose = st.toggle("Verbose Output", value=False, help="Display verbose output")
    cache_mode = st.selectbox(
        "Cache Mode",
        options=[
            "Enabled",
            "Bypass",
            "Disabled"
        ],
        index=0,
        help="Select cache behavior"
    )
    
    # Cache mode mapping
    cache_mode_map = {
        "Enabled": CacheMode.ENABLED,
        "Bypass": CacheMode.BYPASS,
        "Disabled": CacheMode.DISABLED
    }
    
    # Advanced settings
    st.header("Advanced Settings")
    
    # Deep Crawling Configuration
    st.subheader("Deep Crawling")
    enable_deep_crawl = st.toggle("Enable Deep Crawling", value=False, help="Enable deep crawling to explore multiple pages")
    
    if enable_deep_crawl:
        crawl_strategy = st.selectbox(
            "Crawling Strategy",
            options=["BFS (Breadth-First)", "DFS (Depth-First)", "Best-First"],
            help="Select the crawling strategy"
        )
        
        max_depth = st.number_input(
            "Maximum Depth",
            min_value=1,
            max_value=5,
            value=2,
            help="Maximum number of levels to crawl"
        )
        
        max_pages = st.number_input(
            "Maximum Pages",
            min_value=1,
            max_value=100,
            value=50,
            help="Maximum number of pages to crawl"
        )
        
        include_external = st.toggle(
            "Include External Links",
            value=False,
            help="Follow links to external domains"
        )
        
        if crawl_strategy == "Best-First":
            st.subheader("Best-First Configuration")
            keywords = st.text_input(
                "Keywords (comma-separated)",
                placeholder="crawl, example, async, configuration",
                help="Keywords to prioritize pages"
            ).strip()
            keyword_weight = st.slider(
                "Keyword Weight",
                min_value=0.0,
                max_value=1.0,
                value=0.7,
                step=0.1,
                help="Weight of keyword relevance in scoring"
            )

    # Content Filter Type
    content_filter_type = st.selectbox(
        "Content Filter Type",
        options=["Pruning", "BM25"],
        help="Select the type of content filter to use"
    )
    
    if content_filter_type == "Pruning":
        threshold = st.slider(
            "Pruning Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.48,
            step=0.01,
            help="Threshold for the pruning filter"
        )
        threshold_type = st.selectbox(
            "Threshold Type",
            options=["fixed", "auto"],
            help="Threshold type for pruning filter"
        )
        min_word_threshold = st.number_input(
            "Min Word Threshold",
            min_value=0,
            value=0,
            help="Minimum word threshold for the pruning filter"
        )
    elif content_filter_type == "BM25":
        user_query = st.text_input(
            "BM25 Query",
            placeholder="Enter your query for BM25 filtering",
            help="Query for BM25-based filtering"
        )
        bm25_threshold = st.slider(
            "BM25 Threshold",
            min_value=0.1,
            max_value=5.0,
            value=1.0,
            step=0.1,
            help="Threshold for BM25 filter"
        )
    
    # Custom JavaScript execution
    st.header("Custom JavaScript")
    js_code = st.text_area(
        "JavaScript Code",
        placeholder="Enter JavaScript code to execute on the page",
        help="Custom JavaScript code to execute on the page"
    )
    
    # Extraction strategy
    st.header("Data Extraction")
    extraction_type = st.selectbox(
        "Extraction Type",
        options=["None", "LLM", "JSON CSS"],
        help="Select the type of data extraction"
    )
    
    if extraction_type == "LLM":
        st.info("LLM extraction requires API keys to be configured")
        llm_provider = st.selectbox(
            "LLM Provider",
            options=["openai/gpt-4o", "anthropic/claude-3-opus", "ollama/llama3", "mistral/mistral-large"],
            help="Select the LLM provider"
        )
        llm_api_key = st.text_input(
            "API Key",
            type="password",
            placeholder="Enter your API key",
            help="API key for the selected LLM provider"
        )
        llm_instruction = st.text_area(
            "Extraction Instructions",
            placeholder="Enter instructions for the LLM extraction",
            help="Instructions for LLM-based extraction"
        )
    elif extraction_type == "JSON CSS":
        css_schema = st.text_area(
            "CSS Schema (JSON)",
            placeholder='{"name": "Example Schema", "baseSelector": "div.item", "fields": [{"name": "title", "selector": "h2", "type": "text"}]}',
            height=200,
            help="JSON schema for CSS-based extraction"
        )

# Main content area
tab1, tab2, tab3 = st.tabs(["Single URL", "Batch Crawling", "Results"])

# Single URL tab
with tab1:
    st.header("Crawl a Single URL")
    
    url = st.text_input("URL to Crawl", placeholder="https://example.com")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        start_button = st.button("Start Crawling", use_container_width=True)
    
    # Main crawling function
    async def crawl_url(url, config):
        browser_config = BrowserConfig(
            headless=headless,
            verbose=verbose
        )
        
        # Set up the content filter based on selection
        if content_filter_type == "Pruning":
            content_filter = PruningContentFilter(
                threshold=threshold,
                threshold_type=threshold_type,
                min_word_threshold=min_word_threshold
            )
        else:  # BM25
            content_filter = BM25ContentFilter(
                user_query=user_query,
                bm25_threshold=bm25_threshold
            )
        
        # Configure extraction strategy
        extraction_strategy = None
        if extraction_type == "LLM" and llm_api_key and llm_instruction:
            from crawl4ai import LLMConfig
            extraction_strategy = LLMExtractionStrategy(
                llm_config=LLMConfig(provider=llm_provider, api_token=llm_api_key),
                instruction=llm_instruction
            )
        elif extraction_type == "JSON CSS" and css_schema:
            try:
                schema = json.loads(css_schema)
                extraction_strategy = JsonCssExtractionStrategy(schema=schema)
            except json.JSONDecodeError:
                st.error("Invalid JSON schema")
                return None
        
        # Setup deep crawling strategy if enabled
        deep_crawl_strategy = None
        if enable_deep_crawl:
            if crawl_strategy == "BFS (Breadth-First)":
                deep_crawl_strategy = BFSDeepCrawlStrategy(
                    max_depth=max_depth,
                    include_external=include_external,
                    max_pages=max_pages
                )
            elif crawl_strategy == "DFS (Depth-First)":
                deep_crawl_strategy = DFSDeepCrawlStrategy(
                    max_depth=max_depth,
                    include_external=include_external,
                    max_pages=max_pages
                )
            elif crawl_strategy == "Best-First" and keywords:
                keyword_list = [k.strip() for k in keywords.split(",") if k.strip()]
                scorer = KeywordRelevanceScorer(
                    keywords=keyword_list,
                    weight=keyword_weight
                )
                deep_crawl_strategy = BestFirstCrawlingStrategy(
                    max_depth=max_depth,
                    include_external=include_external,
                    max_pages=max_pages,
                    url_scorer=scorer
                )
        
        # Setup the crawler run configuration
        run_config = CrawlerRunConfig(
            cache_mode=cache_mode_map[cache_mode],
            markdown_generator=DefaultMarkdownGenerator(content_filter=content_filter),
            extraction_strategy=extraction_strategy,
            deep_crawl_strategy=deep_crawl_strategy,
            scraping_strategy=LXMLWebScrapingStrategy(),
            stream=True if enable_deep_crawl else False
        )
        
        # Add custom JavaScript if provided
        if js_code:
            run_config.js_code = [js_code]
        
        # Run the crawler
        async with AsyncWebCrawler(config=browser_config) as crawler:
            try:
                if enable_deep_crawl:
                    results = []
                    async for result in await crawler.arun(url=url, config=run_config):
                        results.append(result)
                        # Update progress
                        progress_text.text(f"Crawled {len(results)} pages...")
                    return results[0] if results else None  # Return first result for display
                else:
                    result = await crawler.arun(url=url, config=run_config)
                    return result
            except Exception as e:
                st.error(f"Error during crawling: {str(e)}")
                return None
    
    if start_button and url:
        # Display spinner while crawling
        with st.spinner(f"Crawling {url}..."):
            # Create a placeholder for progress updates
            progress_text = st.empty()
            
            # Run the async crawling function
            result = asyncio.run(crawl_url(url, None))
            
            if result:
                # Store the result in session state for use in other tabs
                if 'results' not in st.session_state:
                    st.session_state['results'] = []
                
                # Append the new result
                st.session_state['results'].append({
                    'url': url,
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'result': result
                })
                
                # Show success message
                st.success(f"Successfully crawled {url}")
                
                # Display some basic information about the result
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**URL:** {url}")
                    st.markdown(f"**Title:** {result.metadata.get('title', 'N/A')}")
                    st.markdown(f"**Content Length:** {len(result.markdown.raw_markdown)} characters")
                    
                with col2:
                    # Check if attributes exist before accessing them
                    images_count = len(result.metadata.get('images', [])) if 'images' in result.metadata else 0
                    links_count = len(result.metadata.get('links', [])) if 'links' in result.metadata else 0
                    st.markdown(f"**Images:** {images_count}")
                    st.markdown(f"**Links:** {links_count}")
                    st.markdown(f"**Successful:** {result.metadata.get('success', False)}")
                
                # Show tabs for different result views
                result_tab1, result_tab2, result_tab3, result_tab4 = st.tabs([
                    "Raw Markdown", "Fit Markdown", "Extracted Data", "Metadata"
                ])
                
                with result_tab1:
                    st.code(result.markdown.raw_markdown, language="markdown")
                    st.download_button(
                        label="Download Raw Markdown",
                        data=result.markdown.raw_markdown,
                        file_name=f"crawl4ai_raw_markdown_{time.strftime('%Y%m%d_%H%M%S')}.md",
                        mime="text/markdown"
                    )
                
                with result_tab2:
                    st.code(result.markdown.fit_markdown, language="markdown")
                    st.download_button(
                        label="Download Fit Markdown",
                        data=result.markdown.fit_markdown,
                        file_name=f"crawl4ai_fit_markdown_{time.strftime('%Y%m%d_%H%M%S')}.md",
                        mime="text/markdown"
                    )
                
                with result_tab3:
                    if result.extracted_content:
                        try:
                            if isinstance(result.extracted_content, str):
                                extracted_data = json.loads(result.extracted_content)
                                st.json(extracted_data)
                            else:
                                st.json(result.extracted_content)
                        except:
                            st.text(result.extracted_content)
                    else:
                        st.info("No data was extracted. Configure an extraction strategy in the sidebar.")
                
                with result_tab4:
                    st.json(result.metadata)

with tab2:
    st.header("Batch Crawling")
    st.info("Coming soon! This feature will allow you to crawl multiple URLs at once.")
    
    # Placeholder for batch crawling functionality
    urls = st.text_area("URLs (one per line)", placeholder="https://example.com\nhttps://example.org")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        batch_button = st.button("Start Batch Crawl", disabled=True)

with tab3:
    st.header("Results")
    
    if 'results' in st.session_state and st.session_state['results']:
        # Create a dataframe for easier viewing
        results_df = pd.DataFrame([
            {
                'URL': r['url'],
                'Timestamp': r['timestamp'],
                'Title': r['result'].metadata.get('title', 'N/A'),
                'Content Length': len(r['result'].markdown.raw_markdown)
            } for r in st.session_state['results']
        ])
        
        st.dataframe(results_df, use_container_width=True)
        
        # Allow selecting and viewing a specific result
        result_indices = list(range(len(st.session_state['results'])))
        selected_result = st.selectbox(
            "Select a result to view",
            options=result_indices,
            format_func=lambda x: f"{st.session_state['results'][x]['url']} ({st.session_state['results'][x]['timestamp']})"
        )
        
        if selected_result is not None:
            result_data = st.session_state['results'][selected_result]['result']
            
            st.subheader("Selected Result Details")
            
            # Display result details in expandable sections
            with st.expander("Markdown Content", expanded=True):
                st.code(result_data.markdown.fit_markdown, language="markdown")
            
            with st.expander("Extracted Data"):
                if result_data.extracted_content:
                    try:
                        if isinstance(result_data.extracted_content, str):
                            extracted_data = json.loads(result_data.extracted_content)
                            st.json(extracted_data)
                        else:
                            st.json(result_data.extracted_content)
                    except:
                        st.text(result_data.extracted_content)
                else:
                    st.info("No data was extracted for this URL.")
            
            with st.expander("Images"):
                images = result_data.metadata.get('images', [])
                if images:
                    for i, img in enumerate(images):
                        st.markdown(f"**Image {i+1}:** {img.get('alt', 'No alt text')}")
                        st.markdown(f"**URL:** {img.get('src', 'No source')}")
                        if 'src' in img and img['src'].startswith('http'):
                            st.image(img['src'], use_column_width=True)
                else:
                    st.info("No images were found in this URL.")
            
            with st.expander("Links"):
                links = result_data.metadata.get('links', [])
                if links:
                    links_df = pd.DataFrame([
                        {
                            'URL': link.get('href', 'N/A'),
                            'Text': link.get('text', 'N/A'),
                            'Type': 'External' if link.get('href', '').startswith(('http', 'https')) and not link.get('href', '').startswith(result_data.metadata.get('base_url', '')) else 'Internal'
                        } for link in links if 'href' in link
                    ])
                    st.dataframe(links_df, use_container_width=True)
                else:
                    st.info("No links were found in this URL.")
            
            # Allow downloading the result as JSON
            result_json = json.dumps({
                'url': st.session_state['results'][selected_result]['url'],
                'timestamp': st.session_state['results'][selected_result]['timestamp'],
                'markdown': result_data.markdown.fit_markdown,
                'metadata': result_data.metadata,
                'extracted_content': result_data.extracted_content
            }, default=str, indent=2)
            
            st.download_button(
                label="Download Result as JSON",
                data=result_json,
                file_name=f"crawl4ai_result_{selected_result}.json",
                mime="application/json"
            )
    else:
        st.info("No results yet. Crawl some URLs to see results here.")

# Footer
st.markdown("---")
st.markdown(
    "Built with ❤️ using [Crawl4AI](https://github.com/unclecode/crawl4ai) - "
    "Open-source LLM Friendly Web Crawler & Scraper"
) 