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

# Sidebar for Configuration
with st.sidebar:
    st.header("Crawl Configuration")
    url = st.text_input("URL to Crawl", placeholder="https://example.com")
    
    # Expandable sections for advanced options
    with st.expander("🌐 Browser Settings"):
        headless = st.toggle("Headless Mode", value=True, 
            help="Run browser invisibly in the background")
        verbose = st.toggle("Verbose Output", value=False, 
            help="Show detailed logging information")
        cache_mode = st.selectbox("Cache Mode", 
            options=["Enabled", "Bypass", "Disabled"],
            help="Control browser caching behavior")
        
        # Add proxy configuration section
        use_proxy = st.checkbox("Use Proxy", value=False, 
            help="Enable proxy for web requests")
        
        proxy_server = None
        proxy_username = None
        proxy_password = None
        
        if use_proxy:
            proxy_server = st.text_input("Proxy Server (host:port)", 
                placeholder="http://proxy-server-address:port",
                help="URL of the proxy server including protocol")
            
            proxy_auth = st.checkbox("Proxy Authentication", value=False, 
                help="Enable if your proxy requires username and password")
            
            if proxy_auth:
                proxy_username = st.text_input("Proxy Username", 
                    help="Username for proxy authentication")
                proxy_password = st.text_input("Proxy Password", type="password", 
                    help="Password for proxy authentication")
            
            st.info("💡 Proxies can help access geo-restricted content and avoid IP blocking when crawling.")
            if proxy_server:
                if proxy_auth and proxy_username and proxy_password:
                    st.success(f"✅ Proxy configured with authentication")
                elif proxy_server:
                    st.success(f"✅ Proxy configured: {proxy_server}")

    with st.expander("🧹 Content Filtering"):
        filter_type = st.selectbox("Filter Strategy", 
            options=["Pruning", "BM25"],
            help="Choose how to filter and select content")
        
        if filter_type == "Pruning":
            threshold = st.slider("Pruning Threshold", 0.0, 1.0, 0.48,
                help="How strictly to filter content")
            threshold_type = st.selectbox("Threshold Type", 
                options=["fixed", "auto"],
                help="Method of applying the threshold")
            bm25_threshold = 1.0  # Default value when not using BM25
            user_query = None
        else:
            user_query = st.text_input("BM25 Query", 
                help="Specify keywords to prioritize content")
            bm25_threshold = st.slider("BM25 Threshold", 0.1, 5.0, 1.0,
                help="Sensitivity of BM25 content filtering")
            threshold = 0.48  # Default value when not using Pruning
            threshold_type = "fixed"  # Default value when not using Pruning

    with st.expander("🤖 Extraction Strategy"):
        extraction_type = st.selectbox("Extraction Method", 
            options=["None", "LLM", "JSON CSS"],
            help="Choose how to extract structured data")
        
        if extraction_type == "LLM":
            llm_provider = st.selectbox("LLM Provider", 
                options=["openai/gpt-4o", "anthropic/claude-3-opus", "ollama/llama3"],
                help="Choose the AI model for extraction")
            
            # Dynamically load API keys from environment variables
            if llm_provider.startswith("openai"):
                llm_api_key = os.getenv("OPENAI_API_KEY", "")
                st.info("OpenAI API key loaded from environment")
            elif llm_provider.startswith("anthropic"):
                llm_api_key = os.getenv("ANTHROPIC_API_KEY", "")
                st.info("Anthropic API key loaded from environment")
            else:
                llm_api_key = st.text_input("API Key", type="password",
                    help="API key for the selected LLM")
            
            llm_instruction = st.text_area("Extraction Instructions",
                help="Specify how the AI should extract information")
        
        elif extraction_type == "JSON CSS":
            css_schema = st.text_area("CSS Schema (JSON)", height=200,
                help="Define JSON schema for CSS-based extraction")

    with st.expander("🔍 Deep Crawling"):
        enable_deep_crawl = st.toggle("Enable Deep Crawling", 
            help="Crawl multiple interconnected pages")
        
        if enable_deep_crawl:
            crawl_strategy = st.selectbox("Crawling Strategy", 
                options=["BFS (Breadth-First)", "DFS (Depth-First)", "Best-First"],
                help="Choose how to navigate through pages")
            
            max_depth = st.number_input("Maximum Depth", 1, 5, 2,
                help="How many link levels to traverse")
            max_pages = st.number_input("Maximum Pages", 1, 100, 10,
                help="Limit total number of pages to crawl")
            include_external = st.toggle("Include External Links",
                help="Allow crawling to different domains")
            
            if crawl_strategy == "Best-First":
                keywords = st.text_input("Keywords", 
                    help="Comma-separated keywords to prioritize pages")
        else:
            # Default values when deep crawling is disabled
            max_depth = 2
            max_pages = 10
            include_external = False
            keywords = None
            crawl_strategy = None

    with st.expander("💻 Custom JavaScript"):
        js_code = st.text_area("Custom JS Code", 
            help="Execute custom JavaScript during page load")

    # Crawl Button
    start_button = st.sidebar.button("Start Crawling")

# Feature Explanations
st.markdown("## 🚀 Crawling Features Explained")

features = [
    {
        "title": "Browser Settings",
        "description": "Control how the browser behaves during crawling. Headless mode runs invisibly, while verbose output provides detailed logs."
    },
    {
        "title": "Proxy Support",
        "description": "Connect through proxy servers to access geo-restricted content or avoid IP blocking. Supports authentication for private proxies."
    },
    {
        "title": "Content Filtering",
        "description": "Refine the content you extract. Pruning filter removes less relevant content, while BM25 allows keyword-based selection."
    },
    {
        "title": "Extraction Strategies",
        "description": "Transform web content into structured data. Use AI (LLM) for intelligent extraction or CSS selectors for precise data picking."
    },
    {
        "title": "Deep Crawling",
        "description": "Explore interconnected web pages. Choose between Breadth-First, Depth-First, or Best-First strategies to navigate links."
    },
    {
        "title": "Custom JavaScript",
        "description": "Execute custom scripts during page load to interact with dynamic content or prepare pages for crawling."
    }
]

cols = st.columns(3)
for i, feature in enumerate(features):
    with cols[i % 3]:
        st.markdown(f"""
        <div class="feature-card">
            <div class="feature-title">{feature['title']}</div>
            {feature['description']}
        </div>
        """, unsafe_allow_html=True)

# Crawling Logic
if start_button and url:
    with st.spinner(f"Crawling {url}..."):
        # Dynamically create CrawlConfig based on selected options
        config = CrawlConfig(
            url=url,
            headless=headless,
            verbose=verbose,
            cache_mode=cache_mode,
            content_filter_type=filter_type,
            threshold=threshold,
            threshold_type=threshold_type,
            user_query=user_query,
            bm25_threshold=bm25_threshold,
            
            # Add proxy settings
            use_proxy=use_proxy,
            proxy_server=proxy_server if use_proxy else None,
            proxy_username=proxy_username if use_proxy and proxy_auth else None,
            proxy_password=proxy_password if use_proxy and proxy_auth else None,
            
            extraction_type=extraction_type,
            llm_provider=llm_provider if extraction_type == "LLM" else None,
            llm_api_key=llm_api_key if extraction_type == "LLM" else None,
            llm_instruction=llm_instruction if extraction_type == "LLM" else None,
            css_schema=css_schema if extraction_type == "JSON CSS" else None,
            enable_deep_crawl=enable_deep_crawl,
            crawl_strategy=crawl_strategy,
            max_depth=max_depth,
            max_pages=max_pages,
            include_external=include_external,
            keywords=keywords if crawl_strategy == "Best-First" else None,
            js_code=js_code
        )
        
        # Run the crawl
        result = asyncio.run(crawl_url(config))
        
        if result:
            # Check if result is a dictionary with a success flag
            if isinstance(result, dict) and result.get('success', False):
                # Store crawl statistics in session state if available
                if 'crawl_stats' in result:
                    st.session_state.crawl_stats = result['crawl_stats']
                
                # Create result display sections
                st.markdown("## 📄 Crawl Results")
                
                # Add a Statistics section for deep crawling
                if "crawl_stats" in st.session_state and st.session_state.crawl_stats:
                    with st.expander("📊 Crawl Statistics", expanded=True):
                        stats = st.session_state.crawl_stats
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Total Pages Crawled", stats.get("total_pages", 0))
                            st.metric("Valid Pages Processed", stats.get("valid_pages", 0))
                        with col2:
                            st.metric("Pages Filtered Out", stats.get("filtered_pages", 0))
                            st.metric("Max Depth Reached", stats.get("max_depth_reached", 0))
                
                # Metadata Section
                with st.expander("📋 Metadata", expanded=True):
                    st.markdown("""
                    <div class="result-section">
                    """, unsafe_allow_html=True)
                    
                    # Display key metadata
                    st.markdown(f"**Title:** {result.get('metadata', {}).get('title', 'N/A')}")
                    st.markdown(f"**Description:** {result.get('metadata', {}).get('description', 'N/A')}")
                    
                    # Full metadata JSON
                    st.json(result.get('metadata', {}))
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                
                # Raw Markdown Section
                with st.expander("📝 Raw Markdown", expanded=False):
                    st.markdown("""
                    <div class="result-section">
                    """, unsafe_allow_html=True)
                    
                    # Display raw markdown content directly from result
                    raw_markdown = result.get('raw_content', '')
                    if raw_markdown:
                        st.code(raw_markdown, language="markdown")
                    else:
                        st.warning("No raw markdown content available")
                    
                    # Download button for raw markdown
                    if raw_markdown:
                        custom_filename = st.text_input("Custom filename prefix for download (optional)", 
                                                       value="crawl4ai", 
                                                       key="raw_custom_filename")
                        st.download_button(
                            label="Download Raw Markdown",
                            data=raw_markdown,
                            file_name=f"{custom_filename}_raw_{time.strftime('%Y%m%d_%H%M%S')}.md",
                            mime="text/markdown"
                        )
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                
                # Fit Markdown Section
                with st.expander("📝 Processed Markdown", expanded=False):
                    st.markdown("""
                    <div class="result-section">
                    """, unsafe_allow_html=True)
                    
                    # Display fit markdown content directly from result
                    fit_markdown = result.get('fit_content', '')
                    if fit_markdown:
                        st.code(fit_markdown, language="markdown")
                    else:
                        st.warning("No processed markdown content available")
                    
                    # Download button for fit markdown
                    if fit_markdown:
                        custom_filename = st.text_input("Custom filename prefix for download (optional)", 
                                                       value="crawl4ai", 
                                                       key="fit_custom_filename")
                        st.download_button(
                            label="Download Processed Markdown",
                            data=fit_markdown,
                            file_name=f"{custom_filename}_fit_{time.strftime('%Y%m%d_%H%M%S')}.md",
                            mime="text/markdown"
                        )
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                
                # Extracted Content Section
                if result.get('extracted_content'):
                    with st.expander("🔍 Extracted Content", expanded=False):
                        st.markdown("""
                        <div class="result-section">
                        """, unsafe_allow_html=True)
                        
                        st.json(result['extracted_content'])
                        
                        st.markdown("</div>", unsafe_allow_html=True)
            else:
                # If the result is not a successful dictionary or is an error
                st.error("Crawl failed")
                st.json(result)