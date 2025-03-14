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
        cache_mode = st.selectbox("Cache Mode", ["ENABLED", "BYPASS", "DISABLED", "READ_ONLY", "WRITE_ONLY"], index=0)
        
        # Proxy settings
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
    
    with st.expander("🔧 Advanced Settings", expanded=False):
        extraction_strategy = st.selectbox("Extraction Strategy", ["Basic", "LLM", "JSON CSS"], index=0)
        
        if extraction_strategy == "LLM":
            st.warning("LLM extraction requires additional configuration. Some features may be unavailable.")
        elif extraction_strategy == "JSON CSS":
            css_selector = st.text_area("CSS Selectors (JSON)", value='{"title": "h1", "content": "main"}')
        
        js_code = st.text_area("Custom JavaScript (runs before extraction)")
        delay_before_return_html = st.slider("Delay Before Return HTML (seconds)", 0, 30, 0)
        wait_for = st.text_input("Wait for Element (CSS selector or XPath)")
        
        magic = st.checkbox("Magic Mode", value=False, help="Try different techniques to extract content")
        remove_overlay_elements = st.checkbox("Remove Overlay Elements", value=False)
        save_raw_markdown = st.checkbox("Save Raw Markdown to File", value=False)
        
        word_count_threshold = st.number_input("Word Count Threshold", 0, 1000, 0)
        excluded_tags = st.multiselect("Excluded Tags", ["script", "style", "svg", "path", "noscript"], default=["script", "style", "svg", "noscript"])

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
            
            # Advanced settings
            extraction_strategy=extraction_strategy,
            css_selector=css_selector if extraction_strategy == "JSON CSS" else "",
            js_code=js_code,
            delay_before_return_html=delay_before_return_html,
            wait_for=wait_for,
            magic=magic,
            remove_overlay_elements=remove_overlay_elements,
            save_raw_markdown=save_raw_markdown,
            word_count_threshold=word_count_threshold,
            excluded_tags=excluded_tags
        )
        
        # Run the crawl
        result = asyncio.run(crawl_url(config))
        
        if result.get("status") == "success":
            # Stats tab
            tabs = st.tabs(["Results", "Raw Markdown", "Processed Markdown", "Stats", "AI Enhanced Content", "AI Answer"])
            
            with tabs[0]:
                st.success(f"Successfully crawled {url}")
                
                # Display stats in a card
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
            
            with tabs[1]:
                if result.get("raw_content"):
                    st.markdown("### Raw Markdown Content")
                    st.text_area("Raw Markdown", result["raw_content"], height=500)
                else:
                    st.info("No raw markdown content generated.")
            
            with tabs[2]:
                if result.get("fit_content"):
                    st.markdown("### Processed Markdown Content")
                    st.text_area("Processed Markdown", result["fit_content"], height=500)
                else:
                    st.info("No processed markdown content generated.")
            
            with tabs[3]:
                st.markdown("### Crawl Statistics")
                
                # Detailed stats
                if "stats" in result:
                    st.json(result["stats"])
                else:
                    st.info("No detailed statistics available.")
            
            with tabs[4]:
                if result.get("ai_enhanced_content"):
                    st.markdown("### AI Enhanced Content")
                    st.markdown(result["ai_enhanced_content"])
                else:
                    if use_ai_agent and enhance_content:
                        st.info("AI enhancement was enabled but no enhanced content was generated.")
                    else:
                        st.info("Enable the AI Agent and Content Enhancement option to generate AI-enhanced content.")
            
            with tabs[5]:
                if result.get("ai_answer"):
                    st.markdown("### AI Answer")
                    st.markdown(result["ai_answer"])
                elif ai_question:
                    st.info("No answer was generated for your question.")
                else:
                    st.info("Enter a question in the AI Agent section to get an answer based on the crawled content.")
        
        else:
            st.error(f"Error during crawling: {result.get('error', 'Unknown error')}")
            if "raw_content" in result and result["raw_content"]:
                st.markdown("### Partial Raw Content")
                st.text_area("Partial Raw Content", result["raw_content"], height=300)