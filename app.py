import streamlit as st
import asyncio
import json
from crawl_agent import crawl_url, CrawlConfig

st.set_page_config(page_title="Crawl4AI", page_icon="🕸️")

st.title("Web Crawler")

url = st.text_input("URL to Crawl", placeholder="https://example.com")
start_button = st.button("Crawl")

if start_button and url:
    with st.spinner(f"Crawling {url}..."):
        # Create a basic CrawlConfig with default settings
        config = CrawlConfig(url=url)
        
        # Run the crawl
        result = asyncio.run(crawl_url(config))
        
        if result:
            # Check if result is a dictionary (error case) or an object
            if isinstance(result, dict):
                # Display error information
                st.error("Crawl failed")
                st.json(result)
            else:
                # Display raw markdown
                st.markdown("### Raw Markdown")
                st.code(result.markdown.raw_markdown, language="markdown")
                
                # Display metadata
                st.markdown("### Metadata")
                st.json(result.metadata)
                
                # Display extracted content if available
                if hasattr(result, 'extracted_content') and result.extracted_content:
                    st.markdown("### Extracted Content")
                    st.json(result.extracted_content)