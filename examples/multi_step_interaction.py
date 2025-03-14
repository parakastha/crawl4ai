#!/usr/bin/env python3
"""
Example script demonstrating multi-step interaction with a website using Crawl4AI.
This example shows how to:
1. Load a page
2. Scroll to the bottom
3. Click a "Load More" button
4. Wait for new content
5. Extract content after each step
"""

import asyncio
import logging
from crawl_agent import CrawlConfig, crawl_url

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def multi_step_demo():
    """Run a multi-step interaction example."""
    # Example: Crawl Hacker News, click the "More" link to load more stories
    logger.info("Running multi-step interaction example with Hacker News")
    
    # Step 1: Scroll to bottom JavaScript
    scroll_to_bottom_js = "window.scrollTo(0, document.body.scrollHeight);"
    
    # Step 2: Click the "More" link JavaScript
    click_more_js = """
    const moreLink = document.querySelector('a.morelink');
    if (moreLink) {
        console.log("Found 'More' link, clicking it...");
        moreLink.click();
        return true;
    } else {
        console.log("'More' link not found");
        return false;
    }
    """
    
    # Wait condition: Wait for more items to load after clicking "More"
    wait_for_more_items = """js:() => {
        const items = document.querySelectorAll('.athing');
        // Wait until we have at least 60 items (default page has 30)
        return items.length > 60;
    }"""
    
    # Configure the multi-step crawl
    config = CrawlConfig(
        url="https://news.ycombinator.com/",
        headless=False,  # Set to False to see the browser in action
        verbose=True,
        cache_mode="BYPASS",  # Always get fresh content
        
        # Enable multi-step interaction
        multi_step_enabled=True,
        session_id="hn_session",  # Unique session ID for this crawl
        
        # Define the steps
        multi_step_js_actions=[
            scroll_to_bottom_js,  # Step 1: Scroll to bottom
            click_more_js,        # Step 2: Click "More" link
        ],
        
        # Define wait conditions for each step
        multi_step_wait_conditions=[
            "css:a.morelink",    # Step 1: Wait for "More" link to be visible
            wait_for_more_items  # Step 2: Wait for more items to load
        ],
        
        # Delay after each step (in seconds)
        multi_step_delays=[1, 2],
        
        # Basic settings
        js_only=True,  # Use JS-only mode for subsequent steps
        delay_before_return_html=2,  # Wait 2 seconds before capturing final state
        word_count_threshold=10,  # Filter out very short blocks
        excluded_tags=["script", "style", "svg", "noscript"],
        save_raw_markdown=True
    )
    
    # Run the multi-step crawl
    result = await crawl_url(config)
    
    # Check the results
    if result.get("status") == "success":
        logger.info("Multi-step crawl successful!")
        
        if "stats" in result:
            logger.info(f"Stats: {result['stats']}")
        
        # Check content length
        raw_content = result.get("raw_content", "")
        logger.info(f"Raw content length: {len(raw_content)} characters")
        
        # Print the first 500 characters to see what we got
        if raw_content:
            logger.info(f"Content preview: {raw_content[:500]}...")
        
        # Save to file for inspection
        output_file = "hn_multi_step_result.md"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(raw_content)
        logger.info(f"Saved content to {output_file}")
    else:
        logger.error(f"Multi-step crawl failed: {result.get('error', 'Unknown error')}")

async def advanced_demo():
    """Run a more advanced multi-step interaction example with a site that has infinite scroll."""
    # Example: GitHub discussions - scroll multiple times to load more content
    logger.info("Running advanced multi-step interaction with GitHub discussions")
    
    # Multiple scroll steps to trigger infinite scroll
    scroll_step_js = """
    // Get current scroll position
    const curScroll = window.scrollY;
    // Scroll down by 1000px
    window.scrollTo(0, curScroll + 1000);
    console.log(`Scrolled from ${curScroll} to ${curScroll + 1000}`);
    // Return true for success
    return true;
    """
    
    # Wait condition: Wait for new content to load after scrolling
    wait_for_new_content = """js:() => {
        // Count visible discussion items
        const discussions = document.querySelectorAll('.discussion-list-item');
        // Store count in window object to track progress
        if (!window.lastDiscussionCount) {
            window.lastDiscussionCount = discussions.length;
            console.log(`Initial discussions count: ${discussions.length}`);
            return false;
        }
        
        const newCount = discussions.length;
        console.log(`Current discussions count: ${newCount}, previous: ${window.lastDiscussionCount}`);
        
        // If we found more discussions, we've loaded new content
        if (newCount > window.lastDiscussionCount) {
            window.lastDiscussionCount = newCount;
            return true;
        }
        
        // No new content yet
        return false;
    }"""
    
    # Configure the advanced multi-step crawl
    config = CrawlConfig(
        url="https://github.com/langchain-ai/langchain/discussions",
        headless=False,  # Set to False to see the browser in action
        verbose=True,
        cache_mode="BYPASS",  # Always get fresh content
        
        # Enable multi-step interaction
        multi_step_enabled=True,
        session_id="github_discussions",  # Unique session ID
        
        # Define multiple scroll steps
        multi_step_js_actions=[scroll_step_js] * 5,  # Repeat the scroll 5 times
        
        # Wait for new content after each scroll
        multi_step_wait_conditions=[wait_for_new_content] * 5,
        
        # Delay after each step (in seconds)
        multi_step_delays=[2] * 5,
        
        # Basic settings
        js_only=True,
        delay_before_return_html=2,
        page_timeout=120000,  # 2 minutes timeout for loading
        simulate_user=True,   # Try to appear more human-like
        word_count_threshold=10,
        excluded_tags=["script", "style", "svg", "noscript"],
        save_raw_markdown=True
    )
    
    # Run the advanced multi-step crawl
    result = await crawl_url(config)
    
    # Check the results
    if result.get("status") == "success":
        logger.info("Advanced multi-step crawl successful!")
        
        # Save to file for inspection
        output_file = "github_discussions_result.md"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(result.get("raw_content", ""))
        logger.info(f"Saved content to {output_file}")
    else:
        logger.error(f"Advanced multi-step crawl failed: {result.get('error', 'Unknown error')}")

async def main():
    """Run the examples."""
    # Run basic multi-step example
    await multi_step_demo()
    
    # Run advanced multi-step example
    await advanced_demo()

if __name__ == "__main__":
    asyncio.run(main()) 