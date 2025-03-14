# Crawl4AI Examples

This directory contains example scripts demonstrating various features of Crawl4AI.

## Page Interaction Examples

The examples demonstrate advanced page interaction capabilities using Crawl4AI:

1. [Multi-Step Interaction](multi_step_interaction.py) - Shows how to implement multi-step interactions, such as scrolling, clicking "Load More" buttons, and waiting for new content.

## Page Interaction Features

Crawl4AI supports various page interaction features:

### Basic JavaScript Execution

```python
from crawl_agent import CrawlConfig, crawl_url

config = CrawlConfig(
    url="https://example.com",
    js_code="window.scrollTo(0, document.body.scrollHeight);"
)
result = await crawl_url(config)
```

### Wait Conditions

Wait for elements or JavaScript conditions:

```python
# Wait for a CSS selector
config = CrawlConfig(
    url="https://example.com",
    wait_for="css:.article-content"
)

# Wait for a JavaScript condition
config = CrawlConfig(
    url="https://example.com",
    wait_for="""js:() => {
        return document.querySelectorAll('.article').length > 10;
    }"""
)
```

### Multi-Step Interaction

Execute a sequence of actions with different wait conditions:

```python
config = CrawlConfig(
    url="https://example.com",
    multi_step_enabled=True,
    session_id="my_session",
    js_only=True,
    
    # Define steps
    multi_step_js_actions=[
        "window.scrollTo(0, document.body.scrollHeight);",
        "document.querySelector('button.load-more').click();"
    ],
    
    # Wait conditions for each step
    multi_step_wait_conditions=[
        "css:button.load-more",
        "js:() => document.querySelectorAll('.article').length > 20;"
    ],
    
    # Delays after each step
    multi_step_delays=[1, 2]
)
```

### Form Interaction

Fill out and submit forms:

```python
form_fill_js = """
document.querySelector('#search-input').value = 'example query';
document.querySelector('form.search').submit();
"""

config = CrawlConfig(
    url="https://example.com/search",
    js_code=form_fill_js,
    wait_for="css:.search-results"
)
```

### Advanced Interaction Options

```python
config = CrawlConfig(
    url="https://example.com",
    
    # Timing control
    page_timeout=60000,  # 60 seconds max page load time
    delay_before_return_html=2,  # Wait 2 seconds after page load
    
    # Anti-bot features
    simulate_user=True,  # Simulate human-like behavior
    override_navigator=True,  # Override navigator properties
    magic=True,  # Enable magic mode for anti-bot detection
    
    # Content processing
    process_iframes=True,  # Extract content from iframes
    remove_overlay_elements=True  # Remove modals/overlays
)
```

### Session Management

Maintain browser state across multiple calls:

```python
# First request
config1 = CrawlConfig(
    url="https://example.com/login",
    session_id="my_session",
    js_code="document.querySelector('#login-form').submit();"
)
result1 = await crawl_url(config1)

# Second request uses same session (cookies, localStorage preserved)
config2 = CrawlConfig(
    url="https://example.com/dashboard",
    session_id="my_session",
    js_only=True  # Don't navigate, just run JS in current page
)
result2 = await crawl_url(config2)
```

## Running the Examples

To run these examples:

```bash
python examples/multi_step_interaction.py
```

For more details, see the [Crawl4AI Page Interaction Documentation](https://docs.crawl4ai.com/core/page-interaction/). 