# Crawl4AI Agent

A powerful AI agent that can crawl any website using the Crawl4AI library. This agent leverages the existing functionality of Crawl4AI to extract content from websites, filter it, and return it in a structured format.

## Features

- **Website Crawling**: Crawl any website and extract content
- **Deep Crawling**: Explore multiple pages using BFS, DFS, or Best-First strategies
- **Content Filtering**: Filter content using Pruning or BM25 algorithms
- **Data Extraction**: Extract structured data using LLMs or CSS selectors
- **Custom JavaScript**: Execute custom JavaScript code on pages during crawling
- **Flexible Configuration**: Configure all aspects of the crawling process

## Installation

1. Ensure you have Python 3.8+ installed
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Set up environment variables (optional):
   - Copy `.env.example` to `.env`
   - Add your API keys for LLM providers if you plan to use LLM extraction

## Usage

### Command Line Interface

The agent can be used from the command line with various options:

```bash
python crawl_agent.py https://example.com --headless --content-filter=Pruning
```

#### Basic Options

- `url`: The URL to crawl (required)
- `--headless`: Run browser in headless mode
- `--verbose`: Enable verbose output
- `--cache-mode`: Cache mode (Enabled, Bypass, Disabled)

#### Content Filter Options

- `--content-filter`: Content filter type (Pruning, BM25)
- `--threshold`: Pruning threshold (default: 0.48)
- `--threshold-type`: Threshold type (fixed, auto)
- `--min-word-threshold`: Min word threshold
- `--user-query`: BM25 query
- `--bm25-threshold`: BM25 threshold (default: 1.0)

#### Extraction Options

- `--extraction-type`: Extraction type (None, LLM, JSON CSS)
- `--llm-provider`: LLM provider
- `--llm-api-key`: LLM API key
- `--llm-instruction`: LLM instruction
- `--css-schema`: CSS schema (JSON)

#### Deep Crawling Options

- `--deep-crawl`: Enable deep crawling
- `--crawl-strategy`: Crawling strategy (BFS (Breadth-First), DFS (Depth-First), Best-First)
- `--max-depth`: Maximum depth (default: 2)
- `--max-pages`: Maximum pages (default: 10)
- `--include-external`: Include external links
- `--keywords`: Keywords (comma-separated)
- `--keyword-weight`: Keyword weight (default: 0.7)

#### Custom JavaScript

- `--js-code`: Custom JavaScript code to execute on the page

### Python API

You can also use the agent programmatically in your Python code:

```python
import asyncio
from crawl_agent import CrawlConfig, crawl_url

async def main():
    config = CrawlConfig(
        url="https://example.com",
        headless=True,
        content_filter_type="Pruning",
        threshold=0.48,
        enable_deep_crawl=True,
        crawl_strategy="BFS (Breadth-First)",
        max_depth=2,
        max_pages=10
    )
    
    result = await crawl_url(config)
    print(result)

if __name__ == "__main__":
    asyncio.run(main())
```

## Output

The agent saves the extracted content to markdown files and returns a structured result with:

- Success status
- Paths to the saved markdown files
- Metadata about the crawled pages
- Extracted structured data (if extraction was configured)

## Examples

### Basic Crawling

```bash
python crawl_agent.py https://example.com
```

### Deep Crawling with BFS Strategy

```bash
python crawl_agent.py https://example.com --deep-crawl --crawl-strategy="BFS (Breadth-First)" --max-depth=3 --max-pages=20
```

### Using LLM Extraction

```bash
python crawl_agent.py https://example.com --extraction-type=LLM --llm-provider="openai/gpt-4o" --llm-instruction="Extract the main article content and key points"
```

### Using Custom JavaScript

```bash
python crawl_agent.py https://example.com --js-code="window.scrollTo(0, document.body.scrollHeight); await new Promise(r => setTimeout(r, 2000));"
```

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.
