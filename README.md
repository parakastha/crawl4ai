# Crawl4AI Agent

A powerful AI agent that can crawl any website using the Crawl4AI library. This agent leverages the existing functionality of Crawl4AI to extract content from websites, filter it, and return it in a structured format. The agent now includes true AI capabilities that enhance the crawling process with intelligent decision-making.

## Features

- **Website Crawling**: Crawl any website and extract content
- **Deep Crawling**: Explore multiple pages using BFS, DFS, or Best-First strategies
- **Content Filtering**: Filter content using Pruning or BM25 algorithms
- **Data Extraction**: Extract structured data using LLMs or CSS selectors
- **Custom JavaScript**: Execute custom JavaScript code on pages during crawling
- **Flexible Configuration**: Configure all aspects of the crawling process
- **AI-Powered Analysis**: Automatically analyze websites to determine optimal crawling strategies
- **Content Enhancement**: Use AI to improve and organize extracted content
- **Knowledge Retention**: Store and retrieve information from past crawls
- **Question Answering**: Ask questions about crawled content and get AI-generated answers

## Installation

1. Ensure you have Python 3.8+ installed
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Set up environment variables:
   - Copy `.env.example` to `.env`
   - Add your OpenAI API key to enable AI agent capabilities:
     ```
     OPENAI_API_KEY=your_openai_api_key_here
     ```

## Usage

### Web Interface

The easiest way to use Crawl4AI Agent is through its web interface:

```bash
python dev.py
```

This will start a Streamlit app accessible at http://localhost:8501 where you can configure all crawling options and use the AI agent features.

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

#### AI Agent Options

- `--use-ai-agent`: Enable AI agent capabilities
- `--analyze-website`: Analyze website structure to determine optimal crawling strategy
- `--enhance-content`: Improve and organize extracted content
- `--store-results`: Store crawl results for future reference
- `--ai-question`: Ask a question about the crawled content

#### Content Filter Options

- `--content-filter`: Content filter type (Pruning, BM25)
- `--threshold`: Pruning threshold (default: 0.48)
- `--min-word-threshold`: Min word threshold
- `--user-query`: BM25 query
- `--bm25-threshold`: BM25 threshold (default: 0.1)

#### Extraction Options

- `--extraction-strategy`: Extraction strategy (Basic, LLM, JSON CSS)
- `--css-selector`: CSS selector for JSON CSS extraction

#### Deep Crawling Options

- `--deep-crawl`: Enable deep crawling
- `--crawl-strategy`: Crawling strategy (BFS, DFS, Best-First)
- `--max-depth`: Maximum depth (default: 2)
- `--max-pages`: Maximum pages (default: 10)
- `--follow-external-links`: Follow external links

#### JavaScript & Output Options

- `--js-code`: Custom JavaScript code to execute on the page
- `--delay-before-return-html`: Delay before returning HTML (seconds)
- `--wait-for`: CSS selector or XPath to wait for
- `--magic`: Enable magic mode for better extraction
- `--remove-overlay-elements`: Remove overlay elements
- `--save-raw-markdown`: Save raw markdown to file
- `--word-count-threshold`: Word count threshold
- `--excluded-tags`: Excluded tags (e.g., script, style)

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
        deep_crawl=True,
        deep_crawl_strategy="BFS",
        max_depth=2,
        max_pages=10,
        # AI Agent features
        use_ai_agent=True,
        analyze_website=True,
        enhance_content=True,
        ai_question="What is the main topic of this website?"
    )
    
    result = await crawl_url(config)
    
    # Access AI-enhanced content and answers
    print(result["ai_enhanced_content"])
    print(result["ai_answer"])

if __name__ == "__main__":
    asyncio.run(main())
```

## Output

The agent returns a structured result with:

- Raw and processed markdown content
- AI-enhanced content
- Answers to questions about the content
- Crawl statistics
- Success status and any error information

## Examples

### Basic Crawling

```bash
python crawl_agent.py https://example.com
```

### Using the AI Agent

```bash
python crawl_agent.py https://example.com --use-ai-agent --analyze-website --enhance-content --ai-question="What are the main products offered on this website?"
```

### Deep Crawling with AI-Powered Link Prioritization

```bash
python crawl_agent.py https://example.com --deep-crawl --crawl-strategy="Best-First" --max-depth=3 --max-pages=20 --use-ai-agent
```

### Content Enhancement with Proxy Support

```bash
python crawl_agent.py https://example.com --use-ai-agent --enhance-content --use-proxy --proxy-server="http://proxy.example.com:8080"
```

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.
