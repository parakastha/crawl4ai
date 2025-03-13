# Crawl4AI Web Interface

A professional and interactive web interface for the Crawl4AI library, built with Streamlit.

## Features

- **Single URL Crawling**: Crawl any website and view the results in real-time
- **Advanced Configuration**: Configure crawling behavior, content filtering, and extraction strategies
- **Content Filtering**: Use Pruning or BM25 filters to extract the most relevant content
- **Data Extraction**: Extract structured data using LLMs or CSS selectors
- **Custom JavaScript**: Execute custom JavaScript code on the page during crawling
- **Results Management**: View, compare, and download crawling results
- **User-Friendly Interface**: Intuitive UI with helpful tooltips and explanations

## Installation

1. Ensure you have Python 3.8+ installed
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Run the post-installation setup for Crawl4AI:

```bash
crawl4ai-setup
```

## Usage

1. Start the Streamlit app:

```bash
streamlit run app.py
```

2. Open your web browser and go to `http://localhost:8501`

3. Configure the crawler settings in the sidebar:
   - Basic settings (headless mode, cache behavior)
   - Content filtering options
   - Data extraction strategies
   - Custom JavaScript code

4. Enter a URL to crawl in the "Single URL" tab and click "Start Crawling"

5. View the results in various formats:
   - Raw Markdown: The full extracted markdown content
   - Fit Markdown: The filtered, LLM-friendly markdown content
   - Extracted Data: Structured data extracted according to your configuration
   - Metadata: Information about the crawling process and the page

6. View past results in the "Results" tab

## Advanced Features

### LLM Data Extraction

To extract structured data using LLMs:

1. Select "LLM" in the "Extraction Type" dropdown
2. Choose an LLM provider (OpenAI, Anthropic, etc.)
3. Enter your API key
4. Provide detailed extraction instructions

### CSS-Based Data Extraction

To extract structured data using CSS selectors:

1. Select "JSON CSS" in the "Extraction Type" dropdown
2. Define your schema in the JSON format:

```json
{
  "name": "Example Schema",
  "baseSelector": "div.item",
  "fields": [
    {
      "name": "title",
      "selector": "h2",
      "type": "text"
    },
    {
      "name": "price",
      "selector": "span.price",
      "type": "text"
    }
  ]
}
```

### Custom JavaScript Execution

You can execute custom JavaScript code on the page by entering it in the "JavaScript Code" field in the sidebar. This is useful for:

- Scrolling through a page to load lazy-loaded content
- Clicking buttons to reveal hidden content
- Expanding sections of a page
- Interacting with page elements

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [Crawl4AI](https://github.com/unclecode/crawl4ai) - The underlying web crawler library
- [Streamlit](https://streamlit.io/) - The framework used to build the web interface
