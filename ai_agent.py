import os
import logging
import json
from typing import Dict, List, Any, Optional, Tuple
from pydantic import BaseModel, Field
import openai
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load OpenAI API key from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY not found in environment variables. AI capabilities will be limited.")

class AgentMemory:
    """Stores and retrieves knowledge from past crawls."""
    
    def __init__(self, collection_name: str = "crawl4ai_memory"):
        """Initialize the agent memory."""
        self.collection_name = collection_name
        try:
            self.embeddings = OpenAIEmbeddings()
            persist_directory = "./.chroma_db"
            self.vectorstore = Chroma(
                collection_name=collection_name, 
                embedding_function=self.embeddings,
                persist_directory=persist_directory
            )
            self.is_initialized = True
        except Exception as e:
            logger.error(f"Failed to initialize AgentMemory: {e}")
            self.is_initialized = False
    
    def add_knowledge(self, url: str, content: str, metadata: Dict[str, Any]) -> bool:
        """Add knowledge to the agent memory."""
        if not self.is_initialized:
            return False
        
        try:
            # Create a document with the content and metadata
            doc = Document(
                page_content=content,
                metadata={"url": url, **metadata}
            )
            self.vectorstore.add_documents([doc])
            return True
        except Exception as e:
            logger.error(f"Failed to add knowledge: {e}")
            return False
    
    def retrieve_relevant_knowledge(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """Retrieve relevant knowledge from the agent memory."""
        if not self.is_initialized:
            return []
        
        try:
            results = self.vectorstore.similarity_search(query, k=k)
            return [{"content": doc.page_content, "metadata": doc.metadata} for doc in results]
        except Exception as e:
            logger.error(f"Failed to retrieve knowledge: {e}")
            return []

class CrawlStrategy(BaseModel):
    """AI-recommended crawling strategy."""
    max_depth: int = Field(2, description="Maximum depth for deep crawling")
    max_pages: int = Field(10, description="Maximum number of pages to crawl")
    strategy_type: str = Field("bfs", description="Type of crawling strategy (bfs, dfs, best-first)")
    content_filter_type: str = Field("Pruning", description="Type of content filter to use")
    threshold: float = Field(0.48, description="Threshold for content filtering")
    focus_keywords: List[str] = Field([], description="Keywords to focus on during crawling")
    
    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()

class AIAgent:
    """Main AI agent that enhances the Crawl4AI capabilities."""
    
    def __init__(self):
        """Initialize the AI agent."""
        self.memory = AgentMemory()
        try:
            self.llm = ChatOpenAI(temperature=0)
            self.is_llm_available = True
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            self.is_llm_available = False
    
    async def analyze_website(self, url: str) -> Optional[CrawlStrategy]:
        """Analyze a website and recommend a crawling strategy."""
        if not self.is_llm_available:
            logger.warning("LLM not available, using default crawl strategy")
            return CrawlStrategy()
        
        # Retrieve relevant knowledge from memory
        relevant_knowledge = self.memory.retrieve_relevant_knowledge(f"Crawling strategy for {url}")
        
        try:
            system_message = """You are an AI assistant specialized in web crawling.
Based on the URL and any relevant knowledge, recommend an optimal crawling strategy.
Consider the website type, likely structure, and content."""
            
            user_message = f"""URL to crawl: {url}
            
Relevant past knowledge: {json.dumps(relevant_knowledge) if relevant_knowledge else "No relevant knowledge found."}

Determine the best crawling strategy, including:
1. Strategy type (BFS, DFS, or Best-First)
2. Maximum depth
3. Maximum pages
4. Content filter type and threshold
5. Important keywords to focus on

Explain your reasoning briefly."""
            
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ]
            
            response = await self.llm.ainvoke(messages)
            strategy = self._parse_strategy_from_response(response.content)
            return strategy
        except Exception as e:
            logger.error(f"Failed to analyze website: {e}")
            return CrawlStrategy()
    
    def _parse_strategy_from_response(self, response: str) -> CrawlStrategy:
        """Parse the crawl strategy from LLM response."""
        try:
            strategy = CrawlStrategy()
            
            if "bfs" in response.lower():
                strategy.strategy_type = "bfs"
            elif "dfs" in response.lower():
                strategy.strategy_type = "dfs"
            elif "best-first" in response.lower() or "best first" in response.lower():
                strategy.strategy_type = "best-first"
            
            depth_match = re.search(r"depth[:\s]+(\d+)", response.lower())
            if depth_match:
                strategy.max_depth = int(depth_match.group(1))
            
            pages_match = re.search(r"pages[:\s]+(\d+)", response.lower())
            if pages_match:
                strategy.max_pages = int(pages_match.group(1))
            
            if "pruning" in response.lower():
                strategy.content_filter_type = "Pruning"
            elif "bm25" in response.lower():
                strategy.content_filter_type = "BM25"
            
            threshold_match = re.search(r"threshold[:\s]+(0\.\d+)", response.lower())
            if threshold_match:
                strategy.threshold = float(threshold_match.group(1))
            
            keywords_match = re.search(r"keywords[:\s]+\[([^\]]+)\]", response.lower())
            if keywords_match:
                keywords_str = keywords_match.group(1)
                strategy.focus_keywords = [k.strip() for k in keywords_str.split(",")]
            
            return strategy
        except Exception as e:
            logger.error(f"Failed to parse strategy from response: {e}")
            return CrawlStrategy()
    
    async def enhance_content(self, raw_content: str, query: str = "") -> str:
        """Enhance the extracted content based on query and context."""
        if not self.is_llm_available or not raw_content:
            return raw_content
        
        try:
            system_message = """You are an AI assistant specialized in content enhancement.
Your task is to improve the scraped content by organizing it, removing noise, and highlighting the most relevant information."""
            
            user_message = f"""Here is the raw content scraped from a website:

{raw_content[:10000]}  # Limit content length to avoid token limits

{f"User query: {query}" if query else "No specific query provided."}

Enhance this content by:
1. Organizing it into clear sections
2. Removing redundant or noisy information
3. Highlighting the most relevant parts
4. Maintaining factual accuracy

Return only the enhanced content, formatted in Markdown."""
            
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ]
            
            response = await self.llm.ainvoke(messages)
            return response.content
        except Exception as e:
            logger.error(f"Failed to enhance content: {e}")
            return raw_content
    
    def analyze_link_relevance(self, page_url: str, links: List[Dict[str, str]], query: str = "") -> List[Dict[str, Any]]:
        """Analyze and score the relevance of links for intelligent crawling."""
        if not self.is_llm_available or not links:
            return links
        
        scored_links = []
        for link in links:
            url = link.get("url", "")
            text = link.get("text", "")
            
            score = 0.5
            
            if query and any(term.lower() in text.lower() for term in query.split()):
                score += 0.3
            
            if any(term in url.lower() for term in ["login", "signin", "register", "cart", "privacy", "terms"]):
                score -= 0.3
            
            url_depth = url.count("/")
            if url_depth > 2:
                score += 0.1
            
            score = max(0.0, min(1.0, score))
            
            scored_links.append({**link, "relevance_score": score})
        
        return sorted(scored_links, key=lambda x: x.get("relevance_score", 0), reverse=True)
    
    def store_crawl_results(self, url: str, content: str, metadata: Dict[str, Any]) -> None:
        """Store crawl results in memory for future reference."""
        self.memory.add_knowledge(url, content, metadata)
        
    async def answer_question(self, question: str, context: str) -> str:
        """Answer a question based on the crawled content."""
        if not self.is_llm_available:
            return "AI capabilities unavailable. Please check your OpenAI API key."
        
        try:
            system_message = """You are an AI assistant that answers questions based on provided context.
Be concise, accurate, and helpful in your responses. Only use the information from the provided context."""
            
            user_message = f"""Context information:
{context[:15000]}  # Limit context to avoid token limits

Question: {question}

Please provide a helpful, accurate, and concise answer based only on the information in the context."""
            
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ]
            
            response = await self.llm.ainvoke(messages)
            return response.content
        except Exception as e:
            logger.error(f"Failed to answer question: {e}")
            return f"An error occurred while answering your question: {str(e)}"

# Create an instance of the AI agent
ai_agent = AIAgent() 