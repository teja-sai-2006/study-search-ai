import logging
from typing import Dict, List, Any, Optional
import requests
from bs4 import BeautifulSoup
import trafilatura

logger = logging.getLogger(__name__)

class WebScraper:
    """Enhanced web scraping for clean content extraction"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
    
    def extract_clean_content(self, url: str) -> Dict[str, Any]:
        """Extract clean text content from website using trafilatura"""
        try:
            # Download content
            downloaded = trafilatura.fetch_url(url)
            if not downloaded:
                return {"error": "Failed to fetch URL", "content": ""}
            
            # Extract main content
            text = trafilatura.extract(downloaded)
            if not text:
                # Fallback to basic extraction
                text = self._fallback_extraction(url)
            
            # Extract metadata
            metadata = trafilatura.extract_metadata(downloaded)
            
            return {
                "content": text or "",
                "title": metadata.title if metadata else "",
                "author": metadata.author if metadata else "",
                "date": metadata.date if metadata else "",
                "url": url,
                "error": None
            }
        
        except Exception as e:
            logger.error(f"Content extraction failed for {url}: {e}")
            return {
                "error": str(e),
                "content": "",
                "url": url
            }
    
    def _fallback_extraction(self, url: str) -> str:
        """Fallback content extraction using BeautifulSoup"""
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get text content
            text = soup.get_text()
            
            # Clean up text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            return text
        
        except Exception as e:
            logger.error(f"Fallback extraction failed: {e}")
            return ""
    
    def scrape_multiple_urls(self, urls: List[str]) -> List[Dict[str, Any]]:
        """Scrape multiple URLs and return clean content"""
        results = []
        
        for url in urls:
            try:
                content_data = self.extract_clean_content(url)
                results.append(content_data)
            except Exception as e:
                logger.error(f"Failed to scrape {url}: {e}")
                results.append({
                    "error": str(e),
                    "content": "",
                    "url": url
                })
        
        return results
    
    def search_and_scrape(self, query: str, search_engine: str = "duckduckgo", max_results: int = 5) -> List[Dict[str, Any]]:
        """Search and scrape results from search engine"""
        try:
            if search_engine == "duckduckgo":
                return self._search_and_scrape_duckduckgo(query, max_results)
            else:
                logger.error(f"Unsupported search engine: {search_engine}")
                return []
        
        except Exception as e:
            logger.error(f"Search and scrape failed: {e}")
            return []
    
    def _search_and_scrape_duckduckgo(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Search DuckDuckGo and scrape the results"""
        try:
            from duckduckgo_search import DDGS
            
            results = []
            with DDGS() as ddgs:
                search_results = ddgs.text(query, max_results=max_results)
                
                for result in search_results:
                    url = result.get('href', '')
                    if url:
                        # Scrape the actual content
                        content_data = self.extract_clean_content(url)
                        content_data.update({
                            'search_title': result.get('title', ''),
                            'search_snippet': result.get('body', ''),
                            'search_url': url
                        })
                        results.append(content_data)
            
            return results
        
        except ImportError:
            logger.error("duckduckgo-search not installed")
            return []
        except Exception as e:
            logger.error(f"DuckDuckGo search and scrape failed: {e}")
            return []
    
    def extract_specific_content(self, url: str, selectors: Dict[str, str]) -> Dict[str, Any]:
        """Extract specific content using CSS selectors"""
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            extracted = {}
            
            for key, selector in selectors.items():
                elements = soup.select(selector)
                if elements:
                    if len(elements) == 1:
                        extracted[key] = elements[0].get_text(strip=True)
                    else:
                        extracted[key] = [elem.get_text(strip=True) for elem in elements]
                else:
                    extracted[key] = None
            
            return {
                "extracted_data": extracted,
                "url": url,
                "error": None
            }
        
        except Exception as e:
            logger.error(f"Specific content extraction failed: {e}")
            return {
                "extracted_data": {},
                "url": url,
                "error": str(e)
            }
    
    def check_url_accessibility(self, url: str) -> Dict[str, Any]:
        """Check if URL is accessible and get basic info"""
        try:
            response = self.session.head(url, timeout=5)
            
            return {
                "accessible": response.status_code == 200,
                "status_code": response.status_code,
                "content_type": response.headers.get('content-type', ''),
                "content_length": response.headers.get('content-length', ''),
                "error": None
            }
        
        except Exception as e:
            return {
                "accessible": False,
                "status_code": None,
                "error": str(e)
            }
    
    def extract_links(self, url: str, filter_domain: bool = True) -> List[str]:
        """Extract all links from a webpage"""
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            links = []
            
            for link in soup.find_all('a', href=True):
                href = link['href']
                
                # Convert relative URLs to absolute
                if href.startswith('/'):
                    from urllib.parse import urljoin
                    href = urljoin(url, href)
                
                if href.startswith('http'):
                    if filter_domain:
                        from urllib.parse import urlparse
                        if urlparse(href).netloc == urlparse(url).netloc:
                            links.append(href)
                    else:
                        links.append(href)
            
            return list(set(links))  # Remove duplicates
        
        except Exception as e:
            logger.error(f"Link extraction failed: {e}")
            return []
