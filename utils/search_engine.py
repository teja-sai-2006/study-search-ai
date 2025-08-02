import logging
from typing import List, Dict, Any, Optional
import streamlit as st

logger = logging.getLogger(__name__)

class SearchEngine:
    """Enhanced search engine with vector search and web fallback"""
    
    def __init__(self):
        self.vector_db = None
        self._initialize_vector_db()
    
    def _initialize_vector_db(self):
        """Initialize vector database"""
        try:
            from utils.vector_db import VectorDatabase
            self.vector_db = VectorDatabase()
        except Exception as e:
            logger.error(f"Failed to initialize vector database: {e}")
    
    def search_documents(self, query: str, documents: List[Dict], top_k: int = 5) -> List[Dict[str, Any]]:
        """Search through uploaded documents using vector similarity"""
        if not documents:
            return []
        
        try:
            # Try vector search first if available
            if self.vector_db and hasattr(self.vector_db, 'search'):
                try:
                    doc_texts = []
                    doc_metadata = []
                    
                    for doc in documents:
                        content = doc.get('content', '')
                        if isinstance(content, str) and content.strip():
                            doc_texts.append(content)
                            doc_metadata.append({
                                'name': doc.get('name', 'Unknown Document'),
                                'type': doc.get('type', 'document'),
                                'source_doc': doc
                            })
                        elif isinstance(content, dict):
                            # Handle dict content
                            text_content = content.get('text', content.get('content', str(content)))
                            if text_content:
                                doc_texts.append(str(text_content))
                                doc_metadata.append({
                                    'name': doc.get('name', 'Unknown Document'),
                                    'type': doc.get('type', 'document'),
                                    'source_doc': doc
                                })
                    
                    if doc_texts:
                        self.vector_db.add_documents(doc_texts, doc_metadata)
                        results = self.vector_db.search(query, top_k=top_k)
                        
                        # Ensure results have proper format
                        formatted_results = []
                        for result in results:
                            formatted_results.append({
                                'content': result.get('content', '')[:1000],
                                'score': result.get('score', 0.5),
                                'title': result.get('metadata', {}).get('name', 'Document'),
                                'metadata': result.get('metadata', {})
                            })
                        
                        if formatted_results:
                            return formatted_results
                
                except Exception as vector_error:
                    logger.error(f"Vector search error: {vector_error}")
            
            # Always fallback to simple text search
            return self._simple_text_search(query, documents, top_k)
        
        except Exception as e:
            logger.error(f"Document search failed: {e}")
            return self._simple_text_search(query, documents, top_k)
    
    def _simple_text_search(self, query: str, documents: List[Dict], top_k: int = 5) -> List[Dict[str, Any]]:
        """Simple text-based search fallback"""
        results = []
        query_words = query.lower().split()
        
        for doc in documents:
            content = doc.get('content', '')
            
            # Handle different content types
            if isinstance(content, dict):
                content_text = str(content.get('text', content.get('content', '')))
            else:
                content_text = str(content)
            
            if not content_text.strip():
                continue
                
            content_lower = content_text.lower()
            score = 0
            
            # Score based on word matches
            for word in query_words:
                if len(word) > 2:  # Skip very short words
                    score += content_lower.count(word)
            
            # Bonus for phrase matches
            if query.lower() in content_lower:
                score += 10
            
            if score > 0:
                results.append({
                    'content': content_text[:1000] + ("..." if len(content_text) > 1000 else ""),
                    'score': score,
                    'title': doc.get('name', 'Document'),
                    'metadata': {
                        'name': doc.get('name', 'Unknown Document'),
                        'type': doc.get('type', 'document'),
                        'source_doc': doc
                    }
                })
        
        # Sort by score and return top_k
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_k]
    
    def web_search(self, query: str, sources: List[str] = None) -> List[Dict[str, Any]]:
        """Search the web using multiple sources"""
        if sources is None:
            sources = ['duckduckgo', 'wikipedia', 'arxiv']
        
        all_results = []
        
        for source in sources:
            try:
                if source == 'duckduckgo':
                    results = self._search_duckduckgo(query)
                elif source == 'wikipedia':
                    results = self._search_wikipedia(query)
                elif source == 'arxiv':
                    results = self._search_arxiv(query)
                else:
                    continue
                
                # Add source information
                for result in results:
                    result['source'] = source
                
                all_results.extend(results)
            
            except Exception as e:
                logger.error(f"Error searching {source}: {e}")
        
        return all_results
    
    def _search_duckduckgo(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Search using DuckDuckGo"""
        try:
            from duckduckgo_search import DDGS
            
            results = []
            with DDGS() as ddgs:
                search_results = ddgs.text(query, max_results=max_results)
                
                for result in search_results:
                    results.append({
                        'title': result.get('title', ''),
                        'content': result.get('body', ''),
                        'url': result.get('href', ''),
                        'score': 1.0
                    })
            
            return results
        
        except ImportError:
            logger.error("duckduckgo-search not installed")
            return []
        except Exception as e:
            logger.error(f"DuckDuckGo search error: {e}")
            return []
    
    def _search_wikipedia(self, query: str, max_results: int = 3) -> List[Dict[str, Any]]:
        """Search Wikipedia"""
        try:
            import wikipedia
            
            results = []
            search_results = wikipedia.search(query, results=max_results)
            
            for title in search_results:
                try:
                    page = wikipedia.page(title)
                    results.append({
                        'title': page.title,
                        'content': page.summary[:1000] + "...",
                        'url': page.url,
                        'score': 1.0
                    })
                except wikipedia.exceptions.DisambiguationError as e:
                    # Try the first option
                    try:
                        page = wikipedia.page(e.options[0])
                        results.append({
                            'title': page.title,
                            'content': page.summary[:1000] + "...",
                            'url': page.url,
                            'score': 0.8
                        })
                    except:
                        continue
                except:
                    continue
            
            return results
        
        except ImportError:
            logger.error("wikipedia package not installed")
            return []
        except Exception as e:
            logger.error(f"Wikipedia search error: {e}")
            return []
    
    def _search_arxiv(self, query: str, max_results: int = 3) -> List[Dict[str, Any]]:
        """Search arXiv for academic papers"""
        try:
            import arxiv
            
            results = []
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance
            )
            
            for result in search.results():
                results.append({
                    'title': result.title,
                    'content': result.summary[:1000] + "...",
                    'url': result.entry_id,
                    'score': 1.0,
                    'authors': [author.name for author in result.authors],
                    'published': result.published.strftime('%Y-%m-%d')
                })
            
            return results
        
        except ImportError:
            logger.error("arxiv package not installed")
            return []
        except Exception as e:
            logger.error(f"arXiv search error: {e}")
            return []
    
    def search_topics(self, topics: List[str], documents: List[Dict] = None) -> Dict[str, List[Dict]]:
        """Search for multiple topics across documents and web"""
        results = {}
        
        for topic in topics:
            topic_results = []
            
            # Search documents if available
            if documents:
                doc_results = self.search_documents(topic, documents, top_k=3)
                topic_results.extend(doc_results)
            
            # Search web
            web_results = self.web_search(topic)
            topic_results.extend(web_results)
            
            results[topic] = topic_results
        
        return results
    
    def get_relevant_context(self, query: str, documents: List[Dict], max_length: int = 2000) -> str:
        """Get relevant context for a query from documents"""
        search_results = self.search_documents(query, documents, top_k=3)
        
        context = ""
        for result in search_results:
            content = result.get('content', '')
            if len(context) + len(content) <= max_length:
                context += content + "\n\n"
            else:
                remaining = max_length - len(context)
                context += content[:remaining] + "..."
                break
        
        return context.strip()
