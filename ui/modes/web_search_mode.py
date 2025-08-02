import streamlit as st
from typing import List, Dict, Any
import logging
from utils.search_engine import SearchEngine
from utils.web_scraper import WebScraper
from ui.components.export_manager import render_quick_export_buttons

logger = logging.getLogger(__name__)

def render_web_search_mode():
    """Render web search mode for comprehensive web research"""
    st.markdown("## 🌐 Web Search")
    st.markdown('<div class="mode-description">Search the web for comprehensive information using multiple sources. Get clean, ad-free results from DuckDuckGo, Wikipedia, and arXiv with intelligent filtering and export options.</div>', unsafe_allow_html=True)
    
    # Search input
    st.markdown("### 🔍 Search Query")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        search_queries = st.text_area(
            "📝 Enter search topics (comma-separated)",
            height=100,
            placeholder="artificial intelligence, machine learning, quantum computing",
            help="Enter multiple topics separated by commas for comprehensive research"
        )
    
    with col2:
        st.markdown("#### 💡 Search Tips")
        st.info("""
        • Use specific terms
        • Mix broad and narrow topics
        • Include technical terms
        • Separate with commas
        """)
    
    # Parse queries
    queries = []
    if search_queries:
        queries = [query.strip() for query in search_queries.split(',') if query.strip()]
        
        if queries:
            st.success(f"✅ {len(queries)} search queries ready")
            
            # Show queries as tags
            query_tags = " • ".join([f"**{query}**" for query in queries])
            st.markdown(f"**Queries:** {query_tags}")
    
    # Search configuration
    st.markdown("### ⚙️ Search Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        search_sources = st.multiselect(
            "🌐 Search Sources",
            ["DuckDuckGo", "Wikipedia", "arXiv"],
            default=["DuckDuckGo", "Wikipedia"],
            help="Select which sources to search"
        )
        
        results_per_query = st.slider(
            "📊 Results per query",
            min_value=1,
            max_value=15,
            value=5,
            help="Number of results to fetch per query"
        )
    
    with col2:
        result_style = st.selectbox(
            "📋 Result Style",
            ["Basic", "Intermediate", "Advanced", "Original Result"],
            help="Choose how to process and present results"
        )
        
        content_filter = st.selectbox(
            "🔍 Content Filter",
            ["All Content", "Educational Only", "Recent (Last Year)", "Academic Sources"],
            help="Filter results by content type"
        )
    
    # Advanced options
    with st.expander("🔧 Advanced Options"):
        col1, col2 = st.columns(2)
        
        with col1:
            include_full_content = st.checkbox(
                "📄 Include full content",
                value=False,
                help="Scrape and include full article content (slower but more comprehensive)"
            )
            
            remove_duplicates = st.checkbox(
                "🔄 Remove duplicates",
                value=True,
                help="Filter out similar results across sources"
            )
            
            language_preference = st.selectbox(
                "🌍 Language",
                ["Any", "English", "Spanish", "French", "German"],
                help="Preferred language for results"
            )
        
        with col2:
            max_content_length = st.slider(
                "📏 Max content length",
                min_value=500,
                max_value=5000,
                value=2000,
                help="Maximum characters per result"
            )
            
            include_related = st.checkbox(
                "🔗 Include related searches",
                value=False,
                help="Add related search suggestions"
            )
            
            export_preparation = st.selectbox(
                "📤 Export as",
                ["Raw Results", "Summary", "Flashcards", "Research Notes"],
                help="How to prepare results for export"
            )
    
    # Search execution
    if st.button("🔍 Start Web Search", type="primary") and queries:
        search_engine = SearchEngine()
        web_scraper = WebScraper()
        
        with st.spinner(f"Searching {len(queries)} queries across {len(search_sources)} sources..."):
            all_results = perform_comprehensive_web_search(
                queries,
                search_sources,
                search_engine,
                web_scraper,
                results_per_query,
                result_style,
                content_filter,
                include_full_content,
                remove_duplicates,
                max_content_length,
                include_related
            )
        
        if all_results:
            display_web_search_results(all_results, result_style, export_preparation)
        else:
            st.error("❌ No results found. Please try different search terms or sources.")

def perform_comprehensive_web_search(
    queries: List[str],
    sources: List[str],
    search_engine: SearchEngine,
    web_scraper: WebScraper,
    results_per_query: int,
    result_style: str,
    content_filter: str,
    include_full_content: bool,
    remove_duplicates: bool,
    max_content_length: int,
    include_related: bool
) -> Dict[str, Dict[str, Any]]:
    """Perform comprehensive web search across multiple sources"""
    
    all_results = {}
    
    try:
        for query in queries:
            query_results = {
                'raw_results': [],
                'processed_results': [],
                'summary': '',
                'related_searches': [],
                'source_breakdown': {}
            }
            
            # Search each source
            for source in sources:
                try:
                    source_results = []
                    
                    if source == "DuckDuckGo":
                        source_results = search_engine._search_duckduckgo(query, results_per_query)
                    elif source == "Wikipedia":
                        source_results = search_engine._search_wikipedia(query, min(results_per_query, 3))
                    elif source == "arXiv":
                        source_results = search_engine._search_arxiv(query, min(results_per_query, 3))
                    
                    # Add source information
                    for result in source_results:
                        result['search_source'] = source
                        result['query'] = query
                    
                    query_results['raw_results'].extend(source_results)
                    query_results['source_breakdown'][source] = len(source_results)
                
                except Exception as e:
                    logger.error(f"Search failed for {source} with query '{query}': {e}")
                    query_results['source_breakdown'][source] = 0
            
            # Process results based on style
            if query_results['raw_results']:
                query_results['processed_results'] = process_search_results(
                    query_results['raw_results'],
                    result_style,
                    content_filter,
                    include_full_content,
                    web_scraper,
                    max_content_length
                )
                
                # Generate summary
                if result_style != "Original Result":
                    query_results['summary'] = generate_search_summary(
                        query,
                        query_results['processed_results'],
                        result_style
                    )
                
                # Find related searches
                if include_related:
                    query_results['related_searches'] = find_related_searches(
                        query,
                        query_results['processed_results']
                    )
            
            # Remove duplicates if requested
            if remove_duplicates:
                query_results['processed_results'] = remove_duplicate_results(
                    query_results['processed_results']
                )
            
            all_results[query] = query_results
        
        return all_results
    
    except Exception as e:
        logger.error(f"Web search failed: {e}")
        return {}

def process_search_results(
    raw_results: List[Dict],
    style: str,
    content_filter: str,
    include_full_content: bool,
    web_scraper: WebScraper,
    max_length: int
) -> List[Dict[str, Any]]:
    """Process raw search results based on style and filters"""
    
    processed = []
    
    for result in raw_results:
        try:
            processed_result = result.copy()
            
            # Apply content filtering
            if not passes_content_filter(result, content_filter):
                continue
            
            # Get full content if requested
            if include_full_content and result.get('url'):
                try:
                    full_content = web_scraper.extract_clean_content(result['url'])
                    if full_content.get('content'):
                        processed_result['full_content'] = full_content['content'][:max_length]
                except Exception as e:
                    logger.warning(f"Failed to extract full content from {result.get('url')}: {e}")
            
            # Truncate content to max length
            if 'content' in processed_result:
                processed_result['content'] = processed_result['content'][:max_length]
            
            # Style-specific processing
            if style == "Basic":
                processed_result = simplify_result_basic(processed_result)
            elif style == "Intermediate":
                processed_result = enhance_result_intermediate(processed_result)
            elif style == "Advanced":
                processed_result = analyze_result_advanced(processed_result)
            
            processed.append(processed_result)
        
        except Exception as e:
            logger.warning(f"Failed to process result: {e}")
    
    return processed

def passes_content_filter(result: Dict, filter_type: str) -> bool:
    """Check if result passes content filter"""
    
    if filter_type == "All Content":
        return True
    
    title = result.get('title', '').lower()
    content = result.get('content', '').lower()
    url = result.get('url', '').lower()
    
    if filter_type == "Educational Only":
        educational_keywords = [
            'education', 'learning', 'tutorial', 'course', 'university',
            'academic', 'research', 'study', 'lesson', 'guide'
        ]
        return any(keyword in title or keyword in content for keyword in educational_keywords)
    
    elif filter_type == "Recent (Last Year)":
        # Check if result has recent date or is from known current sources
        recent_indicators = ['2024', '2023', 'recent', 'latest', 'current']
        return any(indicator in title or indicator in content for indicator in recent_indicators)
    
    elif filter_type == "Academic Sources":
        academic_domains = [
            'arxiv.org', 'scholar.google', 'researchgate', 'ieee.org',
            'acm.org', 'nature.com', 'science.org', 'pubmed', 'doi.org'
        ]
        return any(domain in url for domain in academic_domains)
    
    return True

def simplify_result_basic(result: Dict) -> Dict:
    """Simplify result for basic style"""
    return {
        'title': result.get('title', 'No title'),
        'summary': result.get('content', '')[:200] + "...",
        'url': result.get('url', ''),
        'source': result.get('search_source', 'Unknown')
    }

def enhance_result_intermediate(result: Dict) -> Dict:
    """Enhance result for intermediate style"""
    enhanced = result.copy()
    
    # Add relevance score
    enhanced['relevance'] = calculate_relevance_score(result)
    
    # Add key points extraction
    content = result.get('content', '')
    if content:
        enhanced['key_points'] = extract_key_points_from_content(content)
    
    return enhanced

def analyze_result_advanced(result: Dict) -> Dict:
    """Analyze result for advanced style"""
    analyzed = result.copy()
    
    # Add comprehensive analysis
    try:
        from utils.llm_engine import LLMEngine
        llm_engine = LLMEngine()
        
        content = result.get('content', '')
        title = result.get('title', '')
        
        if content:
            analysis_prompt = f"""Analyze this search result for advanced research purposes:

Title: {title}
Content: {content[:1000]}

Provide:
1. Credibility assessment
2. Key insights and findings
3. Relevance to research
4. Potential applications
5. Related concepts to explore

Format as structured analysis."""
            
            analysis = llm_engine.generate_response(analysis_prompt, task_type="analyze")
            analyzed['advanced_analysis'] = analysis
    
    except Exception as e:
        logger.error(f"Advanced analysis failed: {e}")
        analyzed['advanced_analysis'] = "Analysis unavailable"
    
    return analyzed

def calculate_relevance_score(result: Dict) -> float:
    """Calculate relevance score for result"""
    score = 0.5  # Base score
    
    title = result.get('title', '').lower()
    content = result.get('content', '').lower()
    query = result.get('query', '').lower()
    
    if query:
        query_words = query.split()
        
        # Title relevance (higher weight)
        title_matches = sum(1 for word in query_words if word in title)
        score += (title_matches / len(query_words)) * 0.3
        
        # Content relevance
        content_matches = sum(1 for word in query_words if word in content)
        score += (content_matches / len(query_words)) * 0.2
    
    # Source credibility
    source = result.get('search_source', '').lower()
    if source == 'wikipedia':
        score += 0.1
    elif source == 'arxiv':
        score += 0.15
    
    return min(score, 1.0)

def extract_key_points_from_content(content: str) -> List[str]:
    """Extract key points from content"""
    try:
        from utils.llm_engine import LLMEngine
        llm_engine = LLMEngine()
        
        prompt = f"""Extract 3-5 key points from this content:

{content[:1500]}

Format as bullet points with the most important information."""
        
        response = llm_engine.generate_response(prompt, task_type="analyze")
        
        # Parse bullet points
        lines = response.split('\n')
        key_points = []
        for line in lines:
            line = line.strip()
            if line.startswith('-') or line.startswith('•') or line.startswith('*'):
                key_points.append(line[1:].strip())
        
        return key_points[:5]
    
    except Exception as e:
        logger.error(f"Key point extraction failed: {e}")
        return []

def generate_search_summary(query: str, results: List[Dict], style: str) -> str:
    """Generate summary of search results"""
    
    try:
        from utils.llm_engine import LLMEngine
        llm_engine = LLMEngine()
        
        # Combine content from top results
        combined_content = []
        for result in results[:5]:  # Top 5 results
            title = result.get('title', '')
            content = result.get('content', result.get('summary', ''))
            if content:
                combined_content.append(f"{title}: {content}")
        
        if not combined_content:
            return f"No content available to summarize for '{query}'"
        
        content_text = '\n\n'.join(combined_content)
        
        if style == "Basic":
            prompt = f"""Provide a basic summary of web search results for '{query}':

{content_text}

Create a clear, simple overview suitable for general understanding."""
        
        elif style == "Intermediate":
            prompt = f"""Provide an intermediate-level summary of web search results for '{query}':

{content_text}

Include key concepts, findings, and practical applications."""
        
        elif style == "Advanced":
            prompt = f"""Provide an advanced analysis of web search results for '{query}':

{content_text}

Include detailed analysis, trends, implications, and research insights."""
        
        summary = llm_engine.generate_response(prompt, task_type="summarize")
        return summary
    
    except Exception as e:
        logger.error(f"Summary generation failed: {e}")
        return f"Summary generation failed for '{query}'"

def find_related_searches(query: str, results: List[Dict]) -> List[str]:
    """Find related search suggestions"""
    
    try:
        from utils.llm_engine import LLMEngine
        llm_engine = LLMEngine()
        
        # Use content from results to suggest related searches
        content_sample = ' '.join([
            result.get('title', '') + ' ' + result.get('content', '')[:200]
            for result in results[:3]
        ])
        
        prompt = f"""Based on the search query '{query}' and these results:

{content_sample}

Suggest 5 related search terms that would provide complementary information.
Format as a simple list."""
        
        response = llm_engine.generate_response(prompt, task_type="generate")
        
        # Parse suggestions
        lines = response.split('\n')
        suggestions = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#'):
                if line.startswith('-') or line.startswith('•'):
                    line = line[1:].strip()
                suggestions.append(line)
        
        return suggestions[:5]
    
    except Exception as e:
        logger.error(f"Related search generation failed: {e}")
        return []

def remove_duplicate_results(results: List[Dict]) -> List[Dict]:
    """Remove duplicate results based on title and URL similarity"""
    
    unique_results = []
    seen_titles = set()
    seen_urls = set()
    
    for result in results:
        title = result.get('title', '').lower().strip()
        url = result.get('url', '').strip()
        
        # Check for exact duplicates
        if title in seen_titles or url in seen_urls:
            continue
        
        # Check for similar titles (simple similarity)
        is_similar = False
        for seen_title in seen_titles:
            if title and seen_title and (
                title in seen_title or seen_title in title
            ) and len(title) > 10:
                is_similar = True
                break
        
        if not is_similar:
            unique_results.append(result)
            if title:
                seen_titles.add(title)
            if url:
                seen_urls.add(url)
    
    return unique_results

def display_web_search_results(results: Dict[str, Dict], style: str, export_preparation: str):
    """Display comprehensive web search results"""
    
    st.markdown("## 🌐 Web Search Results")
    
    # Overall summary
    total_queries = len(results)
    total_results = sum(len(data.get('processed_results', [])) for data in results.values())
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Queries Searched", total_queries)
    with col2:
        st.metric("Total Results", total_results)
    with col3:
        avg_per_query = total_results / total_queries if total_queries > 0 else 0
        st.metric("Avg per Query", f"{avg_per_query:.1f}")
    
    # Results for each query
    for query, query_data in results.items():
        with st.expander(f"🔍 {query.title()}", expanded=True):
            
            # Query summary
            if query_data.get('summary') and style != "Original Result":
                st.markdown("### 📋 Summary")
                st.markdown(query_data['summary'])
            
            # Source breakdown
            if query_data.get('source_breakdown'):
                st.markdown("### 📊 Source Breakdown")
                cols = st.columns(len(query_data['source_breakdown']))
                for i, (source, count) in enumerate(query_data['source_breakdown'].items()):
                    with cols[i]:
                        st.metric(source, count)
            
            # Individual results
            st.markdown("### 🔗 Results")
            
            for i, result in enumerate(query_data.get('processed_results', [])):
                with st.expander(f"📄 {result.get('title', f'Result {i+1}')}"):
                    
                    # Basic info
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        if result.get('url'):
                            st.markdown(f"🔗 **[View Source]({result['url']})**")
                    with col2:
                        st.markdown(f"**Source:** {result.get('source', 'Unknown')}")
                    
                    # Content based on style
                    if style == "Basic":
                        st.markdown(result.get('summary', result.get('content', '')))
                    
                    elif style == "Intermediate":
                        st.markdown(result.get('content', ''))
                        if result.get('key_points'):
                            st.markdown("**Key Points:**")
                            for point in result['key_points']:
                                st.markdown(f"• {point}")
                        if result.get('relevance'):
                            st.markdown(f"**Relevance Score:** {result['relevance']:.2f}")
                    
                    elif style == "Advanced":
                        st.markdown(result.get('content', ''))
                        if result.get('advanced_analysis'):
                            st.markdown("**Advanced Analysis:**")
                            st.markdown(result['advanced_analysis'])
                        if result.get('full_content'):
                            with st.expander("📖 Full Content"):
                                st.markdown(result['full_content'])
                    
                    else:  # Original Result
                        st.markdown(result.get('content', ''))
            
            # Related searches
            if query_data.get('related_searches'):
                st.markdown("### 🔗 Related Searches")
                related_tags = " • ".join([f"`{rs}`" for rs in query_data['related_searches']])
                st.markdown(related_tags)
    
    # Export options
    if export_preparation != "Raw Results":
        st.markdown("### 📤 Export Results")
        export_data = prepare_web_search_export(results, export_preparation)
        render_quick_export_buttons(export_data, f"web_search_{export_preparation.lower().replace(' ', '_')}")

def prepare_web_search_export(results: Dict[str, Dict], export_type: str) -> Dict[str, Any]:
    """Prepare web search results for export"""
    
    if export_type == "Summary":
        return {
            'search_summaries': {query: data.get('summary', '') for query, data in results.items()},
            'source_breakdown': {query: data.get('source_breakdown', {}) for query, data in results.items()},
            'related_searches': {query: data.get('related_searches', []) for query, data in results.items()}
        }
    
    elif export_type == "Flashcards":
        flashcards = []
        for query, data in results.items():
            # Summary card
            if data.get('summary'):
                flashcards.append({
                    'question': f"What did web search reveal about {query}?",
                    'answer': data['summary']
                })
            
            # Key findings cards
            for result in data.get('processed_results', [])[:3]:
                if result.get('key_points'):
                    for point in result['key_points'][:2]:
                        flashcards.append({
                            'question': f"Key finding about {query}:",
                            'answer': point
                        })
        
        return {'flashcards': flashcards}
    
    elif export_type == "Research Notes":
        research_notes = {}
        for query, data in results.items():
            notes = []
            
            # Add summary
            if data.get('summary'):
                notes.append(f"## Summary\n{data['summary']}")
            
            # Add key sources
            notes.append("## Key Sources")
            for result in data.get('processed_results', [])[:5]:
                title = result.get('title', 'Unknown')
                url = result.get('url', '')
                content = result.get('content', '')[:300]
                notes.append(f"### {title}\n**URL:** {url}\n**Summary:** {content}...")
            
            # Add related searches
            if data.get('related_searches'):
                notes.append("## Related Research Topics")
                for related in data['related_searches']:
                    notes.append(f"- {related}")
            
            research_notes[query] = '\n\n'.join(notes)
        
        return {'research_notes': research_notes}
    
    return results
