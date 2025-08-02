import streamlit as st
from typing import List, Dict, Any
import logging
from utils.search_engine import SearchEngine
from ui.components.export_manager import render_quick_export_buttons

logger = logging.getLogger(__name__)

def render_topic_search_mode():
    """Render topic search mode for multi-topic research"""
    st.markdown("## 🎯 Topic Search")
    st.markdown('<div class="mode-description">Search for multiple topics simultaneously across your documents and the web. Get comprehensive information from local documents, Wikipedia, arXiv, and DuckDuckGo.</div>', unsafe_allow_html=True)
    
    # Topic input
    st.markdown("### 🔍 Topic Input")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        topics_input = st.text_area(
            "📝 Enter topics to search (comma-separated)",
            height=100,
            placeholder="artificial intelligence, machine learning, neural networks, deep learning, computer vision",
            help="Enter up to 20 topics separated by commas"
        )
    
    with col2:
        st.markdown("#### 💡 Tips")
        st.info("""
        • Use specific terms
        • Separate with commas
        • Max 20 topics
        • Mix broad and specific topics
        """)
    
    # Parse topics
    topics = []
    if topics_input:
        topics = [topic.strip() for topic in topics_input.split(',') if topic.strip()]
        topics = topics[:20]  # Limit to 20 topics
        
        if topics:
            st.success(f"✅ {len(topics)} topics ready to search")
            
            # Show topics as tags
            topic_tags = " • ".join([f"**{topic}**" for topic in topics])
            st.markdown(f"**Topics:** {topic_tags}")
    
    # Search configuration
    st.markdown("### ⚙️ Search Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        search_sources = st.multiselect(
            "🌐 Search Sources",
            ["Local Documents", "DuckDuckGo", "Wikipedia", "arXiv"],
            default=["Local Documents", "DuckDuckGo", "Wikipedia"],
            help="Select which sources to search"
        )
        
        summary_type = st.selectbox(
            "📊 Summary Type",
            ["Basic", "Intermediate", "Advanced", "Full Web Dump"],
            help="Choose the detail level for results"
        )
    
    with col2:
        max_results_per_source = st.slider(
            "📈 Max results per source",
            min_value=1,
            max_value=10,
            value=3,
            help="Limit results to manage response size"
        )
        
        merge_similar = st.checkbox(
            "🔄 Merge similar results",
            value=True,
            help="Combine similar information from different sources"
        )
    
    # Advanced options
    with st.expander("🔧 Advanced Options"):
        col1, col2 = st.columns(2)
        
        with col1:
            include_related = st.checkbox(
                "🔗 Include related topics",
                value=False,
                help="Search for topics related to your input"
            )
            
            depth_analysis = st.checkbox(
                "🧠 Depth analysis",
                value=False,
                help="Provide deeper analysis of each topic"
            )
        
        with col2:
            export_format = st.selectbox(
                "📤 Export preparation",
                ["None", "Summary", "Flashcards", "Both"],
                help="Prepare results for export"
            )
            
            language_filter = st.selectbox(
                "🌍 Language",
                ["All", "English", "Spanish", "French", "German"],
                help="Filter results by language"
            )
    
    # Search execution
    if st.button("🔍 Search Topics", type="primary") and topics:
        search_engine = SearchEngine()
        
        with st.spinner(f"Searching {len(topics)} topics across {len(search_sources)} sources..."):
            all_results = perform_topic_search(
                topics,
                search_sources,
                search_engine,
                max_results_per_source,
                summary_type,
                merge_similar,
                include_related,
                depth_analysis
            )
        
        if all_results:
            display_search_results(all_results, summary_type, export_format)
        else:
            st.error("❌ No results found. Please try different topics or sources.")

def perform_topic_search(
    topics: List[str],
    sources: List[str],
    search_engine: SearchEngine,
    max_results: int,
    summary_type: str,
    merge_similar: bool,
    include_related: bool,
    depth_analysis: bool
) -> Dict[str, Dict[str, Any]]:
    """Perform comprehensive topic search"""
    
    all_results = {}
    
    try:
        for topic in topics:
            topic_results = {
                'local': [],
                'web': [],
                'summary': '',
                'key_points': [],
                'related_topics': []
            }
            
            # Search local documents
            if "Local Documents" in sources and st.session_state.documents:
                try:
                    local_results = search_engine.search_documents(
                        topic, 
                        st.session_state.documents, 
                        top_k=max_results
                    )
                    topic_results['local'] = local_results
                except Exception as e:
                    logger.error(f"Local search failed for {topic}: {e}")
            
            # Search web sources
            web_sources = []
            if "DuckDuckGo" in sources:
                web_sources.append("duckduckgo")
            if "Wikipedia" in sources:
                web_sources.append("wikipedia")
            if "arXiv" in sources:
                web_sources.append("arxiv")
            
            if web_sources:
                try:
                    web_results = search_engine.web_search(topic, web_sources)
                    topic_results['web'] = web_results[:max_results]
                except Exception as e:
                    logger.error(f"Web search failed for {topic}: {e}")
            
            # Generate summary and analysis
            if topic_results['local'] or topic_results['web']:
                topic_results['summary'] = generate_topic_summary(
                    topic,
                    topic_results,
                    summary_type,
                    depth_analysis
                )
                
                topic_results['key_points'] = extract_key_points(
                    topic,
                    topic_results
                )
                
                if include_related:
                    topic_results['related_topics'] = find_related_topics(
                        topic,
                        topic_results
                    )
            
            all_results[topic] = topic_results
        
        return all_results
    
    except Exception as e:
        logger.error(f"Topic search failed: {e}")
        return {}

def generate_topic_summary(
    topic: str,
    results: Dict[str, Any],
    summary_type: str,
    depth_analysis: bool
) -> str:
    """Generate comprehensive summary for a topic"""
    
    from utils.llm_engine import LLMEngine
    llm_engine = LLMEngine()
    
    try:
        # Prepare content from all sources
        content_parts = []
        
        # Add local document results
        for result in results.get('local', []):
            content = result.get('content', '')
            source = result.get('metadata', {}).get('name', 'Local Document')
            if content:
                content_parts.append(f"[{source}] {content}")
        
        # Add web results
        for result in results.get('web', []):
            content = result.get('content', '')
            source = result.get('title', result.get('source', 'Web Source'))
            if content:
                content_parts.append(f"[{source}] {content}")
        
        if not content_parts:
            return f"No detailed information found for '{topic}'"
        
        combined_content = '\n\n'.join(content_parts[:5])  # Limit to avoid token limits
        
        # Build prompt based on summary type
        if summary_type == "Basic":
            prompt = f"""Provide a basic overview of '{topic}' based on the following information:

{combined_content}

Format as a clear, concise summary suitable for general understanding."""
        
        elif summary_type == "Intermediate":
            prompt = f"""Provide an intermediate-level explanation of '{topic}' based on the following information:

{combined_content}

Include key concepts, applications, and important details. Format for someone with some background knowledge."""
        
        elif summary_type == "Advanced":
            prompt = f"""Provide an advanced analysis of '{topic}' based on the following information:

{combined_content}

Include technical details, current research, implications, and expert-level insights."""
        
        elif summary_type == "Full Web Dump":
            return combined_content  # Return raw content
        
        if depth_analysis:
            prompt += """
            
Additionally, provide deeper analysis including:
- Historical context and evolution
- Current trends and developments  
- Future implications and potential
- Cross-connections with other fields"""
        
        # Generate summary using LLM
        response = llm_engine.generate_response(prompt, "", "summarize")
        return response
    
    except Exception as e:
        logger.error(f"Summary generation failed for {topic}: {e}")
        return f"Error generating summary for '{topic}': {str(e)}"

def extract_key_points(topic: str, results: Dict[str, Any]) -> List[str]:
    """Extract key points from search results"""
    key_points = []
    
    # Extract from local results
    for result in results.get('local', []):
        content = result.get('content', '')
        if content:
            # Simple extraction - look for sentences with topic
            sentences = content.split('.')
            for sentence in sentences:
                if topic.lower() in sentence.lower() and len(sentence.strip()) > 20:
                    key_points.append(sentence.strip())
    
    # Extract from web results
    for result in results.get('web', []):
        title = result.get('title', '')
        content = result.get('content', '')
        if title and topic.lower() in title.lower():
            key_points.append(f"• {title}")
        if content:
            sentences = content.split('.')
            for sentence in sentences[:2]:  # Limit to first 2 sentences per source
                if len(sentence.strip()) > 20:
                    key_points.append(f"• {sentence.strip()}")
    
    return key_points[:10]  # Limit to 10 key points

def find_related_topics(topic: str, results: Dict[str, Any]) -> List[str]:
    """Find topics related to the search topic"""
    related = set()
    
    # Simple keyword extraction from results
    all_content = ""
    for result in results.get('local', []) + results.get('web', []):
        content = result.get('content', '')
        title = result.get('title', '')
        all_content += f" {content} {title}"
    
    # Basic related topic extraction (simplified)
    words = all_content.lower().split()
    topic_words = topic.lower().split()
    
    # Look for capitalized words that might be related concepts
    import re
    potential_topics = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', all_content)
    
    for potential in potential_topics[:20]:  # Limit processing
        if potential.lower() not in topic.lower() and len(potential) > 3:
            related.add(potential)
    
    return list(related)[:8]  # Limit to 8 related topics

def display_search_results(
    all_results: Dict[str, Dict[str, Any]], 
    summary_type: str, 
    export_format: str
):
    """Display comprehensive search results"""
    
    st.markdown("## 📊 Search Results")
    
    if not all_results:
        st.error("No results found for your topics.")
        return
    
    # Summary statistics
    total_topics = len(all_results)
    topics_with_results = sum(1 for r in all_results.values() if r.get('local') or r.get('web'))
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Topics Searched", total_topics)
    with col2:
        st.metric("Topics with Results", topics_with_results)
    with col3:
        total_sources = sum(len(r.get('local', [])) + len(r.get('web', [])) for r in all_results.values())
        st.metric("Total Sources Found", total_sources)
    
    # Display results for each topic
    for topic, results in all_results.items():
        with st.expander(f"🎯 {topic}", expanded=True):
            
            # Check if we have any results
            has_local = bool(results.get('local'))
            has_web = bool(results.get('web'))
            
            if not has_local and not has_web:
                st.warning(f"No results found for '{topic}'")
                continue
            
            # Tabs for different views
            tab1, tab2, tab3, tab4 = st.tabs(["📝 Summary", "📚 Sources", "🔑 Key Points", "🔗 Related"])
            
            with tab1:
                summary = results.get('summary', '')
                if summary:
                    st.markdown(summary)
                else:
                    st.info("No summary available")
            
            with tab2:
                # Local sources
                if has_local:
                    st.markdown("### 📁 Local Documents")
                    for i, result in enumerate(results['local']):
                        st.markdown(f"**Source {i+1}:** {result.get('metadata', {}).get('name', 'Unknown')}")
                        st.markdown(f"**Relevance:** {result.get('score', 0):.2f}")
                        with st.expander("View content"):
                            st.text(result.get('content', '')[:1000] + "...")
                
                # Web sources
                if has_web:
                    st.markdown("### 🌐 Web Sources")
                    for i, result in enumerate(results['web']):
                        st.markdown(f"**{result.get('title', 'Unknown Title')}**")
                        st.markdown(f"Source: {result.get('source', 'Unknown')}")
                        if result.get('url'):
                            st.markdown(f"🔗 [Link]({result['url']})")
                        with st.expander("View content"):
                            st.text(result.get('content', '')[:1000] + "...")
            
            with tab3:
                key_points = results.get('key_points', [])
                if key_points:
                    for point in key_points:
                        st.markdown(f"• {point}")
                else:
                    st.info("No key points extracted")
            
            with tab4:
                related_topics = results.get('related_topics', [])
                if related_topics:
                    st.markdown("**Related topics found:**")
                    for related in related_topics:
                        st.markdown(f"• {related}")
                else:
                    st.info("No related topics found")
    
    # Export options
    if export_format != "None":
        st.markdown("---")
        try:
            render_quick_export_buttons(all_results, export_format)
        except Exception as e:
            st.error(f"Export functionality not available: {str(e)}")
            st.info("You can copy and paste the results above for manual export.")
