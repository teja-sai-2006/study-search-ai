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

Additionally provide:
1. Current trends and developments
2. Challenges and limitations
3. Future implications
4. Related fields and connections
"""
        
        summary = llm_engine.generate_response(prompt, task_type="analyze")
        return summary
    
    except Exception as e:
        logger.error(f"Summary generation failed for {topic}: {e}")
        return f"Failed to generate summary for '{topic}': {str(e)}"

def extract_key_points(topic: str, results: Dict[str, Any]) -> List[str]:
    """Extract key points from topic results"""
    
    try:
        from utils.llm_engine import LLMEngine
        llm_engine = LLMEngine()
        
        # Combine content
        all_content = []
        for result in results.get('local', []) + results.get('web', []):
            content = result.get('content', '')
            if content:
                all_content.append(content)
        
        if not all_content:
            return []
        
        combined = '\n'.join(all_content[:3])  # Limit content
        
        prompt = f"""Extract 5-7 key points about '{topic}' from this information:

{combined}

Format as a bullet list of the most important facts, concepts, or insights."""
        
        response = llm_engine.generate_response(prompt, task_type="analyze")
        
        # Parse bullet points
        lines = response.split('\n')
        key_points = []
        for line in lines:
            line = line.strip()
            if line.startswith('-') or line.startswith('•') or line.startswith('*'):
                key_points.append(line[1:].strip())
            elif line and not line.startswith('#'):
                key_points.append(line)
        
        return key_points[:7]  # Limit to 7 points
    
    except Exception as e:
        logger.error(f"Key point extraction failed: {e}")
        return []

def find_related_topics(topic: str, results: Dict[str, Any]) -> List[str]:
    """Find topics related to the search topic"""
    
    try:
        from utils.llm_engine import LLMEngine
        llm_engine = LLMEngine()
        
        prompt = f"""Based on the topic '{topic}', suggest 5 related topics that someone studying this subject might want to explore.

Consider:
- Subtopics and specialized areas
- Related fields and disciplines
- Prerequisite knowledge
- Advanced applications

Format as a simple list of topics."""
        
        response = llm_engine.generate_response(prompt, task_type="generate")
        
        # Parse topics
        lines = response.split('\n')
        related_topics = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#'):
                # Clean up formatting
                if line.startswith('-') or line.startswith('•') or line.startswith('*'):
                    line = line[1:].strip()
                if line:
                    related_topics.append(line)
        
        return related_topics[:5]  # Limit to 5 topics
    
    except Exception as e:
        logger.error(f"Related topic generation failed: {e}")
        return []

def display_search_results(results: Dict[str, Dict], summary_type: str, export_format: str):
    """Display comprehensive search results"""
    
    st.markdown("## 🔍 Search Results")
    
    # Results overview
    total_topics = len(results)
    topics_with_results = sum(1 for r in results.values() if r.get('local') or r.get('web'))
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Topics Searched", total_topics)
    with col2:
        st.metric("Topics with Results", topics_with_results)
    with col3:
        total_sources = sum(len(r.get('local', [])) + len(r.get('web', [])) for r in results.values())
        st.metric("Total Sources Found", total_sources)
    
    # Individual topic results
    for topic, topic_results in results.items():
        with st.expander(f"🎯 {topic.title()}", expanded=True):
            
            # Summary
            if topic_results.get('summary'):
                st.markdown("### 📋 Summary")
                st.markdown(topic_results['summary'])
            
            # Key points
            if topic_results.get('key_points'):
                st.markdown("### ⭐ Key Points")
                for point in topic_results['key_points']:
                    st.markdown(f"• {point}")
            
            # Sources
            col1, col2 = st.columns(2)
            
            with col1:
                if topic_results.get('local'):
                    st.markdown("#### 📄 Local Documents")
                    for result in topic_results['local']:
                        source_name = result.get('metadata', {}).get('name', 'Unknown')
                        score = result.get('score', 0)
                        st.markdown(f"• **{source_name}** (Score: {score:.2f})")
            
            with col2:
                if topic_results.get('web'):
                    st.markdown("#### 🌐 Web Sources")
                    for result in topic_results['web']:
                        title = result.get('title', 'Unknown')
                        source = result.get('source', 'web')
                        url = result.get('url', '')
                        if url:
                            st.markdown(f"• **[{title}]({url})** ({source})")
                        else:
                            st.markdown(f"• **{title}** ({source})")
            
            # Related topics
            if topic_results.get('related_topics'):
                st.markdown("#### 🔗 Related Topics")
                related_tags = " • ".join([f"`{rt}`" for rt in topic_results['related_topics']])
                st.markdown(related_tags)
    
    # Export options
    if export_format != "None":
        st.markdown("### 📤 Export Results")
        
        export_data = prepare_export_data(results, export_format)
        render_quick_export_buttons(export_data, f"topic_search_{export_format.lower()}")

def prepare_export_data(results: Dict[str, Dict], export_format: str) -> Dict[str, Any]:
    """Prepare search results for export"""
    
    if export_format == "Summary":
        return {
            'search_summaries': {topic: data.get('summary', '') for topic, data in results.items()},
            'key_points': {topic: data.get('key_points', []) for topic, data in results.items()},
            'related_topics': {topic: data.get('related_topics', []) for topic, data in results.items()}
        }
    
    elif export_format == "Flashcards":
        flashcards = []
        for topic, data in results.items():
            # Create topic overview card
            if data.get('summary'):
                flashcards.append({
                    'question': f"What is {topic}?",
                    'answer': data['summary'][:500] + "..." if len(data['summary']) > 500 else data['summary']
                })
            
            # Create key point cards
            for point in data.get('key_points', [])[:3]:  # Limit to 3 points per topic
                flashcards.append({
                    'question': f"Key point about {topic}:",
                    'answer': point
                })
        
        return {'flashcards': flashcards}
    
    elif export_format == "Both":
        summary_data = prepare_export_data(results, "Summary")
        flashcard_data = prepare_export_data(results, "Flashcards")
        return {**summary_data, **flashcard_data}
    
    return results
