import streamlit as st
from typing import List, Dict, Any, Optional
import logging
import random
from datetime import datetime
import pandas as pd
from utils.llm_engine import LLMEngine
from utils.search_engine import SearchEngine
from ui.components.file_uploader import render_document_selector
from ui.components.export_manager import render_quick_export_buttons

logger = logging.getLogger(__name__)

def render_flashcards_mode():
    """Render flashcards mode for creating and studying flashcards"""
    st.markdown("## 🧠 Flashcards & Notes")
    st.markdown('<div class="mode-description">Create intelligent flashcards from your documents and topics. Multiple question types, difficulty levels, and smart spaced repetition for effective learning.</div>', unsafe_allow_html=True)
    
    # Mode selection
    mode = st.radio(
        "Choose mode:",
        ["📝 Create Flashcards", "📚 Study Flashcards", "📊 Flashcard Analytics"],
        help="Select whether to create new flashcards or study existing ones"
    )
    
    if mode == "📝 Create Flashcards":
        render_flashcard_creation()
    elif mode == "📚 Study Flashcards":
        render_flashcard_study()
    else:
        render_flashcard_analytics()

def render_flashcard_creation():
    """Render flashcard creation interface"""
    st.markdown("### 📝 Create Flashcards")
    
    # Content source selection
    source_type = st.radio(
        "Content source:",
        ["📄 From uploaded documents", "✍️ Custom topics", "🔍 Web search topics"],
        help="Choose the source for flashcard content"
    )
    
    selected_content = []
    content_type = ""
    
    if source_type == "📄 From uploaded documents":
        if not st.session_state.documents:
            st.warning("📚 No documents available. Please upload documents on the Home page first.")
            return
        
        selected_indices = render_document_selector(st.session_state.documents, "flashcards")
        if selected_indices:
            selected_content = [st.session_state.documents[i] for i in selected_indices]
            content_type = "documents"
    
    elif source_type == "✍️ Custom topics":
        topics_input = st.text_area(
            "📝 Enter topics (comma-separated)",
            height=100,
            placeholder="machine learning, neural networks, data structures, algorithms",
            help="Enter topics you want to create flashcards for"
        )
        
        if topics_input:
            selected_content = [topic.strip() for topic in topics_input.split(',') if topic.strip()]
            content_type = "topics"
    
    else:  # Web search topics
        search_topics = st.text_area(
            "🔍 Enter search topics (comma-separated)",
            height=100,
            placeholder="artificial intelligence, machine learning concepts",
            help="Enter topics to search web for flashcard content"
        )
        
        if search_topics and st.button("🔍 Search Web Content"):
            with st.spinner("Searching web for content..."):
                search_engine = SearchEngine()
                topics = [topic.strip() for topic in search_topics.split(',') if topic.strip()]
                web_content = []
                
                for topic in topics:
                    results = search_engine.web_search(topic, sources=['wikipedia', 'duckduckgo'])
                    for result in results[:2]:  # Top 2 results per topic
                        web_content.append({
                            'topic': topic,
                            'content': result.get('content', ''),
                            'title': result.get('title', ''),
                            'source': result.get('source', '')
                        })
                
                if web_content:
                    selected_content = web_content
                    content_type = "web_search"
                    st.success(f"✅ Found content for {len(web_content)} results")
    
    if not selected_content:
        st.info("👆 Please select content source to create flashcards.")
        return
    
    # Flashcard configuration
    st.markdown("### ⚙️ Flashcard Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        max_cards = st.slider(
            "📊 Maximum cards",
            min_value=5,
            max_value=100,
            value=20,
            help="Maximum number of flashcards to generate"
        )
        
        difficulty = st.selectbox(
            "📈 Difficulty Level",
            ["Easy", "Intermediate", "Difficult", "Mixed"],
            help="Choose the difficulty level for questions"
        )
        
        include_definitions = st.checkbox(
            "📖 Include definitions",
            value=True,
            help="Include definition-based flashcards"
        )
    
    with col2:
        card_types = st.multiselect(
            "🃏 Flashcard Types",
            ["True/False", "Q&A", "One-word", "Fill-in-the-blank", "Application-based", "Definitions", "Formulas"],
            default=["Q&A", "Definitions", "Fill-in-the-blank"],
            help="Select types of flashcards to generate"
        )
        
        language_style = st.selectbox(
            "🌍 Language Style",
            ["Simple", "Academic", "Technical", "Conversational"],
            help="Choose the language style for flashcards"
        )
        
        include_hints = st.checkbox(
            "💡 Include hints",
            value=False,
            help="Add hints to difficult questions"
        )
    
    # Advanced options
    with st.expander("🔧 Advanced Options"):
        col1, col2 = st.columns(2)
        
        with col1:
            focus_areas = st.text_area(
                "🎯 Focus Areas (optional)",
                placeholder="specific concepts, formulas, or topics to emphasize",
                help="Specific areas to focus on when creating flashcards"
            )
            
            avoid_topics = st.text_area(
                "🚫 Avoid Topics (optional)",
                placeholder="topics or concepts to avoid",
                help="Topics to avoid when creating flashcards"
            )
        
        with col2:
            custom_instructions = st.text_area(
                "📝 Custom Instructions",
                placeholder="Any specific requirements for flashcard creation",
                help="Additional instructions for flashcard generation"
            )
            
            spaced_repetition = st.checkbox(
                "🔄 Enable spaced repetition",
                value=True,
                help="Create flashcards optimized for spaced repetition learning"
            )
    
    # Generate flashcards
    if st.button("🧠 Generate Flashcards", type="primary"):
        with st.spinner("Creating intelligent flashcards..."):
            flashcards = generate_flashcards(
                selected_content,
                content_type,
                max_cards,
                card_types,
                difficulty,
                language_style,
                include_definitions,
                include_hints,
                focus_areas,
                avoid_topics,
                custom_instructions,
                spaced_repetition
            )
        
        if flashcards:
            display_generated_flashcards(flashcards)
            
            # Save to session state
            if 'flashcards' not in st.session_state:
                st.session_state.flashcards = []
            
            # Add metadata to flashcards
            for card in flashcards:
                card['created_date'] = datetime.now().isoformat()
                card['study_count'] = 0
                card['correct_count'] = 0
                card['last_studied'] = None
                card['difficulty_rating'] = map_difficulty_to_rating(difficulty)
                card['spaced_repetition_due'] = datetime.now().isoformat()
            
            st.session_state.flashcards.extend(flashcards)
            
            # Export options
            st.markdown("### 📤 Export Flashcards")
            export_data = {
                'flashcards': flashcards,
                'creation_settings': {
                    'max_cards': max_cards,
                    'card_types': card_types,
                    'difficulty': difficulty,
                    'language_style': language_style
                }
            }
            render_quick_export_buttons(export_data, "flashcards")
            
            st.success(f"✅ Generated {len(flashcards)} flashcards! Use Study Flashcards mode to practice.")
        else:
            st.error("❌ Failed to generate flashcards. Please try again.")

def generate_flashcards(
    content: List,
    content_type: str,
    max_cards: int,
    card_types: List[str],
    difficulty: str,
    language_style: str,
    include_definitions: bool,
    include_hints: bool,
    focus_areas: str,
    avoid_topics: str,
    custom_instructions: str,
    spaced_repetition: bool
) -> List[Dict[str, Any]]:
    """Generate flashcards from content using AI"""
    
    try:
        llm_engine = LLMEngine()
        all_flashcards = []
        
        # Process content based on type
        if content_type == "documents":
            combined_content = []
            for doc in content:
                doc_content = doc.get('content', '')
                if isinstance(doc_content, str) and doc_content:
                    combined_content.append(f"Document: {doc.get('name', 'Unknown')}\n{doc_content[:2000]}")
                elif isinstance(doc_content, dict):
                    text_content = doc_content.get('text', doc_content.get('content', ''))
                    if text_content:
                        combined_content.append(f"Document: {doc.get('name', 'Unknown')}\n{str(text_content)[:2000]}")
            content_text = '\n\n'.join(combined_content)
            
        elif content_type == "topics":
            content_text = "Topics to create flashcards for:\n" + '\n'.join([f"- {topic}" for topic in content])
            
        elif content_type == "web_search":
            combined_content = []
            for item in content:
                combined_content.append(f"Topic: {item.get('topic', '')}\nTitle: {item.get('title', '')}\nContent: {item.get('content', '')}")
            content_text = '\n\n'.join(combined_content)
        
        else:
            content_text = str(content)
        
        # Generate flashcards for each type
        cards_per_type = max(1, max_cards // len(card_types)) if card_types else 5
        
        for card_type in card_types:
            type_flashcards = generate_flashcards_by_type(
                content_text,
                card_type,
                cards_per_type,
                difficulty,
                language_style,
                include_hints,
                focus_areas,
                avoid_topics,
                custom_instructions,
                llm_engine
            )
            
            if type_flashcards:
                all_flashcards.extend(type_flashcards)
        
        # Limit total cards
        if len(all_flashcards) > max_cards:
            all_flashcards = all_flashcards[:max_cards]
        
        return all_flashcards
    
    except Exception as e:
        logger.error(f"Flashcard generation failed: {e}")
        return []

def generate_flashcards_by_type(
    content: str,
    card_type: str,
    num_cards: int,
    difficulty: str,
    language_style: str,
    include_hints: bool,
    focus_areas: str,
    avoid_topics: str,
    custom_instructions: str,
    llm_engine
) -> List[Dict[str, Any]]:
    """Generate flashcards of a specific type"""
    
    try:
        # Build prompt based on card type
        prompt = f"""Create {num_cards} {card_type} flashcards from this content.

Content: {content[:3000]}...

Requirements:
- Difficulty: {difficulty}
- Language style: {language_style}
- Type: {card_type}
"""
        
        if focus_areas:
            prompt += f"- Focus on: {focus_areas}\n"
        
        if avoid_topics:
            prompt += f"- Avoid: {avoid_topics}\n"
        
        if custom_instructions:
            prompt += f"- Additional instructions: {custom_instructions}\n"
        
        # Add type-specific formatting instructions
        if card_type == "Q&A":
            prompt += """
Format each flashcard as:
QUESTION: [question text]
ANSWER: [answer text]
---
"""
        elif card_type == "Definitions":
            prompt += """
Format each flashcard as:
TERM: [term]
DEFINITION: [definition]
---
"""
        elif card_type == "True/False":
            prompt += """
Format each flashcard as:
STATEMENT: [true/false statement]
ANSWER: [True/False]
EXPLANATION: [brief explanation]
---
"""
        elif card_type == "Fill-in-the-blank":
            prompt += """
Format each flashcard as:
QUESTION: [sentence with _____ blanks]
ANSWER: [words that fill the blanks]
---
"""
        else:
            prompt += """
Format each flashcard as:
FRONT: [question or prompt]
BACK: [answer or response]
---
"""
        
        # Generate flashcards
        response = llm_engine.generate_response(prompt, content[:1000], "generate")
        
        # Parse response into flashcards
        flashcards = parse_flashcard_response(response, card_type)
        
        return flashcards[:num_cards]
    
    except Exception as e:
        logger.error(f"Failed to generate {card_type} flashcards: {e}")
        return []

def parse_flashcard_response(response: str, card_type: str) -> List[Dict[str, Any]]:
    """Parse LLM response into structured flashcards"""
    
    flashcards = []
    
    try:
        # Split by separator
        card_sections = response.split('---')
        
        for section in card_sections:
            section = section.strip()
            if not section:
                continue
            
            card = {
                'type': card_type,
                'front': '',
                'back': '',
                'hint': '',
                'explanation': ''
            }
            
            lines = section.split('\n')
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip().lower()
                    value = value.strip()
                    
                    if key in ['question', 'term', 'statement', 'front']:
                        card['front'] = value
                    elif key in ['answer', 'definition', 'back']:
                        card['back'] = value
                    elif key == 'hint':
                        card['hint'] = value
                    elif key == 'explanation':
                        card['explanation'] = value
            
            # Only add if we have both front and back
            if card['front'] and card['back']:
                flashcards.append(card)
    
    except Exception as e:
        logger.error(f"Failed to parse flashcard response: {e}")
        # Create a simple fallback flashcard
        flashcards.append({
            'type': card_type,
            'front': f"Question about {card_type}",
            'back': response[:200] + "..." if len(response) > 200 else response,
            'hint': '',
            'explanation': ''
        })
    
    return flashcards

def display_generated_flashcards(flashcards: List[Dict[str, Any]]):
    """Display generated flashcards in an interactive format"""
    
    st.markdown("## 🃏 Generated Flashcards")
    st.markdown(f"**Total cards created:** {len(flashcards)}")
    
    # Group by type
    by_type = {}
    for card in flashcards:
        card_type = card.get('type', 'Unknown')
        if card_type not in by_type:
            by_type[card_type] = []
        by_type[card_type].append(card)
    
    # Display by type
    for card_type, type_cards in by_type.items():
        with st.expander(f"📚 {card_type} Cards ({len(type_cards)} cards)", expanded=True):
            for i, card in enumerate(type_cards):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.markdown(f"**Card {i+1}:**")
                    st.markdown(f"**Front:** {card.get('front', 'No question')}")
                    
                    # Show/hide answer
                    if st.button(f"Show Answer", key=f"show_{card_type}_{i}"):
                        st.markdown(f"**Back:** {card.get('back', 'No answer')}")
                        
                        if card.get('hint'):
                            st.markdown(f"**Hint:** {card.get('hint')}")
                        
                        if card.get('explanation'):
                            st.markdown(f"**Explanation:** {card.get('explanation')}")
                
                with col2:
                    # Quick study buttons
                    if st.button("✅", help="Mark as easy", key=f"easy_{card_type}_{i}"):
                        st.success("Marked as easy!")
                    
                    if st.button("❓", help="Mark as difficult", key=f"hard_{card_type}_{i}"):
                        st.warning("Marked as difficult!")
                
                st.markdown("---")

def render_flashcard_study():
    """Render flashcard study interface"""
    st.markdown("### 📚 Study Flashcards")
    
    if 'flashcards' not in st.session_state or not st.session_state.flashcards:
        st.info("📝 No flashcards available. Create some first using 'Create Flashcards' mode.")
        return
    
    flashcards = st.session_state.flashcards
    
    # Study mode selection
    study_mode = st.radio(
        "Study mode:",
        ["🔀 Random Review", "📅 Spaced Repetition", "🎯 Focus on Difficult"],
        help="Choose how to study your flashcards"
    )
    
    # Display flashcards for study
    if study_mode == "🔀 Random Review":
        random.shuffle(flashcards)
        study_cards = flashcards[:10]  # Study 10 cards at a time
    elif study_mode == "📅 Spaced Repetition":
        study_cards = get_due_flashcards(flashcards)
    else:  # Focus on Difficult
        study_cards = [card for card in flashcards if card.get('difficulty_rating', 2) >= 3][:10]
    
    if study_cards:
        st.markdown(f"**Studying {len(study_cards)} cards**")
        for i, card in enumerate(study_cards):
            with st.container():
                st.markdown(f"**Card {i+1}/{len(study_cards)}**")
                st.markdown(f"**Question:** {card.get('front', 'No question')}")
                
                if st.button("Show Answer", key=f"study_show_{i}"):
                    st.markdown(f"**Answer:** {card.get('back', 'No answer')}")
                    
                    # Study feedback
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        if st.button("😰 Hard", key=f"study_hard_{i}"):
                            update_card_difficulty(card, "hard")
                    with col2:
                        if st.button("😐 Medium", key=f"study_medium_{i}"):
                            update_card_difficulty(card, "medium")
                    with col3:
                        if st.button("😊 Easy", key=f"study_easy_{i}"):
                            update_card_difficulty(card, "easy")
                
                st.markdown("---")
    else:
        st.success("🎉 No cards due for review right now!")

def render_flashcard_analytics():
    """Render flashcard analytics interface"""
    st.markdown("### 📊 Flashcard Analytics")
    
    if 'flashcards' not in st.session_state or not st.session_state.flashcards:
        st.info("📝 No flashcards data available.")
        return
    
    flashcards = st.session_state.flashcards
    
    # Basic stats
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Cards", len(flashcards))
    
    with col2:
        studied_cards = [c for c in flashcards if c.get('study_count', 0) > 0]
        st.metric("Cards Studied", len(studied_cards))
    
    with col3:
        avg_difficulty = sum(c.get('difficulty_rating', 2) for c in flashcards) / len(flashcards) if flashcards else 0
        st.metric("Avg Difficulty", f"{avg_difficulty:.1f}")
    
    with col4:
        total_studies = sum(c.get('study_count', 0) for c in flashcards)
        st.metric("Total Studies", total_studies)
    
    # Charts and analysis would go here
    st.markdown("📈 Detailed analytics coming soon!")

def get_due_flashcards(flashcards):
    """Get flashcards that are due for review"""
    # Simple implementation - return all for now
    return flashcards[:10]

def update_card_difficulty(card, difficulty):
    """Update card difficulty based on user feedback"""
    difficulty_map = {"easy": 1, "medium": 2, "hard": 3}
    card['difficulty_rating'] = difficulty_map.get(difficulty, 2)
    card['study_count'] = card.get('study_count', 0) + 1
    card['last_studied'] = datetime.now().isoformat()

def map_difficulty_to_rating(difficulty: str) -> int:
    """Map difficulty string to numeric rating"""
    mapping = {
        'Easy': 1,
        'Intermediate': 2,
        'Difficult': 3,
        'Mixed': 2
    }
    return mapping.get(difficulty, 2)