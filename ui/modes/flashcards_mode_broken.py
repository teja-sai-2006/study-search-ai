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
                if doc_content:
                    combined_content.append(f"Document: {doc.get('name', 'Unknown')}\n{doc_content[:2000]}")
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
        cards_per_type = max(1, max_cards // len(card_types))
        
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
    
    # Build prompt based on card type
    if card_type == "Q&A":
        prompt = f"""Create {num_cards} question and answer flashcards from this content.

Content: {content[:3000]}...

Requirements:
- Difficulty: {difficulty}
- Language style: {language_style}
- Create clear, specific questions with detailed answers
{f'- Focus on: {focus_areas}' if focus_areas else ''}
{f'- Avoid: {avoid_topics}' if avoid_topics else ''}
{f'- Include hints: {include_hints}' if include_hints else ''}
{f'- Additional instructions: {custom_instructions}' if custom_instructions else ''}

Format each flashcard as:
QUESTION: [question text]
ANSWER: [answer text]
{('HINT: [hint text]' if include_hints else '')}
---
"""
    
    elif card_type == "Definitions":
        prompt = f"""Create {num_cards} definition flashcards from key terms in this content.

Content: {content[:3000]}...

Requirements:
- Difficulty: {difficulty}
- Language style: {language_style}
- Extract important terms and create clear definitions
{f'- Focus on: {focus_areas}' if focus_areas else ''}
{f'- Avoid: {avoid_topics}' if avoid_topics else ''}

Format each flashcard as:
TERM: [term]
DEFINITION: [definition]
---
"""
    
    elif card_type == "True/False":
        prompt = f"""Create {num_cards} true/false flashcards from this content.

Content: {content[:3000]}...

Requirements:
- Difficulty: {difficulty}
- Language style: {language_style}
- Create statements that can be clearly true or false
{f'- Focus on: {focus_areas}' if focus_areas else ''}
{f'- Avoid: {avoid_topics}' if avoid_topics else ''}

Format each flashcard as:
STATEMENT: [true/false statement]
ANSWER: [True/False]
EXPLANATION: [brief explanation]
---
"""
    
    elif card_type == "Fill-in-the-blank":
        prompt = f"""Create {num_cards} fill-in-the-blank flashcards from this content.

Content: {content[:3000]}...

Requirements:
- Difficulty: {difficulty}
- Language style: {language_style}
- Create sentences with one or two key blanks
{f'- Focus on: {focus_areas}' if focus_areas else ''}
{f'- Avoid: {avoid_topics}' if avoid_topics else ''}

Format each flashcard as:
QUESTION: [sentence with _____ blanks]
ANSWER: [words that fill the blanks]
---
"""
    
    else:  # Default for other types
        prompt = f"""Create {num_cards} {card_type} flashcards from this content.

Content: {content[:3000]}...

Requirements:
- Difficulty: {difficulty}
- Language style: {language_style}
- Type: {card_type}
{f'- Focus on: {focus_areas}' if focus_areas else ''}
{f'- Avoid: {avoid_topics}' if avoid_topics else ''}

Format each flashcard appropriately for the type.
---
"""
    
    # Generate flashcards
    response = llm_engine.generate_response(prompt, content[:1000], "generate")
    
    # Parse response into flashcards
    flashcards = parse_flashcard_response(response, card_type)
    
    return flashcards

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
            current_field = None
            current_content = []
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                if line.startswith('QUESTION:') or line.startswith('TERM:') or line.startswith('STATEMENT:'):
                    if current_field and current_content:
                        card[current_field] = ' '.join(current_content)
                    current_field = 'front'
                    current_content = [line.split(':', 1)[1].strip()]
                    
                elif line.startswith('ANSWER:') or line.startswith('DEFINITION:'):
                    if current_field and current_content:
                        card[current_field] = ' '.join(current_content)
                    current_field = 'back'
                    current_content = [line.split(':', 1)[1].strip()]
                    
                elif line.startswith('HINT:'):
                    if current_field and current_content:
                        card[current_field] = ' '.join(current_content)
                    current_field = 'hint'
                    current_content = [line.split(':', 1)[1].strip()]
                    
                elif line.startswith('EXPLANATION:'):
                    if current_field and current_content:
                        card[current_field] = ' '.join(current_content)
                    current_field = 'explanation'
                    current_content = [line.split(':', 1)[1].strip()]
                    
                else:
                    if current_field:
                        current_content.append(line)
            
            # Add final field
            if current_field and current_content:
                card[current_field] = ' '.join(current_content)
            
            # Only add if we have both front and back
            if card['front'] and card['back']:
                flashcards.append(card)
    
    except Exception as e:
        logger.error(f"Failed to parse flashcard response: {e}")
        # Create a simple fallback flashcard
        flashcards.append({
            'type': card_type,
            'front': f"Generated {card_type} question from content",
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

def map_difficulty_to_rating(difficulty: str) -> int:
    """Map difficulty string to numeric rating"""
    mapping = {
        'Easy': 1,
        'Intermediate': 2,
        'Difficult': 3,
        'Mixed': 2
    }
    return mapping.get(difficulty, 2)

# Render study flashcards mode
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
        import random
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
        
        if focus_areas:
            prompt += f"- Focus particularly on: {focus_areas}\n"
        
        if avoid_topics:
            prompt += f"- Avoid these topics: {avoid_topics}\n"
        
        if custom_instructions:
            prompt += f"- Additional instructions: {custom_instructions}\n"
        
        # Type-specific instructions
        if card_type == "True/False":
            prompt += """
Format each flashcard as:
Question: [True/False statement]
Answer: [True/False with brief explanation]
Type: True/False
"""
        elif card_type == "Q&A":
            prompt += """
Format each flashcard as:
Question: [Open-ended question]
Answer: [Comprehensive answer]
Type: Q&A
"""
        elif card_type == "One-word":
            prompt += """
Format each flashcard as:
Question: [Question requiring one-word answer]
Answer: [Single word or short phrase]
Type: One-word
"""
        elif card_type == "Fill-in-the-blank":
            prompt += """
Format each flashcard as:
Question: [Statement with blanks to fill]
Answer: [Words that fill the blanks]
Type: Fill-in-the-blank
"""
        elif card_type == "Application-based":
            prompt += """
Format each flashcard as:
Question: [Practical application scenario or problem]
Answer: [Solution or approach]
Type: Application-based
"""
        elif card_type == "Definitions":
            prompt += """
Format each flashcard as:
Question: [Term to define]
Answer: [Clear definition]
Type: Definitions
"""
        elif card_type == "Formulas":
            prompt += """
Format each flashcard as:
Question: [When to use this formula or what it calculates]
Answer: [Mathematical formula with explanation]
Type: Formulas
"""
        
        if include_hints:
            prompt += "- Include a helpful hint for each question\n"
        
        prompt += """
Generate exactly the requested number of cards. Ensure questions are clear and answers are accurate.
Separate each flashcard with "---"
"""
        
        response = llm_engine.generate_response(prompt, task_type="generate")
        
        # Parse flashcards from response
        flashcards = parse_flashcards_from_response(response, card_type)
        
        return flashcards[:num_cards]  # Ensure we don't exceed requested number
    
    except Exception as e:
        logger.error(f"Flashcard generation failed for type {card_type}: {e}")
        return []

def parse_flashcards_from_response(response: str, card_type: str) -> List[Dict[str, Any]]:
    """Parse flashcards from LLM response"""
    
    flashcards = []
    
    # Split by separator
    card_blocks = response.split("---")
    
    for block in card_blocks:
        block = block.strip()
        if not block:
            continue
        
        try:
            # Parse question and answer
            lines = block.split('\n')
            question = ""
            answer = ""
            hint = ""
            
            for line in lines:
                line = line.strip()
                if line.startswith("Question:"):
                    question = line.replace("Question:", "").strip()
                elif line.startswith("Answer:"):
                    answer = line.replace("Answer:", "").strip()
                elif line.startswith("Hint:"):
                    hint = line.replace("Hint:", "").strip()
            
            if question and answer:
                flashcard = {
                    'question': question,
                    'answer': answer,
                    'type': card_type,
                    'hint': hint if hint else None,
                    'tags': extract_tags_from_content(question + " " + answer),
                    'id': f"card_{len(flashcards)}_{datetime.now().timestamp()}"
                }
                flashcards.append(flashcard)
        
        except Exception as e:
            logger.warning(f"Failed to parse flashcard block: {e}")
    
    return flashcards

def extract_tags_from_content(content: str) -> List[str]:
    """Extract relevant tags from flashcard content"""
    
    # Simple keyword extraction
    import re
    
    # Remove common words
    common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'this', 'that', 'these', 'those'}
    
    # Extract words
    words = re.findall(r'\b[a-zA-Z]+\b', content.lower())
    
    # Filter and get important words
    tags = []
    for word in words:
        if len(word) > 3 and word not in common_words:
            tags.append(word)
    
    # Return unique tags, limited to 5
    return list(dict.fromkeys(tags))[:5]

def map_difficulty_to_rating(difficulty: str) -> int:
    """Map difficulty string to numeric rating"""
    mapping = {
        "Easy": 1,
        "Intermediate": 2,
        "Difficult": 3,
        "Mixed": 2
    }
    return mapping.get(difficulty, 2)

def add_spaced_repetition_metadata(flashcards: List[Dict], difficulty: str):
    """Add spaced repetition metadata to flashcards"""
    
    base_interval = 1  # Start with 1 day
    
    for card in flashcards:
        card['spaced_repetition'] = {
            'interval': base_interval,
            'ease_factor': 2.5,
            'repetitions': 0,
            'next_review': datetime.now().isoformat()
        }

def display_generated_flashcards(flashcards: List[Dict[str, Any]]):
    """Display generated flashcards"""
    
    st.markdown("### 🃏 Generated Flashcards")
    st.success(f"✅ Generated {len(flashcards)} flashcards")
    
    # Flashcard type breakdown
    type_counts = {}
    for card in flashcards:
        card_type = card.get('type', 'Unknown')
        type_counts[card_type] = type_counts.get(card_type, 0) + 1
    
    st.markdown("#### 📊 Breakdown by Type")
    cols = st.columns(min(4, len(type_counts)))
    for i, (card_type, count) in enumerate(type_counts.items()):
        with cols[i % len(cols)]:
            st.metric(card_type, count)
    
    # Display sample flashcards
    st.markdown("#### 👀 Preview")
    
    for i, card in enumerate(flashcards[:5]):  # Show first 5
        with st.expander(f"🃏 Card {i+1}: {card.get('type', 'Unknown')}"):
            st.markdown(f"**Question:** {card.get('question', '')}")
            st.markdown(f"**Answer:** {card.get('answer', '')}")
            
            if card.get('hint'):
                st.markdown(f"**Hint:** {card['hint']}")
            
            if card.get('tags'):
                tags = ' • '.join([f"`{tag}`" for tag in card['tags']])
                st.markdown(f"**Tags:** {tags}")
    
    if len(flashcards) > 5:
        st.info(f"... and {len(flashcards) - 5} more flashcards")

def render_flashcard_study():
    """Render flashcard study interface"""
    st.markdown("### 📚 Study Flashcards")
    
    if 'flashcards' not in st.session_state or not st.session_state.flashcards:
        st.info("No flashcards available. Create some flashcards first!")
        return
    
    # Study options
    col1, col2 = st.columns(2)
    
    with col1:
        study_mode = st.selectbox(
            "📖 Study Mode",
            ["All Cards", "By Type", "By Difficulty", "Due for Review", "Random Selection"],
            help="Choose how to select cards for study"
        )
        
        num_cards = st.slider(
            "📊 Number of cards to study",
            min_value=5,
            max_value=min(50, len(st.session_state.flashcards)),
            value=min(20, len(st.session_state.flashcards)),
            help="How many cards to study in this session"
        )
    
    with col2:
        show_hints = st.checkbox(
            "💡 Show hints",
            value=True,
            help="Display hints when available"
        )
        
        shuffle_cards = st.checkbox(
            "🔀 Shuffle cards",
            value=True,
            help="Randomize card order"
        )
    
    # Filter cards based on study mode
    study_cards = filter_cards_for_study(
        st.session_state.flashcards,
        study_mode,
        num_cards,
        shuffle_cards
    )
    
    if not study_cards:
        st.warning("No cards match the selected criteria.")
        return
    
    # Initialize study session state
    if 'study_session' not in st.session_state:
        st.session_state.study_session = {
            'cards': study_cards,
            'current_index': 0,
            'show_answer': False,
            'session_stats': {
                'correct': 0,
                'incorrect': 0,
                'total_studied': 0
            }
        }
    
    # Study session controls
    render_study_session_interface(show_hints)

def filter_cards_for_study(
    all_cards: List[Dict],
    study_mode: str,
    num_cards: int,
    shuffle: bool
) -> List[Dict]:
    """Filter and select cards for study session"""
    
    filtered_cards = all_cards.copy()
    
    if study_mode == "By Type":
        # Allow user to select type
        available_types = list(set(card.get('type', 'Unknown') for card in all_cards))
        selected_type = st.selectbox("Select card type:", available_types)
        filtered_cards = [card for card in all_cards if card.get('type') == selected_type]
    
    elif study_mode == "By Difficulty":
        difficulty_filter = st.selectbox("Select difficulty:", ["Easy", "Intermediate", "Difficult"])
        target_rating = map_difficulty_to_rating(difficulty_filter)
        filtered_cards = [card for card in all_cards if card.get('difficulty_rating', 2) == target_rating]
    
    elif study_mode == "Due for Review":
        # Cards due for spaced repetition review
        current_time = datetime.now().isoformat()
        filtered_cards = [
            card for card in all_cards 
            if card.get('spaced_repetition_due', current_time) <= current_time
        ]
    
    elif study_mode == "Random Selection":
        if shuffle:
            random.shuffle(filtered_cards)
    
    # Shuffle if requested
    if shuffle and study_mode != "Random Selection":
        random.shuffle(filtered_cards)
    
    # Limit to requested number
    return filtered_cards[:num_cards]

def render_study_session_interface(show_hints: bool):
    """Render the study session interface"""
    
    session = st.session_state.study_session
    cards = session['cards']
    current_index = session['current_index']
    
    if current_index >= len(cards):
        # Session complete
        display_study_session_results()
        return
    
    current_card = cards[current_index]
    
    # Progress indicator
    progress = (current_index + 1) / len(cards)
    st.progress(progress, text=f"Card {current_index + 1} of {len(cards)}")
    
    # Current stats
    stats = session['session_stats']
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Correct", stats['correct'])
    with col2:
        st.metric("Incorrect", stats['incorrect'])
    with col3:
        accuracy = stats['correct'] / max(1, stats['correct'] + stats['incorrect']) * 100
        st.metric("Accuracy", f"{accuracy:.1f}%")
    
    # Card display
    st.markdown("### 🃏 Current Card")
    
    # Card type and tags
    card_type = current_card.get('type', 'Unknown')
    tags = current_card.get('tags', [])
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown(f"**Type:** {card_type}")
    with col2:
        if tags:
            tag_text = ' • '.join([f"`{tag}`" for tag in tags[:3]])
            st.markdown(f"**Tags:** {tag_text}")
    
    # Question
    st.markdown("#### ❓ Question")
    st.markdown(f"**{current_card.get('question', '')}**")
    
    # Hint (if available and enabled)
    if show_hints and current_card.get('hint'):
        with st.expander("💡 Hint"):
            st.markdown(current_card['hint'])
    
    # Show answer button
    if not session['show_answer']:
        if st.button("👁️ Show Answer", type="primary"):
            session['show_answer'] = True
            st.rerun()
    
    else:
        # Show answer
        st.markdown("#### ✅ Answer")
        st.markdown(f"**{current_card.get('answer', '')}**")
        
        # Self-assessment
        st.markdown("#### 📊 How did you do?")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("❌ Incorrect", key="incorrect"):
                record_answer(current_card, False)
                next_card()
        
        with col2:
            if st.button("⚠️ Partially", key="partial"):
                record_answer(current_card, True, partial=True)
                next_card()
        
        with col3:
            if st.button("✅ Correct", key="correct"):
                record_answer(current_card, True)
                next_card()

def record_answer(card: Dict, correct: bool, partial: bool = False):
    """Record the user's answer and update statistics"""
    
    session = st.session_state.study_session
    
    # Update session stats
    if correct and not partial:
        session['session_stats']['correct'] += 1
    else:
        session['session_stats']['incorrect'] += 1
    
    session['session_stats']['total_studied'] += 1
    
    # Update card statistics
    card['study_count'] = card.get('study_count', 0) + 1
    card['last_studied'] = datetime.now().isoformat()
    
    if correct and not partial:
        card['correct_count'] = card.get('correct_count', 0) + 1
    
    # Update spaced repetition
    if 'spaced_repetition' in card:
        update_spaced_repetition(card, correct and not partial)

def update_spaced_repetition(card: Dict, correct: bool):
    """Update spaced repetition algorithm"""
    
    sr = card['spaced_repetition']
    
    if correct:
        sr['repetitions'] += 1
        
        if sr['repetitions'] == 1:
            sr['interval'] = 1
        elif sr['repetitions'] == 2:
            sr['interval'] = 6
        else:
            sr['interval'] = int(sr['interval'] * sr['ease_factor'])
        
        # Increase ease factor for correct answers
        sr['ease_factor'] = min(2.5, sr['ease_factor'] + 0.1)
    
    else:
        # Reset for incorrect answers
        sr['repetitions'] = 0
        sr['interval'] = 1
        sr['ease_factor'] = max(1.3, sr['ease_factor'] - 0.2)
    
    # Set next review date
    next_review = datetime.now() + timedelta(days=sr['interval'])
    card['spaced_repetition_due'] = next_review.isoformat()

def next_card():
    """Move to the next card in the study session"""
    
    session = st.session_state.study_session
    session['current_index'] += 1
    session['show_answer'] = False
    st.rerun()

def display_study_session_results():
    """Display results of completed study session"""
    
    session = st.session_state.study_session
    stats = session['session_stats']
    
    st.markdown("## 🎉 Study Session Complete!")
    
    # Results summary
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Cards Studied", stats['total_studied'])
    with col2:
        st.metric("Correct", stats['correct'])
    with col3:
        st.metric("Incorrect", stats['incorrect'])
    with col4:
        accuracy = stats['correct'] / max(1, stats['total_studied']) * 100
        st.metric("Accuracy", f"{accuracy:.1f}%")
    
    # Performance feedback
    if accuracy >= 90:
        st.success("🏆 Excellent performance! You've mastered these concepts.")
    elif accuracy >= 75:
        st.info("👍 Good job! A few more review sessions will help solidify your knowledge.")
    elif accuracy >= 60:
        st.warning("📚 Keep studying! Focus on the concepts you found challenging.")
    else:
        st.error("💪 Don't give up! Review the material and try again.")
    
    # Restart options
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 Study Again"):
            del st.session_state.study_session
            st.rerun()
    
    with col2:
        if st.button("📊 View Analytics"):
            st.session_state.current_mode = 'flashcards'
            st.rerun()

def render_flashcard_analytics():
    """Render flashcard analytics and statistics"""
    st.markdown("### 📊 Flashcard Analytics")
    
    if 'flashcards' not in st.session_state or not st.session_state.flashcards:
        st.info("No flashcard data available. Create and study some flashcards first!")
        return
    
    flashcards = st.session_state.flashcards
    
    # Overall statistics
    st.markdown("#### 📈 Overall Statistics")
    
    total_cards = len(flashcards)
    studied_cards = len([card for card in flashcards if card.get('study_count', 0) > 0])
    total_studies = sum(card.get('study_count', 0) for card in flashcards)
    total_correct = sum(card.get('correct_count', 0) for card in flashcards)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Cards", total_cards)
    with col2:
        st.metric("Cards Studied", studied_cards)
    with col3:
        st.metric("Total Studies", total_studies)
    with col4:
        overall_accuracy = (total_correct / max(1, total_studies)) * 100
        st.metric("Overall Accuracy", f"{overall_accuracy:.1f}%")
    
    # Cards by type
    st.markdown("#### 🃏 Cards by Type")
    type_data = {}
    for card in flashcards:
        card_type = card.get('type', 'Unknown')
        if card_type not in type_data:
            type_data[card_type] = {'count': 0, 'studied': 0, 'accuracy': 0}
        
        type_data[card_type]['count'] += 1
        if card.get('study_count', 0) > 0:
            type_data[card_type]['studied'] += 1
            
        studies = card.get('study_count', 0)
        correct = card.get('correct_count', 0)
        if studies > 0:
            type_data[card_type]['accuracy'] += (correct / studies)
    
    # Calculate average accuracy per type
    for type_name, data in type_data.items():
        if data['studied'] > 0:
            data['accuracy'] = (data['accuracy'] / data['studied']) * 100
        else:
            data['accuracy'] = 0
    
    # Display type breakdown
    type_df = pd.DataFrame([
        {
            'Type': type_name,
            'Total Cards': data['count'],
            'Cards Studied': data['studied'],
            'Avg Accuracy (%)': f"{data['accuracy']:.1f}"
        }
        for type_name, data in type_data.items()
    ])
    
    st.dataframe(type_df, use_container_width=True)
    
    # Performance over time
    st.markdown("#### 📅 Recent Performance")
    
    # Get cards studied in last 7 days
    recent_cards = []
    week_ago = datetime.now() - timedelta(days=7)
    
    for card in flashcards:
        last_studied = card.get('last_studied')
        if last_studied:
            try:
                last_studied_date = datetime.fromisoformat(last_studied.replace('Z', '+00:00'))
                if last_studied_date >= week_ago:
                    recent_cards.append(card)
            except:
                pass
    
    if recent_cards:
        recent_studies = sum(card.get('study_count', 0) for card in recent_cards)
        recent_correct = sum(card.get('correct_count', 0) for card in recent_cards)
        recent_accuracy = (recent_correct / max(1, recent_studies)) * 100
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Cards Studied (7 days)", len(recent_cards))
        with col2:
            st.metric("Study Sessions", recent_studies)
        with col3:
            st.metric("Recent Accuracy", f"{recent_accuracy:.1f}%")
    else:
        st.info("No recent study activity.")
    
    # Cards needing review
    st.markdown("#### 🔄 Cards Needing Review")
    
    current_time = datetime.now().isoformat()
    due_cards = [
        card for card in flashcards 
        if card.get('spaced_repetition_due', current_time) <= current_time
    ]
    
    if due_cards:
        st.warning(f"⏰ {len(due_cards)} cards are due for review!")
        
        # Show sample of due cards
        for i, card in enumerate(due_cards[:5]):
            with st.expander(f"🃏 {card.get('question', 'Question')[:50]}..."):
                st.markdown(f"**Type:** {card.get('type', 'Unknown')}")
                st.markdown(f"**Last Studied:** {card.get('last_studied', 'Never')}")
                if card.get('spaced_repetition'):
                    interval = card['spaced_repetition'].get('interval', 1)
                    st.markdown(f"**Review Interval:** {interval} days")
        
        if len(due_cards) > 5:
            st.info(f"... and {len(due_cards) - 5} more cards due for review")
    else:
        st.success("✅ All cards are up to date!")
    
    # Export analytics
    st.markdown("#### 📤 Export Analytics")
    
    analytics_data = {
        'overall_stats': {
            'total_cards': total_cards,
            'studied_cards': studied_cards,
            'total_studies': total_studies,
            'overall_accuracy': overall_accuracy
        },
        'type_breakdown': type_data,
        'recent_performance': {
            'cards_studied_week': len(recent_cards),
            'recent_accuracy': recent_accuracy if recent_cards else 0
        },
        'cards_due_review': len(due_cards)
    }
    
    render_quick_export_buttons({'analytics': analytics_data}, "flashcard_analytics")
