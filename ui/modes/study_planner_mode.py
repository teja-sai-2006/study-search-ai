import streamlit as st
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime, timedelta
import pandas as pd
from utils.llm_engine import LLMEngine
from ui.components.file_uploader import render_document_selector
from ui.components.export_manager import render_quick_export_buttons

logger = logging.getLogger(__name__)

def render_study_planner_mode():
    """Render study planner mode for intelligent scheduling"""
    st.markdown("## 📚 Study Planner")
    st.markdown('<div class="mode-description">Create intelligent study plans with personalized schedules, break intervals, and progress tracking. Plan your learning journey with AI-powered recommendations.</div>', unsafe_allow_html=True)
    
    # Planning method selection
    st.markdown("### 📋 Planning Method")
    
    planning_method = st.radio(
        "Choose your planning approach:",
        ["📄 From uploaded documents", "✍️ Custom topics", "🔄 Hybrid approach"],
        help="Select how you want to create your study plan"
    )
    
    selected_topics = []
    selected_documents = []
    
    # Document/topic selection based on method
    if planning_method == "📄 From uploaded documents":
        if not st.session_state.documents:
            st.warning("📚 No documents available. Please upload documents on the Home page first.")
            return
        
        selected_indices = render_document_selector(st.session_state.documents, "study_planner")
        if selected_indices:
            selected_documents = [st.session_state.documents[i] for i in selected_indices]
            
            # Extract topics from documents
            with st.spinner("Analyzing documents for topics..."):
                selected_topics = extract_topics_from_documents(selected_documents)
                
            if selected_topics:
                st.success(f"✅ Extracted {len(selected_topics)} topics from documents")
                topic_tags = " • ".join([f"**{topic}**" for topic in selected_topics[:10]])
                st.markdown(f"**Topics:** {topic_tags}" + ("..." if len(selected_topics) > 10 else ""))
    
    elif planning_method == "✍️ Custom topics":
        custom_topics = st.text_area(
            "📝 Enter study topics (comma-separated)",
            height=100,
            placeholder="calculus, linear algebra, statistics, machine learning, data structures",
            help="Enter the topics you want to study, separated by commas"
        )
        
        if custom_topics:
            selected_topics = [topic.strip() for topic in custom_topics.split(',') if topic.strip()]
            
            if selected_topics:
                st.success(f"✅ {len(selected_topics)} topics added")
    
    else:  # Hybrid approach
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📄 From Documents")
            if st.session_state.documents:
                doc_indices = render_document_selector(st.session_state.documents, "hybrid_docs")
                if doc_indices:
                    selected_documents = [st.session_state.documents[i] for i in doc_indices]
                    doc_topics = extract_topics_from_documents(selected_documents)
                    selected_topics.extend(doc_topics)
            else:
                st.info("No documents available")
        
        with col2:
            st.markdown("#### ✍️ Additional Topics")
            additional_topics = st.text_area(
                "Add custom topics:",
                height=100,
                placeholder="Additional topics...",
                key="additional_topics"
            )
            
            if additional_topics:
                custom_topics = [topic.strip() for topic in additional_topics.split(',') if topic.strip()]
                selected_topics.extend(custom_topics)
        
        # Remove duplicates
        selected_topics = list(dict.fromkeys(selected_topics))
        
        if selected_topics:
            st.success(f"✅ Total: {len(selected_topics)} topics")
    
    if not selected_topics:
        st.info("👆 Please select documents or enter topics to create a study plan.")
        return
    
    # Study plan configuration
    st.markdown("### ⚙️ Study Plan Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        duration_weeks = st.slider(
            "📅 Study Duration (weeks)",
            min_value=1,
            max_value=6,
            value=4,
            help="How many weeks do you want to study?"
        )
        
        daily_hours = st.slider(
            "⏰ Daily Study Hours",
            min_value=1,
            max_value=12,
            value=4,
            help="How many hours per day can you dedicate to studying?"
        )
        
        difficulty_level = st.selectbox(
            "📊 Your Level",
            ["Beginner", "Intermediate", "Advanced"],
            help="Your current knowledge level in these topics"
        )
    
    with col2:
        start_time = st.time_input(
            "🌅 Daily Start Time",
            value=datetime.strptime("09:00", "%H:%M").time(),
            help="When do you prefer to start studying each day?"
        )
        
        break_interval = st.selectbox(
            "☕ Break Interval",
            ["20 minutes", "30 minutes", "40 minutes", "50 minutes"],
            index=1,
            help="How often do you want breaks?"
        )
        
        study_style = st.selectbox(
            "🎯 Study Style",
            ["Balanced", "Intensive", "Gradual", "Review-Heavy"],
            help="What type of study approach do you prefer?"
        )
    
    # Advanced planning options
    with st.expander("🔧 Advanced Planning Options"):
        col1, col2 = st.columns(2)
        
        with col1:
            priority_topics = st.multiselect(
                "⭐ Priority Topics",
                selected_topics,
                help="Select topics that need more focus"
            )
            
            include_weekends = st.checkbox(
                "📅 Include weekends",
                value=True,
                help="Plan study sessions for weekends"
            )
            
            review_frequency = st.selectbox(
                "🔄 Review Frequency",
                ["Daily", "Every 2 days", "Weekly", "Bi-weekly"],
                index=2,
                help="How often to schedule review sessions"
            )
        
        with col2:
            assessment_schedule = st.checkbox(
                "📝 Include assessments",
                value=True,
                help="Schedule self-assessment sessions"
            )
            
            adaptive_planning = st.checkbox(
                "🧠 Adaptive difficulty",
                value=True,
                help="Adjust difficulty based on topic complexity"
            )
            
            integration_goals = st.text_area(
                "🎯 Learning Goals",
                placeholder="What do you want to achieve with this study plan?",
                help="Specific goals or objectives"
            )
    
    # Generate study plan
    if st.button("📅 Generate Study Plan", type="primary"):
        with st.spinner("Creating your personalized study plan..."):
            study_plan = generate_comprehensive_study_plan(
                selected_topics,
                selected_documents,
                duration_weeks,
                daily_hours,
                difficulty_level,
                start_time,
                break_interval,
                study_style,
                priority_topics,
                include_weekends,
                review_frequency,
                assessment_schedule,
                adaptive_planning,
                integration_goals
            )
        
        if study_plan:
            display_study_plan(study_plan)
            
            # Save to session state
            st.session_state.study_plan = study_plan
            
            # Export options
            st.markdown("### 📤 Export Study Plan")
            render_quick_export_buttons({'study_plan': study_plan}, "study_plan")
        else:
            st.error("❌ Failed to generate study plan. Please try again with different settings.")

def extract_topics_from_documents(documents: List[Dict]) -> List[str]:
    """Extract study topics from uploaded documents"""
    
    try:
        llm_engine = LLMEngine()
        
        # Combine document content
        combined_content = []
        for doc in documents:
            content = doc.get('content', '')
            if content:
                # Take first 1000 chars to get main topics
                combined_content.append(content[:1000])
        
        if not combined_content:
            return []
        
        content_sample = '\n\n'.join(combined_content)
        
        prompt = f"""Analyze the following documents and extract 10-15 key study topics that a student should focus on:

{content_sample}

Extract topics that are:
1. Specific enough to study individually
2. Important concepts or subjects
3. Suitable for creating study sessions
4. Represent the main themes of the content

Format as a comma-separated list of topics."""
        
        response = llm_engine.generate_response(prompt, task_type="analyze")
        
        # Parse topics
        topics = [topic.strip() for topic in response.split(',') if topic.strip()]
        
        # Clean and validate topics
        clean_topics = []
        for topic in topics:
            # Remove common prefixes/suffixes
            topic = topic.replace('Topic:', '').replace('Subject:', '').strip()
            if len(topic) > 3 and len(topic) < 100:
                clean_topics.append(topic)
        
        return clean_topics[:15]  # Limit to 15 topics
    
    except Exception as e:
        logger.error(f"Topic extraction failed: {e}")
        return []

def generate_comprehensive_study_plan(
    topics: List[str],
    documents: List[Dict],
    duration_weeks: int,
    daily_hours: int,
    difficulty_level: str,
    start_time,
    break_interval: str,
    study_style: str,
    priority_topics: List[str],
    include_weekends: bool,
    review_frequency: str,
    assessment_schedule: bool,
    adaptive_planning: bool,
    integration_goals: str
) -> Dict[str, Any]:
    """Generate comprehensive study plan with AI optimization"""
    
    try:
        llm_engine = LLMEngine()
        
        # Calculate planning parameters
        total_days = duration_weeks * (7 if include_weekends else 5)
        total_hours = total_days * daily_hours
        break_minutes = int(break_interval.split()[0])
        
        # Assign time allocation based on priority and complexity
        topic_allocations = calculate_topic_allocations(
            topics, priority_topics, total_hours, difficulty_level, adaptive_planning
        )
        
        # Generate daily schedule template
        daily_template = create_daily_schedule_template(
            daily_hours, start_time, break_minutes
        )
        
        # Create detailed weekly plans
        weekly_plans = create_weekly_plans(
            topics, topic_allocations, duration_weeks, daily_template,
            study_style, review_frequency, assessment_schedule
        )
        
        # Generate study strategies and tips
        study_strategies = generate_study_strategies(
            topics, difficulty_level, study_style, integration_goals
        )
        
        # Create progress tracking structure
        progress_tracking = create_progress_tracking_structure(topics, weekly_plans)
        
        study_plan = {
            'overview': {
                'duration_weeks': duration_weeks,
                'daily_hours': daily_hours,
                'total_hours': total_hours,
                'total_topics': len(topics),
                'difficulty_level': difficulty_level,
                'study_style': study_style,
                'created_date': datetime.now().isoformat()
            },
            'topics': topics,
            'topic_allocations': topic_allocations,
            'daily_template': daily_template,
            'weekly_plans': weekly_plans,
            'study_strategies': study_strategies,
            'progress_tracking': progress_tracking,
            'milestones': create_milestones(topics, duration_weeks),
            'resources': {
                'documents': [doc.get('name', '') for doc in documents],
                'priority_topics': priority_topics
            }
        }
        
        return study_plan
    
    except Exception as e:
        logger.error(f"Study plan generation failed: {e}")
        return {}

def calculate_topic_allocations(
    topics: List[str],
    priority_topics: List[str],
    total_hours: int,
    difficulty_level: str,
    adaptive_planning: bool
) -> Dict[str, Dict[str, Any]]:
    """Calculate time allocation for each topic"""
    
    allocations = {}
    
    # Base allocation
    base_hours = total_hours / len(topics)
    
    for topic in topics:
        # Start with base allocation
        allocated_hours = base_hours
        
        # Adjust for priority
        if topic in priority_topics:
            allocated_hours *= 1.3
        
        # Adjust for difficulty if adaptive planning is enabled
        if adaptive_planning:
            complexity_factor = estimate_topic_complexity(topic, difficulty_level)
            allocated_hours *= complexity_factor
        
        allocations[topic] = {
            'hours': allocated_hours,
            'priority': topic in priority_topics,
            'complexity': estimate_topic_complexity(topic, difficulty_level),
            'sessions_needed': max(1, int(allocated_hours / 2))  # 2 hours per session
        }
    
    # Normalize allocations to fit total hours
    total_allocated = sum(data['hours'] for data in allocations.values())
    if total_allocated > 0:
        for topic in allocations:
            allocations[topic]['hours'] = (allocations[topic]['hours'] / total_allocated) * total_hours
    
    return allocations

def estimate_topic_complexity(topic: str, difficulty_level: str) -> float:
    """Estimate complexity factor for a topic"""
    
    # Complex topics that might need more time
    complex_keywords = [
        'calculus', 'physics', 'chemistry', 'algorithm', 'programming',
        'statistics', 'machine learning', 'quantum', 'advanced', 'theory'
    ]
    
    # Simple topics that might need less time
    simple_keywords = [
        'introduction', 'basic', 'overview', 'fundamentals', 'concept'
    ]
    
    topic_lower = topic.lower()
    
    complexity = 1.0  # Base complexity
    
    # Adjust based on keywords
    if any(keyword in topic_lower for keyword in complex_keywords):
        complexity += 0.3
    elif any(keyword in topic_lower for keyword in simple_keywords):
        complexity -= 0.2
    
    # Adjust based on user's difficulty level
    if difficulty_level == "Beginner":
        complexity += 0.2
    elif difficulty_level == "Advanced":
        complexity -= 0.1
    
    return max(0.5, min(2.0, complexity))

def create_daily_schedule_template(daily_hours: int, start_time, break_minutes: int) -> Dict[str, Any]:
    """Create daily schedule template with breaks"""
    
    start_datetime = datetime.combine(datetime.today(), start_time)
    
    # Calculate study blocks
    study_block_duration = break_minutes  # Study for break_minutes, then break
    break_duration = 10  # 10-minute breaks
    
    schedule = []
    current_time = start_datetime
    remaining_minutes = daily_hours * 60
    
    while remaining_minutes > 0:
        # Study block
        block_duration = min(study_block_duration, remaining_minutes)
        end_time = current_time + timedelta(minutes=block_duration)
        
        schedule.append({
            'type': 'study',
            'start': current_time.strftime('%H:%M'),
            'end': end_time.strftime('%H:%M'),
            'duration_minutes': block_duration,
            'topic': None  # To be filled in specific plans
        })
        
        current_time = end_time
        remaining_minutes -= block_duration
        
        # Break (if more study time remains)
        if remaining_minutes > 0:
            break_end = current_time + timedelta(minutes=break_duration)
            schedule.append({
                'type': 'break',
                'start': current_time.strftime('%H:%M'),
                'end': break_end.strftime('%H:%M'),
                'duration_minutes': break_duration
            })
            current_time = break_end
    
    return {
        'total_hours': daily_hours,
        'start_time': start_time.strftime('%H:%M'),
        'schedule_blocks': schedule,
        'break_duration': break_duration,
        'study_block_duration': study_block_duration
    }

def create_weekly_plans(
    topics: List[str],
    allocations: Dict[str, Dict],
    duration_weeks: int,
    daily_template: Dict,
    study_style: str,
    review_frequency: str,
    assessment_schedule: bool
) -> List[Dict[str, Any]]:
    """Create detailed weekly study plans"""
    
    weekly_plans = []
    
    # Distribute topics across weeks
    topics_per_week = distribute_topics_across_weeks(topics, allocations, duration_weeks)
    
    for week_num in range(1, duration_weeks + 1):
        week_topics = topics_per_week.get(week_num, [])
        
        # Create daily plans for the week
        daily_plans = create_week_daily_plans(
            week_topics, allocations, daily_template, study_style, week_num
        )
        
        # Add review sessions
        if should_add_review(week_num, review_frequency):
            add_review_sessions(daily_plans, topics[:week_num*3])  # Review previous topics
        
        # Add assessments
        if assessment_schedule and week_num % 2 == 0:  # Every 2 weeks
            add_assessment_session(daily_plans, week_topics)
        
        weekly_plans.append({
            'week_number': week_num,
            'focus_topics': week_topics,
            'daily_plans': daily_plans,
            'week_goals': generate_week_goals(week_topics, week_num, duration_weeks),
            'deliverables': generate_week_deliverables(week_topics)
        })
    
    return weekly_plans

def distribute_topics_across_weeks(
    topics: List[str],
    allocations: Dict[str, Dict],
    duration_weeks: int
) -> Dict[int, List[str]]:
    """Distribute topics across weeks based on complexity and priority"""
    
    # Sort topics by priority and complexity
    topic_scores = []
    for topic in topics:
        allocation = allocations.get(topic, {})
        priority_score = 2 if allocation.get('priority', False) else 1
        complexity_score = allocation.get('complexity', 1)
        
        # Combine scores - higher scores should be scheduled earlier
        combined_score = priority_score * complexity_score
        topic_scores.append((topic, combined_score))
    
    # Sort by score (highest first)
    topic_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Distribute evenly across weeks
    topics_per_week = {}
    for i, (topic, score) in enumerate(topic_scores):
        week_num = (i % duration_weeks) + 1
        if week_num not in topics_per_week:
            topics_per_week[week_num] = []
        topics_per_week[week_num].append(topic)
    
    return topics_per_week

def create_week_daily_plans(
    week_topics: List[str],
    allocations: Dict[str, Dict],
    daily_template: Dict,
    study_style: str,
    week_num: int
) -> Dict[str, List[Dict]]:
    """Create daily plans for a specific week"""
    
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    daily_plans = {}
    
    # Cycle through topics for the week
    topic_cycle = week_topics * 10  # Repeat topics to fill the week
    topic_index = 0
    
    for day in days:
        day_schedule = []
        
        # Copy template schedule
        for block in daily_template['schedule_blocks']:
            if block['type'] == 'study':
                # Assign topic to study block
                if topic_index < len(topic_cycle):
                    current_topic = topic_cycle[topic_index]
                    topic_index += 1
                    
                    study_block = block.copy()
                    study_block['topic'] = current_topic
                    study_block['activities'] = generate_study_activities(
                        current_topic, study_style, block['duration_minutes']
                    )
                    day_schedule.append(study_block)
                else:
                    # Review session if no new topics
                    review_block = block.copy()
                    review_block['topic'] = 'Review Session'
                    review_block['activities'] = ['Review previous topics', 'Practice problems', 'Q&A session']
                    day_schedule.append(review_block)
            else:
                day_schedule.append(block)
        
        daily_plans[day] = day_schedule
    
    return daily_plans

def generate_study_activities(topic: str, study_style: str, duration_minutes: int) -> List[str]:
    """Generate specific study activities for a topic"""
    
    activities = []
    
    if study_style == "Balanced":
        activities = [
            f"Read/review materials on {topic}",
            f"Take notes on key concepts",
            f"Practice problems or exercises",
            f"Summarize main points"
        ]
    elif study_style == "Intensive":
        activities = [
            f"Deep dive into {topic} theory",
            f"Work through complex problems",
            f"Create detailed mind maps",
            f"Explain concepts out loud"
        ]
    elif study_style == "Gradual":
        activities = [
            f"Introduction to {topic}",
            f"Basic concept review",
            f"Simple practice exercises",
            f"Connect to previous knowledge"
        ]
    elif study_style == "Review-Heavy":
        activities = [
            f"Review {topic} fundamentals",
            f"Practice previous problems",
            f"Identify knowledge gaps",
            f"Reinforce learning"
        ]
    
    # Adjust activities based on duration
    if duration_minutes < 30:
        activities = activities[:2]
    elif duration_minutes > 60:
        activities.extend([f"Advanced {topic} applications", "Create flashcards"])
    
    return activities

def should_add_review(week_num: int, review_frequency: str) -> bool:
    """Determine if review session should be added"""
    
    if review_frequency == "Daily":
        return True
    elif review_frequency == "Every 2 days":
        return week_num % 2 == 0
    elif review_frequency == "Weekly":
        return True  # Each week has review
    elif review_frequency == "Bi-weekly":
        return week_num % 2 == 0
    
    return False

def add_review_sessions(daily_plans: Dict, review_topics: List[str]):
    """Add review sessions to daily plans"""
    
    # Add review to Friday
    if 'Friday' in daily_plans:
        # Replace last study block with review
        friday_plan = daily_plans['Friday']
        for i, block in enumerate(friday_plan):
            if block.get('type') == 'study':
                # Make last study block a review
                friday_plan[-1] = {
                    'type': 'review',
                    'start': block['start'],
                    'end': block['end'],
                    'duration_minutes': block['duration_minutes'],
                    'topics': review_topics[-5:],  # Review last 5 topics
                    'activities': ['Review key concepts', 'Practice problems', 'Self-assessment']
                }
                break

def add_assessment_session(daily_plans: Dict, week_topics: List[str]):
    """Add assessment session to weekly plan"""
    
    # Add assessment to Sunday
    if 'Sunday' in daily_plans:
        sunday_plan = daily_plans['Sunday']
        # Replace first study block with assessment
        for i, block in enumerate(sunday_plan):
            if block.get('type') == 'study':
                sunday_plan[i] = {
                    'type': 'assessment',
                    'start': block['start'],
                    'end': block['end'],
                    'duration_minutes': block['duration_minutes'],
                    'topics': week_topics,
                    'activities': ['Self-test', 'Practice quiz', 'Knowledge check', 'Progress evaluation']
                }
                break

def generate_week_goals(topics: List[str], week_num: int, total_weeks: int) -> List[str]:
    """Generate specific goals for the week"""
    
    goals = []
    
    # Topic-specific goals
    for topic in topics:
        goals.append(f"Master fundamental concepts of {topic}")
        goals.append(f"Complete practice exercises for {topic}")
    
    # Week-specific goals
    if week_num == 1:
        goals.append("Establish study routine and habits")
        goals.append("Set up study environment")
    elif week_num == total_weeks:
        goals.append("Complete comprehensive review")
        goals.append("Prepare for final assessment")
    else:
        goals.append("Build on previous week's knowledge")
        goals.append("Maintain consistent study pace")
    
    return goals[:5]  # Limit to 5 goals

def generate_week_deliverables(topics: List[str]) -> List[str]:
    """Generate deliverables for the week"""
    
    deliverables = []
    
    for topic in topics:
        deliverables.append(f"Summary notes for {topic}")
        deliverables.append(f"Practice problem solutions for {topic}")
    
    deliverables.append("Weekly self-assessment completed")
    deliverables.append("Progress tracking updated")
    
    return deliverables

def generate_study_strategies(
    topics: List[str],
    difficulty_level: str,
    study_style: str,
    integration_goals: str
) -> Dict[str, Any]:
    """Generate study strategies and tips"""
    
    try:
        llm_engine = LLMEngine()
        
        topics_text = ', '.join(topics[:10])  # First 10 topics
        
        prompt = f"""Create study strategies for a {difficulty_level} level student studying these topics: {topics_text}

Study style preference: {study_style}
Learning goals: {integration_goals}

Provide:
1. 5 general study strategies
2. 3 topic-specific techniques
3. 3 retention and recall methods
4. 2 progress monitoring approaches
5. Tips for maintaining motivation

Format as structured recommendations."""
        
        response = llm_engine.generate_response(prompt, task_type="generate")
        
        return {
            'general_strategies': response,
            'study_tips': [
                "Use active recall techniques",
                "Space out your learning sessions",
                "Teach concepts to others",
                "Practice regularly",
                "Take breaks to consolidate memory"
            ],
            'resources': [
                "Create flashcards for key concepts",
                "Use mind maps for complex topics",
                "Join study groups or forums",
                "Seek additional practice problems",
                "Use online educational resources"
            ]
        }
    
    except Exception as e:
        logger.error(f"Strategy generation failed: {e}")
        return {
            'general_strategies': "Study consistently, take breaks, and practice actively.",
            'study_tips': ["Stay organized", "Review regularly", "Ask questions"],
            'resources': ["Use textbooks", "Practice problems", "Online resources"]
        }

def create_progress_tracking_structure(topics: List[str], weekly_plans: List[Dict]) -> Dict[str, Any]:
    """Create structure for tracking study progress"""
    
    tracking = {
        'topics_progress': {},
        'weekly_completion': {},
        'milestones': [],
        'metrics': {
            'total_study_hours': 0,
            'completed_sessions': 0,
            'topics_mastered': 0,
            'assessments_completed': 0
        }
    }
    
    # Initialize topic progress
    for topic in topics:
        tracking['topics_progress'][topic] = {
            'completion_percentage': 0,
            'sessions_completed': 0,
            'last_studied': None,
            'mastery_level': 'Not Started',
            'notes': ''
        }
    
    # Initialize weekly completion
    for i, week_plan in enumerate(weekly_plans):
        week_num = week_plan['week_number']
        tracking['weekly_completion'][week_num] = {
            'completion_percentage': 0,
            'goals_achieved': 0,
            'total_goals': len(week_plan.get('week_goals', [])),
            'deliverables_completed': 0,
            'total_deliverables': len(week_plan.get('deliverables', []))
        }
    
    return tracking

def create_milestones(topics: List[str], duration_weeks: int) -> List[Dict[str, Any]]:
    """Create study milestones"""
    
    milestones = []
    
    # Week-based milestones
    for week in range(1, duration_weeks + 1):
        if week == 1:
            milestones.append({
                'week': week,
                'title': 'Study Foundation',
                'description': 'Establish study routine and cover initial topics',
                'target_completion': 25
            })
        elif week == duration_weeks // 2:
            milestones.append({
                'week': week,
                'title': 'Midpoint Mastery',
                'description': 'Complete half of topics and assess progress',
                'target_completion': 50
            })
        elif week == duration_weeks:
            milestones.append({
                'week': week,
                'title': 'Complete Mastery',
                'description': 'Master all topics and complete final assessment',
                'target_completion': 100
            })
    
    # Topic-based milestones
    milestone_topics = topics[::max(1, len(topics)//3)]  # Every 1/3 of topics
    for i, topic in enumerate(milestone_topics):
        milestones.append({
            'topic': topic,
            'title': f'{topic} Mastery',
            'description': f'Complete understanding and application of {topic}',
            'target_completion': (i + 1) * (100 // len(milestone_topics))
        })
    
    return milestones

def display_study_plan(study_plan: Dict[str, Any]):
    """Display the generated study plan"""
    
    st.markdown("## 📅 Your Personalized Study Plan")
    
    # Overview
    overview = study_plan.get('overview', {})
    st.markdown("### 📊 Plan Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Duration", f"{overview.get('duration_weeks', 0)} weeks")
    with col2:
        st.metric("Daily Hours", f"{overview.get('daily_hours', 0)} hours")
    with col3:
        st.metric("Total Hours", f"{overview.get('total_hours', 0)} hours")
    with col4:
        st.metric("Topics", f"{overview.get('total_topics', 0)} topics")
    
    # Topics and allocations
    st.markdown("### 📚 Topic Allocations")
    
    allocations = study_plan.get('topic_allocations', {})
    if allocations:
        allocation_data = []
        for topic, data in allocations.items():
            allocation_data.append({
                'Topic': topic,
                'Hours': f"{data.get('hours', 0):.1f}",
                'Sessions': data.get('sessions_needed', 0),
                'Priority': '⭐' if data.get('priority', False) else '',
                'Complexity': f"{data.get('complexity', 1):.1f}"
            })
        
        df = pd.DataFrame(allocation_data)
        st.dataframe(df, use_container_width=True)
    
    # Weekly plans
    st.markdown("### 📅 Weekly Plans")
    
    weekly_plans = study_plan.get('weekly_plans', [])
    for week_plan in weekly_plans:
        week_num = week_plan.get('week_number', 0)
        focus_topics = week_plan.get('focus_topics', [])
        
        with st.expander(f"📅 Week {week_num} - Focus: {', '.join(focus_topics[:3])}" + ("..." if len(focus_topics) > 3 else "")):
            
            # Week goals
            week_goals = week_plan.get('week_goals', [])
            if week_goals:
                st.markdown("#### 🎯 Week Goals")
                for goal in week_goals:
                    st.markdown(f"• {goal}")
            
            # Daily schedule
            st.markdown("#### 📅 Daily Schedule")
            daily_plans = week_plan.get('daily_plans', {})
            
            for day, schedule in daily_plans.items():
                with st.expander(f"📅 {day}"):
                    for block in schedule:
                        if block.get('type') == 'study':
                            topic = block.get('topic', 'Unknown')
                            start = block.get('start', '')
                            end = block.get('end', '')
                            activities = block.get('activities', [])
                            
                            st.markdown(f"**📚 {start} - {end}: {topic}**")
                            for activity in activities:
                                st.markdown(f"  • {activity}")
                        
                        elif block.get('type') == 'break':
                            start = block.get('start', '')
                            end = block.get('end', '')
                            st.markdown(f"☕ {start} - {end}: Break")
                        
                        elif block.get('type') == 'review':
                            start = block.get('start', '')
                            end = block.get('end', '')
                            st.markdown(f"🔄 {start} - {end}: Review Session")
                        
                        elif block.get('type') == 'assessment':
                            start = block.get('start', '')
                            end = block.get('end', '')
                            st.markdown(f"📝 {start} - {end}: Assessment")
            
            # Deliverables
            deliverables = week_plan.get('deliverables', [])
            if deliverables:
                st.markdown("#### 📋 Week Deliverables")
                for deliverable in deliverables:
                    st.markdown(f"☐ {deliverable}")
    
    # Study strategies
    strategies = study_plan.get('study_strategies', {})
    if strategies:
        st.markdown("### 🧠 Study Strategies & Tips")
        
        general_strategies = strategies.get('general_strategies', '')
        if general_strategies:
            st.markdown("#### 📝 Personalized Strategies")
            st.markdown(general_strategies)
        
        study_tips = strategies.get('study_tips', [])
        if study_tips:
            st.markdown("#### 💡 Study Tips")
            for tip in study_tips:
                st.markdown(f"• {tip}")
    
    # Milestones
    milestones = study_plan.get('milestones', [])
    if milestones:
        st.markdown("### 🏆 Milestones")
        
        for milestone in milestones:
            title = milestone.get('title', 'Milestone')
            description = milestone.get('description', '')
            target = milestone.get('target_completion', 0)
            
            st.markdown(f"**{title}** (Target: {target}%)")
            st.markdown(f"  {description}")
    
    # Progress tracking setup
    st.markdown("### 📈 Progress Tracking")
    st.info("""
    📊 **Track Your Progress:**
    • Use the Study Progress mode to monitor your advancement
    • Mark completed sessions and topics
    • Update your learning goals
    • Assess your understanding regularly
    """)
    
    st.success("✅ Study plan generated successfully! Use the Study Progress mode to track your advancement.")
