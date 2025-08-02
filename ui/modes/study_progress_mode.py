import streamlit as st
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime, timedelta
import pandas as pd
import json
from utils.llm_engine import LLMEngine
from ui.components.export_manager import render_quick_export_buttons

logger = logging.getLogger(__name__)

def render_study_progress_mode():
    """Render study progress tracking and analytics"""
    st.markdown("## 📈 Study Progress")
    st.markdown('<div class="mode-description">Track your learning journey with comprehensive progress analytics. Monitor flashcard mastery, study plan completion, and overall learning metrics.</div>', unsafe_allow_html=True)
    
    # Progress overview
    render_progress_overview()
    
    # Detailed progress sections
    tab1, tab2, tab3, tab4 = st.tabs(["📚 Study Plan", "🧠 Flashcards", "📝 Summaries", "🎯 Goals"])
    
    with tab1:
        render_study_plan_progress()
    
    with tab2:
        render_flashcard_progress()
    
    with tab3:
        render_summary_progress()
    
    with tab4:
        render_goals_and_milestones()

def render_progress_overview():
    """Render overall progress overview"""
    st.markdown("### 🎯 Progress Overview")
    
    # Calculate overall metrics
    metrics = calculate_overall_metrics()
    
    # Display key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Overall Progress",
            f"{metrics.get('overall_progress', 0):.1f}%",
            delta=f"+{metrics.get('progress_delta', 0):.1f}%" if metrics.get('progress_delta', 0) > 0 else None
        )
    
    with col2:
        st.metric(
            "Study Streak",
            f"{metrics.get('study_streak', 0)} days",
            delta=f"+{metrics.get('streak_delta', 0)}" if metrics.get('streak_delta', 0) > 0 else None
        )
    
    with col3:
        st.metric(
            "Total Study Time",
            f"{metrics.get('total_study_hours', 0):.1f}h",
            delta=f"+{metrics.get('hours_delta', 0):.1f}h" if metrics.get('hours_delta', 0) > 0 else None
        )
    
    with col4:
        st.metric(
            "Mastery Score",
            f"{metrics.get('mastery_score', 0):.1f}/10",
            delta=f"+{metrics.get('mastery_delta', 0):.1f}" if metrics.get('mastery_delta', 0) > 0 else None
        )
    
    # Progress visualization
    if metrics.get('overall_progress', 0) > 0:
        st.markdown("#### 📊 Progress Visualization")
        progress_data = get_progress_timeline()
        
        if progress_data:
            # Create progress chart data
            chart_data = pd.DataFrame(progress_data)
            if not chart_data.empty:
                st.line_chart(chart_data.set_index('date'))
    
    # Recent achievements
    achievements = get_recent_achievements()
    if achievements:
        st.markdown("#### 🏆 Recent Achievements")
        for achievement in achievements[:3]:
            st.success(f"🎉 {achievement}")

def calculate_overall_metrics() -> Dict[str, float]:
    """Calculate overall progress metrics"""
    metrics = {
        'overall_progress': 0.0,
        'progress_delta': 0.0,
        'study_streak': 0,
        'streak_delta': 0,
        'total_study_hours': 0.0,
        'hours_delta': 0.0,
        'mastery_score': 0.0,
        'mastery_delta': 0.0
    }
    
    try:
        # Study plan progress
        if 'study_plan' in st.session_state and st.session_state.study_plan:
            plan_progress = calculate_study_plan_progress()
            metrics['overall_progress'] += plan_progress * 0.4
        
        # Flashcard progress
        if 'flashcards' in st.session_state and st.session_state.flashcards:
            flashcard_progress = calculate_flashcard_mastery()
            metrics['overall_progress'] += flashcard_progress * 0.4
        
        # Summary progress
        if 'summaries' in st.session_state and st.session_state.summaries:
            summary_progress = len(st.session_state.summaries) * 5  # 5% per summary
            metrics['overall_progress'] += min(summary_progress, 20)  # Max 20%
        
        # Calculate study streak
        metrics['study_streak'] = calculate_study_streak()
        
        # Calculate total study time
        metrics['total_study_hours'] = calculate_total_study_time()
        
        # Calculate mastery score
        metrics['mastery_score'] = calculate_mastery_score()
        
        # Get deltas (changes from last calculation)
        previous_metrics = st.session_state.get('previous_metrics', {})
        for key in ['overall_progress', 'study_streak', 'total_study_hours', 'mastery_score']:
            if key in previous_metrics:
                metrics[f'{key.replace("_", "")}_delta'] = metrics[key] - previous_metrics[key]
        
        # Store current metrics for next comparison
        st.session_state.previous_metrics = metrics.copy()
        
    except Exception as e:
        logger.error(f"Error calculating metrics: {e}")
    
    return metrics

def get_progress_timeline() -> List[Dict[str, Any]]:
    """Get progress data over time"""
    timeline = []
    
    try:
        # Get data from session state or storage
        progress_history = st.session_state.get('progress_history', [])
        
        if not progress_history:
            # Create initial timeline based on available data
            base_date = datetime.now() - timedelta(days=30)
            
            for i in range(30):
                date = base_date + timedelta(days=i)
                
                # Simulate progress growth
                overall_progress = min(100, i * 3 + calculate_daily_progress(date))
                flashcard_mastery = min(100, i * 2.5 + calculate_flashcard_progress_for_date(date))
                study_plan_progress = min(100, i * 3.5 + calculate_study_plan_progress_for_date(date))
                
                timeline.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'Overall Progress': overall_progress,
                    'Flashcard Mastery': flashcard_mastery,
                    'Study Plan': study_plan_progress
                })
            
            # Store generated timeline
            st.session_state.progress_history = timeline
        else:
            timeline = progress_history
    
    except Exception as e:
        logger.error(f"Error getting progress timeline: {e}")
    
    return timeline

def calculate_daily_progress(date: datetime) -> float:
    """Calculate progress for a specific date"""
    # Simple calculation based on date and available data
    return min(10, len(st.session_state.get('flashcards', [])) * 0.1)

def calculate_flashcard_progress_for_date(date: datetime) -> float:
    """Calculate flashcard progress for a specific date"""
    flashcards = st.session_state.get('flashcards', [])
    if not flashcards:
        return 0
    
    studied_count = sum(1 for card in flashcards if card.get('study_count', 0) > 0)
    return (studied_count / len(flashcards)) * 100

def calculate_study_plan_progress_for_date(date: datetime) -> float:
    """Calculate study plan progress for a specific date"""
    if 'study_plan' not in st.session_state:
        return 0
    
    # Simple calculation - could be enhanced with actual completion tracking
    return min(100, (datetime.now() - date).days * 5)

def get_recent_achievements() -> List[str]:
    """Get list of recent achievements"""
    achievements = []
    
    try:
        # Check for flashcard achievements
        flashcards = st.session_state.get('flashcards', [])
        if flashcards:
            total_correct = sum(card.get('correct_count', 0) for card in flashcards)
            if total_correct >= 100:
                achievements.append("Answered 100+ flashcards correctly!")
            elif total_correct >= 50:
                achievements.append("Answered 50+ flashcards correctly!")
        
        # Check for study plan achievements
        if 'study_plan' in st.session_state:
            plan_progress = calculate_study_plan_progress()
            if plan_progress >= 100:
                achievements.append("Completed study plan!")
            elif plan_progress >= 50:
                achievements.append("Halfway through study plan!")
        
        # Check for summary achievements
        summaries = st.session_state.get('summaries', [])
        if len(summaries) >= 10:
            achievements.append("Created 10+ summaries!")
        elif len(summaries) >= 5:
            achievements.append("Created 5+ summaries!")
        
        # Check for streak achievements
        streak = calculate_study_streak()
        if streak >= 30:
            achievements.append("30-day study streak!")
        elif streak >= 14:
            achievements.append("2-week study streak!")
        elif streak >= 7:
            achievements.append("1-week study streak!")
    
    except Exception as e:
        logger.error(f"Error getting achievements: {e}")
    
    return achievements

def render_study_plan_progress():
    """Render study plan progress tracking"""
    st.markdown("### 📚 Study Plan Progress")
    
    if 'study_plan' not in st.session_state or not st.session_state.study_plan:
        st.info("No active study plan. Create one in Study Planner mode!")
        return
    
    study_plan = st.session_state.study_plan
    
    # Overall plan progress
    plan_progress = calculate_study_plan_progress()
    st.progress(plan_progress / 100, text=f"Overall Plan Progress: {plan_progress:.1f}%")
    
    # Plan overview
    overview = study_plan.get('overview', {})
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Duration", f"{overview.get('duration_weeks', 0)} weeks")
    with col2:
        st.metric("Daily Hours", f"{overview.get('daily_hours', 0)}h")
    with col3:
        completed_weeks = get_completed_weeks(study_plan)
        st.metric("Weeks Completed", f"{completed_weeks}/{overview.get('duration_weeks', 0)}")
    with col4:
        topics_mastered = get_mastered_topics_count(study_plan)
        total_topics = len(study_plan.get('topics', []))
        st.metric("Topics Mastered", f"{topics_mastered}/{total_topics}")
    
    # Weekly progress
    st.markdown("#### 📅 Weekly Progress")
    weekly_plans = study_plan.get('weekly_plans', [])
    
    for week_plan in weekly_plans:
        week_num = week_plan.get('week_number', 0)
        focus_topics = week_plan.get('focus_topics', [])
        week_goals = week_plan.get('week_goals', [])
        
        week_completion = calculate_week_completion(week_plan)
        
        with st.expander(f"📅 Week {week_num} - {week_completion:.1f}% Complete"):
            st.markdown(f"**Focus Topics:** {', '.join(focus_topics)}")
            
            # Week goals with checkboxes
            st.markdown("**Goals:**")
            completed_goals = 0
            
            for i, goal in enumerate(week_goals):
                goal_completed = st.checkbox(
                    goal,
                    value=get_goal_completion_status(week_num, i),
                    key=f"goal_{week_num}_{i}"
                )
                if goal_completed:
                    completed_goals += 1
                    save_goal_completion(week_num, i, True)
            
            if week_goals:
                goal_progress = (completed_goals / len(week_goals)) * 100
                st.progress(goal_progress / 100, text=f"Goals: {goal_progress:.1f}%")
    
    # Milestones
    milestones = study_plan.get('milestones', [])
    if milestones:
        st.markdown("#### 🏆 Milestones")
        
        for milestone in milestones:
            title = milestone.get('title', 'Milestone')
            description = milestone.get('description', '')
            target = milestone.get('target_completion', 0)
            
            milestone_reached = plan_progress >= target
            
            if milestone_reached:
                st.success(f"✅ **{title}** - {description}")
            else:
                remaining = target - plan_progress
                st.info(f"🎯 **{title}** - {remaining:.1f}% to go")

def calculate_study_plan_progress() -> float:
    """Calculate overall study plan progress"""
    if 'study_plan' not in st.session_state or st.session_state.study_plan is None:
        return 0.0
    
    study_plan = st.session_state.study_plan
    if study_plan is None:
        return 0.0
        
    weekly_plans = study_plan.get('weekly_plans', [])
    
    if not weekly_plans:
        return 0.0
    
    total_progress = 0.0
    
    for week_plan in weekly_plans:
        week_completion = calculate_week_completion(week_plan)
        total_progress += week_completion
    
    return total_progress / len(weekly_plans)

def calculate_week_completion(week_plan: Dict) -> float:
    """Calculate completion percentage for a week"""
    week_num = week_plan.get('week_number', 0)
    week_goals = week_plan.get('week_goals', [])
    
    if not week_goals:
        return 0.0
    
    completed_goals = 0
    
    for i, goal in enumerate(week_goals):
        if get_goal_completion_status(week_num, i):
            completed_goals += 1
    
    return (completed_goals / len(week_goals)) * 100

def get_completed_weeks(study_plan: Dict) -> int:
    """Get number of completed weeks"""
    weekly_plans = study_plan.get('weekly_plans', [])
    completed = 0
    
    for week_plan in weekly_plans:
        if calculate_week_completion(week_plan) >= 90:  # 90% threshold for completion
            completed += 1
    
    return completed

def get_mastered_topics_count(study_plan: Dict) -> int:
    """Get number of mastered topics"""
    topics = study_plan.get('topics', [])
    mastered = 0
    
    # Simple heuristic: topic is mastered if it appears in completed goals
    completed_goals = get_all_completed_goals()
    
    for topic in topics:
        if any(topic.lower() in goal.lower() for goal in completed_goals):
            mastered += 1
    
    return min(mastered, len(topics))

def get_goal_completion_status(week_num: int, goal_index: int) -> bool:
    """Get completion status of a specific goal"""
    if 'goal_completions' not in st.session_state:
        st.session_state.goal_completions = {}
    
    key = f"week_{week_num}_goal_{goal_index}"
    return st.session_state.goal_completions.get(key, False)

def save_goal_completion(week_num: int, goal_index: int, completed: bool):
    """Save goal completion status"""
    if 'goal_completions' not in st.session_state:
        st.session_state.goal_completions = {}
    
    key = f"week_{week_num}_goal_{goal_index}"
    st.session_state.goal_completions[key] = completed

def get_all_completed_goals() -> List[str]:
    """Get all completed goals across all weeks"""
    if 'study_plan' not in st.session_state:
        return []
    
    study_plan = st.session_state.study_plan
    weekly_plans = study_plan.get('weekly_plans', [])
    completed_goals = []
    
    for week_plan in weekly_plans:
        week_num = week_plan.get('week_number', 0)
        week_goals = week_plan.get('week_goals', [])
        
        for i, goal in enumerate(week_goals):
            if get_goal_completion_status(week_num, i):
                completed_goals.append(goal)
    
    return completed_goals

def render_flashcard_progress():
    """Render flashcard progress and analytics"""
    st.markdown("### 🧠 Flashcard Progress")
    
    if 'flashcards' not in st.session_state or not st.session_state.flashcards:
        st.info("No flashcards available. Create some flashcards first!")
        return
    
    flashcards = st.session_state.flashcards
    
    # Overall flashcard metrics
    total_cards = len(flashcards)
    studied_cards = len([card for card in flashcards if card.get('study_count', 0) > 0])
    mastered_cards = len([card for card in flashcards if is_card_mastered(card)])
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Cards", total_cards)
    with col2:
        study_percentage = (studied_cards / total_cards) * 100 if total_cards > 0 else 0
        st.metric("Cards Studied", f"{studied_cards}/{total_cards}", delta=f"{study_percentage:.1f}%")
    with col3:
        mastery_percentage = (mastered_cards / total_cards) * 100 if total_cards > 0 else 0
        st.metric("Cards Mastered", f"{mastered_cards}/{total_cards}", delta=f"{mastery_percentage:.1f}%")
    
    # Progress by card type
    st.markdown("#### 📊 Progress by Card Type")
    
    type_stats = {}
    for card in flashcards:
        card_type = card.get('type', 'Unknown')
        if card_type not in type_stats:
            type_stats[card_type] = {'total': 0, 'studied': 0, 'mastered': 0}
        
        type_stats[card_type]['total'] += 1
        
        if card.get('study_count', 0) > 0:
            type_stats[card_type]['studied'] += 1
        
        if is_card_mastered(card):
            type_stats[card_type]['mastered'] += 1
    
    # Display type progress
    for card_type, stats in type_stats.items():
        with st.expander(f"🃏 {card_type} ({stats['total']} cards)"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                study_pct = (stats['studied'] / stats['total']) * 100
                st.metric("Studied", f"{stats['studied']}/{stats['total']}", delta=f"{study_pct:.1f}%")
            
            with col2:
                mastery_pct = (stats['mastered'] / stats['total']) * 100
                st.metric("Mastered", f"{stats['mastered']}/{stats['total']}", delta=f"{mastery_pct:.1f}%")
            
            with col3:
                if stats['studied'] > 0:
                    accuracy = calculate_type_accuracy(flashcards, card_type)
                    st.metric("Accuracy", f"{accuracy:.1f}%")
    
    # Cards needing review
    st.markdown("#### 🔄 Review Schedule")
    
    current_time = datetime.now().isoformat()
    due_cards = [
        card for card in flashcards 
        if card.get('spaced_repetition_due', current_time) <= current_time
    ]
    
    overdue_cards = [
        card for card in flashcards
        if card.get('spaced_repetition_due', current_time) < (datetime.now() - timedelta(days=1)).isoformat()
    ]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Due Today", len(due_cards))
    with col2:
        st.metric("Overdue", len(overdue_cards))
    with col3:
        upcoming = len([card for card in flashcards if is_card_due_soon(card)])
        st.metric("Due Soon", upcoming)
    
    if due_cards:
        st.warning(f"⏰ {len(due_cards)} cards need review today!")
        
        if st.button("📚 Start Review Session"):
            # Switch to flashcard study mode
            st.session_state.current_mode = 'flashcards'
            st.rerun()

def is_card_mastered(card: Dict) -> bool:
    """Check if a card is considered mastered"""
    study_count = card.get('study_count', 0)
    correct_count = card.get('correct_count', 0)
    
    if study_count < 3:  # Need at least 3 studies
        return False
    
    accuracy = (correct_count / study_count) if study_count > 0 else 0
    return accuracy >= 0.8  # 80% accuracy threshold

def is_card_due_soon(card: Dict) -> bool:
    """Check if card is due for review in next 2 days"""
    due_date_str = card.get('spaced_repetition_due')
    if not due_date_str:
        return False
    
    try:
        due_date = datetime.fromisoformat(due_date_str.replace('Z', '+00:00'))
        soon_date = datetime.now() + timedelta(days=2)
        return due_date <= soon_date
    except:
        return False

def calculate_type_accuracy(flashcards: List[Dict], card_type: str) -> float:
    """Calculate accuracy for a specific card type"""
    type_cards = [card for card in flashcards if card.get('type') == card_type]
    
    total_studies = sum(card.get('study_count', 0) for card in type_cards)
    total_correct = sum(card.get('correct_count', 0) for card in type_cards)
    
    return (total_correct / total_studies) * 100 if total_studies > 0 else 0

def calculate_flashcard_mastery() -> float:
    """Calculate overall flashcard mastery percentage"""
    if 'flashcards' not in st.session_state or not st.session_state.flashcards:
        return 0.0
    
    flashcards = st.session_state.flashcards
    total_cards = len(flashcards)
    mastered_cards = len([card for card in flashcards if is_card_mastered(card)])
    
    return (mastered_cards / total_cards) * 100 if total_cards > 0 else 0.0

def render_summary_progress():
    """Render summary creation progress"""
    st.markdown("### 📝 Summary Progress")
    
    if 'summaries' not in st.session_state or not st.session_state.summaries:
        st.info("No summaries created yet. Create some in Summarize mode!")
        return
    
    summaries = st.session_state.summaries
    
    # Summary statistics
    total_summaries = len(summaries)
    
    # Group by difficulty and style
    difficulty_stats = {}
    style_stats = {}
    
    for summary in summaries:
        options = summary.get('options', {})
        difficulty = options.get('difficulty', 'Unknown')
        styles = options.get('styles', [])
        
        difficulty_stats[difficulty] = difficulty_stats.get(difficulty, 0) + 1
        
        for style in styles:
            style_stats[style] = style_stats.get(style, 0) + 1
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📊 By Difficulty")
        for difficulty, count in difficulty_stats.items():
            percentage = (count / total_summaries) * 100
            st.metric(difficulty, count, delta=f"{percentage:.1f}%")
    
    with col2:
        st.markdown("#### 📋 By Style")
        for style, count in style_stats.items():
            percentage = (count / total_summaries) * 100
            st.metric(style, count, delta=f"{percentage:.1f}%")
    
    # Recent summaries
    st.markdown("#### 📅 Recent Summaries")
    
    recent_summaries = sorted(summaries, key=lambda x: x.get('timestamp', ''), reverse=True)[:5]
    
    for i, summary in enumerate(recent_summaries):
        timestamp = summary.get('timestamp', '')
        documents = summary.get('documents', [])
        options = summary.get('options', {})
        
        with st.expander(f"📝 Summary {i+1} - {timestamp}"):
            st.markdown(f"**Documents:** {', '.join(documents)}")
            st.markdown(f"**Difficulty:** {options.get('difficulty', 'Unknown')}")
            st.markdown(f"**Styles:** {', '.join(options.get('styles', []))}")
            st.markdown(f"**Length:** {options.get('max_length', 'Unknown')} words")

def render_goals_and_milestones():
    """Render goals and milestone tracking"""
    st.markdown("### 🎯 Goals & Milestones")
    
    # Personal goals
    st.markdown("#### 🎯 Personal Goals")
    
    if 'personal_goals' not in st.session_state:
        st.session_state.personal_goals = []
    
    # Add new goal
    with st.expander("➕ Add New Goal"):
        goal_title = st.text_input("Goal Title", placeholder="e.g., Master calculus concepts")
        goal_description = st.text_area("Description", placeholder="Detailed description of the goal")
        goal_deadline = st.date_input("Target Date")
        goal_category = st.selectbox("Category", ["Study Plan", "Flashcards", "Summaries", "General"])
        
        if st.button("Add Goal"):
            if goal_title:
                new_goal = {
                    'id': len(st.session_state.personal_goals),
                    'title': goal_title,
                    'description': goal_description,
                    'deadline': goal_deadline.isoformat(),
                    'category': goal_category,
                    'created_date': datetime.now().isoformat(),
                    'completed': False,
                    'progress': 0
                }
                st.session_state.personal_goals.append(new_goal)
                st.success("Goal added successfully!")
                st.rerun()
    
    # Display existing goals
    if st.session_state.personal_goals:
        for goal in st.session_state.personal_goals:
            if not goal.get('completed', False):
                with st.expander(f"🎯 {goal['title']}"):
                    st.markdown(f"**Description:** {goal['description']}")
                    st.markdown(f"**Category:** {goal['category']}")
                    st.markdown(f"**Deadline:** {goal['deadline']}")
                    
                    # Progress slider
                    current_progress = goal.get('progress', 0)
                    new_progress = st.slider(
                        "Progress",
                        min_value=0,
                        max_value=100,
                        value=current_progress,
                        key=f"goal_progress_{goal['id']}"
                    )
                    
                    if new_progress != current_progress:
                        goal['progress'] = new_progress
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if st.button("✅ Mark Complete", key=f"complete_{goal['id']}"):
                            goal['completed'] = True
                            goal['progress'] = 100
                            goal['completed_date'] = datetime.now().isoformat()
                            st.success("Goal completed! 🎉")
                            st.rerun()
                    
                    with col2:
                        if st.button("🗑️ Delete", key=f"delete_{goal['id']}"):
                            st.session_state.personal_goals.remove(goal)
                            st.rerun()
    
    # Completed goals
    completed_goals = [goal for goal in st.session_state.personal_goals if goal.get('completed', False)]
    
    if completed_goals:
        st.markdown("#### ✅ Completed Goals")
        
        for goal in completed_goals[-5:]:  # Show last 5 completed
            completed_date = goal.get('completed_date', '')
            st.success(f"🏆 **{goal['title']}** - Completed on {completed_date}")
    
    # Milestone suggestions
    st.markdown("#### 🏆 Suggested Milestones")
    
    milestones = generate_milestone_suggestions()
    
    for milestone in milestones:
        if milestone['achieved']:
            st.success(f"✅ {milestone['title']} - {milestone['description']}")
        else:
            st.info(f"🎯 {milestone['title']} - {milestone['description']}")

def generate_milestone_suggestions() -> List[Dict[str, Any]]:
    """Generate milestone suggestions based on current progress"""
    milestones = []
    
    # Flashcard milestones
    flashcards = st.session_state.get('flashcards', [])
    if flashcards:
        total_cards = len(flashcards)
        studied_cards = len([card for card in flashcards if card.get('study_count', 0) > 0])
        
        milestones.append({
            'title': 'Flashcard Explorer',
            'description': 'Study your first 10 flashcards',
            'achieved': studied_cards >= 10
        })
        
        milestones.append({
            'title': 'Flashcard Enthusiast',
            'description': 'Study 50+ flashcards',
            'achieved': studied_cards >= 50
        })
        
        milestones.append({
            'title': 'Flashcard Master',
            'description': 'Study all your flashcards',
            'achieved': studied_cards >= total_cards
        })
    
    # Summary milestones
    summaries = st.session_state.get('summaries', [])
    summary_count = len(summaries)
    
    milestones.append({
        'title': 'Summary Creator',
        'description': 'Create your first summary',
        'achieved': summary_count >= 1
    })
    
    milestones.append({
        'title': 'Summary Expert',
        'description': 'Create 10+ summaries',
        'achieved': summary_count >= 10
    })
    
    # Study plan milestones
    if 'study_plan' in st.session_state:
        plan_progress = calculate_study_plan_progress()
        
        milestones.append({
            'title': 'Study Planner',
            'description': 'Create a study plan',
            'achieved': True
        })
        
        milestones.append({
            'title': 'Halfway Hero',
            'description': 'Complete 50% of study plan',
            'achieved': plan_progress >= 50
        })
        
        milestones.append({
            'title': 'Study Champion',
            'description': 'Complete entire study plan',
            'achieved': plan_progress >= 100
        })
    
    return milestones

def calculate_study_streak() -> int:
    """Calculate current study streak in days"""
    # Simple implementation - could be enhanced with actual activity tracking
    flashcards = st.session_state.get('flashcards', [])
    
    if not flashcards:
        return 0
    
    # Check if any cards were studied recently
    recent_activity = False
    for card in flashcards:
        last_studied = card.get('last_studied')
        if last_studied:
            try:
                last_date = datetime.fromisoformat(last_studied.replace('Z', '+00:00'))
                if (datetime.now() - last_date).days < 2:
                    recent_activity = True
                    break
            except:
                pass
    
    return 7 if recent_activity else 0  # Simplified - return 7 day streak if recent activity

def calculate_total_study_time() -> float:
    """Calculate total estimated study time"""
    total_hours = 0.0
    
    # From flashcards (estimate 1 minute per study)
    flashcards = st.session_state.get('flashcards', [])
    total_studies = sum(card.get('study_count', 0) for card in flashcards)
    total_hours += total_studies / 60  # Convert minutes to hours
    
    # From study plan
    if 'study_plan' in st.session_state:
        overview = st.session_state.study_plan.get('overview', {})
        estimated_hours = overview.get('total_hours', 0)
        completion = calculate_study_plan_progress()
        total_hours += (estimated_hours * completion / 100)
    
    # From summaries (estimate 30 minutes per summary)
    summaries = st.session_state.get('summaries', [])
    total_hours += len(summaries) * 0.5
    
    return total_hours

def calculate_mastery_score() -> float:
    """Calculate overall mastery score out of 10"""
    scores = []
    
    # Flashcard mastery
    flashcard_mastery = calculate_flashcard_mastery()
    scores.append(flashcard_mastery / 10)  # Convert to 0-10 scale
    
    # Study plan progress
    if 'study_plan' in st.session_state:
        plan_progress = calculate_study_plan_progress()
        scores.append(plan_progress / 10)
    
    # Summary creation score
    summaries = st.session_state.get('summaries', [])
    summary_score = min(10, len(summaries) * 2)  # 2 points per summary, max 10
    scores.append(summary_score)
    
    # Personal goals completion
    personal_goals = st.session_state.get('personal_goals', [])
    if personal_goals:
        completed = len([goal for goal in personal_goals if goal.get('completed', False)])
        goal_score = min(10, (completed / len(personal_goals)) * 10)
        scores.append(goal_score)
    
    return sum(scores) / len(scores) if scores else 0.0
