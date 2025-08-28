import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import json
from datetime import datetime
import requests
import os
from urllib.parse import parse_qs, urlparse

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import our enhanced modules
from core.environment import AdaptiveAssessmentEnv
from core.agent import RLAssessmentAgent, AdaptiveStrategy
from data.questions import get_adaptive_question, _question_manager, get_question_statistics

# Page configuration
st.set_page_config(
    page_title="🎯 Adaptive Assessment System",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    
    .question-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #e9ecef;
        margin: 1rem 0;
    }
    
    .success-message {
        background: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #c3e6cb;
    }
    
    .warning-message {
        background: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #ffeaa7;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def initialize_session_state():
    """Initialize all session state variables"""
    defaults = {
        "initialized": False,
        "answer_confirmed": False,
        "show_results": False,
        "current_question": None,
        "selected_answer": None,
        "assessment_complete": False,
        "show_analytics": False,
        "agent_type": "main",
        "adaptation_strategy": "rl_based",
        "env": None,
        "agent": None
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

# Sidebar configuration
def render_sidebar():
    """Render the sidebar with configuration options"""
    st.sidebar.title("⚙️ Assessment Settings")
    
    # Track selection
    available_tracks = _question_manager.generator.get_available_tracks()
    track_descriptions = {
        "web": "🌐 Web Development",
        "ai": "🤖 Artificial Intelligence",
        "cyber": "🔐 Cybersecurity",
        "data": "📊 Data Science",
        "mobile": "📱 Mobile Development",
        "devops": "☁️ DevOps & Cloud"
    }
    
    selected_track = st.sidebar.selectbox(
        "chosse track :",
        options=available_tracks,
        format_func=lambda x: track_descriptions.get(x, x.title()),
        key="track_selector"
    )
    
    # Agent configuration
    st.sidebar.subheader("🤖 Agent Settings")
    
    agent_type = st.sidebar.selectbox(
        "kind of agent :",
        options=["main", "conservative", "aggressive", "ensemble"],
        format_func=lambda x: {
            "main": "🎯 main",
            "conservative": "🛡️ conservative", 
            "aggressive": "⚡ aggressive",
            "ensemble": "🎭 ensemble"
        }.get(x, x)
    )
    
    adaptation_strategy = st.sidebar.selectbox(
        " strategy:",
        options=["rl_based", "conservative", "aggressive", "ability_based"],
        format_func=lambda x: {
            "rl_based": "🧠  rl_based",
            "conservative": "🐌 conservative",
            "aggressive": "🚀 aggressive", 
            "ability_based": "📈 ability_based "
        }.get(x, x)
    )
    
    # Assessment parameters
    st.sidebar.subheader("📋 Assessment Parameters")
    
    max_questions = st.sidebar.slider(
        "maximum of questions  :",
        min_value=5,
        max_value=20,
        value=10,
        help="Maximum number of questions in the assessment"
    )
    
    confidence_threshold = st.sidebar.slider(
        "Confidence :",
        min_value=0.5,
        max_value=0.95,
        value=0.8,
        step=0.05,
        help="Confidence threshold for early termination"
    )
    
    return selected_track, agent_type, adaptation_strategy, max_questions, confidence_threshold

# Analytics and visualizations
def render_analytics():
    """Render analytics dashboard"""
    if not st.session_state.get("env") or not st.session_state.env.question_history:
        st.warning("no data")
        return
    
    env = st.session_state.env
    agent = st.session_state.agent
    
    st.header("📊 Analytics Dashboard")
    
    # Performance metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_questions = len(env.question_history)
        st.metric(" total questions", total_questions)
    
    with col2:
        correct_answers = sum(1 for q in env.question_history if q['is_correct'])
        accuracy = correct_answers / total_questions if total_questions > 0 else 0
        st.metric("accuracy", f"{accuracy:.1%}")
    
    with col3:
        st.metric("ability ", f"{env.student_ability:.1%}")
    
    with col4:
        st.metric("confident score ", f"{env.confidence_score:.1%}")
    
    # Progress visualization
    if len(env.performance_history) > 1:
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Ability Progression", "Performance by Level", 
                          "Confidence Over Time", "Question Difficulty"),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"type": "domain"}]]
        )
        
        # Ability progression
        questions = [p['question_number'] for p in env.performance_history]
        abilities = [p['ability'] for p in env.performance_history]
        
        fig.add_trace(
            go.Scatter(x=questions, y=abilities, mode='lines+markers',
                      name='Student Ability', line=dict(color='blue')),
            row=1, col=1
        )
        
        # Performance by level
        level_data = {}
        for q in env.question_history:
            level = q['level']
            if level not in level_data:
                level_data[level] = {'correct': 0, 'total': 0}
            level_data[level]['total'] += 1
            if q['is_correct']:
                level_data[level]['correct'] += 1
        
        levels = list(level_data.keys())
        accuracies = [level_data[l]['correct'] / level_data[l]['total'] for l in levels]
        
        fig.add_trace(
            go.Bar(x=[f"Level {l}" for l in levels], y=accuracies,
                   name='Accuracy by Level', marker_color='green'),
            row=1, col=2
        )
        
        # Confidence over time
        confidences = [p['confidence'] for p in env.performance_history]
        
        fig.add_trace(
            go.Scatter(x=questions, y=confidences, mode='lines+markers',
                      name='Confidence', line=dict(color='orange')),
            row=2, col=1
        )
        
        # Question difficulty distribution
        difficulty_counts = [0, 0, 0]  # Easy, Medium, Hard
        for q in env.question_history:
            difficulty_counts[q['level'] - 1] += 1
        
        fig.add_trace(
            go.Pie(labels=['Easy', 'Medium', 'Hard'], values=difficulty_counts,
                   name='Question Distribution'),
            row=2, col=2
        )
        
        fig.update_layout(height=600, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
    
    # Agent performance
    st.subheader("🤖 Agent Performance")
    agent_metrics = agent.get_performance_metrics()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.json(agent_metrics)
    
    with col2:
        # Q-table visualization
        q_summary = agent.get_q_table_summary()
        if q_summary:
            q_df = pd.DataFrame(q_summary).T
            st.subheader("Q-Table Values")
            st.dataframe(q_df)
    
    # Detailed question history
    st.subheader("📝 Question History")
    
    history_data = []
    for i, q in enumerate(env.question_history, 1):
        history_data.append({
            'Question #': i,
            'Level': q['level'],
            'Question': q['question']['text'][:50] + "...",
            'Your Answer': q['answer'],
            'Correct Answer': q['question']['correct_answer'],
            'Result': "✅" if q['is_correct'] else "❌",
            'Ability After': f"{q['student_ability_after']:.2%}"
        })
    
    if history_data:
        df = pd.DataFrame(history_data)
        st.dataframe(df, use_container_width=True)

# Question rendering
def render_question():
    """Render the current question"""
    if not st.session_state.current_question:
        st.error("error")
        return
    
    q = st.session_state.current_question
    env = st.session_state.env
    
    # Question header
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.markdown(f"###  question number {env.total_questions_asked}")
    
    with col2:
        level_emoji = {1: "🟢", 2: "🟡", 3: "🔴"}
        level_name = {1: "easy", 2: "medium", 3: "hard"}
        st.markdown(f"**levels:** {level_emoji.get(env.current_level, '⚪')} {level_name.get(env.current_level, ' uncertained')}")
    
    with col3:
        st.markdown(f"**ability:** {env.student_ability:.1%}")
    
    # Question content
    st.markdown(f"""
    <div class="question-card">
        <h4>{q['text']}</h4>
    </div>
    """, unsafe_allow_html=True)
    
    # Answer options
    if "selected_answer" not in st.session_state:
        st.session_state.selected_answer = None
    
    st.session_state.selected_answer = st.radio(
        " choesse answer :",
        q["options"],
        key=f"question_{env.total_questions_asked}",
        index=None
    )
    
    # Action buttons
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        confirm_disabled = (st.session_state.selected_answer is None or 
                          st.session_state.answer_confirmed)
        
        if st.button("✅ confirm ", disabled=confirm_disabled):
            st.session_state.answer_confirmed = True
            st.rerun()
    
    with col2:
        if st.session_state.answer_confirmed:
            if st.button("➡️  next question"):
                process_answer()
    
    with col3:
        if st.session_state.answer_confirmed:
            # Show correct answer
            is_correct = q['correct_answer'] == st.session_state.selected_answer
            if is_correct:
                st.success("🎉 correct !")
            else:
                st.error(f"❌ wrong ,the correct is  : {q['correct_answer']}")
            
            # Show explanation if available
            if 'explanation' in q:
                st.info(f"💡 {q['explanation']}")

def process_answer():
    """Process the student's answer and update the system"""
    env = st.session_state.env
    agent = st.session_state.agent
    q = st.session_state.current_question
    answer = st.session_state.selected_answer
    
    # Get current state
    current_state = agent.get_state()
    
    # Submit answer to environment
    reward, done = env.submit_answer(q, answer)
    
    # Get next state
    next_state = agent.get_state()
    
    # Choose action based on strategy
    if st.session_state.adaptation_strategy == "rl_based":
        action = agent.choose_action(next_state)
    elif st.session_state.adaptation_strategy == "conservative":
        action = AdaptiveStrategy.conservative_strategy(next_state)
    elif st.session_state.adaptation_strategy == "aggressive":
        action = AdaptiveStrategy.aggressive_strategy(next_state)
    elif st.session_state.adaptation_strategy == "ability_based":
        action = AdaptiveStrategy.ability_based_strategy(next_state)
    else:
        action = "auto"
    
    # Update Q-table if using RL agent
    if st.session_state.adaptation_strategy == "rl_based":
        agent.update_q_table(current_state, action, reward, next_state)
    
    # Adjust difficulty
    agent.adjust_difficulty(action)
    
    # Reset for next question
    st.session_state.answer_confirmed = False
    st.session_state.selected_answer = None
    
    if done:
        st.session_state.assessment_complete = True
        st.session_state.show_results = True
    else:
        # Get next question
        next_question = env.get_question()
        if next_question:
            st.session_state.current_question = next_question
        else:
            st.session_state.assessment_complete = True
            st.session_state.show_results = True
    
    st.rerun()

def render_results():
    """Render the final results and summary"""
    env = st.session_state.env
    agent = st.session_state.agent
    
    st.markdown("""
    <div class="success-message">
        <h2 style="margin: 0;">🎉 exam is done sucessfully   !</h2>
        <p style="margin: 0.5rem 0 0 0;"> your results  :</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Get assessment summary
    summary = env.get_assessment_summary()
    
    # Main metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "final score ",
            f"{summary['correct_answers']}/{summary['total_questions']}",
            f"{summary['final_score']:.1%}"
        )
    
    with col2:
        st.metric(
            " ability",
            f"{summary['final_ability']:.1%}",
        )
    
    with col3:
        st.metric(
            "confident ",
            f"{summary['confidence_score']:.1%}",
        )
    
    with col4:
        recommended_level = summary['recommended_level']
        level_names = {1: "beginner", 2: "intermediate", 3: "advanced"}
        st.metric(
            "  recommded level",
            level_names.get(recommended_level, "uncertained ")
        )
    
    # Performance by level
    st.subheader("📊  Performance by level ")
    
    if summary['level_performance']:
        level_data = []
        for level, perf in summary['level_performance'].items():
            level_data.append({
                'Level': f"Level {level}",
                ' questions': perf['questions'],
                ' correct answers': perf['correct'],
                ' accuracy': f"{perf['accuracy']:.1%}"
            })
        
        df = pd.DataFrame(level_data)
        st.dataframe(df, use_container_width=True)
        
        # Visualization
        fig = px.bar(
            df, 
            x='Level', 
            y=' accuracy',
            title=" Performance by level",
            color=' accuracy',
            color_continuous_scale='RdYlGn'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed recommendations
    st.subheader("💡 recommendations and next step ")

    
    # Restart option
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 Restart exam ", type="primary"):
            # Reset session state
            for key in list(st.session_state.keys()):
                if key not in ['track_selector']:  # Keep track selection
                    del st.session_state[key]
            st.rerun()
    
    with col2:
        if st.button("📊 show analytics  "):
            st.session_state.show_analytics = True
            st.rerun()

def main():
    """Main application function"""
    initialize_session_state()
    # Header
    st.markdown("""
    <div class="main-header">
        <h1> smart adaptive learning system </h1>
        <p>evaluate your technical skills </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    selected_track, agent_type, adaptation_strategy, max_questions, confidence_threshold = render_sidebar()
    
    # Update session state with sidebar values
    st.session_state.agent_type = agent_type
    st.session_state.adaptation_strategy = adaptation_strategy
    
    # Navigation
    if st.session_state.show_analytics:
        if st.button("back to test"):
            st.session_state.show_analytics = False
            st.rerun()
        render_analytics()
        return
    
    # Main content based on state
    if not st.session_state.initialized:
        # Welcome screen
        st.header("🚀 start your carrer  ")
        
        # Track statistics
        track_stats = get_question_statistics(selected_track)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(f"📊 statisc {selected_track.upper()}")
            st.write(f"** total questions:** {track_stats['total_questions']}")
            
            for level, info in track_stats['levels'].items():
                st.write(f"**{info['difficulty']}:** {info['count']} question")
        
        with col2:
            st.subheader("ℹ️ test info")
            st.write(f"** max questions:** {max_questions}")

        
        if st.button("🎯 start test", type="primary", use_container_width=True):
        # Initialize environment and agent
            st.session_state.env = AdaptiveAssessmentEnv(
                track=selected_track
            )

            st.session_state.agent = RLAssessmentAgent(
                env=st.session_state.env,
            )
            
            st.session_state.initialized = True
            st.success("✅ Assessment initialized! Ready to start.")
            st.session_state.env = AdaptiveAssessmentEnv(track=selected_track)
            st.session_state.env.max_questions = max_questions
            st.session_state.env.confidence_threshold = confidence_threshold
            
            # Initialize agent based on type
            if agent_type == "ensemble":
                st.session_state.agent = RLAssessmentAgent(st.session_state.env)
            
            # Get first question
            first_question = st.session_state.env.get_question()
            if first_question:
                st.session_state.current_question = first_question
                st.session_state.initialized = True
                st.rerun()
            else:
                st.error("❌ error")
    
    elif st.session_state.show_results:
        
        final_results = {
            'student_id': st.session_state.get('student_id'),
            'app_type': 'assessment',
            'score': env.get_final_score(),
            'ability_level': env.student_ability,
            'questions_answered': env.total_questions_asked,
            'time_spent': env.get_time_spent(),
            'details': env.get_assessment_summary()
        }
        
        render_results()    
    else:
        # Main assessment interface
        if st.session_state.current_question:
            render_question()
        else:
            st.error("❌ error")
    
    # Progress indicator
    if st.session_state.initialized and not st.session_state.show_results:
        env = st.session_state.env
        progress = min(env.total_questions_asked / env.max_questions, 1.0)
        
        st.sidebar.markdown("---")
        st.sidebar.subheader("📈 progress")
        st.sidebar.progress(progress)
        st.sidebar.write(f"question  {env.total_questions_asked} من {env.max_questions}")
        
        # Real-time metrics
        if env.question_history:
            correct = sum(1 for q in env.question_history if q['is_correct'])
            accuracy = correct / len(env.question_history)
            
            st.sidebar.metric(" accuracy", f"{accuracy:.1%}")
            st.sidebar.metric(" student ability", f"{env.student_ability:.1%}")
            st.sidebar.metric(" confidence score", f"{env.confidence_score:.1%}")

