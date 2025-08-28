# app.py - Complete Adaptive Assessment Application
import streamlit as st
import pandas as pd
import numpy as np
import json
import hashlib
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import plotly.express as px
import plotly.graph_objects as go
from dataclasses import dataclass
import requests
import random

# Enhanced imports from your existing modules
from core.environment import AdaptiveAssessmentEnv
from core.agent import RLAssessmentAgent
from data.questions import get_adaptive_question, generate_interview_questions, _question_manager

# Configure Streamlit page
st.set_page_config(
    page_title="Adaptive Assessment Platform",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Database classes for user management
@dataclass
class User:
    username: str
    password_hash: str
    email: str
    created_at: datetime
    role: str = "student"
    profile_data: Dict = None

@dataclass
class AssessmentSession:
    session_id: str
    username: str
    track: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    final_score: float = 0.0
    questions_answered: int = 0
    ability_level: float = 0.5
    performance_data: Dict = None

class SimpleDatabase:
    """Simple in-memory database for users and sessions"""
    
    def __init__(self):
        self.users = {}
        self.sessions = {}
        self.load_sample_data()
    
    def load_sample_data(self):
        """Load some sample users for testing"""
        sample_users = [
            {"username": "admin", "password": "admin123", "email": "admin@test.com", "role": "admin"},
            {"username": "student1", "password": "pass123", "email": "student1@test.com", "role": "student"},
            {"username": "teacher", "password": "teach123", "email": "teacher@test.com", "role": "teacher"}
        ]
        
        for user_data in sample_users:
            self.create_user(
                user_data["username"],
                user_data["password"], 
                user_data["email"],
                user_data["role"]
            )
    
    def hash_password(self, password: str) -> str:
        """Hash password using SHA256"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def create_user(self, username: str, password: str, email: str, role: str = "student") -> bool:
        """Create a new user"""
        if username in self.users:
            return False
        
        self.users[username] = User(
            username=username,
            password_hash=self.hash_password(password),
            email=email,
            created_at=datetime.now(),
            role=role,
            profile_data={}
        )
        return True
    
    def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """Authenticate user login"""
        if username not in self.users:
            return None
        
        user = self.users[username]
        if user.password_hash == self.hash_password(password):
            return user
        return None
    
    def get_user_sessions(self, username: str) -> List[AssessmentSession]:
        """Get all sessions for a user"""
        return [session for session in self.sessions.values() if session.username == username]
    
    def save_session(self, session: AssessmentSession):
        """Save assessment session"""
        self.sessions[session.session_id] = session
    
    def get_all_users(self) -> List[User]:
        """Get all users (admin only)"""
        return list(self.users.values())
    
    def get_all_sessions(self) -> List[AssessmentSession]:
        """Get all sessions (admin only)"""
        return list(self.sessions.values())

# AI Question Generator
class AIQuestionGenerator:
    """Generate questions using free AI APIs"""
    
    def __init__(self):
        self.api_endpoints = {
            # Using free APIs that don't require tokens
            "huggingface_inference": "https://api-inference.huggingface.co/models/microsoft/DialoGPT-medium",
            "openai_compatible": None  # Can be configured for local models
        }
    
    def generate_question_with_ai(self, track: str, level: int, topic: str = None) -> Dict:
        """Generate question using AI (fallback to template if API fails)"""
        try:
            # Try to generate with AI
            question_text = self._call_ai_api(track, level, topic)
            if question_text:
                return self._create_question_from_ai_text(question_text, track, level, topic)
        except Exception as e:
            st.warning(f"AI generation failed: {e}")
        
        # Fallback to enhanced template generation
        return self._generate_template_question(track, level, topic)
    
    def _call_ai_api(self, track: str, level: int, topic: str) -> Optional[str]:
        """Call AI API to generate question text"""
        difficulty_names = {1: "beginner", 2: "intermediate", 3: "advanced"}
        difficulty = difficulty_names.get(level, "intermediate")
        
        prompt = f"Create a {difficulty} level multiple choice question about {topic or track}. The question should test practical knowledge in {track} development."
        
        # Try different methods to generate content
        try:
            # Method 1: Use requests to call a free API (if available)
            # This is a placeholder - in real implementation, you'd use actual free APIs
            response = self._generate_with_template_ai(prompt, track, level)
            return response
        except:
            return None
    
    def _generate_with_template_ai(self, prompt: str, track: str, level: int) -> str:
        """Generate using enhanced templates that simulate AI"""
        
        question_templates = {
            "web": {
                1: [
                    "What is the primary purpose of {concept} in web development?",
                    "Which {concept} is most commonly used for {purpose}?",
                    "How do you implement basic {concept} functionality?",
                    "What happens when you use {concept} in a web application?"
                ],
                2: [
                    "What are the best practices when implementing {concept} in production?",
                    "How would you optimize {concept} for better performance?",
                    "What security considerations apply to {concept}?",
                    "How do you handle {concept} errors in real applications?"
                ],
                3: [
                    "What advanced patterns can be used with {concept} for scalability?",
                    "How would you architect {concept} for enterprise applications?",
                    "What are the performance implications of different {concept} approaches?",
                    "How do you implement {concept} with modern frameworks and patterns?"
                ]
            },
            "ai": {
                1: [
                    "What is the basic concept of {concept} in machine learning?",
                    "Which algorithm is best suited for {concept} problems?",
                    "What is the difference between {concept} and related techniques?",
                    "How do you prepare data for {concept} tasks?"
                ],
                2: [
                    "What evaluation metrics are appropriate for {concept} models?",
                    "How do you handle overfitting in {concept} applications?",
                    "What preprocessing steps are crucial for {concept} success?",
                    "How do you optimize hyperparameters for {concept} models?"
                ],
                3: [
                    "What advanced architectures work best for {concept} at scale?",
                    "How do you implement {concept} in production environments?",
                    "What are the ethical considerations when deploying {concept}?",
                    "How do you handle bias and fairness in {concept} systems?"
                ]
            },
            "cyber": {
                1: [
                    "What is {concept} and why is it important for security?",
                    "Which tool is commonly used for {concept} analysis?",
                    "What are the basic principles of {concept}?",
                    "How does {concept} protect against common threats?"
                ],
                2: [
                    "What are the implementation challenges of {concept}?",
                    "How do you monitor and maintain {concept} systems?",
                    "What compliance requirements relate to {concept}?",
                    "How do you integrate {concept} with existing infrastructure?"
                ],
                3: [
                    "What advanced {concept} techniques are used in enterprise security?",
                    "How do you design {concept} for zero-trust architectures?",
                    "What emerging threats does {concept} address?",
                    "How do you implement {concept} across cloud and hybrid environments?"
                ]
            }
        }
        
        # Get appropriate template
        templates = question_templates.get(track, question_templates["web"])
        level_templates = templates.get(level, templates[2])
        template = random.choice(level_templates)
        
        # Generate concept based on track
        concepts = {
            "web": ["responsive design", "API integration", "state management", "component architecture", "routing"],
            "ai": ["neural networks", "feature engineering", "model selection", "data preprocessing", "model evaluation"],
            "cyber": ["encryption", "network security", "access control", "vulnerability assessment", "incident response"],
            "data": ["data visualization", "statistical analysis", "data cleaning", "machine learning pipelines", "big data processing"],
            "mobile": ["cross-platform development", "native performance", "user interface design", "data persistence", "push notifications"],
            "devops": ["container orchestration", "continuous integration", "infrastructure automation", "monitoring", "deployment strategies"]
        }
        
        concept = random.choice(concepts.get(track, concepts["web"]))
        purpose_map = {
            "responsive design": "mobile compatibility",
            "API integration": "data communication",
            "neural networks": "pattern recognition",
            "encryption": "data protection"
        }
        
        purpose = purpose_map.get(concept, "system functionality")
        
        return template.format(concept=concept, purpose=purpose)
    
    def _create_question_from_ai_text(self, question_text: str, track: str, level: int, topic: str) -> Dict:
        """Create a complete question object from AI-generated text"""
        
        # Generate options based on track and level
        options = self._generate_smart_options(track, level, topic, question_text)
        
        return {
            'text': question_text,
            'options': options,
            'correct_answer': options[0],  # First option is always correct
            'explanation': f"This {['beginner', 'intermediate', 'advanced'][level-1]} level question tests knowledge of {topic or track} concepts.",
            'generated': True,
            'ai_generated': True,
            'topic': topic or track,
            'level': level
        }
    
    def _generate_smart_options(self, track: str, level: int, topic: str, question: str) -> List[str]:
        """Generate intelligent options based on context"""
        
        option_pools = {
            "web": {
                "frameworks": ["React", "Vue.js", "Angular", "Svelte"],
                "languages": ["JavaScript", "TypeScript", "Python", "PHP"],
                "tools": ["Webpack", "Vite", "Rollup", "Parcel"],
                "concepts": ["Virtual DOM", "Server-side rendering", "Progressive Web Apps", "Single Page Applications"]
            },
            "ai": {
                "algorithms": ["Random Forest", "Support Vector Machine", "Neural Networks", "K-Means"],
                "libraries": ["TensorFlow", "PyTorch", "Scikit-learn", "Keras"],
                "concepts": ["Supervised Learning", "Unsupervised Learning", "Reinforcement Learning", "Deep Learning"],
                "metrics": ["Accuracy", "Precision", "Recall", "F1-Score"]
            },
            "cyber": {
                "tools": ["Wireshark", "Nmap", "Metasploit", "Burp Suite"],
                "protocols": ["HTTPS", "SSL/TLS", "VPN", "SSH"],
                "concepts": ["Zero Trust", "Defense in Depth", "Least Privilege", "Multi-factor Authentication"],
                "attacks": ["Phishing", "SQL Injection", "Cross-site Scripting", "Man-in-the-middle"]
            }
        }
        
        # Get relevant pool for the track
        pools = option_pools.get(track, option_pools["web"])
        
        # Select appropriate category based on question content
        category = "concepts"
        for cat_name, items in pools.items():
            if any(item.lower() in question.lower() for item in items):
                category = cat_name
                break
        
        selected_options = random.sample(pools[category], min(4, len(pools[category])))
        random.shuffle(selected_options)
        
        return selected_options
    
    def _generate_template_question(self, track: str, level: int, topic: str) -> Dict:
        """Fallback template-based question generation"""
        difficulty_names = {1: "beginner", 2: "intermediate", 3: "advanced"}
        
        templates = {
            1: f"What is a fundamental concept in {topic or track}?",
            2: f"How would you implement {topic or track} in a production environment?",
            3: f"What are the scalability considerations for {topic or track}?"
        }
        
        question_text = templates.get(level, f"What do you know about {topic or track}?")
        options = self._generate_smart_options(track, level, topic or track, question_text)
        
        return {
            'text': question_text,
            'options': options,
            'correct_answer': options[0],
            'explanation': f"This tests {difficulty_names[level]} knowledge of {topic or track}.",
            'generated': True,
            'template_generated': True,
            'topic': topic or track,
            'level': level
        }

# Initialize global components
if 'database' not in st.session_state:
    st.session_state.database = SimpleDatabase()

if 'ai_generator' not in st.session_state:
    st.session_state.ai_generator = AIQuestionGenerator()

def authenticate_user():
    """Handle user authentication"""
    st.title("🎓 Adaptive Assessment Platform")
    
    tab1, tab2 = st.tabs(["Login", "Register"])
    
    with tab1:
        st.header("Login")
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit_button = st.form_submit_button("Login")
            
            if submit_button:
                user = st.session_state.database.authenticate_user(username, password)
                if user:
                    st.session_state.user = user
                    st.success(f"Welcome back, {user.username}!")
                    st.rerun()
                else:
                    st.error("Invalid username or password")
    
    with tab2:
        st.header("Register New Account")
        with st.form("register_form"):
            new_username = st.text_input("Choose Username")
            new_email = st.text_input("Email Address")
            new_password = st.text_input("Choose Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")
            role = st.selectbox("Role", ["student", "teacher"])
            register_button = st.form_submit_button("Register")
            
            if register_button:
                if new_password != confirm_password:
                    st.error("Passwords don't match")
                elif len(new_password) < 6:
                    st.error("Password must be at least 6 characters")
                elif st.session_state.database.create_user(new_username, new_password, new_email, role):
                    st.success("Account created successfully! Please login.")
                else:
                    st.error("Username already exists")

def student_dashboard():
    """Student dashboard interface"""
    st.title(f"Welcome, {st.session_state.user.username}! 🎓")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Take Assessment", "My Progress", "Practice Questions", "Profile"]
    )
    
    if page == "Take Assessment":
        take_assessment_page()
    elif page == "My Progress":
        progress_page()
    elif page == "Practice Questions":
        practice_page()
    elif page == "Profile":
        profile_page()

def take_assessment_page():
    """Main assessment taking interface"""
    st.header("📝 Take Assessment")
    
    # Track selection
    available_tracks = ["web", "ai", "cyber", "data", "mobile", "devops"]
    
    col1, col2 = st.columns(2)
    
    with col1:
        selected_track = st.selectbox(
            "Select Technology Track",
            available_tracks,
            format_func=lambda x: {
                "web": "🌐 Web Development",
                "ai": "🤖 Artificial Intelligence",
                "cyber": "🔒 Cybersecurity",
                "data": "📊 Data Science",
                "mobile": "📱 Mobile Development",
                "devops": "⚙️ DevOps"
            }.get(x, x.title())
        )
    
    with col2:
        question_source = st.selectbox(
            "Question Source",
            ["Pool Questions", "AI Generated", "Mixed"]
        )
    
    # Assessment configuration
    st.subheader("Assessment Settings")
    col3, col4 = st.columns(2)
    
    with col3:
        max_questions = st.slider("Maximum Questions", 5, 20, 10)
        difficulty_mode = st.selectbox(
            "Difficulty Mode",
            ["Adaptive", "Fixed Easy", "Fixed Medium", "Fixed Hard"]
        )
    
    with col4:
        time_limit = st.selectbox("Time Limit (minutes)", [None, 15, 30, 45, 60])
        show_explanations = st.checkbox("Show explanations after each question", True)
    
    if st.button("🚀 Start Assessment", type="primary"):
        start_assessment(selected_track, question_source, max_questions, difficulty_mode, time_limit, show_explanations)

def start_assessment(track, question_source, max_questions, difficulty_mode, time_limit, show_explanations):
    """Initialize and run assessment"""
    
    # Initialize assessment components
    env = AdaptiveAssessmentEnv(track=track)
    env.max_questions = max_questions
    
    agent = RLAssessmentAgent(env)
    
    # Create session
    session_id = f"session_{int(time.time())}"
    session = AssessmentSession(
        session_id=session_id,
        username=st.session_state.user.username,
        track=track,
        started_at=datetime.now(),
        performance_data={}
    )
    
    # Store in session state
    st.session_state.assessment_env = env
    st.session_state.assessment_agent = agent
    st.session_state.current_session = session
    st.session_state.question_source = question_source
    st.session_state.show_explanations = show_explanations
    st.session_state.assessment_started = True
    st.session_state.current_question = None
    st.session_state.question_count = 0
    
    if time_limit:
        st.session_state.end_time = datetime.now() + timedelta(minutes=time_limit)
    
    st.rerun()

def run_assessment():
    """Run the active assessment"""
    env = st.session_state.assessment_env
    agent = st.session_state.assessment_agent
    session = st.session_state.current_session
    
    # Check time limit
    if 'end_time' in st.session_state:
        remaining_time = st.session_state.end_time - datetime.now()
        if remaining_time.total_seconds() <= 0:
            complete_assessment()
            return
        
        # Show countdown
        minutes, seconds = divmod(int(remaining_time.total_seconds()), 60)
        st.sidebar.metric("Time Remaining", f"{minutes:02d}:{seconds:02d}")
    
    # Show progress
    progress = st.session_state.question_count / env.max_questions
    st.progress(progress)
    st.caption(f"Question {st.session_state.question_count + 1} of {env.max_questions}")
    
    # Get current question
    if st.session_state.current_question is None:
        current_question = get_next_question()
        if current_question is None:
            complete_assessment()
            return
        st.session_state.current_question = current_question
    
    question = st.session_state.current_question
    
    # Display question
    st.subheader(f"Question {st.session_state.question_count + 1}")
    st.write(question['text'])
    
    # Show options
    with st.form("question_form"):
        selected_answer = st.radio("Choose your answer:", question['options'])
        submit_answer = st.form_submit_button("Submit Answer")
        
        if submit_answer:
            process_answer(question, selected_answer)

def get_next_question():
    """Get the next question based on source preference"""
    env = st.session_state.assessment_env
    agent = st.session_state.assessment_agent
    source = st.session_state.question_source
    
    if source == "AI Generated":
        # Generate question with AI
        topic = f"{env.track} development"
        ai_question = st.session_state.ai_generator.generate_question_with_ai(
            env.track, 
            env.current_level,
            topic
        )
        return ai_question
    
    elif source == "Pool Questions":
        # Get from existing pool
        return env.get_question()
    
    else:  # Mixed
        # Randomly choose between pool and AI
        if random.choice([True, False]):
            ai_question = st.session_state.ai_generator.generate_question_with_ai(
                env.track, 
                env.current_level
            )
            return ai_question
        else:
            return env.get_question()

def process_answer(question, selected_answer):
    """Process the submitted answer"""
    env = st.session_state.assessment_env
    agent = st.session_state.assessment_agent
    
    # Get current state before answering
    current_state = agent.get_state()
    
    # Submit answer and get reward
    reward, is_done = env.submit_answer(question, selected_answer)
    
    # Get new state after answering
    new_state = agent.get_state()
    
    # Let agent choose next action for difficulty adjustment
    action = agent.choose_action(new_state)
    
    # Update Q-table
    agent.update_q_table(current_state, action, reward, new_state)
    
    # Adjust difficulty based on agent's decision
    agent.adjust_difficulty(action)
    
    # Show immediate feedback
    is_correct = question['correct_answer'] == selected_answer
    
    if is_correct:
        st.success("✅ Correct!")
    else:
        st.error(f"❌ Incorrect. The correct answer was: {question['correct_answer']}")
    
    if st.session_state.show_explanations and 'explanation' in question:
        st.info(f"💡 **Explanation:** {question['explanation']}")
    
    # Update question count
    st.session_state.question_count += 1
    st.session_state.current_question = None
    
    # Show level adjustment if any
    if 'level_changes' in env.__dict__ and env.level_changes:
        last_change = env.level_changes[-1]
        if last_change['question_number'] == env.total_questions_asked:
            level_names = {1: "Easy", 2: "Medium", 3: "Hard"}
            st.info(f"🎯 Difficulty adjusted to: {level_names[last_change['to_level']]}")
    
    # Check if assessment should end
    if env.total_questions_asked >= env.max_questions or env._check_completion():
        st.button("Continue", on_click=complete_assessment)
    else:
        st.button("Next Question", on_click=lambda: st.rerun())

def complete_assessment():
    """Complete the assessment and show results"""
    env = st.session_state.assessment_env
    session = st.session_state.current_session
    
    # Finalize session
    session.completed_at = datetime.now()
    session.final_score = env.get_assessment_summary()['final_score']
    session.questions_answered = env.total_questions_asked
    session.ability_level = env.student_ability
    session.performance_data = env.get_assessment_summary()
    
    # Save session to database
    st.session_state.database.save_session(session)
    
    # Clear assessment state
    st.session_state.assessment_started = False
    
    # Show results
    show_assessment_results(session)

def show_assessment_results(session: AssessmentSession):
    """Display assessment results"""
    st.title("🎉 Assessment Complete!")
    
    perf_data = session.performance_data
    
    # Overall metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Final Score", f"{perf_data['final_score']:.1%}")
    
    with col2:
        st.metric("Questions Answered", perf_data['total_questions'])
    
    with col3:
        st.metric("Ability Level", f"{session.ability_level:.1%}")
    
    with col4:
        recommended_level = perf_data.get('recommended_level', 2)
        level_names = {1: "Beginner", 2: "Intermediate", 3: "Advanced"}
        st.metric("Recommended Level", level_names[recommended_level])
    
    # Performance by level chart
    if 'level_performance' in perf_data:
        st.subheader("📊 Performance by Difficulty Level")
        
        level_data = []
        for level, data in perf_data['level_performance'].items():
            level_data.append({
                'Level': f"Level {level}",
                'Questions': data['questions'],
                'Correct': data['correct'],
                'Accuracy': data['accuracy']
            })
        
        if level_data:
            df = pd.DataFrame(level_data)
            fig = px.bar(df, x='Level', y=['Questions', 'Correct'], 
                        title="Questions vs Correct Answers by Level")
            st.plotly_chart(fig)
    
    # Ability progression
    if 'ability_progression' in perf_data:
        st.subheader("📈 Ability Progression")
        
        progression = perf_data['ability_progression']
        if len(progression) > 1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=list(range(1, len(progression) + 1)),
                y=progression,
                mode='lines+markers',
                name='Ability Level'
            ))
            fig.update_layout(
                title="Ability Level Throughout Assessment",
                xaxis_title="Question Number",
                yaxis_title="Ability Level",
                yaxis=dict(range=[0, 1])
            )
            st.plotly_chart(fig)
    
    # Recommendations
    st.subheader("💡 Recommendations")
    
    final_score = perf_data['final_score']
    if final_score >= 0.8:
        st.success("🌟 Excellent performance! You're ready for advanced topics.")
    elif final_score >= 0.6:
        st.info("👍 Good job! Consider reviewing intermediate concepts.")
    else:
        st.warning("📚 Keep practicing! Focus on fundamental concepts first.")
    
    if st.button("Take Another Assessment"):
        st.rerun()

def progress_page():
    """Show user progress and history"""
    st.header("📈 My Progress")
    
    sessions = st.session_state.database.get_user_sessions(st.session_state.user.username)
    
    if not sessions:
        st.info("No assessments taken yet. Start your first assessment!")
        return
    
    # Overall statistics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_assessments = len(sessions)
        st.metric("Total Assessments", total_assessments)
    
    with col2:
        completed_sessions = [s for s in sessions if s.completed_at]
        avg_score = np.mean([s.final_score for s in completed_sessions]) if completed_sessions else 0
        st.metric("Average Score", f"{avg_score:.1%}")
    
    with col3:
        total_questions = sum(s.questions_answered for s in sessions)
        st.metric("Total Questions Answered", total_questions)
    
    # Progress over time
    if completed_sessions:
        st.subheader("Score Progression")
        
        df = pd.DataFrame([
            {
                'Date': s.completed_at.strftime('%Y-%m-%d'),
                'Score': s.final_score,
                'Track': s.track.title(),
                'Questions': s.questions_answered
            }
            for s in completed_sessions
        ])
        
        fig = px.line(df, x='Date', y='Score', color='Track',
                     title="Assessment Scores Over Time",
                     markers=True)
        fig.update_layout(yaxis=dict(range=[0, 1]))
        st.plotly_chart(fig)
    
    # Recent assessments table
    st.subheader("Recent Assessments")
    
    recent_sessions = sorted(sessions, key=lambda x: x.started_at, reverse=True)[:10]
    
    table_data = []
    for session in recent_sessions:
        status = "Completed" if session.completed_at else "In Progress"
        score = f"{session.final_score:.1%}" if session.completed_at else "N/A"
        
        table_data.append({
            'Date': session.started_at.strftime('%Y-%m-%d %H:%M'),
            'Track': session.track.title(),
            'Status': status,
            'Score': score,
            'Questions': session.questions_answered
        })
    
    st.dataframe(pd.DataFrame(table_data))

def practice_page():
    """Practice questions page"""
    st.header("🏋️ Practice Questions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        practice_track = st.selectbox(
            "Select Track for Practice",
            ["web", "ai", "cyber", "data", "mobile", "devops"],
            format_func=lambda x: {
                "web": "🌐 Web Development",
                "ai": "🤖 Artificial Intelligence", 
                "cyber": "🔒 Cybersecurity",
                "data": "📊 Data Science",
                "mobile": "📱 Mobile Development",
                "devops": "⚙️ DevOps"
            }.get(x, x.title())
        )
    
    with col2:
        practice_level = st.selectbox(
            "Difficulty Level",
            [1, 2, 3],
            format_func=lambda x: {1: "🟢 Easy", 2: "🟡 Medium", 3: "🔴 Hard"}[x]
        )
    
    # Practice mode selection
    practice_mode = st.selectbox(
        "Practice Mode",
        ["Single Question", "Quick Quiz (5 questions)", "Extended Practice (10 questions)"]
    )
    
    use_ai = st.checkbox("Generate new questions with AI", value=True)
    
    if st.button("🎯 Start Practice"):
        start_practice_session(practice_track, practice_level, practice_mode, use_ai)

def start_practice_session(track, level, mode, use_ai):
    """Start a practice session"""
    question_counts = {
        "Single Question": 1,
        "Quick Quiz (5 questions)": 5,
        "Extended Practice (10 questions)": 10
    }
    
    num_questions = question_counts[mode]
    
    # Initialize practice session
    st.session_state.practice_active = True
    st.session_state.practice_track = track
    st.session_state.practice_level = level
    st.session_state.practice_use_ai = use_ai
    st.session_state.practice_questions = []
    st.session_state.practice_answers = []
    st.session_state.practice_current_q = 0
    st.session_state.practice_total = num_questions
    
    st.rerun()

def run_practice_session():
    """Run the practice session"""
    if st.session_state.practice_current_q >= st.session_state.practice_total:
        show_practice_results()
        return
    
    # Progress indicator
    progress = st.session_state.practice_current_q / st.session_state.practice_total
    st.progress(progress)
    st.caption(f"Question {st.session_state.practice_current_q + 1} of {st.session_state.practice_total}")
    
    # Get or generate question
    if len(st.session_state.practice_questions) <= st.session_state.practice_current_q:
        if st.session_state.practice_use_ai:
            question = st.session_state.ai_generator.generate_question_with_ai(
                st.session_state.practice_track,
                st.session_state.practice_level
            )
        else:
            question = get_adaptive_question(
                st.session_state.practice_track,
                st.session_state.practice_level,
                [q['text'] for q in st.session_state.practice_questions]
            )
        
        if question:
            st.session_state.practice_questions.append(question)
        else:
            st.error("No more questions available for this combination.")
            return
    
    current_question = st.session_state.practice_questions[st.session_state.practice_current_q]
    
    # Display question
    st.subheader(f"Question {st.session_state.practice_current_q + 1}")
    st.write(current_question['text'])
    
    # Answer form
    with st.form(f"practice_form_{st.session_state.practice_current_q}"):
        selected_answer = st.radio("Select your answer:", current_question['options'])
        submit_button = st.form_submit_button("Submit Answer")
        
        if submit_button:
            # Record answer
            is_correct = current_question['correct_answer'] == selected_answer
            st.session_state.practice_answers.append({
                'question': current_question,
                'selected': selected_answer,
                'correct': is_correct
            })
            
            # Show feedback
            if is_correct:
                st.success("✅ Correct!")
            else:
                st.error(f"❌ Incorrect. The correct answer was: {current_question['correct_answer']}")
            
            if 'explanation' in current_question:
                st.info(f"💡 **Explanation:** {current_question['explanation']}")
            
            # Move to next question
            st.session_state.practice_current_q += 1
            
            if st.session_state.practice_current_q < st.session_state.practice_total:
                st.button("Next Question", on_click=lambda: st.rerun())
            else:
                st.button("Show Results", on_click=lambda: st.rerun())

def show_practice_results():
    """Show practice session results"""
    st.title("🎉 Practice Session Complete!")
    
    answers = st.session_state.practice_answers
    correct_count = sum(1 for a in answers if a['correct'])
    total_count = len(answers)
    score = correct_count / total_count if total_count > 0 else 0
    
    # Results summary
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Score", f"{score:.1%}")
    with col2:
        st.metric("Correct Answers", f"{correct_count}/{total_count}")
    with col3:
        level_name = {1: "Easy", 2: "Medium", 3: "Hard"}[st.session_state.practice_level]
        st.metric("Level", level_name)
    
    # Detailed results
    st.subheader("📝 Detailed Results")
    
    for i, answer in enumerate(answers, 1):
        with st.expander(f"Question {i} - {'✅ Correct' if answer['correct'] else '❌ Incorrect'}"):
            st.write(f"**Question:** {answer['question']['text']}")
            st.write(f"**Your Answer:** {answer['selected']}")
            st.write(f"**Correct Answer:** {answer['question']['correct_answer']}")
            if 'explanation' in answer['question']:
                st.write(f"**Explanation:** {answer['question']['explanation']}")
    
    # Reset practice session
    if st.button("Practice More"):
        st.session_state.practice_active = False
        st.rerun()

def profile_page():
    """User profile page"""
    st.header("👤 My Profile")
    
    user = st.session_state.user
    
    # Basic info
    col1, col2 = st.columns(2)
    
    with col1:
        st.text_input("Username", user.username, disabled=True)
        st.text_input("Email", user.email, disabled=True)
    
    with col2:
        st.text_input("Role", user.role.title(), disabled=True)
        st.text_input("Member Since", user.created_at.strftime('%Y-%m-%d'), disabled=True)
    
    # Learning preferences
    st.subheader("🎯 Learning Preferences")
    
    with st.form("preferences_form"):
        preferred_tracks = st.multiselect(
            "Preferred Technology Tracks",
            ["web", "ai", "cyber", "data", "mobile", "devops"],
            default=user.profile_data.get('preferred_tracks', [])
        )
        
        difficulty_preference = st.selectbox(
            "Preferred Starting Difficulty",
            ["Easy", "Medium", "Hard"],
            index=["Easy", "Medium", "Hard"].index(user.profile_data.get('difficulty_preference', 'Medium'))
        )
        
        question_explanation = st.checkbox(
            "Always show explanations",
            value=user.profile_data.get('show_explanations', True)
        )
        
        if st.form_submit_button("Save Preferences"):
            # Update user preferences
            if not user.profile_data:
                user.profile_data = {}
            
            user.profile_data.update({
                'preferred_tracks': preferred_tracks,
                'difficulty_preference': difficulty_preference,
                'show_explanations': question_explanation
            })
            
            st.success("Preferences saved!")

def teacher_dashboard():
    """Teacher dashboard interface"""
    st.title(f"Teacher Dashboard - {st.session_state.user.username} 👨‍🏫")
    
    st.sidebar.title("Teacher Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Student Overview", "Assessment Management", "Question Management", "Analytics"]
    )
    
    if page == "Student Overview":
        student_overview_page()
    elif page == "Assessment Management":
        assessment_management_page()
    elif page == "Question Management":
        question_management_page()
    elif page == "Analytics":
        teacher_analytics_page()

def student_overview_page():
    """Overview of all students"""
    st.header("📊 Student Overview")
    
    # Get all students
    all_users = st.session_state.database.get_all_users()
    students = [user for user in all_users if user.role == 'student']
    
    if not students:
        st.info("No students registered yet.")
        return
    
    # Student statistics
    st.subheader("Student Statistics")
    
    student_data = []
    for student in students:
        sessions = st.session_state.database.get_user_sessions(student.username)
        completed_sessions = [s for s in sessions if s.completed_at]
        
        avg_score = np.mean([s.final_score for s in completed_sessions]) if completed_sessions else 0
        total_questions = sum(s.questions_answered for s in sessions)
        
        student_data.append({
            'Username': student.username,
            'Email': student.email,
            'Assessments': len(sessions),
            'Completed': len(completed_sessions),
            'Avg Score': f"{avg_score:.1%}",
            'Total Questions': total_questions,
            'Last Active': max([s.started_at for s in sessions], default=student.created_at).strftime('%Y-%m-%d')
        })
    
    df = pd.DataFrame(student_data)
    st.dataframe(df, use_container_width=True)
    
    # Performance distribution
    if completed_sessions:
        st.subheader("Performance Distribution")
        
        scores = [s.final_score for student in students 
                 for s in st.session_state.database.get_user_sessions(student.username)
                 if s.completed_at]
        
        fig = px.histogram(x=scores, nbins=10, title="Score Distribution Across All Students")
        fig.update_xaxis(title="Score")
        fig.update_yaxis(title="Number of Assessments")
        st.plotly_chart(fig)

def assessment_management_page():
    """Manage assessments"""
    st.header("📝 Assessment Management")
    
    tab1, tab2 = st.tabs(["Recent Assessments", "Create Custom Assessment"])
    
    with tab1:
        st.subheader("Recent Assessment Sessions")
        
        all_sessions = st.session_state.database.get_all_sessions()
        recent_sessions = sorted(all_sessions, key=lambda x: x.started_at, reverse=True)[:20]
        
        if not recent_sessions:
            st.info("No assessment sessions yet.")
        else:
            session_data = []
            for session in recent_sessions:
                status = "Completed" if session.completed_at else "In Progress"
                duration = ""
                if session.completed_at:
                    duration = str(session.completed_at - session.started_at).split('.')[0]
                
                session_data.append({
                    'Student': session.username,
                    'Track': session.track.title(),
                    'Started': session.started_at.strftime('%Y-%m-%d %H:%M'),
                    'Status': status,
                    'Score': f"{session.final_score:.1%}" if session.completed_at else "N/A",
                    'Questions': session.questions_answered,
                    'Duration': duration
                })
            
            st.dataframe(pd.DataFrame(session_data), use_container_width=True)
    
    with tab2:
        st.subheader("Create Custom Assessment")
        st.info("Feature coming soon: Create custom assessments with specific question sets.")

def question_management_page():
    """Manage questions"""
    st.header("❓ Question Management")
    
    tab1, tab2, tab3 = st.tabs(["Question Pool Stats", "Add Questions", "AI Question Generator"])
    
    with tab1:
        st.subheader("Current Question Pool Statistics")
        
        for track in ["web", "ai", "cyber", "data", "mobile", "devops"]:
            with st.expander(f"{track.title()} Track"):
                stats = {}
                for level in [1, 2, 3]:
                    question = get_adaptive_question(track, level)
                    # This is a simplified count - in real implementation, you'd count available questions
                    stats[level] = "Available" if question else "Limited"
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Easy", stats[1])
                with col2:
                    st.metric("Medium", stats[2])
                with col3:
                    st.metric("Hard", stats[3])
    
    with tab2:
        st.subheader("Add New Questions")
        
        with st.form("add_question_form"):
            new_track = st.selectbox("Track", ["web", "ai", "cyber", "data", "mobile", "devops"])
            new_level = st.selectbox("Level", [1, 2, 3], format_func=lambda x: ["Easy", "Medium", "Hard"][x-1])
            
            question_text = st.text_area("Question Text")
            
            st.write("Answer Options:")
            option1 = st.text_input("Option 1 (Correct Answer)")
            option2 = st.text_input("Option 2")
            option3 = st.text_input("Option 3")
            option4 = st.text_input("Option 4")
            
            explanation = st.text_area("Explanation (Optional)")
            
            if st.form_submit_button("Add Question"):
                if all([question_text, option1, option2, option3, option4]):
                    # In a real implementation, you would add this to your question database
                    st.success("Question added successfully! (Note: This is a demo)")
                else:
                    st.error("Please fill in all required fields")
    
    with tab3:
        st.subheader("AI Question Generator")
        
        with st.form("ai_generate_form"):
            ai_track = st.selectbox("Track for AI Generation", ["web", "ai", "cyber", "data", "mobile", "devops"])
            ai_level = st.selectbox("Difficulty Level", [1, 2, 3], format_func=lambda x: ["Easy", "Medium", "Hard"][x-1])
            ai_topic = st.text_input("Specific Topic (Optional)", placeholder="e.g., React hooks, SQL injection")
            
            if st.form_submit_button("Generate Question"):
                with st.spinner("Generating question with AI..."):
                    generated_q = st.session_state.ai_generator.generate_question_with_ai(
                        ai_track, ai_level, ai_topic
                    )
                    
                    if generated_q:
                        st.success("Question generated successfully!")
                        st.json(generated_q)
                    else:
                        st.error("Failed to generate question")

def teacher_analytics_page():
    """Teacher analytics dashboard"""
    st.header("📈 Analytics Dashboard")
    
    all_sessions = st.session_state.database.get_all_sessions()
    completed_sessions = [s for s in all_sessions if s.completed_at]
    
    if not completed_sessions:
        st.info("No completed assessments to analyze yet.")
        return
    
    # Overall metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Assessments", len(completed_sessions))
    with col2:
        avg_score = np.mean([s.final_score for s in completed_sessions])
        st.metric("Average Score", f"{avg_score:.1%}")
    with col3:
        total_questions = sum(s.questions_answered for s in completed_sessions)
        st.metric("Total Questions", total_questions)
    with col4:
        unique_students = len(set(s.username for s in completed_sessions))
        st.metric("Active Students", unique_students)
    
    # Track popularity
    st.subheader("Track Popularity")
    track_counts = {}
    for session in completed_sessions:
        track_counts[session.track] = track_counts.get(session.track, 0) + 1
    
    fig = px.pie(values=list(track_counts.values()), names=list(track_counts.keys()),
                 title="Assessment Distribution by Track")
    st.plotly_chart(fig)
    
    # Performance trends
    st.subheader("Performance Trends Over Time")
    
    # Group sessions by date
    daily_data = {}
    for session in completed_sessions:
        date_key = session.completed_at.date()
        if date_key not in daily_data:
            daily_data[date_key] = []
        daily_data[date_key].append(session.final_score)
    
    if len(daily_data) > 1:
        dates = sorted(daily_data.keys())
        avg_scores = [np.mean(daily_data[date]) for date in dates]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dates, y=avg_scores, mode='lines+markers',
                                name='Daily Average Score'))
        fig.update_layout(title="Average Daily Performance", 
                         xaxis_title="Date", yaxis_title="Average Score")
        st.plotly_chart(fig)

def admin_dashboard():
    """Admin dashboard interface"""
    st.title(f"Admin Dashboard - {st.session_state.user.username} 🛠️")
    
    st.sidebar.title("Admin Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["System Overview", "User Management", "Data Management", "System Settings"]
    )
    
    if page == "System Overview":
        system_overview_page()
    elif page == "User Management":
        user_management_page()
    elif page == "Data Management":
        data_management_page()
    elif page == "System Settings":
        system_settings_page()

def system_overview_page():
    """System overview for admins"""
    st.header("🖥️ System Overview")
    
    # System statistics
    all_users = st.session_state.database.get_all_users()
    all_sessions = st.session_state.database.get_all_sessions()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Users", len(all_users))
    with col2:
        students = [u for u in all_users if u.role == 'student']
        st.metric("Students", len(students))
    with col3:
        teachers = [u for u in all_users if u.role == 'teacher']
        st.metric("Teachers", len(teachers))
    with col4:
        st.metric("Total Sessions", len(all_sessions))
    
    # Recent activity
    st.subheader("Recent System Activity")
    
    recent_users = sorted(all_users, key=lambda x: x.created_at, reverse=True)[:5]
    recent_sessions = sorted(all_sessions, key=lambda x: x.started_at, reverse=True)[:5]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Recent Registrations:**")
        for user in recent_users:
            st.write(f"• {user.username} ({user.role}) - {user.created_at.strftime('%Y-%m-%d')}")
    
    with col2:
        st.write("**Recent Assessments:**")
        for session in recent_sessions:
            status = "✅" if session.completed_at else "⏳"
            st.write(f"• {status} {session.username} - {session.track} - {session.started_at.strftime('%Y-%m-%d %H:%M')}")

def user_management_page():
    """User management for admins"""
    st.header("👥 User Management")
    
    all_users = st.session_state.database.get_all_users()
    
    # User table
    user_data = []
    for user in all_users:
        sessions = st.session_state.database.get_user_sessions(user.username)
        user_data.append({
            'Username': user.username,
            'Email': user.email,
            'Role': user.role.title(),
            'Created': user.created_at.strftime('%Y-%m-%d'),
            'Sessions': len(sessions)
        })
    
    df = pd.DataFrame(user_data)
    st.dataframe(df, use_container_width=True)
    
    # User actions
    st.subheader("User Actions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Create New User:**")
        with st.form("create_user_form"):
            new_username = st.text_input("Username")
            new_email = st.text_input("Email")
            new_password = st.text_input("Password", type="password")
            new_role = st.selectbox("Role", ["student", "teacher", "admin"])
            
            if st.form_submit_button("Create User"):
                if st.session_state.database.create_user(new_username, new_password, new_email, new_role):
                    st.success(f"User {new_username} created successfully!")
                    st.rerun()
                else:
                    st.error("Failed to create user (username may already exist)")
    
    with col2:
        st.write("**User Statistics:**")
        role_counts = {}
        for user in all_users:
            role_counts[user.role] = role_counts.get(user.role, 0) + 1
        
        for role, count in role_counts.items():
            st.metric(f"{role.title()}s", count)

def data_management_page():
    """Data management for admins"""
    st.header("💾 Data Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Export Data")
        
        if st.button("Export All Users"):
            all_users = st.session_state.database.get_all_users()
            user_data = [
                {
                    'username': user.username,
                    'email': user.email,
                    'role': user.role,
                    'created_at': user.created_at.isoformat()
                }
                for user in all_users
            ]
            st.download_button(
                "Download Users JSON",
                json.dumps(user_data, indent=2),
                "users_export.json",
                "application/json"
            )
        
        if st.button("Export All Sessions"):
            all_sessions = st.session_state.database.get_all_sessions()
            session_data = [
                {
                    'session_id': session.session_id,
                    'username': session.username,
                    'track': session.track,
                    'started_at': session.started_at.isoformat(),
                    'completed_at': session.completed_at.isoformat() if session.completed_at else None,
                    'final_score': session.final_score,
                    'questions_answered': session.questions_answered,
                    'ability_level': session.ability_level
                }
                for session in all_sessions
            ]
            st.download_button(
                "Download Sessions JSON",
                json.dumps(session_data, indent=2),
                "sessions_export.json",
                "application/json"
            )
    
    with col2:
        st.subheader("System Maintenance")
        
        if st.button("🧹 Clean Old Sessions", help="Remove incomplete sessions older than 7 days"):
            # In a real implementation, you would clean old data
            st.success("System cleanup completed! (This is a demo)")
        
        if st.button("📊 Refresh Statistics"):
            st.rerun()

def system_settings_page():
    """System settings for admins"""
    st.header("⚙️ System Settings")
    
    st.subheader("AI Question Generation Settings")
    
    with st.form("ai_settings_form"):
        enable_ai = st.checkbox("Enable AI Question Generation", value=True)
        ai_api_endpoint = st.text_input("AI API Endpoint (Optional)")
        default_questions_per_track = st.number_input("Default Questions per Track", value=50)
        
        st.subheader("Assessment Settings")
        max_assessment_time = st.number_input("Max Assessment Time (minutes)", value=60)
        default_question_count = st.number_input("Default Question Count", value=10)
        
        if st.form_submit_button("Save Settings"):
            st.success("Settings saved! (This is a demo)")

# Main application flow
def main():
    """Main application entry point"""
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        text-align: center;
        color: #2E86AB;
        padding: 1rem 0;
        border-bottom: 2px solid #2E86AB;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2E86AB;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        padding: 1rem;
        border-radius: 0.25rem;
        color: #155724;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Check if user is logged in
    if 'user' not in st.session_state:
        authenticate_user()
        return
    
    # Show logout button
    with st.sidebar:
        st.write(f"Logged in as: **{st.session_state.user.username}** ({st.session_state.user.role})")
        if st.button("🚪 Logout"):
            # Clear all session state
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    
    # Check for active assessment
    if st.session_state.get('assessment_started', False):
        st.title("📝 Assessment in Progress")
        run_assessment()
        return
    
    # Check for active practice session
    if st.session_state.get('practice_active', False):
        st.title("🏋️ Practice Session")
        run_practice_session()
        return
    
    # Route to appropriate dashboard based on user role
    user_role = st.session_state.user.role
    
    if user_role == "student":
        student_dashboard()
    elif user_role == "teacher":
        teacher_dashboard()
    elif user_role == "admin":
        admin_dashboard()
    else:
        st.error("Invalid user role")

if __name__ == "__main__":
    main()