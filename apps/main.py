import streamlit as st
import sqlite3
import hashlib
from datetime import datetime
import os
import subprocess
import sys
import importlib.util

# Database setup
DATABASE_FILE = "students.db"

class DatabaseManager:
    def __init__(self):
        self.init_database()
    
    def init_database(self):
        """Initialize the database with required tables"""
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        
        # Create users table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            full_name TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            is_new_user INTEGER DEFAULT 1,
            assessment_completed INTEGER DEFAULT 0,
            skill_level INTEGER DEFAULT 0
        )
        ''')
        
        # Create assessment_results table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS assessment_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            total_questions INTEGER,
            correct_answers INTEGER,
            score_percentage REAL,
            difficulty_level INTEGER,
            completed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
        ''')
        
        # Create skill_gaps table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS skill_gaps (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            missing_skills TEXT,
            matched_skills TEXT,
            skill_analysis TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
        ''')
        
        conn.commit()
        conn.close()
    
    def hash_password(self, password):
        """Hash password using SHA256"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def create_user(self, username, email, password, full_name):
        """Create new user"""
        try:
            conn = sqlite3.connect(DATABASE_FILE)
            cursor = conn.cursor()
            
            password_hash = self.hash_password(password)
            cursor.execute('''
            INSERT INTO users (username, email, password_hash, full_name)
            VALUES (?, ?, ?, ?)
            ''', (username, email, password_hash, full_name))
            
            conn.commit()
            user_id = cursor.lastrowid
            conn.close()
            return user_id
        except sqlite3.IntegrityError:
            return None
    
    def authenticate_user(self, username, password):
        """Authenticate user login"""
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        
        password_hash = self.hash_password(password)
        cursor.execute('''
        SELECT id, username, email, full_name, is_new_user, assessment_completed, skill_level
        FROM users WHERE username = ? AND password_hash = ?
        ''', (username, password_hash))
        
        user = cursor.fetchone()
        conn.close()
        
        if user:
            return {
                'id': user[0],
                'username': user[1],
                'email': user[2],
                'full_name': user[3],
                'is_new_user': bool(user[4]),
                'assessment_completed': bool(user[5]),
                'skill_level': user[6]
            }
        return None
    
    def update_user_status(self, user_id, is_new_user=None, assessment_completed=None, skill_level=None):
        """Update user status after completing assessments"""
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        
        updates = []
        params = []
        
        if is_new_user is not None:
            updates.append("is_new_user = ?")
            params.append(int(is_new_user))
        
        if assessment_completed is not None:
            updates.append("assessment_completed = ?")
            params.append(int(assessment_completed))
        
        if skill_level is not None:
            updates.append("skill_level = ?")
            params.append(skill_level)
        
        if updates:
            params.append(user_id)
            cursor.execute(f'''
            UPDATE users SET {", ".join(updates)} WHERE id = ?
            ''', params)
            conn.commit()
        
        conn.close()
    
    def save_assessment_result(self, user_id, total_questions, correct_answers, score_percentage, difficulty_level):
        """Save assessment results"""
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        
        cursor.execute('''
        INSERT INTO assessment_results (user_id, total_questions, correct_answers, score_percentage, difficulty_level)
        VALUES (?, ?, ?, ?, ?)
        ''', (user_id, total_questions, correct_answers, score_percentage, difficulty_level))
        
        conn.commit()
        conn.close()

def init_session_state():
    """Initialize session state variables"""
    if 'db_manager' not in st.session_state:
        st.session_state.db_manager = DatabaseManager()
    
    # Initialize database reference for compatibility with mcqs.py
    if 'database' not in st.session_state:
        st.session_state.database = st.session_state.db_manager
    
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    
    if 'user' not in st.session_state:
        st.session_state.user = None
    
    if 'current_app' not in st.session_state:
        st.session_state.current_app = None
    
    if 'assessment_completed' not in st.session_state:
        st.session_state.assessment_completed = False

def login_page():
    """Login and registration page"""
    st.title("🎯 AI Learning Platform")
    st.markdown("### Welcome! Please login or create an account")
    
    tab1, tab2 = st.tabs(["Login", "Sign Up"])
    
    with tab1:
        st.subheader("Login to your account")
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            login_button = st.form_submit_button("Login", type="primary")
            
            if login_button:
                if username and password:
                    user = st.session_state.db_manager.authenticate_user(username, password)
                    if user:
                        st.session_state.logged_in = True
                        st.session_state.user = user
                        st.success(f"Welcome back, {user['full_name']}!")
                        st.rerun()
                    else:
                        st.error("Invalid username or password")
                else:
                    st.error("Please fill in all fields")
    
    with tab2:
        st.subheader("Create new account")
        with st.form("signup_form"):
            new_username = st.text_input("Username*")
            new_email = st.text_input("Email*")
            new_password = st.text_input("Password*", type="password")
            confirm_password = st.text_input("Confirm Password*", type="password")
            full_name = st.text_input("Full Name*")
            
            signup_button = st.form_submit_button("Create Account", type="primary")
            
            if signup_button:
                if not all([new_username, new_email, new_password, confirm_password, full_name]):
                    st.error("Please fill in all required fields")
                elif new_password != confirm_password:
                    st.error("Passwords don't match")
                elif len(new_password) < 6:
                    st.error("Password must be at least 6 characters long")
                else:
                    user_id = st.session_state.db_manager.create_user(
                        new_username, new_email, new_password, full_name
                    )
                    if user_id:
                        st.success("Account created successfully! Please login.")
                    else:
                        st.error("Username or email already exists")

def main_dashboard():
    """Main dashboard after login"""
    user = st.session_state.user
    
    # Sidebar with user info
    with st.sidebar:
        st.markdown(f"### Welcome, {user.get('full_name', 'User')}")
        st.markdown(f"**Username:** {user.get('username', 'N/A')}")
        st.markdown(f"**Email:** {user.get('email', 'N/A')}")
        
        if st.button("Logout", type="secondary"):
            st.session_state.logged_in = False
            st.session_state.user = None
            st.session_state.current_app = None
            st.session_state.assessment_completed = False
            st.rerun()
    
    # Check if user is new and needs assessment
    if user.get('is_new_user') and not user.get('assessment_completed'):
        show_new_user_flow()
    else:
        show_main_menu()

def show_new_user_flow():
    """Show assessment for new users"""
    st.title("🎯 Welcome to AI Learning Platform!")
    st.markdown("### Let's start by assessing your current skill level")
    
    st.info("""
    **First Time Setup:**
    
    Before we begin, we need to understand your current skill level through a quick assessment.
    This will help us personalize your learning experience and provide better recommendations.
    
    **What happens next:**
    1. Take a skill assessment (10-15 questions)
    2. Get your skill level evaluation
    3. Access personalized learning tools
    """)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if st.button("🚀 Start Skill Assessment", type="primary", use_container_width=True):
            # Redirect to MCQs app
            run_mcqs_app()

def show_main_menu():
    """Show main menu for existing users"""
    user = st.session_state.user
    
    st.title("🎓 AI Learning Platform Dashboard")
    st.markdown(f"### Welcome back, {user.get('full_name', 'User')}!")
    
    # Show user's current level
    level_names = {0: "Not Assessed", 1: "Beginner", 2: "Intermediate", 3: "Advanced"}
    current_level = level_names.get(user.get('skill_level', 0), "Unknown")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Current Level", current_level)
    with col2:
        status = "Completed" if user.get('assessment_completed') else "Pending"
        st.metric("Assessment Status", status)
    with col3:
        st.metric("Account Type", "Student")
    
    st.markdown("---")
    
    # Main menu options with beautiful buttons
    st.markdown("### 🛠 Available Tools")
    
    # Create 4 beautiful buttons in a grid
    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style='text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    border-radius: 10px; color: white; margin-bottom: 20px;'>
            <h3>📝 Skill Assessment</h3>
            <p>Test your knowledge with our AI-powered assessment</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Start Assessment", key="mcqs_btn", use_container_width=True):
            run_mcqs_app()
    
    with col2:
        st.markdown("""
        <div style='text-align: center; padding: 20px; background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                    border-radius: 10px; color: white; margin-bottom: 20px;'>
            <h3>🔍 Skill Gap Analysis</h3>
            <p>Identify gaps between your skills and job requirements</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Analyze Skills", key="skillgap_btn", use_container_width=True):
            run_skillgap_app()
    
    with col3:
        st.markdown("""
        <div style='text-align: center; padding: 20px; background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                    border-radius: 10px; color: white; margin-bottom: 20px;'>
            <h3>💼 Interview Prep</h3>
            <p>Practice with AI-powered mock interviews</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Practice Interview", key="interview_btn", use_container_width=True):
            run_interview_app()
    
    with col4:
        st.markdown("""
        <div style='text-align: center; padding: 20px; background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); 
                    border-radius: 10px; color: white; margin-bottom: 20px;'>
            <h3>📚 Course Recommendations</h3>
            <p>Get personalized learning recommendations</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Find Courses", key="recommend_btn", use_container_width=True):
            run_recommend_app()
    
    # Additional section for retaking assessment
    st.markdown("---")
    st.markdown("### 🔄 Update Your Progress")
    st.markdown("""
    If you've been learning and improving your skills, you can retake the assessment 
    to update your skill level and get new recommendations.
    """)
    
    if st.button("🔄 Retake Skill Assessment", type="secondary", use_container_width=True):
        run_mcqs_app()

def run_mcqs_app():
    """Run the MCQs assessment application"""
    st.session_state.current_app = "mcqs"
    st.rerun()

def run_skillgap_app():
    """Run the skill gap analysis application"""
    st.session_state.current_app = "skillgap"
    st.rerun()

def run_interview_app():
    """Run the interview preparation application"""
    st.session_state.current_app = "interview"
    st.rerun()

def run_recommend_app():
    """Run the course recommendation application"""
    st.session_state.current_app = "recommend"
    st.rerun()

def load_external_app(app_name):
    """Load external application modules"""
    try:
        if app_name == "mcqs":
            # Clear any conflicting session state before loading mcqs
            if 'assessment_started' in st.session_state:
                del st.session_state.assessment_started
            if 'practice_active' in st.session_state:
                del st.session_state.practice_active
            
            # Import and run MCQs app
            try:
                from apps.mcqs import main as mcqs_main
                mcqs_main()
            except ImportError:
                # Try direct import if apps package doesn't exist
                from mcqs import main as mcqs_main
                mcqs_main()
            
        elif app_name == "skillgap":
            # Import and run skill gap app
            try:
                from apps.skillgap import main as skillgap_main
                skillgap_main()
            except ImportError:
                from skillgap import main as skillgap_main
                skillgap_main()
            
        elif app_name == "interview":
            # Import and run interview app
            try:
                from apps.interview import main as interview_main
                interview_main()
            except ImportError:
                from interview import main as interview_main
                interview_main()
            
        elif app_name == "recommend":
            # Import and run recommendation app
            try:
                from apps.recommend import main as recommend_main
                recommend_main()
            except ImportError:
                from recommend import main as recommend_main
                recommend_main()
            
    except ImportError as e:
        st.error(f"Could not load {app_name} application: {e}")
        st.markdown("Please make sure all application files are in the same directory.")
        if st.button("🏠 Back to Dashboard"):
            st.session_state.current_app = None
            st.rerun()
    except Exception as e:
        st.error(f"Error running {app_name}: {e}")
        if st.button("🏠 Back to Dashboard"):
            st.session_state.current_app = None
            st.rerun()

def show_navigation():
    """Show navigation to return to main dashboard"""
    with st.sidebar:
        st.markdown("---")
        if st.button("🏠 Back to Dashboard"):
            st.session_state.current_app = None
            st.rerun()

def main():
    """Main application entry point"""
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        text-align: center;
        color: #2E86AB;
        padding: 1rem 0;
        border-bottom: 2px solid #2E86AB;
        margin-bottom: 2rem;
    }
    .feature-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #2E86AB;
        margin-bottom: 1rem;
        transition: transform 0.2s;
    }
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    .stButton button {
        transition: all 0.3s ease;
    }
    .stButton button:hover {
        transform: scale(1.05);
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    init_session_state()
    
    # Check if we're running an external app
    if st.session_state.get('current_app'):
        show_navigation()
        load_external_app(st.session_state.current_app)
    else:
        # Main application flow
        if not st.session_state.logged_in:
            login_page()
        else:
            main_dashboard()

if __name__ == "__main__":
    main()