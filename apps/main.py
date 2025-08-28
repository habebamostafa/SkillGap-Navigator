import streamlit as st
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'data'))
from database import db

# Page configuration
st.set_page_config(
    page_title="Student Management System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Home"
if 'current_student' not in st.session_state:
    st.session_state.current_student = None

# Sidebar navigation
st.sidebar.title("🎓 Navigation")
page = st.sidebar.radio("Go to", [
    "Home", 
    "Register", 
    "Login", 
    "View Students", 
    "Manage Students"
])

# Home Page
if page == "Home":
    st.title("Welcome to Student Management System")
    st.write("""
    ### Features:
    - ✅ Student Registration
    - ✅ Login System
    - ✅ Student Management
    - ✅ Database Storage
    - ✅ Multiple Pages
    """)
    
    # Display statistics
    students = db.get_all_students()
    st.info(f"**Total Students:** {len(students)}")

# Register Page
elif page == "Register":
    st.title("Student Registration")
    
    with st.form("register_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            name = st.text_input("Full Name")
            email = st.text_input("Email Address")
        
        with col2:
            password = st.text_input("Password", type="password")
            track = st.selectbox("Track", [
                "AI & Machine Learning",
                "Web Development", 
                "Data Science",
                "Cyber Security",
                "Mobile Development"
            ])
        
        submitted = st.form_submit_button("Register Student")
        
        if submitted:
            if all([name, email, password, track]):
                try:
                    student = db.add_student(name, email, password, track)
                    st.success(f"✅ Student {name} registered successfully!")
                    st.json(student)
                except ValueError as e:
                    st.error(f"❌ {e}")
            else:
                st.warning("⚠️ Please fill all fields")

# Login Page
elif page == "Login":
    st.title("Student Login")
    
    with st.form("login_form"):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        
        submitted = st.form_submit_button("Login")
        
        if submitted:
            if email and password:
                student = db.verify_login(email, password)
                if student:
                    st.session_state.current_student = student
                    st.success(f"🎉 Welcome back {student['name']}!")
                    st.json(student)
                else:
                    st.error("❌ Invalid email or password")
            else:
                st.warning("⚠️ Please enter email and password")

# View Students Page
elif page == "View Students":
    st.title("All Students")
    
    students = db.get_all_students()
    
    if students:
        st.write(f"**Total Students:** {len(students)}")
        
        # Search and filter
        col1, col2 = st.columns(2)
        with col1:
            search_name = st.text_input("Search by name")
        with col2:
            filter_track = st.selectbox("Filter by track", 
                ["All"] + list(set(s['track'] for s in students)))
        
        # Filter students
        filtered_students = students
        if search_name:
            filtered_students = [s for s in filtered_students 
                               if search_name.lower() in s['name'].lower()]
        if filter_track != "All":
            filtered_students = [s for s in filtered_students 
                               if s['track'] == filter_track]
        
        # Display students
        for student in filtered_students:
            with st.expander(f"{student['name']} - {student['track']}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Email:** {student['email']}")
                    st.write(f"**Track:** {student['track']}")
                with col2:
                    st.write(f"**Registered:** {student['created_at'][:10]}")
                    st.write(f"**Status:** {student['status']}")

# Manage Students Page
elif page == "Manage Students":
    st.title("Manage Students")
    
    students = db.get_all_students()
    
    if students:
        # Student selection
        student_options = {f"{s['name']} (ID: {s['id']})": s['id'] for s in students}
        selected = st.selectbox("Select Student", list(student_options.keys()))
        
        if selected:
            student_id = student_options[selected]
            student = db.get_student(student_id)
            
            if student:
                st.write("### Current Information")
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Name:** {student['name']}")
                    st.write(f"**Email:** {student['email']}")
                with col2:
                    st.write(f"**Track:** {student['track']}")
                    st.write(f"**Status:** {student['status']}")
                
                st.divider()
                
                # Update form
                st.write("### Update Information")
                col1, col2 = st.columns(2)
                
                with col1:
                    new_track = st.selectbox("New Track", [
                        "AI & Machine Learning",
                        "Web Development", 
                        "Data Science",
                        "Cyber Security",
                        "Mobile Development"
                    ], index=[
                        "AI & Machine Learning",
                        "Web Development", 
                        "Data Science",
                        "Cyber Security",
                        "Mobile Development"
                    ].index(student['track']))
                
                with col2:
                    new_status = st.selectbox("New Status", 
                        ["active", "inactive"], 
                        index=0 if student['status'] == 'active' else 1)
                
                if st.button("💾 Save Changes"):
                    updated = db.update_student(
                        student_id, 
                        track=new_track, 
                        status=new_status
                    )
                    if updated:
                        st.success("✅ Student updated successfully!")
                        st.rerun()
                    else:
                        st.error("❌ Error updating student")
                
                st.divider()
                
                # Danger zone
                st.write("### Danger Zone")
                if st.button("🗑️ Delete Student", type="secondary"):
                    if db.delete_student(student_id):
                        st.success("✅ Student deleted successfully!")
                        st.rerun()
                    else:
                        st.error("❌ Error deleting student")
    
    else:
        st.info("No students found in database")

# Footer
st.sidebar.divider()
st.sidebar.info("""
**Student Management System**  
Built with Streamlit & SQLite  
Database: `students.db`
""")