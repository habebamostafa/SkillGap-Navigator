"""
AI Skill-Gap Analyzer - Streamlit Web Application
Author: Senior ML Engineer
Purpose: Deploy BERT-based skill matching model with interactive UI
"""

import streamlit as st
import pandas as pd
import pickle
import numpy as np
import joblib
## ...existing code...
import numpy as np
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import re
import json
import base64
from io import BytesIO
import PyPDF2
import docx
from typing import Dict, List, Tuple, Optional
import hashlib
import gdown
import os
# Page configuration
st.set_page_config(
    page_title="AI Skill-Gap Analyzer",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1e3d59;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #f5f0e1;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .skill-tag {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        margin: 0.2rem;
        background-color: #e3f2fd;
        border-radius: 15px;
        font-size: 0.9rem;
        color: #1976d2;
    }
    .missing-skill-tag {
        background-color: #ffebee;
        color: #c62828;
    }
    .recommendation-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
file_id = "12J2NhvcXlIc5ol6GIBsocr-YWQ1MAoZI"
output_path = "best_model.pth"

# Download model if not already downloaded
if not os.path.exists(output_path):
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, output_path, quiet=False)

# Configuration
# MODEL_PATH = "C:\\Games\\New folder\\Job_Matching\\Skill Matching Model\\best_model.pth"
MAX_LENGTH = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


SKILL_CATEGORIES = {
    'Programming Languages': [
        'python', 'java', 'javascript', 'c++', 'c#', 'ruby', 'go', 'rust', 
        'scala', 'kotlin', 'swift', 'r', 'matlab', 'typescript', 'php'
    ],
    'ML/AI Frameworks': [
        'tensorflow', 'pytorch', 'keras', 'scikit-learn', 'xgboost', 'lightgbm',
        'catboost', 'huggingface', 'transformers', 'opencv', 'spacy', 'nltk'
    ],
    'Data Tools': [
        'pandas', 'numpy', 'sql', 'spark', 'hadoop', 'hive', 'airflow',
        'databricks', 'snowflake', 'tableau', 'power bi', 'looker'
    ],
    'Cloud & DevOps': [
        'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'terraform', 'jenkins',
        'git', 'ci/cd', 'ansible', 'helm', 'prometheus', 'grafana'
    ],
    'Databases': [
        'mysql', 'postgresql', 'mongodb', 'redis', 'cassandra', 'elasticsearch',
        'dynamodb', 'neo4j', 'influxdb'
    ],
    'Web Technologies': [
        'react', 'angular', 'vue', 'node.js', 'express', 'django', 'flask',
        'fastapi', 'rest api', 'graphql', 'html', 'css', 'bootstrap'
    ]
}

@st.cache_resource
def load_model():
    """Load the trained BERT model"""
    try:
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased",
            num_labels=5  # 5 classes for match scores 1-5
        )
        
        # Load trained weights
        checkpoint = torch.load(output_path, map_location=DEVICE)
        model.load_state_dict(checkpoint)
        model.to(DEVICE)
        model.eval()
        
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        # Return a dummy model for demonstration
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased",
            num_labels=5
        )
        model.to(DEVICE)
        model.eval()
        return model, tokenizer

def extract_text_from_pdf(file):
    """Extract text from PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except:
        return ""

def extract_text_from_docx(file):
    """Extract text from DOCX file"""
    try:
        doc = docx.Document(file)
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        return text
    except:
        return ""

def extract_skills(text: str) -> List[str]:
    """Extract skills from text"""
    text_lower = text.lower()
    found_skills = []
    
    for category, skills in SKILL_CATEGORIES.items():
        for skill in skills:
            pattern = r'\b' + re.escape(skill) + r'\b'
            if re.search(pattern, text_lower):
                found_skills.append(skill)
    
    return list(set(found_skills))

def clean_text(text: str) -> str:
    """Clean and preprocess text"""
    # Remove URLs and emails
    text = re.sub(r'http[s]?://\S+|www\.\S+', ' ', text)
    text = re.sub(r'\S+@\S+\.\S+', ' ', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Keep only relevant characters
    text = ''.join(ch if ch.isalnum() or ch.isspace() or ch in '#+.-_' else ' ' for ch in text)
    
    return text.strip()

def predict_match_score(resume_text: str, job_text: str, model, tokenizer) -> Tuple[float, np.ndarray]:
    """Predict match score between resume and job description"""
    
    # Clean texts
    resume_clean = clean_text(resume_text)
    job_clean = clean_text(job_text)
    
    # Prepare input
    pair_text = f"[CLS] {job_clean[:200]} [SEP] {resume_clean[:200]}"
    
    # Tokenize
    encoding = tokenizer(
        pair_text,
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )
    
    # Move to device
    input_ids = encoding['input_ids'].to(DEVICE)
    attention_mask = encoding['attention_mask'].to(DEVICE)
    
    # Predict
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        
        # Get prediction (1-5 scale)
        pred_class = torch.argmax(probs, dim=-1).item() + 1
        confidence = probs[0].cpu().numpy()
    
    return pred_class, confidence

def calculate_skill_gap(resume_skills: List[str], job_skills: List[str]) -> Dict:
    """Calculate skill gap analysis"""
    resume_set = set(resume_skills)
    job_set = set(job_skills)
    
    matched_skills = resume_set.intersection(job_set)
    missing_skills = job_set - resume_set
    additional_skills = resume_set - job_set
    
    match_percentage = (len(matched_skills) / len(job_set) * 100) if job_set else 0
    
    return {
        'matched_skills': list(matched_skills),
        'total_required': len(job_set),
        'total_matched': len(matched_skills)
    }


def create_skill_radar_chart(skill_categories: Dict[str, List[str]], 
                           resume_skills: List[str]) -> go.Figure:
    """Create radar chart for skill analysis"""
    categories = list(skill_categories.keys())
    values = []
    
    for category, skills in skill_categories.items():
        category_skills = set(skills)
        matched = len(set(resume_skills).intersection(category_skills))
        total = len(category_skills)
        percentage = (matched / total * 100) if total > 0 else 0
        values.append(percentage)
    
    fig = go.Figure(data=go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        marker=dict(color='rgb(103, 126, 234)')
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        showlegend=False,
        title="Skill Coverage by Category"
    )
    
    return fig

def create_match_gauge(match_score: int) -> go.Figure:
    """Create gauge chart for match score"""
    color = "green" if match_score >= 4 else "orange" if match_score >= 3 else "red"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=match_score * 20,  # Convert to percentage
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Match Score"},
        delta={'reference': 60},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 40], 'color': "lightgray"},
                {'range': [40, 60], 'color': "gray"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 80
            }
        }
    ))
    
    fig.update_layout(height=300)
    return fig

def main():
    """Main application"""
    
    # Header
    st.markdown('<h1 class="main-header">🎯 AI Skill-Gap Analyzer</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Bridge the gap between your skills and dream job</p>', unsafe_allow_html=True)
    
    # Load model
    if not st.session_state.model_loaded:
        with st.spinner("Loading AI model..."):
            model, tokenizer = load_model()
            st.session_state.model = model
            st.session_state.tokenizer = tokenizer
            st.session_state.model_loaded = True
    else:
        model = st.session_state.model
        tokenizer = st.session_state.tokenizer
    
    # Sidebar
    with st.sidebar:
        st.image("https://via.placeholder.com/300x100/667eea/ffffff?text=Skill+Gap+Analyzer", width=300)
        st.markdown("###  Tips")
        st.info(
            "• Upload your resume in PDF, DOCX, or paste text\n"
            "• Include all relevant skills and technologies\n"
            "• Be specific about your experience level"
        )
    
    # Main content
    tabs = st.tabs(["📝 Normal Analysis", "📊 Results"])
    
    with tabs[0]:  # Analysis Tab
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📄 Resume Input")
            resume_input_method = st.radio(
                "Choose input method:",
                ["Upload File", "Paste Text"],
                key="resume_method"
            )
            
            resume_text = ""
            if resume_input_method == "Upload File":
                uploaded_file = st.file_uploader(
                    "Upload your resume",
                    type=['pdf', 'docx', 'txt'],
                    key="resume_upload"
                )
                if uploaded_file:
                    if uploaded_file.type == "application/pdf":
                        resume_text = extract_text_from_pdf(uploaded_file)
                    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                        resume_text = extract_text_from_docx(uploaded_file)
                    else:
                        resume_text = str(uploaded_file.read(), "utf-8")
                    
                    st.success("✅ Resume uploaded successfully!")
            else:
                resume_text = st.text_area(
                    "Paste your resume here:",
                    height=300,
                    placeholder="Enter your resume text...",
                    key="resume_text"
                )
        
        with col2:
            st.markdown("### 💼 Job Description")
            job_input_method = st.radio(
                "Choose input method:",
                ["Paste Text", "Use Template"],
                key="job_method"
            )
            
            job_text = ""
            if job_input_method == "Use Template":
                template = st.selectbox(
                    "Select job template:",
                    ["Data Scientist", "ML Engineer", "Software Engineer", "DevOps Engineer", "Full Stack Developer"]
                )
                
                # Template job descriptions
                templates = {
                    "Data Scientist": "We are looking for a Data Scientist with experience in Python, machine learning, deep learning, SQL, and data visualization. Required skills: pandas, numpy, scikit-learn, tensorflow, pytorch, tableau, statistical analysis, A/B testing.",
                    "ML Engineer": "Seeking ML Engineer proficient in Python, tensorflow, pytorch, docker, kubernetes, MLOps, model deployment, AWS/GCP, REST APIs, and CI/CD pipelines.",
                    "Software Engineer": "Software Engineer position requiring Java, Python, Spring Boot, microservices, REST APIs, SQL, Git, agile methodology, and cloud platforms.",
                    "DevOps Engineer": "DevOps Engineer with expertise in Docker, Kubernetes, Terraform, AWS/Azure/GCP, CI/CD, Jenkins, monitoring tools, and infrastructure as code.",
                    "Full Stack Developer": "Full Stack Developer skilled in React, Node.js, JavaScript, TypeScript, MongoDB, PostgreSQL, REST APIs, and responsive web design."
                }
                job_text = templates[template]
                st.text_area("Job Description:", value=job_text, height=200, disabled=True)
            else:
                job_text = st.text_area(
                    "Paste job description here:",
                    height=300,
                    placeholder="Enter job description...",
                    key="job_text"
                )
        
        # Analysis button
        if st.button("🚀 Analyze Skill Gap", type="primary", use_container_width=True):
            if resume_text and job_text:
                with st.spinner("Analyzing your skills..."):
                    # Extract skills
                    resume_skills = extract_skills(resume_text)
                    job_skills = extract_skills(job_text)
                    
                    # Predict match score
                    match_score, confidence = predict_match_score(resume_text, job_text, model, tokenizer)
                    
                    # Calculate skill gap
                    skill_gap = calculate_skill_gap(resume_skills, job_skills)
                    
                    # Store in session state
                    st.session_state.current_analysis = {
                        'match_score': match_score,
                        'confidence': confidence,
                        'resume_skills': resume_skills,
                        'job_skills': job_skills,
                        'skill_gap': skill_gap,
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M")
                    }
                    
                    # Add to history
                    st.session_state.analysis_history.append({
                        'score': match_score,
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M")
                    })
                    
                    st.success("✅ Analysis complete! Check the Results tab.")
            else:
                st.error("Please provide both resume and job description.")
    
    with tabs[1]:  # Results Tab
        if hasattr(st.session_state, 'current_analysis'):
            analysis = st.session_state.current_analysis
            # Skill Analysis
            st.markdown("### 🎯 Skill Analysis")
            col1, col2 = st.columns(2)
            with col1:
                # Matched Skills
                st.markdown("#### ✅ Matched Skills")
                if analysis['skill_gap']['matched_skills']:
                    skills_html = ""
                    for skill in analysis['skill_gap']['matched_skills']:
                        skills_html += f'<span class="skill-tag">{skill}</span>'
                    st.markdown(skills_html, unsafe_allow_html=True)
                else:
                    st.info("No matching skills found")
                # Additional Skills
                st.markdown("#### 💪 Your Additional Skills")
                if analysis['skill_gap']['additional_skills']:
                    skills_html = ""
                    for skill in analysis['skill_gap']['additional_skills']:
                        skills_html += f'<span class="skill-tag">{skill}</span>'
                    st.markdown(skills_html, unsafe_allow_html=True)
                else:
                    st.info("No additional skills")
            with col2:
                # Missing Skills
                st.markdown("#### ❌ Missing Skills")
                if analysis['skill_gap']['missing_skills']:
                    skills_html = ""
                    for skill in analysis['skill_gap']['missing_skills']:
                        skills_html += f'<span class="skill-tag missing-skill-tag">{skill}</span>'
                    st.markdown(skills_html, unsafe_allow_html=True)
                else:
                    st.success("You have all required skills!")
            # Radar Chart
            st.markdown("### 📊 Skill Coverage by Category")
            radar_fig = create_skill_radar_chart(SKILL_CATEGORIES, analysis['resume_skills'])
            st.plotly_chart(radar_fig, use_container_width=True)
        else:
            st.info("No analysis results yet. Please go to the Analysis tab to start.")
    
    
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>Built with ❤️ by </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()