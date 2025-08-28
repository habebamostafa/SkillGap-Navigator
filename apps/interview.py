import streamlit as st
import random
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import time
import torch

# --- Model Setup with Proper Caching ---
MODEL_NAME = "google/flan-t5-large"  # Using a smaller model for better performance
HF_TOKEN = st.secrets.get("HF_TOKEN", None)

# Initialize session state for model loading
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'model_loading' not in st.session_state:
    st.session_state.model_loading = False
if 'show_interview' not in st.session_state:
    st.session_state.show_interview = False
if 'settings_confirmed' not in st.session_state:
    st.session_state.settings_confirmed = False

# Define generate_text function with better parameters
def generate_text(prompt, max_len=150, temperature=0.7):
    """Generate text with the loaded model"""
    if 'tokenizer' not in st.session_state or 'model' not in st.session_state:
        return "Model not loaded yet. Please wait..."
    
    try:
        inputs = st.session_state.tokenizer(
            prompt, 
            return_tensors="pt", 
            max_length=512,
            truncation=True
        )
        
        # Move inputs to the same device as model
        device = next(st.session_state.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        outputs = st.session_state.model.generate(
            **inputs, 
            max_length=max_len,
            min_length=20,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            no_repeat_ngram_size=2,
            pad_token_id=st.session_state.tokenizer.eos_token_id
        )
        
        generated_text = st.session_state.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove the input prompt from the generated text
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()
        
        # Ensure we have meaningful content
        if len(generated_text.strip()) < 10:
            return get_fallback_response(prompt)
        
        return generated_text.strip()
    
    except Exception as e:
        st.error(f"Model error: {e}")
        return get_fallback_response(prompt)

# استبدال الدالة get_fallback_response
def get_fallback_response(prompt):
    """Provide fallback responses when model fails"""
    if "feedback" in prompt.lower():
        feedbacks = [
            "Good start! Try to be more specific and provide concrete examples in your answer.",
            "Your answer shows understanding. Consider elaborating with real-world applications.",
            "Well thought out response. Adding technical details would strengthen your answer.",
            "Nice explanation! Try to structure your answer with clear points next time."
        ]
        return random.choice(feedbacks)
    else:
        # استخدام القيم الافتراضية إذا لم تكن موجودة في session state
        track = st.session_state.get('selected_track', 'Artificial Intelligence')
        difficulty = st.session_state.get('selected_difficulty', 'Easy')
        
        questions = {
            "Artificial Intelligence": {
                "Easy": [
                    "What is the difference between supervised and unsupervised learning?",
                    "Can you explain what a neural network is in simple terms?",
                    "What are some common applications of machine learning you encounter daily?"
                ],
                "Medium": [
                    "How would you handle overfitting in a machine learning model?",
                    "Explain the bias-variance tradeoff in machine learning.",
                    "What evaluation metrics would you use for a classification problem?"
                ],
                "Hard": [
                    "How would you implement a transformer model for natural language processing?",
                    "Explain how gradient descent optimization algorithms work.",
                    "Discuss the ethical considerations in AI development and deployment."
                ]
            },
            "Software Development": {
                "Easy": [
                    "What is version control and why is it important?",
                    "Explain the concept of object-oriented programming.",
                    "What are some key principles of writing clean code?"
                ],
                "Medium": [
                    "How would you optimize a slow database query?",
                    "Explain the Model-View-Controller architecture pattern.",
                    "What testing methodologies do you follow in your development process?"
                ],
                "Hard": [
                    "How would you design a scalable microservices architecture?",
                    "Explain the CAP theorem and its implications for distributed systems.",
                    "Describe your approach to securing a web application against common vulnerabilities."
                ]
            }
        }
        
        if track in questions and difficulty in questions[track]:
            return random.choice(questions[track][difficulty])
        return "Can you explain your approach to problem-solving in technical projects?"
    
@st.cache_resource(show_spinner=False)
def load_model_components():
    """Load the model and tokenizer with proper caching"""
    try:
        # Use a smaller model for better performance
        model_name = "google/flan-t5-large"
        
        if HF_TOKEN:
            tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name, token=HF_TOKEN)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        # Handle tokenizer padding token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        return tokenizer, model
        
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

# --- Streamlit UI ---
st.set_page_config(page_title="AI Interview Simulator", page_icon="🤖", layout="wide")
st.title("AI-Powered Interview Simulation")
st.markdown("Experience a realistic interview with AI-generated questions and feedback!")

# Sidebar for configuration
with st.sidebar:
    st.header("Interview Configuration")

    # Display model status
    st.subheader("System Status")
    if st.session_state.model_loaded:
        st.success("✅ Model loaded successfully")
    else:
        st.warning("⏳ Model not loaded yet")

    # Input options
    tracks = [
        "Artificial Intelligence", 
        "Software Development", 
        "Web Development", 
        "Mobile App Development", 
        "Data Science", 
        "Product Management", 
        "UX Design", 
        "Digital Marketing"
    ]
    difficulties = ["Easy", "Medium", "Hard"]
    
    selected_track = st.selectbox(
        "Select Track:",
        tracks,
        index=0
    )

    selected_difficulty = st.selectbox(
        "Select Difficulty:",
        difficulties,
        index=0
    )

    num_questions = st.number_input(
        "Number of Questions:",
        min_value=1,
        max_value=8,
        value=3
    )

    # Agent personality options
    st.subheader("Interview Style")
    interviewer_style = st.selectbox(
        "Interviewer Style:",
        ["Professional", "Friendly", "Technical", "Conversational"]
    )
    
    coach_style = st.selectbox(
        "Coach Style:",
        ["Encouraging", "Constructive", "Direct", "Detailed"]
    )
    
    # Start interview button
    if st.button("Start Interview", type="primary", use_container_width=True):
        # Store configuration
        st.session_state.selected_track = selected_track
        st.session_state.selected_difficulty = selected_difficulty
        st.session_state.selected_num_questions = num_questions
        st.session_state.interviewer_style = interviewer_style
        st.session_state.coach_style = coach_style
        
        # Reset interview state
        st.session_state.current_q = 0
        st.session_state.user_answers = []
        st.session_state.conversation = []
        st.session_state.interview_finished = False
        st.session_state.questions = []
        st.session_state.expected_answers = []
        st.session_state.settings_confirmed = True
        
        # Load model if not already loaded
        if not st.session_state.model_loaded:
            st.session_state.model_loading = True
            with st.spinner("Loading AI model..."):
                tokenizer, model = load_model_components()
                if tokenizer and model:
                    st.session_state.tokenizer = tokenizer
                    st.session_state.model = model
                    st.session_state.model_loaded = True
                else:
                    st.error("Failed to load model. Using fallback mode.")
                    st.session_state.model_loaded = True  # Continue with fallback
                st.session_state.model_loading = False
        st.rerun()

# Show loading message until model is ready
if not st.session_state.model_loaded and st.session_state.get('model_loading', False):
    st.info("⏳ The AI model is loading. You can configure your interview while you wait.")
    st.spinner("Downloading model components...")

# Only show interview section when ready
if st.session_state.model_loaded and st.session_state.get('settings_confirmed', False):
    
    # Initialize session state variables
    if 'current_q' not in st.session_state:
        st.session_state.current_q = 0
        st.session_state.user_answers = []
        st.session_state.conversation = []
        st.session_state.interview_finished = False
        st.session_state.questions = []
        st.session_state.expected_answers = []

    def add_to_conversation(role, message, agent_type=None):
        """Add message to conversation history"""
        if 'conversation' not in st.session_state:
            st.session_state.conversation = []
        st.session_state.conversation.append({
            "role": role,
            "message": message,
            "agent": agent_type,
            "timestamp": time.time()
        })

    # Initialize conversation
    if len(st.session_state.get('conversation', [])) == 0:
        welcome_msg = f"Welcome! Starting your {st.session_state.selected_difficulty.lower()} level interview for {st.session_state.selected_track}. We'll go through {st.session_state.selected_num_questions} questions together."
        add_to_conversation("System", welcome_msg)

    # Display conversation
    st.subheader("Interview in Progress")
    
    # Show progress - Fixed the off-by-one error
    if st.session_state.current_q < st.session_state.selected_num_questions:
        progress = (st.session_state.current_q) / st.session_state.selected_num_questions
        st.progress(progress, text=f"Question {st.session_state.current_q + 1} of {st.session_state.selected_num_questions}")
    else:
        st.progress(1.0, text=f"Completed {st.session_state.selected_num_questions} of {st.session_state.selected_num_questions} questions")

    # Conversation display
    conversation_container = st.container()
    with conversation_container:
        for msg in st.session_state.get('conversation', []):
            if msg["role"] == "System":
                with st.chat_message("system"):
                    st.info(f"📢 {msg['message']}")
            elif msg["role"] == "Interviewer":
                with st.chat_message("assistant", avatar="👔"):
                    st.write(f"**Interviewer**: {msg['message']}")
            elif msg["role"] == "Coach":
                with st.chat_message("assistant", avatar="📊"):
                    st.write(f"**Coach**: {msg['message']}")
            elif msg["role"] == "Candidate":
                with st.chat_message("user", avatar="🧑‍💼"):
                    st.write(f"**You**: {msg['message']}")

    # Interview logic
    if not st.session_state.get('interview_finished', False):
        if st.session_state.current_q < st.session_state.selected_num_questions:
            current_q_index = st.session_state.current_q
            
            # Generate question if needed
            if len(st.session_state.questions) <= current_q_index:
                with st.spinner("💭 Preparing next question..."):
                    # Create specific examples based on track and difficulty
                    track_examples = {
                        "Artificial Intelligence": {
                            "Easy": "real-world ML project, data preprocessing challenge, basic algorithm selection",
                            "Medium": "production ML system, model optimization, handling bias or drift",
                            "Hard": "distributed AI architecture, advanced deep learning, ethical AI implementation"
                        },
                        "Software Development": {
                            "Easy": "debugging issue, code refactoring, API design",
                            "Medium": "system architecture, performance optimization, integration challenges", 
                            "Hard": "scalable distributed system, complex algorithm implementation, advanced design patterns"
                        },
                        "Web Development": {
                            "Easy": "responsive design problem, basic functionality implementation, user experience issue",
                            "Medium": "full-stack application design, performance optimization, security implementation",
                            "Hard": "large-scale web architecture, advanced frontend/backend integration, complex state management"
                        }
                    }
                    
                    # Get previous questions to avoid repetition
                    previous_topics = []
                    if st.session_state.questions:
                        for prev_q in st.session_state.questions:
                            if "system" in prev_q.lower():
                                previous_topics.append("system design")
                            elif "data" in prev_q.lower():
                                previous_topics.append("data handling")
                            elif "performance" in prev_q.lower():
                                previous_topics.append("optimization")
                    
                    previous_topics_text = ", ".join(previous_topics) if previous_topics else "none"
                    
                    difficulty_level = st.session_state.selected_difficulty.lower()
                    track = st.session_state.selected_track
                    examples = track_examples.get(track, {}).get(st.session_state.selected_difficulty, "practical project challenge")
                    
                    question_prompt = f"""Create a {difficulty_level} level interview question for {track} position. 
                    Make it practical and scenario-based. Examples: {examples}
                    Avoid topics: {previous_topics_text}
                    Question:"""
                    
                    question = generate_text(question_prompt, max_len=100, temperature=0.8)
                    
                    # If question is too short or generic, use fallback
                    if len(question.split()) < 5:
                        question = get_fallback_response("")
                    
                    # Generate expected answer for scoring
                    answer_prompt = f"""As a {track} expert, provide key points for answering: "{question}"
                    List the main technical concepts and approaches:"""
                    
                    expected_answer = generate_text(answer_prompt, max_len=150)
                    
                    st.session_state.questions.append(question)
                    st.session_state.expected_answers.append(expected_answer)

            # Ask question if not already asked
            current_question = st.session_state.questions[current_q_index]
            question_asked = any(
                msg.get("question_id") == current_q_index and msg["role"] == "Interviewer"
                for msg in st.session_state.get('conversation', [])
            )
            
            if not question_asked:
                add_to_conversation("Interviewer", current_question, "Interviewer")
                st.session_state.conversation[-1]["question_id"] = current_q_index
                st.rerun()
            
            # Answer form
            with st.form(key=f"answer_form_{current_q_index}"):
                user_answer = st.text_area(
                    "Your answer:", 
                    key=f"answer_{current_q_index}", 
                    height=120,
                    placeholder="Take your time to provide a detailed answer..."
                )
                
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.caption("💡 Tip: Use specific examples and explain your reasoning")
                with col2:
                    submit_answer = st.form_submit_button("Submit Answer", type="primary")
                
                if submit_answer and user_answer.strip():
                    # Add user answer to conversation
                    add_to_conversation("Candidate", user_answer, "Candidate")
                    st.session_state.user_answers.append(user_answer)
                    
                    # Generate coach feedback
                    with st.spinner("📝 Coach analyzing your response..."):
                        feedback_prompt = f"""You are an interview coach providing feedback after this exchange:

QUESTION: {current_question}
CANDIDATE'S ANSWER: {user_answer}

Provide specific, actionable feedback focusing on:
- What they did well in their answer
- 1-2 specific areas for improvement  
- One concrete suggestion for how to strengthen their response

Keep it supportive but honest. Feedback:"""
                        
                        feedback = generate_text(feedback_prompt, max_len=150, temperature=0.7)
                        
                        # Ensure feedback is constructive and not repetitive
                        if len(feedback.strip()) < 20 or "feedback" in feedback.lower():
                            feedback = get_fallback_response("feedback")
                    
                    add_to_conversation("Coach", feedback, "Coach")
                    st.session_state.current_q += 1
                    
                    # Check if interview is finished
                    if st.session_state.current_q >= st.session_state.selected_num_questions:
                        st.session_state.interview_finished = True
                    st.rerun()

        else:
            # Interview completed
            st.session_state.interview_finished = True
            
            # Generate final feedback
            if 'final_feedback_generated' not in st.session_state:
                # Create a summary of their performance
                answer_lengths = [len(ans.split()) for ans in st.session_state.user_answers]
                avg_length = sum(answer_lengths) / len(answer_lengths) if answer_lengths else 0
                detailed_answers = sum(1 for length in answer_lengths if length > 25)
                
                # Quick assessment based on performance metrics
                performance_level = "good" if avg_length > 20 else "basic"
                engagement_level = "high" if detailed_answers > st.session_state.selected_num_questions // 2 else "moderate"
                
                # Fast, template-based feedback with personalization
                overall_feedback = f"""**Overall Performance**: You completed this {st.session_state.selected_difficulty.lower()} level {st.session_state.selected_track} interview with {performance_level} engagement.

**Key Strengths**: 
• Active participation in all {st.session_state.selected_num_questions} questions
• {"Detailed responses showing good technical thinking" if engagement_level == "high" else "Willingness to tackle technical challenges"}

**Areas for Improvement**:
• Structure answers using STAR method (Situation, Task, Action, Result)  
• Include more specific examples from your experience
• {"Focus on explaining implementation details" if st.session_state.selected_track in ["Software Development", "Web Development"] else "Discuss real-world applications and case studies"}

**Next Steps**: Practice mock interviews, prepare concrete examples, and focus on explaining your problem-solving process step-by-step."""

                st.session_state.final_feedback_generated = overall_feedback
            
            add_to_conversation("Coach", f"🎯 **Final Assessment**: {st.session_state.final_feedback_generated}", "Coach")
            st.rerun()

    else:
        # Show completion
        st.success("🎉 Interview completed successfully!")
        
        # Performance summary
        st.subheader("📈 Your Performance Summary")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Questions Completed", 
                f"{st.session_state.selected_num_questions}/{st.session_state.selected_num_questions}",
                "100%"
            )
        
        with col2:
            # Calculate engagement score based on answer length
            if st.session_state.user_answers:
                avg_length = sum(len(ans.split()) for ans in st.session_state.user_answers) / len(st.session_state.user_answers)
                engagement_score = min(95, max(65, int(avg_length * 2)))
            else:
                engagement_score = 70
            st.metric("Engagement Score", f"{engagement_score}%")
        
        with col3:
            improvement_areas = random.randint(2, 4)
            st.metric("Growth Areas Identified", improvement_areas)
        
        # Detailed feedback section
        with st.expander("📋 Detailed Performance Analysis"):
            st.markdown("### Strengths Observed:")
            st.write("✅ Active participation in the interview process")
            st.write("✅ Willingness to engage with technical questions")
            if any(len(ans.split()) > 30 for ans in st.session_state.user_answers):
                st.write("✅ Provided detailed responses to questions")
            
            st.markdown("### Recommendations for Improvement:")
            st.write("📈 Practice the STAR method (Situation, Task, Action, Result)")
            st.write("📈 Include more specific examples from your experience")
            st.write("📈 Focus on explaining your thought process clearly")
            
            st.markdown("### Next Steps:")
            st.write(f"🎯 Continue practicing {st.session_state.selected_track} concepts")
            st.write("🎯 Work on structuring your responses more effectively")
            st.write("🎯 Practice explaining complex topics in simple terms")
        
        # Reset button
        if st.button("🔄 Start New Interview", type="primary", use_container_width=True):
            # Clear interview-related session state
            for key in ['current_q', 'user_answers', 'conversation', 'interview_finished', 
                       'questions', 'expected_answers', 'settings_confirmed', 
                       'final_feedback_generated']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

# Interview tips (always visible)
with st.expander("💡 Interview Success Tips"):
    st.markdown("""
    **Before You Start:**
    - Take a moment to understand each question fully
    - Think about real examples from your experience
    - Structure your thoughts before answering
    
    **During the Interview:**
    - Use the STAR method: Situation, Task, Action, Result
    - Be specific with numbers, technologies, and outcomes
    - It's okay to pause and think before answering
    - Ask for clarification if a question is unclear
    
    **Answer Structure:**
    - Start with a direct answer to the question
    - Provide context and background
    - Explain your approach or methodology
    - Share the outcome and what you learned
    
    **Common Mistakes to Avoid:**
    - Being too vague or general in your responses
    - Not providing concrete examples
    - Rushing through your answers
    - Forgetting to explain your reasoning process
    """)

# Footer
st.markdown("---")
st.caption("🤖 AI-Powered Interview Simulator | Realistic practice with instant feedback")