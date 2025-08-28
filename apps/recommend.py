import streamlit as st
import pandas as pd
import joblib
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ============================
# PAGE CONFIGURATION
# ============================
st.set_page_config(
    page_title="Personalized Course Recommendation System",
    page_icon="🎓",
    layout="wide"
)

# ============================
# LOAD DATA & MODEL
# ============================
@st.cache_resource
def load_model_and_data():
    # Load your course dataset and precomputed embeddings
    data = joblib.load("courses_df1.pkl")
    embeddings = joblib.load("embeddings1.pkl")

    # Load SentenceTransformer model directly instead of using embedding_model1.pkl
    model = SentenceTransformer("all-MiniLM-L6-v2")

    return data, embeddings, model

data, embeddings, model = load_model_and_data()

# ============================
# CUSTOM CSS STYLING
# ============================
st.markdown("""
    <style>
    body {
        background-color: #f8f9fa;
        color: #222;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .main-title {
        text-align: center;
        font-size: 38px;
        color: #4a90e2;
        margin-bottom: 20px;
    }
    .sub-title {
        text-align: center;
        font-size: 20px;
        color: #333;
        margin-bottom: 40px;
    }
    .stButton>button {
        background-color: #4a90e2;
        color: white;
        border-radius: 10px;
        padding: 10px 24px;
        font-size: 18px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #357ABD;
    }
    .course-card {
        padding: 15px;
        margin: 10px;
        border-radius: 10px;
        background-color: #ffffff;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# ============================
# HEADER
# ============================
st.markdown('<h1 class="main-title">🎓 Personalized Course Recommendation System</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Find the best courses based on your interests 🚀</p>', unsafe_allow_html=True)

# ============================
# USER INPUT
# ============================
search_query = st.text_input("🔍 Type a keyword, skill, or topic:", "")

# ============================
# RECOMMENDATION FUNCTION
# ============================
def get_recommendations(query, top_n=5):
    query_embedding = model.encode([query])
    similarity_scores = cosine_similarity(query_embedding, embeddings)
    top_indices = similarity_scores[0].argsort()[-top_n:][::-1]
    return data.iloc[top_indices]

# ============================
# SHOW RECOMMENDATIONS
# ============================
if st.button("Find Courses"):
    if search_query.strip() == "":
        st.warning("⚠️ Please enter a keyword to search!")
    else:
        st.subheader("🎯 Recommended Courses for You:")
        results = get_recommendations(search_query, top_n=5)

        for idx, row in results.iterrows():
            st.markdown(f"""
                <div class="course-card">
                    <h3>{row['Course Name']}</h3>
                    <p><strong>University:</strong> {row['University']}</p>
                    <p><strong>Difficulty:</strong> {row['Difficulty Level']}</p>
                    <p>{row['Course Description'][:300]}...</p>
                    <p><strong>Skills:</strong> {row['Skills']}</p>
                </div>
            """, unsafe_allow_html=True)

# ============================
# FOOTER
# ============================
st.markdown("""
    <br><br>
    <hr>
    <p style="text-align: center; color: gray;">
        Built with ❤️ using <strong>Streamlit</strong> & <strong>Sentence Transformers</strong>
    </p>
""", unsafe_allow_html=True)
