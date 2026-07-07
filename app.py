import streamlit as st
import pickle
import pandas as pd

st.set_page_config(page_title="QA Intelligent Framework", layout="centered")

st.title("🧠 Intelligent Test Case Prioritization & Generation")
st.write("An NLP & ML-driven cloud-native QA framework.")

# Load the saved model, vectorizer, and mapping mapping 
@st.cache_resource
def load_artifacts():
    with open('random_forest_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    with open('step_mapping.pkl', 'rb') as f:
        step_mapping = pickle.load(f)
    return model, vectorizer, step_mapping

try:
    model, vectorizer, step_mapping = load_artifacts()
    st.success("Cloud Core Engine Loaded Successfully!")
except Exception as e:
    st.error(f"Error loading model artifacts: {e}")

# User Input
user_story = st.text_area(
    "Enter Unstructured Natural Language Requirement:",
    placeholder="e.g., As a customer I want to process payments securely via credit cards"
)

if st.button("Analyze and Generate"):
    if user_story.strip() == "":
        st.warning("Please enter a requirement specification string.")
    else:
        # Step 1: Vectorization and Priority Prediction
        vec_text = vectorizer.transform([user_story])
        priority_prediction = model.predict(vec_text)[0]
        
        # Color formatting based on priority risk
        color = "red" if priority_prediction == "High" else "orange" if priority_prediction == "Medium" else "green"
        
        st.markdown(f"### 📊 Execution Priority: :{color}[{priority_prediction}]")
        
        # Step 2: Semantic Step Generation Logic
        st.markdown("### 📋 Structured Functional Testing Pathways")
        
        # Check for exact match or fallback to simple token lookups
        if user_story in step_mapping:
            st.info(step_mapping[user_story])
        else:
            # Fallback heuristic rule-based step engine if it's a completely new string
            if "payment" in user_story.lower() or "credit" in user_story.lower():
                st.info("1. Navigate to checkout interface.\n2. Submit payment token parameters.\n3. Verify payment transaction status confirmation.")
            elif "security" in user_story.lower() or "credential" in user_story.lower() or "password" in user_story.lower():
                st.info("1. Initiate access protocol configuration.\n2. Apply modified administrative security tokens.\n3. Save configuration adjustments.")
            else:
                st.info("1. Access the target module interface view.\n2. Execute state adjustment interaction controls.\n3. Observe UI property mutations.")
