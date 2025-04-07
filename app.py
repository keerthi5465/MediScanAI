import streamlit as st
import torch
import numpy as np
from PIL import Image
import os
from dotenv import load_dotenv
import requests
from torchvision import transforms
from models.model import load_model, DISEASE_LABELS
from utils.visualization import apply_gradcam
from utils.report import get_predictions, format_findings


load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "llama-3.1-8b-instant"

def setup_page():
    st.set_page_config(
        page_title="X-ray Diagnosis System",
        page_icon="🏥",
        layout="wide"
    )
    st.title("AI-Driven X-ray Diagnosis & Report Generator")

def load_and_preprocess_image(image_file):
    """Load and preprocess the uploaded X-ray image."""
    image = Image.open(image_file).convert('RGB')
    image = image.resize((384, 384))
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0), image

def call_groq_api(findings, visual_features):
    """Generate diagnostic report using Groq LLM."""
    if not GROQ_API_KEY:
        st.error("Groq API key not found. Please check your .env file.")
        return None
    
    prompt = f"""As a radiologist, generate a detailed medical report based on the following X-ray findings:
    
    Detected abnormalities: {findings}
    Visual features: {visual_features}
    
    Format the report with these sections:
    1. Findings
    2. Impression
    3. Recommendations"""
    
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 1000
    }
    
    try:
        response = requests.post(GROQ_API_URL, headers=headers, json=data)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        st.error(f"Error generating report: {str(e)}")
        return None

def main():
    setup_page()
    
    # Load model
    model = load_model()
    
    # File uploader
    uploaded_file = st.file_uploader("Upload X-ray Image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        # Display original image
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original X-ray")
            st.image(uploaded_file, use_column_width=True)
        
        # Process image
        input_tensor, original_image = load_and_preprocess_image(uploaded_file)
        
        # Get model predictions
        with torch.no_grad():
            outputs = model(input_tensor)
            predictions, probabilities = get_predictions(outputs)
            findings = format_findings(predictions, probabilities, labels=DISEASE_LABELS)
        
        # Generate heatmap and visual features using Grad-CAM++
        heatmap, visual_features = apply_gradcam(model, input_tensor)
        
        with col2:
            st.subheader("Abnormality Detection")
            st.image(heatmap, use_column_width=True)
        
        # Generate report
        report = call_groq_api(findings, visual_features)
        
        if report:
            st.subheader("AI-Generated Diagnostic Report")
            st.markdown(report)
            
            # Download button for the report
            st.download_button(
                label="Download Report (PDF)",
                data=report.encode(),
                file_name="xray_report.txt",
                mime="text/plain"
            )

if __name__ == "__main__":
    main()