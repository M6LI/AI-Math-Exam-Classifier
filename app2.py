import streamlit as st
import os
import tempfile
import Predictor
from Predictor import *
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
import joblib
import torch

# Streamlit UI Setup
st.title("Math Question Topic Classifier")
st.write("Upload a question paper in PDF format, and the app will classify and label each question by topic.")

# Function to predict topics for a given question
def predict_topics(question, model, tokenizer, mlb, threshold=0.5):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)  # Ensure the model is on the same device as inputs
    model.eval()

    with torch.no_grad():
        # Tokenize the input and move to the appropriate device
        encoding = tokenizer(
            question,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Ensure all inputs are on the correct device
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)

        # # Device consistency checks
        # print("Model device:", next(model.parameters()).device)
        # print("Input IDs device:", input_ids.device)
        # print("Attention mask device:", attention_mask.device)

        # Perform inference
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits.to(device)
        probs = torch.sigmoid(logits).squeeze(0)

        # Convert to binary predictions based on the threshold
        predicted_labels = (probs > threshold).int().cpu().numpy()

    # Reshape `predicted_labels` to ensure it is 2D
    predicted_labels = predicted_labels.reshape(1, -1)

    # Map back to topics using the MultiLabelBinarizer
    predicted_topics = mlb.inverse_transform(predicted_labels)

    return predicted_topics[0]

# Load the saved model and tokenizer
model = BertForSequenceClassification.from_pretrained('bert_multilabel_model')
tokenizer = BertTokenizer.from_pretrained('bert_multilabel_model')
mlb = joblib.load('mlb.pkl')

question = "Integrate sin(x)dx"
topics = predict_topics(question, model, tokenizer, mlb, threshold=0.3)
print("Predicted Topics:", [f'{topic}' for topic in topics] )

# File uploader
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

# Option to specify if the PDF has mixed questions per page
mixed_questions = st.checkbox("Does each page have a mix of multiple questions?", value=False)

# Process the uploaded file
if uploaded_file is not None:
    # Save the uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_pdf_path = tmp_file.name
    
    output_path = tmp_pdf_path.replace(".pdf", "_labelled.pdf")
    
    # Run the topic classification
    st.write("Processing your file... This may take a few moments.")
    pdf_to_topic(tmp_pdf_path, mixed_questions, output_path)
    
    # Provide the labelled PDF for download
    with open(output_path, "rb") as f:
        st.download_button(
            label="Download Labelled PDF",
            data=f,
            file_name="labelled_question_paper.pdf",
            mime="application/pdf"
        )

    # Clean up temporary files
    os.remove(tmp_pdf_path)
    os.remove(output_path)
