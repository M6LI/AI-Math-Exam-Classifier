import streamlit as st
import tempfile
import os
from transformers import BertTokenizer, BertForSequenceClassification
import joblib
import fitz  # PyMuPDF
import shutil
import gc

from Predictor import pdfs_to_topic_pdfs  # Ensure this function handles file paths correctly

# Load model and tokenizer
model = BertForSequenceClassification.from_pretrained('bert_multilabel_model')
tokenizer = BertTokenizer.from_pretrained('bert_multilabel_model')
mlb = joblib.load('mlb.pkl')

# Streamlit UI Setup
st.title("Math Question Topic Classifier")
st.write("Upload one or more question papers in PDF format, and the app will classify and sort questions into topic-specific PDFs.")

# File uploader (multiple files allowed)
uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)

# Checkbox for mixed questions per page
mixed_questions = st.checkbox("Does each page have a mix of multiple questions?", value=False)

# Process uploaded files
if uploaded_files:
    temp_dir = tempfile.mkdtemp()  # Create a temporary directory manually

    try:
        pdf_paths = []
        for uploaded_file in uploaded_files:
            temp_pdf_path = os.path.join(temp_dir, uploaded_file.name)
            
            # Save the uploaded file
            with open(temp_pdf_path, "wb") as f:
                f.write(uploaded_file.read())
                f.flush()

            # Check if file is non-empty and valid
            if os.path.getsize(temp_pdf_path) > 0:
                try:
                    pdf_document = fitz.open(temp_pdf_path)
                    st.write(f"Successfully opened {uploaded_file.name} with {pdf_document.page_count} pages.")
                    pdf_paths.append(temp_pdf_path)
                    pdf_document.close()
                except fitz.EmptyFileError:
                    st.error(f"Failed to open {uploaded_file.name}. File might be corrupted or unreadable.")
            else:
                st.error(f"File {uploaded_file.name} is empty. Please upload a valid PDF.")

        if pdf_paths:
            # Output directory for topic PDFs
            output_dir = os.path.join(temp_dir, "topic_pdfs")
            os.makedirs(output_dir, exist_ok=True)

            # Run classification and generate topic-specific PDFs
            st.write("Processing your files... This may take a few moments.")
            pdfs_to_topic_pdfs(pdf_paths, mixed_questions, output_dir)

            # Provide download links for generated PDFs
            st.write("Download your topic-based PDFs:")
            for topic_pdf in os.listdir(output_dir):
                topic_pdf_path = os.path.join(output_dir, topic_pdf)
                
                # Read the file into memory
                with open(topic_pdf_path, "rb") as f:
                    file_data = f.read()

                # Now pass the file data to Streamlit without keeping the file open
                st.download_button(
                    label=f"Download {topic_pdf}",
                    data=file_data,
                    file_name=topic_pdf,
                    mime="application/pdf"
                )

    finally:
        # Force garbage collection to ensure all file handles are released
        gc.collect()

        # Ensure file permissions allow deletion, and handle locked files
        def handle_remove_readonly(func, path, exc_info):
            import stat
            os.chmod(path, stat.S_IWRITE)
            func(path)

        # Remove the temporary directory and handle permission issues
        shutil.rmtree(temp_dir, onerror=handle_remove_readonly)
