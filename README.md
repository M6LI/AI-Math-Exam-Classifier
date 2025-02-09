# AI-Math-Exam-Classifier

An AI-powered web app that classifies A-level math exam questions into their respective topics using BERT (Bidirectional Encoder Representations from Transformers). Simply upload a PDF of a math question paper, and the app will label each question by topic and provide a downloadable, annotated version.

## Demo

(Live app to be added very soon)

## Features

- Automated Topic Classification: Uses BERT for multi-label classification of math questions.
- PDF Processing: Upload your question paper in PDF format and get PDFs back which contain questions on each topic.
- Multi-label Support: Questions belonging to multiple topics are accurately tagged.
- User-friendly Interface: Powered by Streamlit for a seamless experience.

## How It Works

1. PDF Upload: Users upload a math question paper in PDF format.
2. Text Extraction: The app extracts questions using PyMuPDF.
3. Topic Classification: Each question is processed using a fine-tuned BERT model that predicts relevant math topics.
4. PDF Annotation: The predicted topics are added to the original PDF, which can be downloaded.

## Machine Learning Techniques Used

- BERT for Sequence Classification: The model was fine-tuned on a custom dataset of math questions labeled with multiple topics.
- Multi-label Classification: The app supports multi-topic detection using BCEWithLogitsLoss and threshold-based predictions.
- Text Preprocessing: Custom preprocessing for math-specific text structures, ensuring accurate tokenization and classification.

## Getting Started

### Prerequisites

Ensure you have Python 3.7+ installed. Then install the required packages:

```pip install -r requirements.txt```

### Running Locally

Clone the repository:

```git clone https://github.com/your-username/your-repo-name.git cd your-repo-name```

Run the app:

```streamlit run app.py```

Open your browser and go to http://localhost:8501.

### Deploying the App

Streamlit Cloud:

Push this repository to GitHub.

Go to Streamlit Cloud, link your GitHub, and deploy!

### Folder Structure

```.
├── app.py                  # Streamlit application
├── Predictor.py            # Core logic for question extraction and classification
├── bert_multilabel_model/  # Saved BERT model files
├── mlb.pkl                 # MultiLabelBinarizer for decoding topics
├── requirements.txt        # Dependencies
└── README.md               # Project documentation
```
