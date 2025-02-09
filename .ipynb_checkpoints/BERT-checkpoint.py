import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# Load model and tokenizer only once
model = BertForSequenceClassification.from_pretrained('bert_multilabel_model')
tokenizer = BertTokenizer.from_pretrained('bert_multilabel_model')
mlb = joblib.load('mlb.pkl')

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

question = "Integrate sin(x)dx"
topics = predict_topics(question, model, tokenizer, mlb, threshold=0.3)
print("Predicted Topics:", [f'{topic}' for topic in topics] )