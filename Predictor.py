import os
import fitz
import re
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import numpy as np

import string
text_to_remove = ["Turn over", "(i)", "(ii)", "(iii)", ""]
text_to_replace = {'_' : "-", 'xeR' : "x in R", 'Find' : "find", "\n" : "", "Edexcel Internal Review" : "", "PhysicsAndMathsTutor.com" : "", "" : "",
                   'DO NOT WRITE IN THIS AREA' : "", '--' : "", '		 ' : "", '  ' : "", '	' : " ", '\t' : " ", '\xa0' : " ", "," : "",
                  }
M = ("1.", "2.", "3.", "4.", "5.", "6.")
symbols_to_keep = "°<>%()=π≤+-_√≥∫≠∈*θ"
# Define a whitelist of characters: ASCII printable + specified symbols
whitelist = string.ascii_letters + string.digits + string.punctuation + string.whitespace + symbols_to_keep

def remove_empty_strings(d): 
    if isinstance(d, dict): 
        return {k: remove_empty_strings(v) for k, v in d.items() if v != ''} 
    elif isinstance(d, list): 
        return [remove_empty_strings(i) for i in d if i != '']
    else: return d 
    

def extract_question_regions_single(pdf_page):
    """
    Extracts question regions from a given PDF page using PyMuPDF.
    
    :param pdf_page: A fitz.Page object representing the current page.
    :return: A list of tuples [(x1, y1, x2, y2, label)] for each detected question region.
    """
    # Get text blocks from the page
    text_blocks = pdf_page.get_text("dict")["blocks"]
    question_regions = []
    question_marks = []
    page_rect = pdf_page.rect
    current_region = None
    start_found = False

    for block in text_blocks:
        if "lines" not in block:  # Skip blocks that don't have 'lines'
            continue
            
        for line in block["lines"]:
            for span in line["spans"]:
                text = span["text"].strip()
                bbox = span["bbox"]  # Bounding box: (x1, y1, x2, y2)
                x1, y1, x2, y2 = bbox

                # Look for question number pattern like "1.", "2.", etc.
                if re.match(r"^Q?\d+\.$", text) and not start_found:
                    current_region = { "x1" : x1, 
                                       "y1" : y1 - 10, 

                                     }
                    start_found = True
                    match = re.search(r"Q?(\d{1,2})\.", text)
                    if match:
                        label = int(match.group(1))  # Extract the integer part
                    current_region["label"] = label

                #Look for "(x)" pattern if it's a pdf where one question takes up one page
                if re.match(r"^\(\d+\)", text) and start_found:
                    question_marks.append((x2, y2))

    if question_marks:
        x2, y2 = question_marks[-1]
        current_region["x2"] = x2
        current_region["y2"] = y2
    
    # Add the final region if it exists
    if current_region:
        question_regions.append(current_region)

    # Convert to a list of tuples for consistency
    try:
        return [(r["x1"], r["y1"], r["x2"], r["y2"], r["label"]) for r in question_regions]
    except KeyError:
        return False


def extract_question_regions(pdf_page):
    """
    Extracts question regions from a given PDF page using PyMuPDF.

    :param pdf_page: A fitz.Page object representing the current page.
    :return: A list of tuples [(x1, y1, x2, y2, label)] for each detected question region.
    """
    text_blocks = pdf_page.get_text("dict")["blocks"]
    question_regions = []
    current_region = None
    start_found = False

    for block in text_blocks:
        if "lines" not in block:
            continue

        for line in block["lines"]:
            for span in line["spans"]:
                text = span["text"].strip()
                bbox = span["bbox"]  # Bounding box: (x1, y1, x2, y2)
                x1, y1, x2, y2 = bbox

                # Detect end marker "(Total X marks)" or "(X)"
                if re.match(r"^\(Total.*", text): #or re.match(r"^\(\d+\)", text):
                    if current_region:
                        current_region["x2"] = max(current_region["x2"], x2)
                        current_region["y2"] = max(current_region["y2"], y2)
                        question_regions.append(current_region)
                        current_region = None
                        start_found = False

                # Detect start marker like "1.", "2.", etc.
                elif re.match(r"(Q?\d{1,2}\.)", text) and not start_found:
                    if current_region:
                        question_regions.append(current_region)
                    
                    current_region = {
                        "x1": x1,
                        "y1": y1 - 10,  # Small buffer
                        "x2": x2,
                        "y2": y2,
                        #"label": int(re.sub(r"[.]", "", text)),  # Extract number
                        "label": int(re.sub(r"[Q.]", "", text)),
                    }
                    start_found = True

                # Expand the current question region
                if current_region:
                    current_region["x2"] = max(current_region["x2"], x2)
                    current_region["y2"] = max(current_region["y2"], y2)

    # Add the last region if it exists
    if current_region:
        question_regions.append(current_region)

    return [(r["x1"], r["y1"], r["x2"], r["y2"], r["label"]) for r in question_regions]

def clean(question):
    # Clean the text
    for old, new in text_to_replace.items():
        question = question.replace(old, new)
                    
    question = re.sub(r"\(\d{1,2}\)", "", question)
    question = re.sub(r"Q?\d{1,2}\.", "", question)
    question = re.sub(r"\(Total \d{1,2} marks\)", "", question)
    question = re.sub(r"\(Total for question = \d+ marks\)", "", question)
    question = re.sub(r"\([a-fA-F]\)", "", question)
    question = "".join(filter(lambda x: x in whitelist, question))
    question = re.sub(r" {2,}", "  ", question)
    question = re.sub(r" +", " ", question)
    return question

def read_page(pdf_path, page_num, mixed):
    pdf_document = fitz.open(pdf_path)
    page = pdf_document[page_num]  # This is the correct fitz.Page object
    
    if mixed:
        regions = extract_question_regions_single(page)
    else:
        regions = extract_question_regions(page)

    if regions != False:
        for r in regions:
            # Use PDF reader to write the contents of the question to a .csv
            (x1, y1, x2, y2, label) = r
            rect = fitz.Rect(x1, y1, x2, y2)  # Create a rectangle object for the region

            question = clean(page.get_text("text", clip=rect))
            return question
            
    else:
        print(f"No questions found on page {page_num+1}")#, but here's what I could read: ", page.get_text("text"))
    

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

        # Perform inference
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits.to(device)
        probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()

        # Find indices of the top two probabilities
        top_indices = probs.argsort()[-2:][::-1]

        # Create a binary prediction array for these top indices
        predicted_labels = [0] * len(probs)
        for idx in top_indices:
            if probs[idx] > threshold:
                predicted_labels[idx] = 1

    # Reshape `predicted_labels` to ensure it is 2D
    predicted_labels = np.array(predicted_labels).reshape(1, -1)

    # Map back to topics using the MultiLabelBinarizer
    predicted_topics = mlb.inverse_transform(predicted_labels)

    return predicted_topics[0]

def add_text_to_pdf_page(pdf_path, text, page_number):
    """
    Adds text to the top-right corner of a specific page in a PDF and overwrites the original file.
    
    :param pdf_path: Path to the input PDF file.
    :param text: The text to add to the top-right corner.
    :param page_number: The page number (1-based index) where the text should be added.
    """
    pdf_document = fitz.open(pdf_path)

    if page_number < 1 or page_number > len(pdf_document):
        raise ValueError(f"Page number {page_number} is out of range. The document has {len(pdf_document)} pages.")

    page = pdf_document[page_number - 1]
    page_width = page.rect.width
    text_x = page_width - 200
    text_y = 20
    page.insert_text((text_x, text_y), text, fontsize=12, color=(0, 0, 0))

    # Fully rewrite the PDF (disables incremental saving)
    output_path = "output.pdf"
    pdf_document.save(output_path)
    pdf_document.close()
    print(f"Text added and saved to {output_path}.")

# Load the saved model and tokenizer
model = BertForSequenceClassification.from_pretrained('./bert_multilabel_model')
tokenizer = BertTokenizer.from_pretrained('./bert_multilabel_model')
mlb = joblib.load('./mlb.pkl')

def question_to_topic(pdf_path,page_num,mixed):
    question = read_page(pdf_path, page_num,mixed)
    topics = predict_topics(question, model, tokenizer, mlb)
    print("Predicted Topics:", [f'{topic}' for topic in topics] )

def pdfs_to_topic_pdfs(pdf_paths, mixed, output_dir):
    topic_pages = {}

    # First Pass: Classify Questions
    for pdf_path in pdf_paths:
        file_name = os.path.basename(pdf_path)
        try:
            with fitz.open(pdf_path) as pdf_document:
                for i in range(pdf_document.page_count):
                    question = read_page(pdf_path, i, mixed)
                    if question:
                        topics = predict_topics(question, model, tokenizer, mlb, threshold=0.3)
                        for topic in topics:
                            if topic not in topic_pages:
                                topic_pages[topic] = []
                            topic_pages[topic].append((pdf_path, i, file_name))
        except fitz.EmptyFileError:
            print(f"Skipped empty or invalid PDF: {file_name}")
            continue

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Second Pass: Generate Topic-Specific PDFs
    for topic, pages in topic_pages.items():
        output_pdf = fitz.open()
        for pdf_path, page_number, file_name in pages:
            with fitz.open(pdf_path) as source_pdf:
                page = source_pdf.load_page(page_number)
                
                # Insert file name text at the top right of the page
                text_x = 20
                text_y = 20
                page.insert_text((text_x, text_y), file_name[:-4], fontsize=12, color=(0, 0, 0))

                # Add the modified page to the topic-specific PDF
                output_pdf.insert_pdf(source_pdf, from_page=page_number, to_page=page_number)

        # Save and close the topic-specific PDF
        output_path = os.path.join(output_dir, f"{topic}.pdf")
        output_pdf.save(output_path)
        output_pdf.close()
        print(f"Saved {topic} PDF with {len(pages)} pages to {output_path}.")
