import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from transformers import AutoTokenizer, AutoModel
import numpy as np


def preprocess_text(text):
    """Preprocesses text by removing stop words, punctuation, and stemming."""
    # Tokenize text
    tokens = word_tokenize(text)

    # Remove stop words
    stop_words = set(stopwords.words('english')).union(set(stopwords.words('french')))
    tokens = [word.lower() for word in tokens if word.lower() not in stop_words]

    # Stem words
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(word) for word in tokens]

    # Join tokens into a string
    return ' '.join(stemmed_tokens)


def extract_text_from_pdf(file_path):
    with open(file_path, 'rb') as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        txt = ""
        for page in pdf_reader.pages:
            txt += page.extract_text()
        return txt


def extract_text_from_file(file_path):
    with open(file_path, 'r') as txt_file:
        return txt_file.read()


def score_pdf_content_tfid(test_text, ref_text):
    """Scores PDF content against a specific description."""
    # Vectorize text and description
    vectorizer = TfidfVectorizer()
    text_vector = vectorizer.fit_transform([test_text])
    description_vector = vectorizer.transform([ref_text])

    # Calculate similarity
    similarity = cosine_similarity(text_vector, description_vector)[0][0]

    # Assign score
    return similarity * 100  # Convert similarity to percentage


def score_pdf_content_word_embeddings(test_text, ref_text):
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    # Encode the texts
    inputs = tokenizer(test_text, ref_text, return_tensors="pt", padding=True, truncation=True)

    # Get the last hidden state of the model
    outputs = model(**inputs)
    last_hidden_states = outputs.last_hidden_state

    # Average the embeddings for each text
    text1_embedding = last_hidden_states[0, 0, :].mean(dim=0)
    text2_embedding = last_hidden_states[0, 1, :].mean(dim=0)

    # Calculate cosine similarity
    text_embedding_detached = text1_embedding.detach().numpy()
    description_embedding_detached = text2_embedding.detach().numpy()
    similarity = (np.dot(text_embedding_detached, description_embedding_detached)
                  / (np.linalg.norm(text_embedding_detached) * np.linalg.norm(description_embedding_detached)))
    # Assign score
    return similarity * 100


# Example usage
pdf_path = "cv.pdf"
description_path = "description.txt"

# Extract data
cv = extract_text_from_pdf(pdf_path)
description = extract_text_from_file(description_path)

# Preprocess text
preprocessed_cv = preprocess_text(cv)
preprocessed_description = preprocess_text(description)

score1 = score_pdf_content_tfid(preprocessed_cv, preprocessed_description)
print("Score Tfid:", score1)

score2 = score_pdf_content_word_embeddings(cv, description)
print("Score Word Embeddings:", score2)
