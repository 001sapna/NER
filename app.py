import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import spacy
import os
from spacy import displacy
import subprocess

# Function to install Spacy model if not present
def install_spacy_model():
    try:
        nlp = spacy.load('en_core_web_sm')
    except OSError:
        st.warning("Spacy model 'en_core_web_sm' not found. Installing it now...")
        os.system('python -m spacy download en_core_web_sm')
        nlp = spacy.load('en_core_web_sm')
    return nlp

# Function to load Spacy model with additional logging
@st.cache_resource
def load_spacy_model():
    try:
        nlp = spacy.load('en_core_web_sm')
    except OSError:
        st.warning("Spacy model 'en_core_web_sm' not found. Attempting installation...")
        result = subprocess.run(['python', '-m', 'spacy', 'download', 'en_core_web_sm'], capture_output=True, text=True)
        st.write(result.stdout)
        st.write(result.stderr)
        if result.returncode != 0:
            st.error("Failed to install Spacy model. Please check the logs above.")
            return None
        nlp = spacy.load('en_core_web_sm')
    return nlp

# Load your NER model
@st.cache_resource
def load_ner_model():
    return load_model('ner.keras')

# Define POS tag explanations
pos_explanations = {
    'NNS': 'Noun, plural',
    'IN': 'Preposition or subordinating conjunction',
    'VBP': 'Verb, non-3rd person singular present',
    'VBN': 'Verb, past participle',
    'NNP': 'Proper noun, singular',
    'TO': 'to',
    'VB': 'Verb, base form',
    'DT': 'Determiner',
    'NN': 'Noun, singular or mass',
    'CC': 'Coordinating conjunction',
    'JJ': 'Adjective',
    '.': 'Punctuation mark, sentence closer',
    'VBD': 'Verb, past tense',
    'WP': 'Wh-pronoun',
    '`': 'Opening quotation mark',
    'CD': 'Cardinal number',
    'PRP': 'Personal pronoun',
    'VBZ': 'Verb, 3rd person singular present',
    'POS': 'Possessive ending',
    'VBG': 'Verb, gerund or present participle',
    'RB': 'Adverb',
    ',': 'Punctuation mark, comma',
    'WRB': 'Wh-adverb',
    'PRP$': 'Possessive pronoun',
    'MD': 'Modal',
    'WDT': 'Wh-determiner',
    'JJR': 'Adjective, comparative',
    ':': 'Punctuation mark, colon',
    'JJS': 'Adjective, superlative',
    'WP$': 'Possessive wh-pronoun',
    'RP': 'Particle',
    'PDT': 'Predeterminer',
    'NNPS': 'Proper noun, plural',
    'EX': 'Existential there',
    'RBS': 'Adverb, superlative',
    'LRB': 'Left round bracket',
    'RRB': 'Right round bracket',
    '$': 'Dollar sign',
    'RBR': 'Adverb, comparative',
    ';': 'Punctuation mark, semicolon',
    'UH': 'Interjection',
    'nan': 'Not a Number (missing value)',
}

# Load the models
nlp = load_spacy_model()
if nlp is None:
    st.stop()

model = load_ner_model()

# Function to make predictions
def predict_entities(text):
    doc = nlp(text)
    tokens = [token.text for token in doc]
    input_data = np.array(tokens).reshape(1, -1)  # Adjust according to your model's input format
    predictions = model.predict(input_data)  # Adjust based on your model's prediction method
    return tokens, predictions

# Streamlit app interface
st.title("Named Entity Recognition (NER) with POS Explanations")

# User input
user_input = st.text_area("Enter text:", "Hi, My name is Sapna Kumari. I am from India. I want to work with Google. Steve Jobs is my inspiration.")

# Predict button
if st.button("Predict"):
    tokens, predictions = predict_entities(user_input)
    
    # Display results
    for token, pred in zip(tokens, predictions[0]):  # Adjust indexing based on your model's output
        pos_tag = token.tag_
        pos_explanation = pos_explanations.get(pos_tag, "Unknown")
        st.write(f"{token.text}: {pos_tag} - {pos_explanation}")

# Render NER visualization using spaCy's displacy
doc = nlp(user_input)
html = displacy.render(doc, style='ent', jupyter=False)
st.write(html, unsafe_allow_html=True)
