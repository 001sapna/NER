import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import spacy

# Load NER model
@st.cache(allow_output_mutation=True)
def load_ner_model():
    return load_model('ner.keras')

# Define POS tags with their meanings
pos_dict = {
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

# Load spaCy model for tokenization and POS tagging
nlp = spacy.load('en_core_web_sm')

def predict_entities(text, model):
    doc = nlp(text)
    entities = []
    for ent in doc.ents:
        entities.append(f'{ent.text} {ent.label_}')
    return ' '.join(entities)

def pos_meaning(pos):
    if pos in pos_dict:
        return pos_dict[pos]
    else:
        return 'Not Available'

# Streamlit app
def main():
    st.title('Named Entity Recognition with POS Explanation')
    
    # Input text area
    text = st.text_area('Enter text:')
    
    # Load model
    model = load_ner_model()
    
    if st.button('Analyze'):
        if text:
            st.write('**POS Tagging with Meanings:**')
            doc = nlp(text)
            
            # Display tokens with POS tags and meanings
            for token in doc:
                st.write(f'{token.text} ({pos_meaning(token.pos_)})', end=' ')
            
            st.write('\n\n**Named Entities:**')
            entities = predict_entities(text, model)
            st.write(entities)

if __name__ == '__main__':
    main()
