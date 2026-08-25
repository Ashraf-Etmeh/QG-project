'''Shared NLP model instances.

Loaded once at import time and reused across requests instead of being
reloaded (spaCy) or resolved over the network (SentenceTransformer) on
every question-generation call.
'''
import spacy
from sentence_transformers import SentenceTransformer

nlp = spacy.load('en_core_web_md')
semantic_model = SentenceTransformer('./models/all-MiniLM-L6-v2')
