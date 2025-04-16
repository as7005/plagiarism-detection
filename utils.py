# src/utils.py

import numpy as np
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow_hub as hub
import nltk
from nltk.util import ngrams
from typing import List
import re

nltk.download('punkt')
nlp = spacy.load("en_core_web_sm")
use_model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

def clean_text(text: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return text.lower().strip()

def get_tfidf_cosine(text1: str, text2: str) -> float:
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform([text1, text2])
    return cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]

def get_ngrams_overlap(text1: str, text2: str, n: int = 3) -> float:
    ngrams1 = set(ngrams(nltk.word_tokenize(text1), n))
    ngrams2 = set(ngrams(nltk.word_tokenize(text2), n))
    overlap = ngrams1.intersection(ngrams2)
    return len(overlap) / max(len(ngrams1), len(ngrams2), 1)

def get_jaccard_similarity(text1: str, text2: str) -> float:
    tokens1 = set(nltk.word_tokenize(text1))
    tokens2 = set(nltk.word_tokenize(text2))
    intersection = tokens1.intersection(tokens2)
    union = tokens1.union(tokens2)
    return len(intersection) / len(union)

def get_use_similarity(text1: str, text2: str) -> float:
    embeddings = use_model([text1, text2])
    return cosine_similarity([embeddings[0].numpy()], [embeddings[1].numpy()])[0][0]

def get_semantic_roles(text: str) -> List[str]:
    doc = nlp(text)
    roles = []
    for token in doc:
        if token.dep_ in ["nsubj", "dobj", "agent", "iobj"]:
            roles.append(f"{token.text} ({token.dep_})")
    return roles

def detect_plagiarism(text1: str, text2: str) -> dict:
    text1_clean = clean_text(text1)
    text2_clean = clean_text(text2)

    results = {
        "TF-IDF Cosine Similarity": get_tfidf_cosine(text1_clean, text2_clean),
        "N-gram Overlap (trigram)": get_ngrams_overlap(text1_clean, text2_clean),
        "Jaccard Similarity": get_jaccard_similarity(text1_clean, text2_clean),
        "USE Cosine Similarity": get_use_similarity(text1, text2),
        "Semantic Roles Text1": get_semantic_roles(text1),
        "Semantic Roles Text2": get_semantic_roles(text2),
    }
    return results
