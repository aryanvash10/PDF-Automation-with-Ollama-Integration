import fitz  # PyMuPDF
from typing import Set
import ollama
import numpy as np
import sys

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from a PDF file using PyMuPDF."""
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text("text")
        doc.close()
        return text.strip()
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
        return ""

def tokenize_text(text: str) -> Set[str]:
    """Tokenize text into a set of words."""
    return set(text.lower().split())

def jaccard_similarity(set1: Set[str], set2: Set[str]) -> float:
    """Compute Jaccard Similarity between two sets."""
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0

def get_ollama_embeddings(text: str, model: str = "phi:latest") -> np.ndarray:
    """Generate embeddings for text using Ollama."""
    try:
        response = ollama.embeddings(model=model, prompt=text)
        return np.array(response["embedding"])
    except Exception as e:
        print(f"Error generating embeddings with {model}: {e}")
        print(f"Ensure Ollama server is running ('ollama serve') and model '{model}' is pulled.")
        return np.array([])

def compute_cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """Compute Cosine Similarity between two embeddings."""
    if emb1.size == 0 or emb2.size == 0:
        return 0.0
    try:
        return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    except Exception as e:
        print(f"Error computing cosine similarity: {e}")
        return 0.0

def main(pdf1_path: str, pdf2_path: str, model: str = "phi:latest", max_chars: int = 10000):
    """Compare two PDFs using Jaccard Similarity and Ollama embeddings."""
    # Check if Ollama server is running
    try:
        ollama.list()  # Test connection to Ollama
    except Exception as e:
        print(f"Ollama server not running or not installed: {e}")
        print(f"Start Ollama with 'ollama serve' and ensure model '{model}' is pulled.")
        sys.exit(1)
    
    # Extract text from PDFs
    print("Extracting text from PDFs...")
    text1 = extract_text_from_pdf(pdf1_path)
    text2 = extract_text_from_pdf(pdf2_path)
    
    if not text1 or not text2:
        print("Failed to extract text from one or both PDFs.")
        sys.exit(1)
    
    # Compute Jaccard Similarity
    print("Computing Jaccard Similarity...")
    tokens1 = tokenize_text(text1)
    tokens2 = tokenize_text(text2)
    jaccard_score = jaccard_similarity(tokens1, tokens2)
    
    # Truncate text for Ollama to avoid overloading
    text1_truncated = text1[:max_chars]
    text2_truncated = text2[:max_chars]
    
    # Compute Cosine Similarity using Ollama embeddings
    print(f"Generating embeddings with {model}...")
    emb1 = get_ollama_embeddings(text1_truncated, model)
    emb2 = get_ollama_embeddings(text2_truncated, model)
    cosine_score = compute_cosine_similarity(emb1, emb2)
    
    # Output results
    print(f"Jaccard Similarity: {jaccard_score:.3f}")
    print(f"Cosine Similarity (Ollama embeddings): {cosine_score:.3f}")

if __name__ == "__main__":
    # Paths for your PDFs
    pdf1_path = "E:/PDFmuTutorial/large1.pdf"
    pdf2_path = "E:/PDFmuTutorial/large2.pdf"
    main(pdf1_path, pdf2_path, model="phi:latest", max_chars=10000)