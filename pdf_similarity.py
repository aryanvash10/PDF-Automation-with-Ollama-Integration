import fitz  # PyMuPDF
import re
import logging
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from datetime import datetime
import os
from typing import List, Set, Tuple
import argparse

# Configure logging for debugging and monitoring
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pdf_similarity.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from a PDF file using PyMuPDF, optimized for large files."""
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page_num, page in enumerate(doc, 1):
            text += page.get_text()
            if page_num % 100 == 0:
                logger.info(f"Processed {page_num} pages of {pdf_path}")
        doc.close()
        
        if not text.strip():
            logger.warning(f"No text extracted from {pdf_path}")
            return ""
        
        logger.info(f"Extracted {len(text)} characters from {pdf_path}")
        return text
    except FileNotFoundError:
        logger.error(f"File not found: {pdf_path}")
        return ""
    except Exception as e:
        logger.error(f"Error extracting text from {pdf_path}: {e}")
        return ""

def generate_ngrams(text: str, n: int = 5) -> Set[str]:
    """Generate n-grams (1 to n) from text, excluding stop words."""
    try:
        # Convert to lowercase and remove punctuation
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        
        # Split into words
        words = text.split()
        
        # Stop words (extended from your previous script)
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she',
            'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your',
            'his', 'her', 'its', 'our', 'their', 'am', 'as', 'not', 'no', 'yes',
            'while', 'over', 'full'
        }
        
        # Filter words: keep alphanumeric, length > 2, not stop words
        words = [w for w in words if w.isalnum() and len(w) > 2 and w not in stop_words]
        
        # Generate n-grams (1 to n)
        ngrams = set()
        for i in range(1, n + 1):
            for j in range(len(words) - i + 1):
                ngram = ' '.join(words[j:j+i])
                ngrams.add(ngram)
        
        logger.info(f"Generated {len(ngrams)} n-grams (1 to {n}-grams)")
        return ngrams
    except Exception as e:
        logger.error(f"Error generating n-grams: {e}")
        return set()

def get_vector_embeddings(text: str, model: SentenceTransformer) -> np.ndarray:
    """Generate vector embeddings for text using SentenceTransformer."""
    try:
        # Split text into sentences for embedding
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        if not sentences:
            logger.warning("No valid sentences for embedding")
            return np.zeros(384)  # Default dimension for all-MiniLM-L6-v2
        
        embeddings = model.encode(sentences, show_progress_bar=False)
        # Average embeddings for document-level representation
        doc_embedding = np.mean(embeddings, axis=0)
        logger.info(f"Generated vector embedding with shape {doc_embedding.shape}")
        return doc_embedding
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        return np.zeros(384)

def jaccard_similarity(set1: Set[str], set2: Set[str]) -> float:
    """Calculate Jaccard similarity between two sets."""
    if not set1 and not set2:
        return 1.0
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    similarity = len(intersection) / len(union) if union else 0.0
    logger.info(f"Jaccard: {len(intersection)} common, {len(union)} total, similarity: {similarity:.3f}")
    return similarity

def cosine_similarity_vectors(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    try:
        similarity = cosine_similarity([vec1], [vec2])[0][0]
        logger.info(f"Cosine similarity: {similarity:.3f}")
        return similarity
    except Exception as e:
        logger.error(f"Error calculating cosine similarity: {e}")
        return 0.0

def compare_pdfs(pdf1_path: str, pdf2_path: str, model: SentenceTransformer, n: int = 5) -> Tuple[float, float, List[str]]:
    """Compare two PDFs using n-gram Jaccard and vector cosine similarity."""
    # Extract text
    logger.info(f"Comparing {pdf1_path} and {pdf2_path}")
    text1 = extract_text_from_pdf(pdf1_path)
    text2 = extract_text_from_pdf(pdf2_path)
    
    if not text1 or not text2:
        logger.error("Failed to extract text from one or both PDFs")
        return 0.0, 0.0, []
    
    # Generate n-grams
    ngrams1 = generate_ngrams(text1, n)
    ngrams2 = generate_ngrams(text2, n)
    
    # Calculate Jaccard similarity
    jaccard_sim = jaccard_similarity(ngrams1, ngrams2)
    common_ngrams = list(ngrams1.intersection(ngrams2))[:15]
    
    # Generate vector embeddings
    vec1 = get_vector_embeddings(text1, model)
    vec2 = get_vector_embeddings(text2, model)
    
    # Calculate cosine similarity
    cosine_sim = cosine_similarity_vectors(vec1, vec2)
    
    return jaccard_sim, cosine_sim, common_ngrams

def save_results(pdf1_path: str, pdf2_path: str, jaccard_sim: float, cosine_sim: float, common_ngrams: List[str], output_dir: str = "results"):
    """Save comparison results to CSV and JSON."""
    try:
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results = {
            "pdf1": pdf1_path,
            "pdf2": pdf2_path,
            "jaccard_similarity": jaccard_sim,
            "cosine_similarity": cosine_sim,
            "common_ngrams": common_ngrams
        }
        
        # Save to CSV
        df = pd.DataFrame([results])
        csv_path = os.path.join(output_dir, f"similarity_{timestamp}.csv")
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved results to {csv_path}")
        
        # Save to JSON
        json_path = os.path.join(output_dir, f"similarity_{timestamp}.json")
        df.to_json(json_path, orient="records", lines=True)
        logger.info(f"Saved results to {json_path}")
    except Exception as e:
        logger.error(f"Error saving results: {e}")

def main():
    """Main function to compare PDFs for pattern similarity."""
    parser = argparse.ArgumentParser(description="Compare large PDFs for pattern similarity using n-grams and vector embeddings.")
    parser.add_argument("pdf1", help="Path to first PDF file")
    parser.add_argument("pdf2", help="Path to second PDF file")
    parser.add_argument("--ngram", type=int, default=5, help="Maximum n-gram size (default: 5)")
    parser.add_argument("--output-dir", default="results", help="Output directory for results (default: results)")
    args = parser.parse_args()
    
    # Load SentenceTransformer model
    logger.info("Loading SentenceTransformer model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Compare PDFs
    jaccard_sim, cosine_sim, common_ngrams = compare_pdfs(args.pdf1, args.pdf2, model, args.ngram)
    
    # Print results
    print("=== PDF Pattern Similarity Calculator ===")
    print(f"\nPDF 1: {args.pdf1}")
    print(f"PDF 2: {args.pdf2}")
    print(f"\nN-gram Jaccard Similarity: {jaccard_sim:.3f} ({jaccard_sim*100:.1f}%)")
    print(f"Vector Cosine Similarity: {cosine_sim:.3f} ({cosine_sim*100:.1f}%)")
    print(f"Common N-grams (up to 15): {common_ngrams}")
    
    # Interpret results
    print("\n=== Interpretation ===")
    for sim, name in [(jaccard_sim, "Jaccard"), (cosine_sim, "Cosine")]:
        if sim > 0.8:
            print(f"{name}: Documents are very similar")
        elif sim > 0.5:
            print(f"{name}: Documents are moderately similar")
        elif sim > 0.2:
            print(f"{name}: Documents have some similarity")
        else:
            print(f"{name}: Documents are quite different")
    
    # Save results
    save_results(args.pdf1, args.pdf2, jaccard_sim, cosine_sim, common_ngrams, args.output_dir)

if __name__ == "__main__":
    main()