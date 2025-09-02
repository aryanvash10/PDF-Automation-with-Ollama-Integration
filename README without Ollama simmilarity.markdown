# PDF Pattern Similarity Calculator

A Python script to compare large PDF documents (300–400+ pages) for syntactic and semantic similarity using n-grams and vector embeddings. This tool is designed for analyzing energy reports or similar large PDFs, computing Jaccard similarity (based on n-grams) and cosine similarity (based on SentenceTransformer embeddings), and saving results to CSV/JSON files with detailed logging.

## Table of Contents
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Example](#example)
- [Output Files](#output-files)
- [Notes](#notes)
- [Troubleshooting](#troubleshooting)
- [Optional Enhancements](#optional-enhancements)
- [License](#license)

## Features
- **Text Extraction**: Extracts text from large PDFs (300–400+ pages) using `PyMuPDF`.
- **Syntactic Analysis**: Generates 1- to 5-grams to compute Jaccard similarity, capturing pattern overlap.
- **Semantic Analysis**: Uses `SentenceTransformer` (`all-MiniLM-L6-v2`) to generate 384-dimensional embeddings for cosine similarity.
- **Output**: Saves results to CSV and JSON files, including Jaccard and cosine similarities, common n-grams, and interpretation.
- **Logging**: Detailed logs saved to `pdf_similarity.log` for debugging and tracking.
- **Scalability**: Tested on 400-page PDFs (~34,761 characters) with identical content (Jaccard: 1.000, Cosine: 1.000) and differing content (e.g., Jaccard: 0.059, Cosine: 0.830).
- **Portability**: Runs in a Python virtual environment, isolated to a specified drive (e.g., E:).

## Requirements
- **Operating System**: Windows (tested on Windows with Python 3.10).
- **Python**: Version 3.10.
- **System Dependencies**: [Microsoft Visual C++ Redistributable (x64)](https://aka.ms/vs/17/release/vc_redist.x64.exe) for `PyMuPDF`.
- **Python Libraries**:
  - `pymupdf==1.24.7` (includes `PyMuPDFb==1.24.6`)
  - `sentence-transformers==5.0.0`
  - `numpy==1.23.5`
  - `scikit-learn==1.2.1`
  - `pandas==1.5.3`
  - `tf-keras==2.17.0`
  - `huggingface_hub[hf_xet]` (installs `huggingface_hub==0.34.3` and `hf-xet==1.1.7`)

## Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourcompany/pdf-similarity.git
   cd pdf-similarity
   ```

2. **Create a Virtual Environment**:
   ```bash
   python -m venv .venv
   ```

3. **Activate the Virtual Environment**:
   - On Windows:
     ```bash
     .venv\Scripts\activate
     ```
   - You should see `(.venv)` in your prompt, e.g., `(.venv) PS E:\pdf-similarity>`.

4. **Install Dependencies**:
   ```bash
   pip install pymupdf==1.24.7 sentence-transformers==5.0.0 numpy==1.23.5 scikit-learn==1.2.1 pandas==1.5.3 tf-keras==2.17.0 huggingface_hub[hf_xet]
   ```

5. **Verify Installation**:
   ```bash
   pip list
   python -c "import fitz; doc = fitz.open() # Should not raise errors"
   ```

6. **Install System Dependencies**:
   - Download and install the [Microsoft Visual C++ Redistributable (x64)](https://aka.ms/vs/17/release/vc_redist.x64.exe).

## Usage
Run the script from the command line, specifying two PDF files, the maximum n-gram size, and an output directory.

```bash
python pdf_similarity.py <pdf1_path> <pdf2_path> --ngram <max_n_gram> --output-dir <output_directory>
```

- `<pdf1_path>`, `<pdf2_path>`: Paths to the PDF files to compare (e.g., `large1.pdf`, `large2.pdf`).
- `--ngram`: Maximum n-gram size (default: 5, for 1- to 5-grams).
- `--output-dir`: Directory to save CSV/JSON results (default: `results`).

### Example
```bash
python pdf_similarity.py large1.pdf large2.pdf --ngram 5 --output-dir results
```

**Output**:
```
=== PDF Pattern Similarity Calculator ===
PDF 1: large1.pdf
PDF 2: large2.pdf
N-gram Jaccard Similarity: 1.000 (100.0%)
Vector Cosine Similarity: 1.000 (100.0%)
Common N-grams (up to 15): ['energy report data voltage 2400v', 'voltage 1940v current efficiency', 'page 392 sample energy report', ...]
=== Interpretation ===
Jaccard: Documents are very similar
Cosine: Documents are very similar
2025-08-07 12:35:55,813 - INFO - Saved results to results\similarity_20250807_123555.csv
2025-08-07 12:35:55,815 - INFO - Saved results to results\similarity_20250807_123555.json
```

## Output Files
- **CSV**: `results\similarity_YYYYMMDD_HHMMSS.csv` (e.g., `similarity_20250807_123555.csv`).
  - Contains Jaccard and cosine similarities, common n-grams, and metadata.
- **JSON**: `results\similarity_YYYYMMDD_HHMMSS.json` (e.g., `similarity_20250807_123555.json`).
  - Structured data with the same information as the CSV.
- **Log**: `pdf_similarity.log` in the project root.
  - Logs processing steps, page counts, character counts, and n-gram generation.

To inspect:
```bash
dir results
dir pdf_similarity.log
```

## Notes
- **Performance**: Tested on 400-page PDFs (~34,761 characters) with identical content (Jaccard: 1.000, Cosine: 1.000) and differing content (e.g., Jaccard: 0.059, Cosine: 0.830).
- **Text-Based PDFs**: The script assumes text-based PDFs. For scanned PDFs, add OCR support (see [Optional Enhancements](#optional-enhancements)).
- **TensorFlow Warnings**: Benign warnings from `tf-keras` (e.g., `oneDNN`, `tf.losses`) can be ignored or suppressed:
  ```bash
  set TF_ENABLE_ONEDNN_OPTS=0
  ```
  Add to the script:
  ```python
  import os
  os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
  ```
- **Ollama**: `ollama==0.5.2` is installed but not used. See [Optional Enhancements](#optional-enhancements) for LLM integration.

## Troubleshooting
- **DLL Error**: If you see `ImportError: DLL load failed while importing _extra`:
  - Ensure Microsoft Visual C++ Redistributable (x64) is installed.
  - Reinstall `pymupdf`:
    ```bash
    pip uninstall pymupdf PyMuPDFb -y
    pip install pymupdf==1.24.7
    ```
  - Check for antivirus blocking `E:\pdf-similarity\.venv\Lib\site-packages\pymupdf`.
- **Dependency Conflicts**: Ensure exact versions (`numpy==1.23.5`, `scipy==1.15.3`, etc.) to avoid mismatches.
- **Large PDFs**: For memory issues, increase system RAM or optimize text extraction (contact developers).
- **Windows Path Issues**: Use raw paths (e.g., `E:\path\to\file.pdf`) or double backslashes (e.g., `E:\\path\\to\\file.pdf`).

## Optional Enhancements
1. **Stemming** (merge `dog`/`dogs` for better Jaccard scores):
   ```bash
   pip install nltk
   python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords')"
   ```
   - Modify `pdf_similarity.py` to use `nltk.stem.PorterStemmer`.

2. **LLM Tokenization** (use `llama3.1:8b` for embeddings):
   ```bash
   pip install ollama
   ollama pull llama3.1:8b
   ```
   - Update `pdf_similarity.py` to use `ollama.embeddings(model='llama3.1:8b')`.

3. **OCR for Scanned PDFs**:
   ```bash
   pip install pytesseract
   ```
   - Install [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki).
   - Update `pdf_similarity.py` to use `pytesseract` for image-based PDFs.

## License
MIT License (or specify your company’s preferred license).