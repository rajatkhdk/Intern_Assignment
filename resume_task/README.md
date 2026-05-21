# NLP RESUME MATCHER

A python application that compares a candidate's **PDF resume** with a **job description** using techniques such as - text extraction, preprocessing, TF-IDF vectorization and cosine similarity to produce a **match score**.

---
## Architecture and Methodology

---

# Features

- Resume text extraction from PDF files
- NLP preprocessing pipeline
- Tokenization and lemmatization
- Stopword removal
- TF-IDF vectorization
- Cosine similarity scoring
- Unit testing using pytest

---

# Tech Stack

- Python
- NLTK
- Scikit-learn
- pdfplumber
- pytest

---

### Step by step logic

| Step | What happens | Library |
|------|--------------|---------|
| **1. Extract** | PDF pages are read and text concatenated | `pdfplumber` |
| **2. Clean** | Lowercase -> remove whitespace/unicode characters/URLS/emails/ phone numbers / special characters -> tokenize -> remove stop words -> lemmatize | `NLTK` |
| **3. Vectorize** | Both resume and job description texts converted to TF-IDF vectors | `sklearn` |
| **4. Similarity** | Cosine similarity score between TF-IDF vectors | `sklearn` |

## Installation and Running Locally

### Prerequisites:
Complete the installation process in the root folder.

### Change directory
```bash
cd resume_task
```

### Download the NLTK data (auto-downloaded on first run or manually)

```bash
python -c "import nltk; [nltk.download(r) for r in ('punkt','punkt_tab','stopwords','wordnet','omw-1.4')]"
```

### Running tests
```bash
pytest tests/ -v
```

### Running the app
```bash
python app.py
```
