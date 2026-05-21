# Intern Assignment

Four different assignment as per the requirement of the intership.

---

The projects were built using Python and modern AI/ML library such as :

- Scikit-learn
- PyTorch
- OpenCV
- MediaPipe
- Librosa
- FastAPI
- WebSockets
- NLTK
- Playwright
- Django

## Architecture and Methodology

### Step by step logic

| Step | What happens | Library |
|------|--------------|---------|
| **1. Extract** | PDF pages are read and text concatenated | `pdfplumber` |
| **2. Clean** | Lowercase -> remove whitespace/unicode characters/URLS/emails/ phone numbers / special characters -> tokenize -> remove stop words -> lemmatize | `NLTK` |
| **3. Vectorize** | Both resume and job description texts converted to TF-IDF vectors | `sklearn` |
| **4. Similarity** | Cosine similarity score between TF-IDF vectors | `sklearn` |

## Installation and Running Locally

### Prerequisutes
- python **3.9 or higher**
- pip

### 1. Clone the repository

```bash
git clone https://github.com/rajatkhdk/Intern_Assignment
cd intern_tasks
```

### 2. Create and activate a virtual environment

```bash
python -m venv

# Linux
source .venv/bin/activate

# Windows
.venv/Scripts/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```
