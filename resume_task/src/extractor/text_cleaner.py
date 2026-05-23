import re
import nltk
import unicodedata
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import pos_tag
from nltk.chunk import RegexpParser

# ── Download required NLTK data (idempotent) ───────────────────────────────────
for resource in ("punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"):
        nltk.download(resource, quiet=True)
nltk.download("averaged_perceptron_tagger_eng", quiet=True)

_STOP_WORDS = set(stopwords.words("english"))
_LEMMATIZER = WordNetLemmatizer()

IMPORTANT_POS = {
    "NN", "NNS", "NNP", "NNPS",  # Nouns
}

def extract_skills(text):

    cleaned = clean_text(text)
    tokens = word_tokenize(cleaned)
    pos_tags = pos_tag(tokens)

    grammar = r"""
    SKILL: {<NN.*>+}  # Capture sequences of nouns as potential skills
    """

    chunk_parser = RegexpParser(grammar)
    tree = chunk_parser.parse(pos_tags)
    skills = []

    for subtree in tree.subtrees():
        if subtree.label() == "SKILL":
            skill = " ".join(word for word, tag in subtree.leaves())
            # if len(skill) > 2:  # Filter out very short skills
            skills.append(skill)

    skill_concatenated = " ".join(skills)
    clean_skill = clean_text(skill_concatenated)

    return clean_skill



def clean_text(text: str) -> str:
    """NLP preprocessing pipeline: lowercase, remove special chars/numbers, tokenise, remove stop words, lemmatise."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Input text must be a non-empty string.")
    
    # lowercase
    text = text.lower()

    # remove extra whitespace
    text = re.sub(r'\s+', " ", text).strip()

    # Remove unicode characters
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')

    # Remove urls
    text = re.sub(r"http\S+|www\S+|", "", text)

    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)

    # Remove phone numbers
    text = re.sub(r"\+?\d[\d\s\-]{7,}", "", text)

    # remove special characters and numbers, keep only letters and whitespace
    text = re.sub(r"[^a-zA-Z0-9+#.\s]", " ", text)   

    tokens = word_tokenize(text)
    # print("Tokens before stop word removal and lemmatization: ", tokens)
    tokens = [token for token in tokens if token not in _STOP_WORDS]
    # print("Tokens after stop word removal: ", tokens)
    pos_tags = pos_tag(tokens)

    tokens = [
        _LEMMATIZER.lemmatize(word, get_wordnet_pos(tag))
        for word, tag in pos_tags
    ]
    # print("Tokens after lemmatization: ", tokens)
    return " ".join(tokens)

def get_wordnet_pos(tag):
    """
    Map POS tag to first character used by WordNetLemmatizer.
    """

    if tag.startswith('J'):
        return wordnet.ADJ

    elif tag.startswith('V'):
        return wordnet.VERB

    elif tag.startswith('N'):
        return wordnet.NOUN

    elif tag.startswith('R'):
        return wordnet.ADV

    return wordnet.NOUN

if __name__ == '__main__':
    sample_text = """This is a Sample tEXt! It includes shyam@gmail.com "how" "https://www.google.com/" numbers (123) and s  pecial characters 
    
    #hashtag.
    developer developing developed development."""
    resume_text = """
Rajat Kumar Khadka
Kirtipur-04, Kathmandu, Nepal
: rajatkhdk111@gmail.com | 9861484466 | : Github | : LinkedIn
Professional Summary
An enthusiastic AI/ML engineer with a strong foundation in probability and statistics, ML algorithms
and NLP. Experienced in building predictive models like a session calorie predictor which focused on
feature engineering, model comparison and SHAP for XAI to GenAI-related tasks including chatbots that
allow users to upload the documents then query based on the document. Aside from ML tasks, skilled in
API development using FastAPI, Django and web-scraping using BeautifulSoup and Playwright.
Technical Skills
• Programming: Python, JavaScript, C, C++
• Frameworks: FastAPI, Django, TensorFlow
• Libraries & Tools: Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, LangChain
• Core Interest: Computer Vision and NLP
• Web-scraping: Requests, BeautifulSoup, Playwright
• Databases: MongoDB, SQL, PostgreSQL
• Version Control: Git, GitHub
• Other: Retrieval Augmented Generation (RAG), Data Preprocessing, Model Evaluation,
Hyperparameter Tuning, Bagging and Boosting algorithms, Gradient Descent, LaTeX
Experience
Django Intern
Nectar Digit | March 2026 - Present
Research Experience
Undergraduate Researcher - Nepal College of Information Technology
Enhanced Calorie Burn Prediction using XGBoost with Feature Selection and SHAP - View Paper
• Built predictive model for personalized calorie expenditure estimation
• Applied VIF-based feature selection to reduce multicollinearity
• Used SHAP for interpretability and feature importance analysis
• Analyzed trade-off between accuracy and model complexity
Education
Bachelor of Computer Engineering
Pokhara University | CGPA: 3.65 / 4.00
April 2021 - August 2025
Projects
RAG AI Interview Assistant (RAG) - Link
• Designed a modular backend with document ingestion and conversational RAG APIs for multi-turn
Q&A
• Implemented custom retrieval pipeline (no prebuilt chains), embeddings storage, and Redis-based
chat memory
• Added LLM-powered interview booking with structured data extraction and SQL persistence
Nutrition RAG API (RAG) - Link
• Built a backend API using a Retrieval Augmented Generation (RAG) pipeline to answer nutrition-
related queries from document data.
• Implemented document ingestion, text chunking, embeddings and semantic search, feeding
retrieved context into an LLM for accurate responses.
• Showcased end-to-end integration of retrieval + ML embeddings + LLM generation into a
production-ready backend API.
ML from Scratch (ML Algorithms) - Link
• Implemented core ML algorithms (KNN, Linear Regression, Logistic Regression, Decision Tree)
entirely from scratch without using high-level libraries.
• Built end-to-end pipeline: data preprocessing, training, prediction, and evaluation using metrics like
accuracy and MSE.
• Compared results with Scikit-learn implementations to validate correctness and performance.
Personalized Fitness & Recommendation System (Major Project) - Link
• Built ML-based calorie prediction model using XGBoost for personalized nutrition planning
• Designed rule-based workout recommendation engine
• Developed FastAPI microservice for model deployment and API integration
• Delivered personalized daily calorie and macronutrient recommendations for users
Push-up Form Analyzer (Computer Vision) - Link
• Collected and annotated custom workout video dataset
• Extracted body key points using MediaPipe for pose estimation
• Trained classification model to evaluate posture and movement quality
Certifications
• TensorFlow & Keras Bootcamp - OpenCV University
Issued - 22 May 2025
• Flutter Framework - Broadway Infosys
19 December 2023 - 18 January 2024
"""
    # cleaned = clean_text(resume_text)
    # print(cleaned)

    skills = extract_skills(resume_text)
    # for skill in skills:
    print(skills)