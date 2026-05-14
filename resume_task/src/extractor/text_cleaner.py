import re
import nltk
import unicodedata
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import pos_tag

# ── Download required NLTK data (idempotent) ───────────────────────────────────
for resource in ("punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"):
        nltk.download(resource, quiet=True)
nltk.download("averaged_perceptron_tagger_eng", quiet=True)

_STOP_WORDS = set(stopwords.words("english"))
_LEMMATIZER = WordNetLemmatizer()

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
    cleaned = clean_text(sample_text)
    print(cleaned)