import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def create_tfidf_vectors(resume_text: str, jd_text: str):
    """Convert resume and job description text into TF-IDF vectors."""

    if not resume_text or not jd_text:
        raise ValueError("Both resume and job description text must be non-empty.")
    
    documents = [resume_text, jd_text]

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)

    return tfidf_matrix.toarray(), vectorizer.get_feature_names_out()


def compute_cosine_similarity(vectors) -> float:
    """Compute cosine similarity between two TF-IDF vectors."""
    # if vec1.shape != vec2.shape:
    #     raise ValueError("Input vectors must have the same shape.")
    
    similarity = cosine_similarity(vectors)[0][1]  # similarity between resume and job description
    return similarity


if __name__ == "__main__":
    # Example usage
    resume = "Experienced software engineer with expertise in Python and machine learning."
    job_description = "Looking for a software engineer skilled in Python and machine learning."

    vectors, feature_names = create_tfidf_vectors(resume, job_description)
    print("TF-IDF Vectors:\n", vectors)
    print("Feature Names:\n", feature_names)

    similarity_score = compute_cosine_similarity(vectors)
    print(f"Cosine Similarity Score: {similarity_score:.4f}")
