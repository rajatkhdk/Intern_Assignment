from resume_task.src.extractor.pdf_extractor import extract_text_from_pdf
from resume_task.src.extractor.text_cleaner import clean_text
from resume_task.src.scorer.tfidf_cosine import create_tfidf_vectors, compute_cosine_similarity

def main():
    # Example usage
    file_path1 = '/home/rajat/Jnotebook/intern_tasks/resume_task/data/sample_resume_pdf/AI ML CV Rajat (5).pdf'
    file_path2 = '/home/rajat/Jnotebook/intern_tasks/resume_task/data/sample_job_description/job_description.txt'
    
    try:
        # extracted_text1 = extract_text_from_pdf(file_path1)
        # print("Extracted Text:\n", extracted_text1)

        # cleaned_text1 = clean_text(extracted_text1)
        # print("\nCleaned Text:\n", cleaned_text1)

        cleaned_text1 = clean_text("""AI ML Rajat""")
        cleaned_text2 = clean_text("""AI CV Rajat""")

        # For demonstration, using the cleaned text as both resume and job description
        vectors, feature_names = create_tfidf_vectors(cleaned_text1, cleaned_text2)
        print("\nTF-IDF Vectors:\n", vectors)
        print("Feature Names:\n", feature_names)

        similarity_score = compute_cosine_similarity(vectors)
        print(f"\nCosine Similarity Score: {similarity_score:.4f}")
    except ValueError as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
