import gradio as gr

from src.extractor.pdf_extractor import extract_text_from_pdf
from src.extractor.text_cleaner import extract_skills
from src.scorer.tfidf_cosine import (
    create_tfidf_vectors,
    compute_cosine_similarity
)


# Main Resume Matching Function

def calculate_similarity(resume_file, job_description):

    try:

        # Extract resume text from uploaded PDF

        extracted_resume_text = extract_text_from_pdf(
            resume_file.name
        )

        # Clean both texts

        resume_skills = extract_skills(
            extracted_resume_text
        )

        job_skills = extract_skills(
            job_description
        )

        # print(job_skills)
        # print("\n\n")
        # print(resume_skills)
        # print("\n\n")
        filtered_resume_skills = []
            
        for skill in resume_skills.split():
            # for jd_skill in jd_skill_set:
            if (skill in job_skills.split()):
                filtered_resume_skills.append(skill)
        
        print(filtered_resume_skills)

        resume_text = " ".join(filtered_resume_skills)
        job_text = job_skills

        # Create TF-IDF vectors

        vectors, feature_names = create_tfidf_vectors(
            resume_text,
            job_text
        )

        # Compute similarity

        similarity_score = compute_cosine_similarity(
            vectors
        )

        similarity_percentage = round(
            similarity_score * 100,
            2
        )

        # Match category

        if similarity_percentage >= 75:
            match_result = "Excellent Match"
        elif similarity_percentage >= 50:
            match_result = "Good Match"
        elif similarity_percentage >= 30:
            match_result = "Average Match"
        else:
            match_result = "Low Match"

        # Return outputs

        return (
            similarity_percentage,
            match_result,
            resume_text[:3000],
            job_text[:3000]
        )

    except Exception as e:

        return (
            0,
            f"Error: {str(e)}",
            ""
        )


# Gradio UI

with gr.Blocks() as demo:

    gr.Markdown(
        """
        # AI Resume Matcher
        
        Upload a resume PDF and compare it with a job description using:
        
        - NLP
        - TF-IDF Vectorization
        - Cosine Similarity
        """
    )

    with gr.Row():

        with gr.Column():

            resume_input = gr.File(
                label="Upload Resume PDF",
                file_types=[".pdf"]
            )

            job_description_input = gr.Textbox(
                label="Job Description",
                lines=12,
                placeholder="Paste the job description here..."
            )

            submit_button = gr.Button(
                "Analyze Resume"
            )

        with gr.Column():

            similarity_output = gr.Number(
                label="Similarity Percentage"
            )

            match_output = gr.Textbox(
                label="Match Result"
            )

            extracted_text_output = gr.Textbox(
                label="Extracted Resume Text",
                lines=30
            )

            job_text_output = gr.Textbox(
                label="Extracted Job Description Text",
                lines=30
            )

    # Button Action

    submit_button.click(
        fn=calculate_similarity,
        inputs=[
            resume_input,
            job_description_input
        ],
        outputs=[
            similarity_output,
            match_output,
            extracted_text_output,
            job_text_output
        ]
    )


# Launch App

demo.launch()