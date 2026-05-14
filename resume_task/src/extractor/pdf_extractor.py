import os
import pdfplumber

def extract_text_from_pdf(file_path: str) -> str:
    """Extract all text from a PDF file."""

    if not os.path.exists(file_path):
        raise ValueError(f"File not found: {file_path}")
    
    ext = os.path.splitext(file_path)[-1].lower()
    # print("extension : ",ext)

    if ext != ".pdf":
        raise ValueError(f"Unsupported file type '{ext}'. Please upload a PDF (.pdf) file.")
    
    try:
        page_text : list[str] = []

        with pdfplumber.open(file_path) as pdf:
            if len(pdf.pages) == 0:
                raise ValueError("The PDF has no pages.")
            
            for page in pdf.pages:
                page_text.append(page.extract_text())
        return "\n".join(page_text).strip()
    except Exception as e:
        raise ValueError(f"Error occurred while extracting text from PDF: {e}")

if __name__ == '__main__':
    file_path = '/home/rajat/Jnotebook/intern_tasks/resume_task/data/sample_resume_pdf/AI ML CV Rajat (5).pdf'
    extracted_text = extract_text_from_pdf(file_path)
    print(extracted_text)