import pytest
import os
import tempfile
from reportlab.pdfgen import canvas


from src.extractor.pdf_extractor import extract_text_from_pdf
from src.extractor.text_cleaner import clean_text
from src.scorer.tfidf_cosine import create_tfidf_vectors, compute_cosine_similarity

# PDF Extractor Tests

class TestPDFExtractor:
    def test_nonexistent_file_raises(self):
        with pytest.raises(FileNotFoundError):
            extract_text_from_pdf("nonexistent_file.pdf")
    
    def test_wrong_extension_raises(self):
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp_txt:
            tmp_txt.write(b"some text")
            tmp_txt_path = tmp_txt.name
        try:
            with pytest.raises(ValueError, match="Unsupported file type"):
                extract_text_from_pdf(tmp_txt_path)
        finally:
            os.remove(tmp_txt_path) 

    def test_extract_text_from_pdf(self):
        # Create a temporary PDF file with known content
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_pdf:
            
            tmp_pdf_path = tmp_pdf.name

        try:
            c = canvas.Canvas(tmp_pdf_path)

            c.drawString(100, 750, "This is a test PDF.")

            c.save()

            extracted_text = extract_text_from_pdf(tmp_pdf_path)

            assert "This is a test PDF." in extracted_text

        finally:
            os.remove(tmp_pdf_path)


# Text Cleaner Tests

class TestTextCleaner:
    def test_clean_text(self):
        dirty_text = """This is a Sample tEXt! It includes shyam@gmail.com "how" "https://www.google.com/" numbers (123) and s  pecial characters 
    
    #hashtag.
    developer developing developed development."""
        cleaned = clean_text(dirty_text)
        assert cleaned == "sample text include number 123 pecial character # hashtag . developer develop develop development ."


#TF-IDF & Cosine Similarity Tests

class TestTFIDFAndCosine:
    def test_tfidf_and_cosine(self):
        text1 = "AI ML Rajat"
        text2 = "AI CV Rajat"

        vectors, feature_names = create_tfidf_vectors(text1, text2)
        assert vectors.shape == (2, len(feature_names))

        similarity_score = compute_cosine_similarity(vectors)
        assert 0.4 <= similarity_score <= 0.6