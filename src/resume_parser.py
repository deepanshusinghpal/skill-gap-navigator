import pandas as pd
import PyPDF2
import re
import os

def extract_text_from_pdf(pdf_path):
    """Reads a PDF file and extracts all text."""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                if page.extract_text():
                    text += page.extract_text() + " "
            return text.lower()
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return ""

def extract_skills_from_text(text, skills_csv_path=None):
    if skills_csv_path is None:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        skills_csv_path = os.path.join(BASE_DIR, '..', 'data', 'top_skills.csv')
    """Scans the text for skills listed in our top_skills database."""
    try:
        # Load our master list of skills
        top_skills_df = pd.read_csv(skills_csv_path)
        master_skills = top_skills_df['Skill'].tolist()
        
        found_skills = []
        
        # Check if each skill exists in the resume text
        for skill in master_skills:
            # We use regex word boundaries (\b) so we don't accidentally 
            # match "c" inside the word "react"
            pattern = r'\b' + re.escape(skill.lower()) + r'\b'
            if re.search(pattern, text):
                found_skills.append(skill)
                
        return found_skills
        
    except Exception as e:
        print(f"Error extracting skills: {e}")
        return []

# --- Testing the script ---
if __name__ == "__main__":
    print("Testing Resume Parser...")
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    test_pdf_path = os.path.join(BASE_DIR, '..', 'Sample_Resume.pdf') 
    
    print(f"Reading {test_pdf_path}...")
    resume_text = extract_text_from_pdf(test_pdf_path)
    
    if resume_text:
        extracted_skills = extract_skills_from_text(resume_text)
        print("\nSkills found in the resume:")
        for skill in extracted_skills:
            print(f"• {skill}")
    else:
        print("\nCould not read text from the PDF. Make sure 'sample_resume.pdf' exists in your root folder.")