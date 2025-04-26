import json
import os
import sys
from pathlib import Path
import time
from dotenv import load_dotenv
from tqdm import tqdm
from typing import Optional
from pydantic import BaseModel, Field

from rag_vanilla import extract_text_from_pdf

# Add the parent directory to the system path
sys.path.append(str(Path(__file__).parent.parent))

import instructor
import google.generativeai as genai

# Load environment variables
load_dotenv(verbose=True, dotenv_path=".env")

# Initialize Gemini client
google_api_key = os.getenv("GOOGLE_API_KEY")
# print(f"Google API Key: {'Found' if google_api_key else 'Not found'}")
genai.configure(api_key=google_api_key)

client = instructor.from_gemini(
    client=genai.GenerativeModel(
        model_name="models/gemini-1.5-flash-latest",  
    )
)

# Define the response model
class ResumeAnswer(BaseModel):
    """Answer to a question about a resume"""
    answer: str = Field(..., description="The answer to the question based on the resume content")
    confidence: int = Field(..., ge=1, le=10, description="Confidence level in the answer from 1-10")
    reasoning: Optional[str] = Field(None, description="Brief reasoning for the answer")

def load_questions(file_path):
    """Load questions from a JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def get_resume_files(directory):
    """Get all PDF resume files from a directory."""
    resume_files = []
    for file in os.listdir(directory):
        if file.endswith('.pdf'):
            resume_files.append(file)
    return resume_files

def extract_text_from_resume(pdf_path):
    """Extract text from a PDF resume."""
    with open(pdf_path, "rb") as f:
        pdf_data = f.read()
    return extract_text_from_pdf(pdf_data)

def generate_expected_answer(resume_text, question):
    """Generate an expected answer for a question about a resume using Google Gemini with structured output."""
    system_prompt = """You are a resume analysis expert. 
Generate a concise, factual answer to the question based only on the resume content provided.
Your answer should be specific and accurate, focusing only on information explicitly stated in the resume.
If the information is not present in the resume, respond with "Not mentioned in the resume."
Provide a confidence score from 1-10, where 10 means the information is explicitly stated in the resume,
and 1 means you're guessing or the information is not available.
"""

    user_prompt = f"""Based on the following resume content, answer this question:

Question: {question}

Resume Content:
{resume_text}

Provide a brief, factual answer without additional commentary.
"""

    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_model=ResumeAnswer
        )
        
        # Return just the answer for compatibility with existing code
        return response.answer
    except Exception as e:
        print(f"Error generating answer: {e}")
        return "Error generating answer"

def main():
    # Define paths
    base_dir = Path(__file__).parent
    questions_path = base_dir / "data" / "questions.json"
    resumes_dir = base_dir / "resumes"
    output_path = base_dir / "data" / "gold_set.jsonl"
    
    # Load questions
    questions = load_questions(questions_path)
    print(f"Loaded {len(questions)} questions")
    
    # Get resume files
    resume_files = get_resume_files(resumes_dir)
    print(f"Found {len(resume_files)} resume files")
    
    # Create gold set
    gold_set = []
    
    # Process each resume and question
    for resume_file in tqdm(resume_files, desc="Processing resumes"):
        print(f"\nProcessing resume: {resume_file}")
        resume_path = resumes_dir / resume_file
        
        # Extract text from resume
        resume_text = extract_text_from_resume(resume_path)
        
        for q_item in questions:
            question = q_item["question"]
            print(f"  Question: {question}")
            
            # Generate expected answer
            expected_answer = generate_expected_answer(resume_text, question)
            print(f"  Answer: {expected_answer[:50]}..." if len(expected_answer) > 50 else f"  Answer: {expected_answer}")
            
            # Add to gold set
            gold_set.append({
                "resume": resume_file,
                "question": question,
                "expected_answer": expected_answer
            })
            
            # Add a small delay to avoid rate limiting
            time.sleep(0.5)
    
    # Write gold set to JSONL file
    with open(output_path, 'w') as f:
        for item in gold_set:
            f.write(json.dumps(item) + '\n')
    
    print(f"\nGold set written to {output_path}")
    print(f"Created {len(gold_set)} entries for {len(resume_files)} resumes and {len(questions)} questions")

if __name__ == "__main__":
    main()