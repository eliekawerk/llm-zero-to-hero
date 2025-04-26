import json
import os
import sys
from pathlib import Path
import time

from rag_vanilla import extract_text_from_pdf

# Add the parent directory to the system path
sys.path.append(str(Path(__file__).parent.parent))

import instructor

# Replace OpenAI with Instructor for Google Gemini
from instructor import GeminiClient

# Initialize Gemini client
client = GeminiClient(api_key=os.getenv("GEMINI_API_KEY"))

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
    """Generate an expected answer for a question about a resume using Google Gemini."""
    system_prompt = """You are a resume analysis expert. 
Generate a concise, factual answer to the question based only on the resume content provided.
Your answer should be specific and accurate, focusing only on information explicitly stated in the resume.
If the information is not present in the resume, respond with "Not mentioned in the resume."
"""

    user_prompt = f"""Based on the following resume content, answer this question:

Question: {question}

Resume Content:
{resume_text}

Provide a brief, factual answer without additional commentary.
"""

    try:
        completion = client.chat.completions.create(
            model="gemini-1.5-flash",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        return completion.choices[0].message.content.strip()
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
    for resume_file in resume_files:
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