import os
import json
import openai
from tqdm import tqdm
from rag_vanilla import rag_pipeline

openai.api_key = os.getenv("OPENAI_API_KEY")  # set this in your env

# Load questions
with open("data/questions.json", "r") as f:
    questions = json.load(f)

# Directory of resumes (PDFs)
RESUME_DIR = "resumes/"
OUTPUT_PATH = "data/your_predictions.jsonl"

def main():
    output_lines = []

    for filename in tqdm(os.listdir(RESUME_DIR)):
        if not filename.endswith(".pdf"):
            continue

        pdf_path = os.path.join(RESUME_DIR, filename)
        
        # Read PDF file as binary data
        with open(pdf_path, "rb") as f:
            pdf_data = f.read()

        for q in questions:
            qa = {
                "resume": filename,
                "question": q["question"]
            }

            try:
                # Use rag_pipeline instead of ask_llm
                response = rag_pipeline(pdf_data, q["question"])
                qa["predicted_answer"] = response["response"]
            except Exception as e:
                qa["predicted_answer"] = ""
                qa["error"] = str(e)

            output_lines.append(qa)

    with open(OUTPUT_PATH, "w") as f:
        for line in output_lines:
            f.write(json.dumps(line) + "\n")

    print(f"\nDone. Wrote {len(output_lines)} Q&A pairs to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
