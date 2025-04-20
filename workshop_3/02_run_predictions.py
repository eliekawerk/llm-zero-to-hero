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
    
    # Track overall metrics
    total_input_tokens = 0
    total_output_tokens = 0
    total_tokens = 0
    total_retrieval_time = 0
    total_llm_time = 0
    total_time = 0
    total_queries = 0

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
                
                # Store metrics in the output
                qa["input_tokens"] = response.get("input_tokens", 0)
                qa["output_tokens"] = response.get("output_tokens", 0)
                qa["total_tokens"] = response.get("total_tokens", 0)
                qa["retrieval_time"] = response.get("retrieval_time", 0)
                qa["llm_time"] = response.get("llm_time", 0)
                qa["total_time"] = response.get("total_time", 0)
                
                # Accumulate totals
                total_input_tokens += qa["input_tokens"]
                total_output_tokens += qa["output_tokens"]
                total_tokens += qa["total_tokens"]
                total_retrieval_time += qa["retrieval_time"]
                total_llm_time += qa["llm_time"]
                total_time += qa["total_time"]
                total_queries += 1
                
            except Exception as e:
                qa["predicted_answer"] = ""
                qa["error"] = str(e)

            output_lines.append(qa)

    with open(OUTPUT_PATH, "w") as f:
        for line in output_lines:
            f.write(json.dumps(line) + "\n")

    # Print summary statistics
    if total_queries > 0:
        print("\n===== Performance Metrics Summary =====")
        print(f"Total queries: {total_queries}")
        print(f"Average tokens per query: {total_tokens / total_queries:.1f}")
        print(f"  - Input tokens: {total_input_tokens / total_queries:.1f}")
        print(f"  - Output tokens: {total_output_tokens / total_queries:.1f}")
        print(f"Average latency per query: {total_time / total_queries:.2f}s")
        print(f"  - Retrieval: {total_retrieval_time / total_queries:.2f}s")
        print(f"  - LLM inference: {total_llm_time / total_queries:.2f}s")
        
        # Estimate cost (using OpenAI's GPT-4o pricing as of April 2025)
        # These rates should be updated based on current pricing
        input_cost = (total_input_tokens / 1000) * 0.01  # $0.01 per 1K input tokens
        output_cost = (total_output_tokens / 1000) * 0.03  # $0.03 per 1K output tokens
        total_cost = input_cost + output_cost
        print(f"Estimated cost: ${total_cost:.4f}")
        print(f"  - Input cost: ${input_cost:.4f}")
        print(f"  - Output cost: ${output_cost:.4f}")
        print(f"Cost per query: ${total_cost / total_queries:.4f}")

    print(f"\nDone. Wrote {len(output_lines)} Q&A pairs to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
