import gradio as gr
import fitz
import instructor
import google.generativeai as genai
from pydantic import BaseModel
import time
import sys
from pathlib import Path

# Add parent directory to path to make utils importable
sys.path.append(str(Path(__file__).parent.parent))
from utils import hybrid_search, lanceDBConnection, semantic_search, text_chunker

# load_dotenv(verbose=True, dotenv_path=".env")
# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
# print(f"Environment variables loaded from .env file.")

CHUNK_SIZE = 300
PDF_STORAGE_DIR = "pdfs" # Directory inside volume for storing PDFs

# Extract text from PDF
def extract_text_from_pdf(pdf_bytes):
    pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = ""
    for page_num in range(pdf_doc.page_count):
        page = pdf_doc.load_page(page_num)
        text += page.get_text("text")
    return text

class QAModel(BaseModel):
    answer: str

def generate_response(query, context):
    """Generates a response using Google Gemini with retrieved context."""
    
    client = instructor.from_gemini(
        client=genai.GenerativeModel(
            model_name="models/gemini-1.5-flash-latest",  
        )
    )
    
    V2_SYSTEM_PROMPT = """You are a helpful assistant.
Only answer questions based on the context provided.
Be concise, factual, and avoid speculation.
"""
    V2_PROMPT = f"""Answer the following question using the context provided.

  If the answer is not in the context, respond with:
  I don't have enough information to answer that based on the provided context.

  ---

  ### Question:
  {query}

  ### Context:
  {context}
  """

    start_time = time.time()
    completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": V2_SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": V2_PROMPT
            }
        ],
        response_model=QAModel
    )
    llm_time = time.time() - start_time

    return {
      "response": completion.answer,
      "system_prompt": V2_SYSTEM_PROMPT,
      "user_prompt": V2_PROMPT,
      "llm_time": llm_time
    }

def ingest_file(pdf_upload):
    """Ingest PDF file into LanceDB."""
    text_data = extract_text_from_pdf(pdf_upload)

    chunks = text_chunker(text_data, max_chunk_length=CHUNK_SIZE, overlap=50)
    print("Number of Chunks: ", len(chunks))
    # print(pdf_upload)
    metadata = {"file_path": "Uploaded PDF"}

    # create and add table in table
    table = lanceDBConnection(chunks, metadata)

    # Create a fts index before the hybrid search
    table.create_fts_index("text", replace=True)

    print("Ingestion complete. PDF file has been added to LanceDB.")
    return table

def rag_pipeline_vector_db(pdf_upload, query_input, table=None):
    """Main RAG pipeline using vector DB."""
    if pdf_upload:
        table = ingest_file(pdf_upload)
    
    total_start_time = time.time()
    retrieval_start_time = time.time()
    
    # relevant_chunks = semantic_search(table, query_input)
    relevant_chunks = hybrid_search(table, query_input)
    
    retrieval_time = time.time() - retrieval_start_time
    
    # Generation phase
    response = generate_response(query_input, relevant_chunks)
    
    # Calculate total time
    total_time = time.time() - total_start_time
    
    # Get PDF name
    pdf_name = pdf_upload.name if hasattr(pdf_upload, 'name') else "Uploaded PDF"
    
    # Log interaction with metrics
    
    print(f"Response: {response['response']}")  # Print response for debugging
    print(f"Tokens: {response.get('total_tokens', 0)} (input: {response.get('input_tokens', 0)}, output: {response.get('output_tokens', 0)})")
    print(f"Times: total={total_time:.2f}s, retrieval={retrieval_time:.2f}s, llm={response.get('llm_time', 0):.2f}s")
    
    # Add metrics to the response
    response["retrieval_time"] = retrieval_time
    response["total_time"] = total_time
    
    return response

with gr.Blocks() as app:
  gr.Markdown("# (DEMO) Vanilla RAG")
  pdf_upload = gr.File(label="Upload PDF", type="binary")
  query_input = gr.Textbox(label="Ask a question about the PDF")
  output = gr.Textbox(label="Answer")

  query_button = gr.Button("Submit")
  query_button.click(
      lambda pdf, query: rag_pipeline_vector_db(pdf, query)["response"], 
      inputs=[pdf_upload, query_input], 
      outputs=output
  )

if __name__ == "__main__":
    app.launch()

