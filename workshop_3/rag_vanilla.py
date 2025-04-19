import os
import gradio as gr
import fitz
import sqlite3
from datetime import datetime
import uuid
from openai import OpenAI
import requests
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

load_dotenv(verbose=True, dotenv_path=".env")
print(f"Environment variables loaded from .env file.")

DB_FILE = "pdf_vanilla_qa_V2_logs.db"

# Initialize the database
def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS interactions (
                 id TEXT PRIMARY KEY,
                 timestamp TEXT,
                 pdf_name TEXT,
                 system_prompt TEXT,
                 chunk_size INTEGER,
                 user_prompt TEXT,
                 query TEXT,
                 response TEXT)''')
    conn.commit()
    conn.close()

init_db()

# Extract text from PDF
def extract_text_from_pdf(pdf_bytes):
    pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = ""
    for page_num in range(pdf_doc.page_count):
        page = pdf_doc.load_page(page_num)
        text += page.get_text("text")
    return text

def chunk_text(text, chunk_size=1000, overlap=20):
    """Splits text into chunks with overlap. If chunk_size is 0, the whole document is a single chunk with no overlap."""
    if chunk_size == 0:
        print("Chunk size is 0, returning the whole document as a single chunk.")
        return [text], len(text)
    
    print(f"Chunking text of length {len(text)} into chunks of size {chunk_size} with overlap {overlap}.")
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start = max(end - overlap, start + 1)  # Ensure start progresses and avoids infinite loop
    print(f"Created {len(chunks)} chunks.")
    return chunks, chunk_size

# Log to SQLite
def log_interaction(pdf_name, query, response, system_prompt, chunk_size, user_prompt):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    interaction_id = str(uuid.uuid4())
    timestamp = datetime.now().isoformat()
    c.execute("INSERT INTO interactions VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
              (interaction_id, timestamp, pdf_name, system_prompt, chunk_size, user_prompt, query, response)) 
    conn.commit()
    conn.close()
    
def get_text_embedding(text_chunks):
    """Gets text embeddings using the OpenAI API.""" 
    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.embeddings.create(
            model="text-embedding-ada-002",
            input=text_chunks,
            encoding_format="float"
        )
        embeddings = [record.embedding for record in response.data]
        return embeddings
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}    
    
def find_relevant_chunks(query_embedding, chunk_embeddings, chunks, top_k=10):
    """Finds the most relevant text chunks based on cosine similarity."""
    if len(query_embedding) != len(chunk_embeddings[0]):
        raise ValueError("Query embedding and chunk embeddings must have the same dimensions.")
    
    query_embedding = [query_embedding]  # Ensure query_embedding is 2D
    similarities = cosine_similarity(query_embedding, chunk_embeddings)[0]
    relevant_chunks = sorted(zip(similarities, chunks), key=lambda x: x[0], reverse=True)[:top_k]
    return [chunk for _, chunk in relevant_chunks]

def generate_response(query, context):
    """Generates a response using OpenAI's Completion API with retrieved context."""

    V1_SYSTEM_PROMPT = "You are a Q&A assistant. Answer the question based only from context provided."
    V1_PROMPT = f"Question: {query}\n\nContext:\n{' '.join(context)}"
    
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
  {'\n'.join(context).strip()}
  """

    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY")
    )

    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": V2_SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": V2_PROMPT
            }
        ]
    )
    return {
      "response": completion.choices[0].message.content,
      "system_prompt": V2_SYSTEM_PROMPT,
      "user_prompt": V2_PROMPT
      }

def rag_pipeline(pdf_upload, query_input):
    """Main RAG pipeline."""
    pdf_text = extract_text_from_pdf(pdf_upload)
    text_chunks, chunk_size = chunk_text(pdf_text, 0)
    chunk_embeddings = get_text_embedding(text_chunks)
    query_embedding = get_text_embedding([query_input])[0]
    relevant_chunks = find_relevant_chunks(query_embedding, chunk_embeddings, text_chunks)
    response = generate_response(query_input, relevant_chunks)
    pdf_name = pdf_upload.name if hasattr(pdf_upload, 'name') else "Uploaded PDF"
    log_interaction(pdf_name, query_input, response["response"], response["system_prompt"], chunk_size, response["user_prompt"])
    print(f"Response: {response['response']}")  # Print response for debugging
    return response

# Gradio app
with gr.Blocks() as app:
  gr.Markdown("# Vanilla RAG")
  pdf_upload = gr.File(label="Upload PDF", type="binary")
  query_input = gr.Textbox(label="Ask a question about the PDF")
  output = gr.Textbox(label="Answer")

  query_button = gr.Button("Submit")
  query_button.click(
      lambda pdf, query: rag_pipeline(pdf, query)["response"], 
      inputs=[pdf_upload, query_input], 
      outputs=output
  )

if __name__ == "__main__":
    app.launch()

