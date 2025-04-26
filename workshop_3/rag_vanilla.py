import os
import gradio as gr
import fitz
import sqlite3
from datetime import datetime
import uuid
import instructor
import google.generativeai as genai
from openai import OpenAI
from pydantic import BaseModel
import requests
import time
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
from utils import lanceDBConnection, semantic_search, text_chunker

load_dotenv(verbose=True, dotenv_path=".env")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
print(f"Environment variables loaded from .env file.")

DB_FILE = "pdf_vanilla_qa_V2_logs.db"
CHUNK_SIZE = 300

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
                 response TEXT,
                 input_tokens INTEGER,
                 output_tokens INTEGER,
                 total_tokens INTEGER,
                 retrieval_time REAL,
                 llm_time REAL,
                 total_time REAL)''')
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
def log_interaction(pdf_name, query, response, system_prompt, chunk_size, user_prompt, 
                    input_tokens=0, output_tokens=0, total_tokens=0, 
                    retrieval_time=0, llm_time=0, total_time=0):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    interaction_id = str(uuid.uuid4())
    timestamp = datetime.now().isoformat()
    c.execute("INSERT INTO interactions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
              (interaction_id, timestamp, pdf_name, system_prompt, chunk_size, user_prompt, 
               query, response, input_tokens, output_tokens, total_tokens, 
               retrieval_time, llm_time, total_time)) 
    conn.commit()
    conn.close()
    
def get_text_embedding(text_chunks):
    """Gets text embeddings using the OpenAI API.""" 
    try:
        emb_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        start_time = time.time()
        response = emb_client.embeddings.create(
            model="text-embedding-ada-002",
            input=text_chunks,
            encoding_format="float"
        )
        embeddings = [record.embedding for record in response.data]
        embedding_time = time.time() - start_time
        return embeddings, embedding_time
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}, 0
    
def find_relevant_chunks(query_embedding, chunk_embeddings, chunks, top_k=10):
    """Finds the most relevant text chunks based on cosine similarity."""
    if len(query_embedding) != len(chunk_embeddings[0]):
        raise ValueError("Query embedding and chunk embeddings must have the same dimensions.")
    
    query_embedding = [query_embedding]  # Ensure query_embedding is 2D
    similarities = cosine_similarity(query_embedding, chunk_embeddings)[0]
    relevant_chunks = sorted(zip(similarities, chunks), key=lambda x: x[0], reverse=True)[:top_k]
    return [chunk for _, chunk in relevant_chunks]

def get_text_embedding_google(text_chunks):
    """Gets text embeddings using Google's embedding model 'models/text-embedding-004'."""
    try:
        start_time = time.time()
        response = genai.generate_embeddings(
            model="models/text-embedding-004",
            input=text_chunks
        )
        embeddings = response["embeddings"]
        embedding_time = time.time() - start_time
        return embeddings, embedding_time
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}, 0

class QAModel(BaseModel):
    answer: str

def generate_response(query, context):
    """Generates a response using Google Gemini with retrieved context."""
    
    # load_dotenv(verbose=True, dotenv_path=".env")
    # print(f"Environment variables loaded from .env file.")
    # print(f"Google API Key: {os.getenv('GOOGLE_API_KEY')}")
    
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
  {'\n'.join(context).strip()}
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

def ingest_files():
    """Ingest PDF files from the resumes folder into LanceDB."""
    resumes_folder = "./resumes"
    for file_name in os.listdir(resumes_folder):
        if file_name.endswith(".pdf"):
            file_path = os.path.join(resumes_folder, file_name)
    with open(file_path, "rb") as pdf_file:
        pdf_bytes = pdf_file.read()

    text_data = extract_text_from_pdf(pdf_bytes)

    chunks = text_chunker(text_data, max_chunk_length=CHUNK_SIZE, overlap=50)
    print("Number of Chunks: ", len(chunks))
    metadata = {"file_path": file_path, "name": "Alex Thompson"}

    # create and add table in table
    table = lanceDBConnection(chunks, metadata)

    # Create a fts index before the hybrid search
    table.create_fts_index("text", replace=True)

    print("Ingestion complete. All PDF files have been added to LanceDB.")
    return table

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

def rag_pipeline(pdf_upload, query_input):
    """Main RAG pipeline."""
    total_start_time = time.time()
    
    # Extraction and embedding phase
    pdf_text = extract_text_from_pdf(pdf_upload)
    text_chunks, chunk_size = chunk_text(pdf_text, 0)
    
    retrieval_start_time = time.time()
    chunk_embeddings, chunk_embedding_time = get_text_embedding(text_chunks)
    query_embedding, query_embedding_time = get_text_embedding([query_input])
    query_embedding = query_embedding[0]
    
    relevant_chunks = find_relevant_chunks(query_embedding, chunk_embeddings, text_chunks)
    retrieval_time = time.time() - retrieval_start_time
    
    # Generation phase
    response = generate_response(query_input, relevant_chunks)
    
    # Calculate total time
    total_time = time.time() - total_start_time
    
    # Get PDF name
    pdf_name = pdf_upload.name if hasattr(pdf_upload, 'name') else "Uploaded PDF"
    
    # Log interaction with metrics
    log_interaction(
        pdf_name=pdf_name, 
        query=query_input, 
        response=response["response"], 
        system_prompt=response["system_prompt"], 
        chunk_size=chunk_size, 
        user_prompt=response["user_prompt"],
        input_tokens=response.get("input_tokens", 0),
        output_tokens=response.get("output_tokens", 0),
        total_tokens=response.get("total_tokens", 0),
        retrieval_time=retrieval_time,
        llm_time=response.get("llm_time", 0),
        total_time=total_time
    )
    
    print(f"Response: {response['response']}")  # Print response for debugging
    print(f"Tokens: {response.get('total_tokens', 0)} (input: {response.get('input_tokens', 0)}, output: {response.get('output_tokens', 0)})")
    print(f"Times: total={total_time:.2f}s, retrieval={retrieval_time:.2f}s, llm={response.get('llm_time', 0):.2f}s")
    
    # Add metrics to the response
    response["retrieval_time"] = retrieval_time
    response["total_time"] = total_time
    
    return response

def rag_pipeline_vector_db(pdf_upload, query_input, table=None):
    """Main RAG pipeline using vector DB."""
    if pdf_upload:
        table = ingest_file(pdf_upload)
    
    total_start_time = time.time()
    retrieval_start_time = time.time()
    
    relevant_chunks = semantic_search(table, query_input)
    retrieval_time = time.time() - retrieval_start_time
    
    # Generation phase
    response = generate_response(query_input, relevant_chunks)
    
    # Calculate total time
    total_time = time.time() - total_start_time
    
    # Get PDF name
    pdf_name = pdf_upload.name if hasattr(pdf_upload, 'name') else "Uploaded PDF"
    
    # Log interaction with metrics
    log_interaction(
        pdf_name=pdf_name, 
        query=query_input, 
        response=response["response"], 
        system_prompt=response["system_prompt"], 
        chunk_size=CHUNK_SIZE, 
        user_prompt=response["user_prompt"],
        input_tokens=response.get("input_tokens", 0),
        output_tokens=response.get("output_tokens", 0),
        total_tokens=response.get("total_tokens", 0),
        retrieval_time=retrieval_time,
        llm_time=response.get("llm_time", 0),
        total_time=total_time
    )
    
    print(f"Response: {response['response']}")  # Print response for debugging
    print(f"Tokens: {response.get('total_tokens', 0)} (input: {response.get('input_tokens', 0)}, output: {response.get('output_tokens', 0)})")
    print(f"Times: total={total_time:.2f}s, retrieval={retrieval_time:.2f}s, llm={response.get('llm_time', 0):.2f}s")
    
    # Add metrics to the response
    response["retrieval_time"] = retrieval_time
    response["total_time"] = total_time
    
    return response

# Gradio app
with gr.Blocks() as app:
  gr.Markdown("# Vanilla RAG")
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

