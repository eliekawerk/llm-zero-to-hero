from pathlib import Path
import os
import uuid

from fastapi import FastAPI
from gradio.routes import mount_gradio_app
import modal

from rag_demo_vanilla import app as blocks
from rag_demo_vanilla import PDF_STORAGE_DIR

# Create a lightweight Modal image with required dependencies
image = modal.Image.debian_slim().pip_install(
    "gradio<6",
    "pymupdf",
    "google-generativeai",
    "instructor[google-generativeai]",
    "scikit-learn",
    "lancedb",
    "pydantic",
    "tantivy",
    "pylance",
    "load_dotenv"
)

app = modal.App("rag_demo_vanilla", image=image)

pdf_storage = modal.Volume.from_name("pdf-storage-volume", create_if_missing=True)


# Helper function to save uploaded PDF to Modal volume and return filepath
def save_pdf_to_volume(pdf_bytes, pdf_name=""):
    """Save a PDF to the Modal volume and return its unique filename"""
    # Generate a unique filename with UUID to avoid conflicts
    unique_id = str(uuid.uuid4())
    if pdf_name:
        # Clean filename and append UUID
        clean_name = pdf_name.replace(" ", "_").lower()
        filename = f"{clean_name}_{unique_id}.pdf"
    else:
        filename = f"uploaded_pdf_{unique_id}.pdf"
    
    # Ensure PDF storage directory exists
    pdf_dir = Path("/pdfs") / PDF_STORAGE_DIR
    pdf_dir.mkdir(parents=True, exist_ok=True)
    
    # Save PDF file
    filepath = pdf_dir / filename
    with open(filepath, "wb") as f:
        f.write(pdf_bytes)
    
    return str(filepath)


# Override the ingest_file function to save PDFs to volume
def ingest_file_with_storage(pdf_upload):
    """Ingest PDF file into LanceDB and save to volume."""
    from rag_demo_vanilla import extract_text_from_pdf, text_chunker, lanceDBConnection, CHUNK_SIZE
    
    # Get PDF binary data
    if hasattr(pdf_upload, 'read'):
        # Handle file-like object
        pdf_bytes = pdf_upload.read()
        # Reset file pointer for downstream processing
        if hasattr(pdf_upload, 'seek'):
            pdf_upload.seek(0)
    else:
        # Already in bytes form
        pdf_bytes = pdf_upload
    
    # Extract PDF content for processing
    text_data = extract_text_from_pdf(pdf_bytes)
    
    # Save PDF to volume for persistence
    pdf_name = pdf_upload.name if hasattr(pdf_upload, 'name') else "Uploaded_PDF"
    stored_path = save_pdf_to_volume(pdf_bytes, pdf_name)
    
    # Continue with regular processing
    chunks = text_chunker(text_data, max_chunk_length=CHUNK_SIZE, overlap=50)
    print(f"Number of Chunks: {len(chunks)}")
    print(f"PDF saved to: {stored_path}")
    
    # Include the stored path in metadata
    metadata = {"file_path": stored_path}

    # Create and add table in LanceDB
    table = lanceDBConnection(chunks, metadata)
    table.create_fts_index("text", replace=True)

    print("Ingestion complete. PDF file has been added to LanceDB.")
    return table


@app.function(
    scaledown_window=300,
    max_containers=1,
    secrets=[modal.Secret.from_name("google-secrets")],
    volumes={
        "/pdfs": pdf_storage,
    },
)
@modal.concurrent(max_inputs=1000)
@modal.asgi_app()  # Register this as an ASGI app
def serve() -> FastAPI:
    """
    Main server function:
    - Handles PDFs persistence in Modal volume
    - Wraps Gradio inside FastAPI
    - Deploys the API through Modal with a single instance for consistency
    """
    from contextlib import asynccontextmanager
    import asyncio
    
    # Override ingest_file to use our storage version
    import rag_demo_vanilla
    rag_demo_vanilla.ingest_file = ingest_file_with_storage
    
    # Configure Google Gemini API
    import google.generativeai as genai
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    
    # Set up FastAPI with a lifespan manager for persistence
    @asynccontextmanager
    async def lifespan(api: FastAPI):
        print("Starting service with PDF storage enabled")
        yield
        print("Shutting down service")

    api = FastAPI(lifespan=lifespan)
    
    @api.get("/health")
    def health_check():
        return {"status": "healthy", "storage": "enabled"}
    
    # Mount Gradio app at root path
    return mount_gradio_app(app=api, blocks=blocks, path="/")


@app.local_entrypoint()
def main():
    """Local entrypoint for testing"""
    print("Starting local development server")
    print("Note: This won't use Modal volumes when run locally")
    serve.serve()
