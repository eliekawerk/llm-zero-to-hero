from fastapi import FastAPI
from gradio.routes import mount_gradio_app
import modal
from rag_demo_vanilla import app as blocks

# Create a lightweight Modal image (Debian-based) with required dependencies
image = modal.Image.debian_slim().pip_install(
    "gradio<6",
    "pymupdf",
    "google-generativeai",
    "instructor[google-generativeai]",
    "scikit-learn",
    "lancedb",
    "pydantic",
    "tantivy",
    "pylance"
)

app = modal.App("rag_demo_vanilla", image=image)

@app.function(
    max_containers=1,
    allow_concurrent_inputs=1000,
    secrets=[modal.Secret.from_name("google-secrets")]
)
@modal.asgi_app()  # Register this as an ASGI app (compatible with FastAPI)
def serve() -> FastAPI:
    """
    Main server function:
    - Handles movement of our logs DB to and from remote storage on Modal
    - Wraps Gradio inside FastAPI
    - Deploys the API through Modal with a single instance for session consistency
    """
    api = FastAPI(title="PDF Query API")

    return mount_gradio_app(app=api, blocks=blocks, path="/")  # Mount Gradio app at root path


@app.local_entrypoint()
def main():
    """
    Local development entry point:
    - Allows running the app locally for testing
    - Prints the type of Gradio app to confirm readiness
    """
    print(f"{type(blocks)} is ready to go!")
