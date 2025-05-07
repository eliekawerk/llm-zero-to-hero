from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
import modal

# Define paths
BASE_DIR = Path(__file__).parent.parent
VIEWERS_DIR = BASE_DIR / "data_viewers"
DATA_DIR = BASE_DIR / "data"
RESUMES_DIR = BASE_DIR / "resumes"
ARCHIVE_DIR = BASE_DIR / "resume_archive"

# Create a Modal image with FastAPI
image = modal.Image.debian_slim().pip_install("fastapi", "uvicorn")

local_assets_path = Path(__file__).parent / "data_viewers"
local_data_path = Path(__file__).parent / "data"
local_resume_path = Path(__file__).parent.parent / "resumes"
image = image.add_local_dir(local_assets_path, remote_path="/assets")
image = image.add_local_dir(local_data_path, remote_path="/assets/data")
image = image.add_local_dir(local_resume_path, remote_path="/assets/resumes")

# Create a Modal app
app = modal.App("static-data-viewer")

# Create a Modal volume for storing data if needed
data_volume = modal.Volume.from_name("data-viewer-volume", create_if_missing=True)

# Create a FastAPI app
api = FastAPI(title="Data Viewer")

@api.get("/")
async def root():
    """Serve the main index.html file"""
    return FileResponse("/assets/index.html")

@api.get("/error_analysis")
async def error_analysis():
    """Serve the error analysis HTML file"""
    return FileResponse("/assets/error_analysis.html")

@api.get("/evaluation_report")
async def evaluation_report():
    """Serve the evaluation_report HTML file"""
    return FileResponse("/assets/evaluation_report.html")

@api.get("/api/list-json-files")
async def list_json_files():
    """API endpoint to list available JSON files"""
    print("Listing JSON files")
    json_files = []
    
    # In Modal environment, files are in /assets/data
    data_path = Path("/assets/data")
    
    try:
        # List all JSON files in the data directory
        for file_path in data_path.glob("*.json"):
            json_files.append(file_path.name)
        
        print(f"Found {len(json_files)} JSON files: {json_files}")
    except Exception as e:
        print(f"Error listing JSON files: {e}")
    
    return JSONResponse({"files": json_files})

@app.function(
    image=image,
    volumes={"/data": data_volume},
)
@modal.concurrent(max_inputs=100)
@modal.asgi_app()
def serve_app() -> FastAPI:
    api.mount("/", StaticFiles(directory="/assets", html=True))    
    return api


