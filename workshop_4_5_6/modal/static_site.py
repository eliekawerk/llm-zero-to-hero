import os
import json
from pathlib import Path
from fastapi import FastAPI, HTTPException
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

@app.function(
    image=image,
    volumes={"/data": data_volume},
)
@modal.concurrent(max_inputs=100)
@modal.asgi_app()
def serve_app() -> FastAPI:
    # Mount the static files from data_viewers directory
    api.mount("/", StaticFiles(directory="/assets", html=True))
    
    @api.get("/")
    async def root():
        """Serve the main index.html file"""
        return FileResponse("/assets" / "index.html")
    
    # @api.get("/error_analysis")
    # async def error_analysis():
    #     """Serve the error analysis HTML file"""
    #     return FileResponse("/assets" / "error_analysis.html")
    
    # @api.get("/evaluation_report")
    # async def evaluation_report():
    #     """Serve the evaluation_report HTML file"""
    #     return FileResponse("/assets" / "evaluation_report.html")
    
    # @api.get("/api/list-json-files")
    # async def list_json_files():
    #     """API endpoint to list available JSON files"""
    #     json_files = []
    #     try:
    #         if DATA_DIR.exists():
    #             json_files = [
    #                 f"./data/{f}" for f in os.listdir(DATA_DIR) 
    #                 if f.endswith('.json') or f.endswith('.jsonl')
    #             ]
            
    #         # Default file should always be available
    #         if "./data/evaluation_report.json" not in json_files and (DATA_DIR / "evaluation_report.json").exists():
    #             json_files.append("./data/evaluation_report.json")
            
    #         # Sort alphabetically
    #         json_files.sort()
    #     except Exception as e:
    #         print(f"Error listing JSON files: {e}")
        
    #     return JSONResponse({"files": json_files})
    
    # @api.get("/data/{filename:path}")
    # async def get_data_file(filename: str):
    #     """Serve files from the data directory"""
    #     file_path = DATA_DIR / filename.split("?")[0]
    #     if not file_path.exists():
    #         raise HTTPException(status_code=404, detail=f"File {filename} not found")
        
    #     content_type = "application/json" if file_path.suffix in [".json", ".jsonl"] else None
    #     return FileResponse(file_path, media_type=content_type)
    
    # @api.get("/resumes/{filename:path}")
    # async def get_resume_file(filename: str):
    #     """Serve PDF files from the resumes directory"""
    #     file_path = RESUMES_DIR / filename.split("?")[0]
        
    #     # If not found in main resumes dir, try resume_archive
    #     if not file_path.exists() and ARCHIVE_DIR.exists():
    #         file_path = ARCHIVE_DIR / filename.split("?")[0]
        
    #     if not file_path.exists():
    #         raise HTTPException(status_code=404, detail=f"File {filename} not found")
        
    #     return FileResponse(file_path, media_type="application/pdf")
    
    # # Map app.js, styles.css to their proper paths
    # @api.get("/app.js")
    # async def get_app_js():
    #     return FileResponse(VIEWERS_DIR / "app.js", media_type="application/javascript")
    
    # @api.get("/styles.css")
    # async def get_styles_css():
    #     return FileResponse(VIEWERS_DIR / "styles.css", media_type="text/css")
    
    return api

@app.local_entrypoint()
def main():
    """Local entrypoint for testing"""
    print("Starting local development server")
    print("Note: This won't use Modal volumes when run locally")
    app.serve()

if __name__ == "__main__":
    # Try to use Modal's local entrypoint first
    main()


