#!/usr/bin/env python3
import os
import json
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESUMES_DIR = PROJECT_ROOT / "resumes"

class CustomHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        # API endpoint to list available JSON files
        if self.path.startswith("/api/list-json-files"):
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            
            # Get list of JSON files in the data directory
            json_files = []
            try:
                # Find all json files in the data directory
                data_path = DATA_DIR
                if data_path.exists():
                    json_files = [
                        f"./data/{f}" for f in os.listdir(data_path) 
                        if f.endswith('.json') or f.endswith('.jsonl')
                    ]
            except Exception as e:
                print(f"Error listing JSON files: {e}")
            
            # Default file should always be available
            if "./data/evaluation_report.json" not in json_files and os.path.exists(DATA_DIR / "evaluation_report.json"):
                json_files.append("./data/evaluation_report.json")
                
            # Sort alphabetically
            json_files.sort()
            
            # Write the JSON response
            self.wfile.write(json.dumps({"files": json_files}).encode())
            return
        
        # Handle requests for data files
        elif self.path.startswith("/data/"):
            file_path = PROJECT_ROOT / self.path.lstrip("/").split("?")[0]
            if file_path.exists():
                self.send_response(200)
                if file_path.suffix == ".json":
                    self.send_header("Content-type", "application/json")
                elif file_path.suffix == ".jsonl":
                    self.send_header("Content-type", "application/json")
                else:
                    self.send_header("Content-type", self.guess_type(str(file_path)))
                self.end_headers()
                
                with open(file_path, "rb") as f:
                    self.wfile.write(f.read())
                return
            else:
                print(f"File not found: {file_path}")
                
        # Handle requests for resume PDFs
        elif self.path.startswith("/resumes/"):
            file_name = self.path.split("/")[-1].split("?")[0]
            file_path = RESUMES_DIR / file_name
            
            # If not found in main resumes dir, try resume_archive
            if not file_path.exists() and (PROJECT_ROOT / "resume_archive").exists():
                file_path = PROJECT_ROOT / "resume_archive" / file_name
            
            if file_path.exists():
                self.send_response(200)
                self.send_header("Content-type", "application/pdf")
                self.end_headers()
                
                with open(file_path, "rb") as f:
                    self.wfile.write(f.read())
                return
            else:
                print(f"Resume PDF not found: {file_path}")
                self.send_response(404)
                self.send_header('Content-type', 'text/plain')
                self.end_headers()
                self.wfile.write(f"File not found: {file_name}".encode())
                return

        # Handle request for error_analysis.html specifically
        elif self.path == "/error_analysis" or self.path == "/error_analysis.html":
            file_path = Path(__file__).parent / "error_analysis.html"
            if file_path.exists():
                self.send_response(200)
                self.send_header("Content-type", "text/html")
                self.end_headers()
                
                with open(file_path, "rb") as f:
                    self.wfile.write(f.read())
                return
                
        # For all other paths, use the default handler
        return super().do_GET()

def run(server_class=HTTPServer, handler_class=CustomHandler, port=8000):
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    print(f"Starting server at http://localhost:{port}")
    print(f"Error Analysis Dashboard available at http://localhost:{port}/error_analysis")
    httpd.serve_forever()

if __name__ == "__main__":
    run()