#!/usr/bin/env python3
import http.server
import socketserver
import os
import webbrowser
from pathlib import Path

# Get the current directory (where this script is)
current_dir = Path(__file__).parent.absolute()
# Go one directory up to the workshop_3 root
os.chdir(current_dir.parent)

PORT = 8000

Handler = http.server.SimpleHTTPRequestHandler

print(f"Starting server at http://localhost:{PORT}")
print(f"Open your browser and navigate to: http://localhost:{PORT}/data_viewers/")

# Open the browser automatically
webbrowser.open(f'http://localhost:{PORT}/data_viewers/')

with socketserver.TCPServer(("", PORT), Handler) as httpd:
    print("Press Ctrl+C to stop the server")
    httpd.serve_forever()