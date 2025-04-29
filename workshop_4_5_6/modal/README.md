# Modal Deployment with PDF Persistence

This guide explains how to deploy the RAG application to Modal with PDF file persistence.

## Overview

The implementation uses two Modal volumes to ensure data persistence:

1. **LanceDB Volume**: Stores the vector database for text embeddings
2. **PDF Storage Volume**: Stores the actual PDF files that users upload

When a user uploads a PDF:
1. The PDF content is extracted and processed for the RAG pipeline
2. The original PDF file is saved to the PDF storage volume with a unique filename
3. The location of the stored PDF is saved in the LanceDB document metadata
4. The RAG pipeline continues with query processing as usual

## Deployment to Modal

To deploy the application:

```bash
# Check modal is configured
modal config show

# (Optional) If not already logged in
modal token new

# Deploy the application
cd workshop_4_5_6/modal
modal deploy -m rag_demo_vanilla_wrapper
```

## How PDF Persistence Works

1. **PDF Uploads**: When a user uploads a PDF, the original file is stored in a Modal volume
2. **Unique Identifiers**: Each PDF gets a unique filename (using UUID) to prevent conflicts
3. **Metadata Linking**: The PDF's storage path is saved as metadata in the document records
4. **Volume Mounting**: Both volumes are mounted to specific paths in the container:
   - LanceDB mounts to `/lancedb_workspace`
   - PDFs mount to `/pdfs`

## Benefits

- **Persistence Between Sessions**: PDFs remain available even after container restart
- **Document Recovery**: Original documents can be retrieved using the stored path
- **Multiple File Support**: The system can handle multiple PDF uploads with unique IDs
- **Metadata Tracking**: Document metadata keeps track of original file location