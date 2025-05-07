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

# Deploy the eval tools (static site)
modal deploy -m static_site
```
## Running the Application
