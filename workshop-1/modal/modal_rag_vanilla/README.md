### Modal deployment of RAG with LlamaIndex

This is a simple example of how to deploy a RAG (retrieval-augmented generation) application using LlamaIndex and Modal. The application uses the LlamaIndex library to create an index from a set of documents and then uses that index to answer questions about the documents. The application is deployed on Modal, which allows for easy scaling and management of the application.

### Deployment to Modal
To deploy the application, you need to have a Modal account and the Modal CLI installed. You can install the Modal CLI using pip:

```bash
# check modal version
modal --version

# modal deployment
cd workshop-1/modal/modal_rag_vanilla
modal deploy -m app_rag_vanilla_wrapper
```
