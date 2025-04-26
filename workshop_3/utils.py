import nltk
import lancedb
from lancedb.embeddings import get_registry
from lancedb.pydantic import LanceModel, Vector

nltk.download("punkt")
import re

def text_chunker(text, max_chunk_length=1000, overlap=100):
    """
    Helper function for chunking text
    """
    # Initialize result
    result = []

    current_chunk_count = 0
    separator = ["\n", " "]
    _splits = re.split(f"({separator})", text)
    splits = [_splits[i] + _splits[i + 1] for i in range(1, len(_splits), 2)]

    for i in range(len(splits)):
        if current_chunk_count != 0:
            chunk = "".join(
                splits[
                    current_chunk_count
                    - overlap : current_chunk_count
                    + max_chunk_length
                ]
            )
        else:
            chunk = "".join(splits[0:max_chunk_length])

        if len(chunk) > 0:
            result.append("".join(chunk))
        current_chunk_count += max_chunk_length

    return result
  

# embeddings_mdl = (
#     get_registry()
#       .get("sentence-transformers")
#       .create(name="BAAI/bge-small-en-v1.5")
# )
# embeddings_mdl = (
#   get_registry()
#     .get("openai")
#     .create(model="text-embedding-ada-002")
#   )

# Initialize Gemini embedding model
embeddings_mdl = get_registry().get("gemini-text").create()


# this is the schema for the table, where the embeddings are created automatically
# when the data is added
class Documents(LanceModel):
    vector: Vector(embeddings_mdl.ndims()) = embeddings_mdl.VectorField()
    text: str = embeddings_mdl.SourceField()
    file_path: str

def lanceDBConnection(chunks, metadata):
    """
    LanceDB insertion with metadata
    """
    db = lancedb.connect("lancedb_workspace")
    table = db.create_table("documents", schema=Documents, mode="overwrite")

    # Combine chunks with metadata
    # note that the embeddings are created and stored in the table implicitly
    # when the data is added
    data = [{"text": s, **metadata} for s in chunks]
    
    # Ingest data into the table
    table.add(data)
    return table

def full_text_search(table, question):
  # FTS Search
  fts_result = table.search(question, query_type="fts").limit(5).to_list()
  return [r["text"] for r in fts_result]
  
def semantic_search(table, question):
  # Semantic Retriever
  vs_result = table.search(question, query_type="vector").limit(10).to_list()
  return [r["text"] for r in vs_result]
  
def hybrid_search(table, question):
  # Hybrid Retriever
  from lancedb.rerankers import LinearCombinationReranker
  

  reranker = LinearCombinationReranker(
      weight=0.7
  )  # Weight = 0 Means pure Text Search (BM-25) and 1 means pure Sementic (Vector) Search

  hs_result = (
      table.search(
          question,
          query_type="hybrid",
      )
      .rerank(reranker=reranker)
      .limit(5)
      .to_list()
  )

  return [r["text"] for r in hs_result]