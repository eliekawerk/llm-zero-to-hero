from rag_vanilla import ingest_files, rag_pipeline_vector_db
from utils import semantic_search

if __name__ == "__main__":
    print(semantic_search(table, "What is Alex's experience with Python?"))
    response = rag_pipeline_vector_db(
        {},
        "What is this person's experience with Python?"
    )
    
    print(response)