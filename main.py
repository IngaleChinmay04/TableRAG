import os
from rag_pipelines.basic_rag import BasicRAGPipeline

from dotenv import load_dotenv
load_dotenv()

def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    DB_DIR = os.path.join(BASE_DIR, "db/chroma")
    COLLECTION_NAME = "rag_docs"

    rag = BasicRAGPipeline(DATA_DIR, DB_DIR, COLLECTION_NAME)
    print("Basic RAG CLI. Type 'exit' to quit.")

    while True:
        query = input("\nEnter your question: ")
        if query.lower() == "exit":
            break
        answer = rag.run(query)
        print(f"\nAnswer:\n{answer}")

if __name__ == "__main__":
    main()