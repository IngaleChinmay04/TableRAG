import os
import glob
from scripts.pdf_utils import extract_text_from_pdf
from scripts.embedding_utils import get_embedder
from scripts.chroma_utils import get_chroma_client, get_or_create_collection
from langchain.text_splitter import RecursiveCharacterTextSplitter
from groq import Groq
from dotenv import load_dotenv


load_dotenv()

class BasicRAGPipeline:
    def __init__(self, data_dir, db_dir, collection_name, embed_model="sentence-transformers/all-MiniLM-L6-v2"):
        self.data_dir = data_dir
        self.db_dir = db_dir
        self.collection_name = collection_name
        self.embedder = get_embedder(embed_model)
        self.client = get_chroma_client(db_dir)
        self.collection = get_or_create_collection(self.client, collection_name)

    def ingest_pdfs(self, reset_db=True):
        """Extract, chunk, embed, and store all PDFs in data_dir."""
        pdf_files = glob.glob(os.path.join(self.data_dir, "*.pdf"))
        docs = []
        for pdf_path in pdf_files:
            text = extract_text_from_pdf(pdf_path)
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            for chunk in splitter.split_text(text):
                docs.append({"content": chunk, "meta": {"source": os.path.basename(pdf_path)}})
        if reset_db:
            # Optionally: drop and recreate collection instead of resetting entire DB, if needed
            try:
                self.client.delete_collection(self.collection_name)
            except Exception:
                pass  # Collection may not exist yet
            self.collection = get_or_create_collection(self.client, self.collection_name)

        for i, doc in enumerate(docs):
            emb = self.embedder.embed_documents([doc["content"]])[0]
            self.collection.add(documents=[doc["content"]],
                                metadatas=[doc["meta"]],
                                embeddings=[emb],
                                ids=[str(i)])
        print(f"Ingested {len(docs)} chunks from {len(pdf_files)} PDFs into ChromaDB.")

    def retrieve(self, query, k=3):
        """Retrieve top k most relevant docs for a query."""
        query_emb = self.embedder.embed_query(query)
        results = self.collection.query(query_embeddings=[query_emb], n_results=k)
        return [doc for doc in results["documents"][0]]

    
    def query_llm(self, prompt, model="llama-3.3-70b-versatile"):
        client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant."
                },
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model=model,
            max_tokens=256,
            temperature=0.0,
        )
        return chat_completion.choices[0].message.content



    def run(self, query, ingest_if_empty=True):
        # If collection is empty, ingest data
        if ingest_if_empty and not self.collection.count():
            print("No data in vector DB. Ingesting...")
            self.ingest_pdfs(reset_db=True)
        docs = self.retrieve(query, k=3)
        context = "\n\n".join(docs)
        prompt = f"Use the following context to answer the question.\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"
        answer = self.query_llm(prompt)
        return answer