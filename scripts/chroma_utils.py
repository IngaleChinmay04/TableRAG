import chromadb
from chromadb.config import Settings

def get_chroma_client(db_dir: str) -> chromadb.PersistentClient:
    """Create or return a persistent ChromaDB client."""
    return chromadb.PersistentClient(path=db_dir, settings=Settings())

def get_or_create_collection(client, collection_name: str):
    """Get or create a ChromaDB collection."""
    return client.get_or_create_collection(collection_name)