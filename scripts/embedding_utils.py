from langchain_huggingface import HuggingFaceEmbeddings

def get_embedder(model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """Returns a HuggingFace embedding model."""
    return HuggingFaceEmbeddings(model_name=model_name)