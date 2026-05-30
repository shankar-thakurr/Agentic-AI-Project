"""
Ingests data/faq.csv into the 'faq' Chroma collection.
Run once (or whenever the CSV changes): python ingest_faq.py
"""
import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
import chromadb
import pandas as pd
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

_HERE       = os.path.dirname(os.path.abspath(__file__))
CHROMA_DIR  = os.path.join(_HERE, "chroma_store")
COLLECTION  = "faq"
CSV_PATH    = os.path.join(_HERE, "data", "faq.csv")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_faq_documents(csv_path: str) -> list[Document]:
    df = pd.read_csv(csv_path)
    docs = []
    for _, row in df.iterrows():
        content = f"Q: {row['question']}\nA: {row['answer']}"
        docs.append(Document(
            page_content=content,
            metadata={"source": "faq", "category": row["category"], "faq_id": str(row["id"])},
        ))
    return docs


def main():
    print("Loading FAQ documents...")
    docs = load_faq_documents(CSV_PATH)
    print(f"  {len(docs)} FAQ entries loaded.")

    print("Initialising embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    # Clear existing collection to avoid duplicate vectors on re-run
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        client.delete_collection(COLLECTION)
        print(f"  Existing '{COLLECTION}' collection cleared.")
    except Exception:
        pass  # Collection didn't exist yet

    print(f"Embedding and storing in Chroma collection '{COLLECTION}'...")
    Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=COLLECTION,
        persist_directory=CHROMA_DIR,
    )
    print(f"  Done. {len(docs)} vectors stored.")


if __name__ == "__main__":
    main()
