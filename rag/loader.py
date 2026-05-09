from pathlib import Path
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader
)

def load_documents(file_paths):

    documents = []

    for file_path in file_paths:

        file_path = str(file_path)
        ext = Path(file_path).suffix.lower()

        print(f"Loading: {Path(file_path).name}")

        try:

            if ext == ".pdf":
                loader = PyPDFLoader(file_path)

            elif ext == ".docx":
                loader = Docx2txtLoader(file_path)

            elif ext == ".txt":
                loader = TextLoader(file_path, encoding="utf-8")

            else:
                continue

            docs = loader.load()

            for doc in docs:
                doc.metadata["source"] = Path(file_path).name

            documents.extend(docs)

            print(f"✅ Loaded {len(docs)} document units")

        except Exception as e:
            print(f"❌ Error loading file: {e}")

    return documents