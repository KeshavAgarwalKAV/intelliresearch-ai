from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_documents(documents):

    """
    Smaller chunks = more precise retrieval.
    chunk_size=500 gives finer granularity than 800.
    chunk_overlap=150 gives more context overlap.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = splitter.split_documents(documents)

    print(f"✅ Split into {len(chunks)} chunks")

    return chunks