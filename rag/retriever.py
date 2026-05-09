def create_retriever(vector_store):

    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 10,
            "fetch_k": 40,
            "lambda_mult": 0.7
        }
    )

    return retriever