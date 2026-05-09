from langchain_groq import ChatGroq
from rag.prompts import get_evaluation_prompt

def evaluate_answer(question, answer, source_docs, groq_api_key):

    if not source_docs:
        return None

    context = "\n\n".join(
        doc.page_content[:300]
        for doc in source_docs[:3]
    )

    prompt = get_evaluation_prompt(
        question,
        answer,
        context
    )

    try:

        llm = ChatGroq(
            api_key=groq_api_key,
            model="llama-3.1-8b-instant",
            temperature=0
        )

        response = llm.invoke(prompt)

        return response.content

    except Exception as e:

        print(f"Evaluation error: {e}")

        return None