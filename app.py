import os
import gradio as gr
from pathlib import Path

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain

chain = None
chat_history = []
doc_summary = ""

# ── 1. DOCUMENT LOADING ───────────────────────────────────────────────────────

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
            print(f"✅ Loaded {len(docs)} pages from {Path(file_path).name}")
        except Exception as e:
            print(f"❌ Error: {e}")
    return documents


# ── 2. CHUNKING ───────────────────────────────────────────────────────────────

def chunk_documents(documents):
    """
    Smaller chunks = more precise retrieval.
    chunk_size=500 gives finer granularity than 800.
    chunk_overlap=150 gives more context overlap.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(documents)
    print(f"✅ Split into {len(chunks)} chunks")
    return chunks


# ── 3. VECTOR STORE ───────────────────────────────────────────────────────────

def create_vector_store(chunks):
    print("Creating embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    vector_store = FAISS.from_documents(chunks, embeddings)
    print(f"✅ Vector store ready — {len(chunks)} vectors")
    return vector_store, embeddings


# ── 4. CUSTOM PROMPTS ─────────────────────────────────────────────────────────

def get_custom_prompt():
    """
    This is where the magic happens.
    A strong system prompt makes the LLM give detailed, structured answers.
    We tell it exactly how to behave and what to do with the context.
    """
    template = """You are an expert research assistant with deep analytical skills.
Your job is to answer questions based strictly on the provided document context.

INSTRUCTIONS:
- Give detailed, thorough answers — never give one-line responses
- Always structure your answer with clear sections when appropriate
- Quote directly from the documents when relevant (use "quotation marks")
- If comparing multiple documents, explicitly mention each document by name
- If the answer is not in the context, say "This information is not in the provided documents"
- Never hallucinate or make up information not in the context
- End every answer with a "Key Takeaway" summary in one sentence

CONTEXT FROM DOCUMENTS:
{context}

CONVERSATION HISTORY:
{chat_history}

QUESTION: {question}

DETAILED ANSWER:"""

    return PromptTemplate(
        input_variables=["context", "chat_history", "question"],
        template=template
    )


# ── 5. RAG CHAIN ──────────────────────────────────────────────────────────────

def build_rag_chain(vector_store, groq_api_key):
    llm = ChatGroq(
        api_key=groq_api_key,
        model="llama-3.1-8b-instant",
        temperature=0.3,
        max_tokens=1024   # allow longer, more detailed answers
    )

    # MMR retrieval — more diverse chunks, less repetition
    # MMR = Maximal Marginal Relevance
    # Instead of top-k similar chunks, it picks diverse relevant chunks
    # This prevents the same sentence being retrieved multiple times
    retriever = vector_store.as_retriever(
        search_type="mmr",           # diversity-aware retrieval
        search_kwargs={
            "k": 6,                  # retrieve 6 chunks
            "fetch_k": 20,           # consider top 20 before MMR filtering
            "lambda_mult": 0.7       # 0=max diversity, 1=max similarity
        }
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

    rag_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        verbose=False,
        combine_docs_chain_kwargs={"prompt": get_custom_prompt()}
    )

    return rag_chain


# ── 6. PROCESS DOCUMENTS ──────────────────────────────────────────────────────

def process_documents(files, groq_api_key):
    global chain, chat_history, doc_summary

    if not files:
        return "❌ Please upload at least one document."
    if not groq_api_key.strip():
        return "❌ Please enter your Groq API key."

    try:
        file_paths = [f.name for f in files]
        documents = load_documents(file_paths)

        if not documents:
            return "❌ Could not load documents."

        chunks = chunk_documents(documents)
        vector_store, _ = create_vector_store(chunks)
        chain = build_rag_chain(vector_store, groq_api_key)
        chat_history = []

        sources = list(set([Path(f.name).name for f in files]))
        total_chars = sum(len(doc.page_content) for doc in documents)

        summary = f"""✅ **Documents processed successfully!**

| Detail | Value |
|--------|-------|
| Files loaded | {', '.join(sources)} |
| Document units | {len(documents)} |
| Chunks created | {len(chunks)} |
| Total characters | {total_chars:,} |
| Retrieval method | MMR (diversity-aware) |
| Chunks per query | 6 |

**Now ask anything — the AI will give detailed, cited answers.**

**Suggested questions:**
- "Give me a detailed summary of all documents"
- "What are the key skills mentioned?"
- "Compare the main themes across documents"
- "What evidence supports X claim?"
"""
        return summary

    except Exception as e:
        return f"❌ Error: {str(e)}"

def evaluate_answer(question, answer, source_docs, groq_api_key):
    """
    Automatically evaluate RAG answer quality using LLM-as-judge.
    This is a real technique used in production RAG systems.
    
    Metrics:
    - Faithfulness: Is the answer grounded in the source docs?
    - Relevance: Does it actually answer the question?
    - Completeness: Did it use all available context?
    """
    if not source_docs:
        return None
    
    context = "\n\n".join([doc.page_content for doc in source_docs[:3]])
    
    eval_prompt = f"""You are a RAG evaluation expert. Score this Q&A pair.

QUESTION: {question}
ANSWER: {answer}
SOURCE CONTEXT: {context[:1000]}

Rate each metric from 1-5 and explain briefly:
1. Faithfulness (1-5): Is the answer supported by the context? No hallucinations?
2. Relevance (1-5): Does the answer directly address the question?
3. Completeness (1-5): Did it use the context thoroughly?

Respond in exactly this format:
Faithfulness: X/5 — reason
Relevance: X/5 — reason  
Completeness: X/5 — reason
Overall: X/5"""

    try:
        llm = ChatGroq(api_key=groq_api_key, model="llama-3.1-8b-instant", temperature=0)
        response = llm.invoke(eval_prompt)
        return response.content
    except:
        return None


# ── 7. ANSWER QUESTION ────────────────────────────────────────────────────────

def answer_question(question, groq_api_key):
    global chain
    if chain is None:
        return "⚠️ Please upload and process documents first.", ""
    if not question.strip():
        return "Please enter a question.", ""
    try:
        result = chain.invoke({"question": question})
        answer = result["answer"]
        source_docs = result.get("source_documents", [])

        # Evaluate answer quality
        eval_score = evaluate_answer(question, answer, source_docs, groq_api_key)

        citations = ""
        if source_docs:
            citations = "\n\n---\n### 📚 Sources Used:\n"
            seen = set()
            for i, doc in enumerate(source_docs):
                source = doc.metadata.get("source", "Unknown")
                page = doc.metadata.get("page", "")
                page_str = f" · Page {int(page)+1}" if page != "" else ""
                key = f"{source}{page_str}"
                if key not in seen:
                    seen.add(key)
                    preview = doc.page_content[:200].replace("\n", " ").strip()
                    citations += f"\n**[{len(seen)}] {source}{page_str}**\n> {preview}...\n"

        # Add evaluation scores
        if eval_score:
            citations += f"\n\n---\n### 🎯 Answer Quality Evaluation:\n```\n{eval_score}\n```"

        return answer, citations
    except Exception as e:
        return f"❌ Error: {str(e)}", ""


# ── 8. GRADIO UI ──────────────────────────────────────────────────────────────

def chat(question, history, groq_api_key):
    if not question.strip():
        return history, ""
    answer, citations = answer_question(question, groq_api_key)
    full_response = answer + citations
    history = history + [
        {"role": "user", "content": question},
        {"role": "assistant", "content": full_response}
    ]
    return history, ""


with gr.Blocks(title="Multi-Document RAG Assistant") as demo:

    gr.Markdown("""
    # 🧠 Multi-Document AI Research Assistant
    **Production-grade RAG with MMR retrieval, conversation memory & source citations**
    
    Upload PDFs, DOCX, or TXT files and have an intelligent conversation about them.
    Powered by LangChain · FAISS · HuggingFace Embeddings · Groq Llama 3.1
    """)

    with gr.Row():
        groq_key = gr.Textbox(
            label="🔑 Groq API Key (free at console.groq.com)",
            placeholder="gsk_...",
            type="password",
            scale=3
        )

    with gr.Row():
        with gr.Column(scale=1):
            file_upload = gr.Files(
                label="📁 Upload Documents (PDF, DOCX, TXT)",
                file_types=[".pdf", ".docx", ".txt"]
            )
            process_btn = gr.Button("⚙️ Process Documents", variant="primary", size="lg")
            process_status = gr.Markdown(value="*Upload documents and click Process to begin*")

            gr.Markdown("""
            ---
            **💡 Pro Tips:**
            - Upload multiple docs to compare them
            - Ask follow-up questions — it remembers context
            - Ask for specific quotes from documents
            - Ask it to compare two documents directly
            - Ask "what evidence supports X?"
            """)

        with gr.Column(scale=2):
            chatbot = gr.Chatbot(
                label="💬 Chat with your documents",
                height=500
            )
            with gr.Row():
                question_input = gr.Textbox(
                    label="Your question",
                    placeholder="Ask anything about your documents...",
                    scale=5
                )
                submit_btn = gr.Button("Send 📨", variant="primary", scale=1)
            clear_btn = gr.Button("🗑️ Clear Chat")

    # Suggested question buttons
    gr.Markdown("**Quick questions:**")
    with gr.Row():
        q1 = gr.Button("📋 Summarize all documents")
        q2 = gr.Button("🔍 What are the key findings?")
        q3 = gr.Button("⚖️ Compare the documents")
        q4 = gr.Button("💡 What are the main conclusions?")

    # Events
    process_btn.click(
        fn=process_documents,
        inputs=[file_upload, groq_key],
        outputs=process_status
    )
    submit_btn.click(
        fn=chat,
        inputs=[question_input, chatbot, groq_key],
        outputs=[chatbot, question_input]
    )
    question_input.submit(
        fn=chat,
        inputs=[question_input, chatbot, groq_key],
        outputs=[chatbot, question_input]
    )
    clear_btn.click(lambda: ([], ""), outputs=[chatbot, question_input])

    # Quick question buttons
    q1.click(lambda: "Give me a detailed summary of all the uploaded documents", outputs=question_input)
    q2.click(lambda: "What are the key findings and important points across all documents?", outputs=question_input)
    q3.click(lambda: "Compare and contrast the main themes across the different documents", outputs=question_input)
    q4.click(lambda: "What are the main conclusions and recommendations in these documents?", outputs=question_input)


if __name__ == "__main__":
    demo.launch(share=True)