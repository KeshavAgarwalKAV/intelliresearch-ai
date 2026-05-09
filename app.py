import gradio as gr
from pathlib import Path

from rag.loader import load_documents
from rag.chunking import chunk_documents
from rag.embeddings import create_vector_store
from rag.chain import build_rag_chain
from rag.evaluator import evaluate_answer
from analytics.dataset_loader import load_dataset
from analytics.eda import generate_dataset_summary, compute_correlations
from analytics.visualizations import plot_missing_values, plot_correlation_heatmap, plot_feature_importance
from analytics.insights import generate_ai_insights, generate_correlation_insights
from analytics.context_builder import build_dataset_context
from analytics.ml_recommender import detect_target_candidates, generate_ml_recommendations, select_best_target
from ml.preprocessing import preprocess_dataset
from ml.trainer import train_baseline_model
from ml.evaluation import evaluate_model, get_feature_importance
from langchain_groq import ChatGroq
from ui.layout import create_visualization_tabs

chain = None
dataset_contexts = {}

# ── PROCESS DOCUMENTS ──────────────────────────────────────────────────────

def process_documents(files, groq_api_key):

    global chain, dataset_contexts

    documents = []
    chunks = []

    if not files:
        return "❌ Please upload at least one document."

    if not groq_api_key.strip():
        return "❌ Please enter your Groq API key."

    try:

        file_paths = [f.name for f in files]

        document_files = [
            f for f in file_paths
            if f.endswith((".pdf", ".docx", ".txt"))
        ]

        csv_files = [
            f for f in file_paths
            if f.endswith(".csv")
            ]

        if not document_files and not csv_files:
            return "❌ No supported files uploaded."
        

        if document_files:

            documents = load_documents(document_files)

        if documents:

            chunks = chunk_documents(documents)

            vector_store = create_vector_store(chunks)

            chain = build_rag_chain(
                vector_store,
                groq_api_key
            )

        total_chars = sum(
            len(doc.page_content)
            for doc in documents
        )
        summary = "✅ Processing completed!\n\n"

        if documents:

            summary += f"""
        📄 Document Processing

        Files loaded: {', '.join(document_files)}

        Document units: {len(documents)}
        Chunks created: {len(chunks)}
        Total characters: {total_chars:,}

        Retrieval method: MMR
        Chunks per query: 6

        """
        
        analytics_output = ""
        missing_fig = None
        corr_fig = None
        feature_fig = None
        dataset_contexts = {}

        for csv_file in csv_files:

            df = load_dataset(csv_file)

            if df is not None:

                dataset_summary = generate_dataset_summary(df)
                target_candidates = detect_target_candidates(df)

                ml_recommendations = generate_ml_recommendations(
                    dataset_summary,
                    target_candidates,
                    groq_api_key
                )

                corr_matrix = compute_correlations(df)
                corr_insights = generate_correlation_insights(
                    corr_matrix,
                    groq_api_key
                )
                corr_fig = plot_correlation_heatmap(corr_matrix)
                
                best_target = select_best_target(target_candidates)

                feature_fig = None
                metrics = None
                target_column = None
                task_type = None

                if best_target:

                    target_column = best_target["column"]
                    task_type = best_target["task"]

                    X_train, X_test, y_train, y_test = (
                        preprocess_dataset(
                            df,
                            target_column
                        )
                    )

                    model = train_baseline_model(
                        X_train,
                        y_train,
                        task_type
                    )

                    metrics = evaluate_model(
                        model,
                        X_test,
                        y_test,
                        task_type
                    )

                    feature_importance = get_feature_importance(
                        model,
                        X_train.columns
                    )

                    feature_fig = plot_feature_importance(
                        feature_importance
                    )

                dataset_context = build_dataset_context(df, dataset_summary)
                missing_fig = plot_missing_values(df)

                dataset_contexts[Path(csv_file).name] = dataset_context

                ai_insights = generate_ai_insights(
                    dataset_summary,
                    groq_api_key
                )

                formatted_metrics = ""

                if metrics:

                    for metric_name, value in metrics.items():

                        formatted_metrics += (
                            f"- {metric_name}: "
                            f"{value:.4f}\n"
                        )

                analytics_output += f"""

        ---

        # 📊 Dataset Analysis: {Path(csv_file).name}

        Rows: {dataset_summary['rows']}
        Columns: {dataset_summary['columns']}

        Numeric Columns:
        {', '.join(dataset_summary['numeric_columns'])}

        Categorical Columns:
        {', '.join(dataset_summary['categorical_columns'])}

        ---

        # 🤖 AI Insights

        {ai_insights}

        ---

        # 📈 Correlation Analysis

        {corr_insights}

        ---

        # ⚙️ ML Recommendations

        {ml_recommendations}

        ---

        # 📊 Baseline ML Performance

        Target Column: {target_column}
        Task Type: {task_type}

        Metrics:
        {formatted_metrics}

"""

        return (
        summary + analytics_output,
        missing_fig,
        corr_fig,
        feature_fig
    )

    except Exception as e:
        print(f"Processing error: {e}")
        return f"❌ Error: {str(e)}"


# ── ANSWER QUESTION ────────────────────────────────────────────────────────

def answer_question(question, groq_api_key):
    global chain, dataset_contexts
    if chain is None and not dataset_contexts:
        return "⚠️ Please upload and process files first.", ""
    if not question.strip():
        return "Please enter a question.", ""
    try:
        combined_dataset_context = ""
        if dataset_contexts:

            combined_dataset_context = "\n\n".join(
                [
                    f"DATASET: {name}\n{context}"
                    for name, context in dataset_contexts.items()
                ]
            )
        enhanced_question = f"""
        USER QUESTION:
        {question}

        DATASET CONTEXT:
        {combined_dataset_context}
        """
        
        if chain is None and dataset_contexts:
            llm = ChatGroq(
                api_key=groq_api_key,
                model="llama-3.1-8b-instant",
                temperature=0.3
            )

            dataset_prompt = f"""
        You are an expert AI data scientist.

        Use the dataset context below to answer the user's question.

        DATASET CONTEXT:
        {combined_dataset_context}

        QUESTION:
        {question}

        Provide a detailed analytical answer.
        """

            response = llm.invoke(dataset_prompt)

            return response.content, ""
        
        result = chain.invoke({
            "question": enhanced_question
        })
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
                    preview = " ".join(    doc.page_content[:200].split())
                    citations += f"\n**[{len(seen)}] {source}{page_str}**\n> {preview}...\n"

        # Add evaluation scores
        if eval_score:
            citations += f"\n\n---\n### 🎯 Answer Quality Evaluation:\n```\n{eval_score}\n```"

        return answer, citations
    except Exception as e:
        print(f"Processing error: {e}")
        return f"❌ Error: {str(e)}"


# ── GRADIO UI ──────────────────────────────────────────────────────────────

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


with gr.Blocks(title="AI Research & Data Science Assistant") as demo:

    gr.Markdown("""
    # 🧠 AI Research & Data Science Assistant

    Multi-modal AI platform for:

    - 📄 Document Intelligence (RAG)
    - 📊 Dataset Analytics
    - 📈 Automated EDA & Visualizations
    - 🤖 AI-Generated Insights
    - ⚙️ Baseline ML Training & Evaluation

    Powered by LangChain · FAISS · Groq · Scikit-learn
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
                label="📁 Upload Files (PDF, DOCX, TXT, CSV)",
                file_types=[".pdf", ".docx", ".txt", ".csv"]
            )
            process_btn = gr.Button("⚙️ Process Documents", variant="primary", size="lg")
            process_status, missing_plot, correlation_plot, feature_plot = create_visualization_tabs()

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
        outputs=[process_status, missing_plot, correlation_plot, feature_plot]
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