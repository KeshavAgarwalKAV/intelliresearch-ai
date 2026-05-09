# 🧠 AI Research & Data Science Assistant

An end-to-end AI-powered platform that combines **Document Intelligence**, **Dataset Analytics**, and **Automated Machine Learning Workflows** into a unified conversational interface.

---

## 📖 Why This Project?

Most AI assistants focus only on document retrieval or only on machine learning workflows.

This project combines:
- Retrieval-Augmented Generation (RAG)
- Automated dataset analytics
- Automated ML experimentation
- Conversational AI

into a unified intelligent assistant for research and data science tasks.

---

## ✨ Features

### 📄 Document Intelligence (RAG)
- Multi-document upload support (PDF, DOCX, TXT)
- Conversational Retrieval-Augmented Generation (RAG)
- FAISS vector database with MMR (Max Marginal Relevance) retrieval
- Source-aware, context-aware responses
- AI-based answer quality evaluation

### 📊 Dataset Analytics
Automatically performs on CSV datasets:
- Dataset summarization & statistical analysis
- Missing value analysis
- Numeric and categorical feature detection
- Correlation analysis
- AI-generated analytical insights

### 📈 Visualizations
- Missing value analysis plots
- Correlation heatmaps
- Feature importance visualizations

### ⚙️ Automated ML Workflow
- Auto-detects potential target columns
- Infers ML task type (Classification or Regression)
- Preprocesses data and trains Random Forest models
- Evaluates model performance and generates feature importance rankings

### 💬 Conversational AI Interface
- Ask questions about uploaded documents
- Ask analytical questions about datasets
- Request summaries and insights through a natural chatbot UI

---

## 🏗️ Project Architecture

```
AI-Research-Assistant/
│
├── analytics/
│   ├── context_builder.py
│   ├── dataset_loader.py
│   ├── eda.py
│   ├── insights.py
│   ├── ml_recommender.py
│   └── visualizations.py
│
├── ml/
│   ├── evaluation.py
│   ├── preprocessing.py
│   └── trainer.py
│
├── rag/
│   ├── chain.py
│   ├── chunking.py
│   ├── embeddings.py
│   ├── evaluator.py
│   ├── loader.py
│   ├── prompts.py
│   └── retriever.py
│
├── ui/
│   └── layout.py
│
├── utils/
├── .gitignore
├── app.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Tech Stack

| Layer | Tools |
|---|---|
| **AI / LLM** | LangChain, Groq API, Llama 3.1 |
| **Embeddings** | Hugging Face Embeddings |
| **Vector Search** | FAISS, MMR Retrieval |
| **Data Science & ML** | Pandas, NumPy, Scikit-learn, Matplotlib |
| **Frontend** | Gradio |

---

## 🔄 System Workflow

```
Upload Files
    ↓
File Type Detection
    ↓
┌─────────────────────────────┐    ┌─────────────────────────────┐
│     Document Pipeline       │    │      Dataset Pipeline        │
│─────────────────────────────│    │─────────────────────────────│
│ Load Documents              │    │ Load CSV                     │
│ Chunk Text                  │    │ EDA & Statistics             │
│ Generate Embeddings         │    │ Correlation Analysis         │
│ Create FAISS Vector Store   │    │ AI Insight Generation        │
│ Conversational Retrieval QA │    │ Automated ML Training        │
│                             │    │ Evaluation & Feature Ranks   │
└─────────────────────────────┘    └─────────────────────────────┘
```

---

## 🛠️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/KeshavAgarwalKAV/ai-research-assistant.git
cd ai-research-assistant
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Create a `.env` file in the root directory:

```env
GROQ_API_KEY=your_api_key_here
```

### 4. Run the Application

```bash
python app.py
```

---

## 💡 Example Queries

**Document Questions**
- *"Summarize the report"*
- *"What are the key findings?"*
- *"Who is the author?"*
- *"Compare the uploaded documents"*

**Dataset Questions**
- *"Which columns contain missing values?"*
- *"What are the strongest correlations?"*
- *"Which features are most important?"*
- *"What ML task is suitable for this dataset?"*

---

## 🧠 Key Learnings

Through this project, I explored:

- RAG pipeline architecture
- Embedding-based retrieval
- Conversational memory systems
- Automated EDA workflows
- ML preprocessing and evaluation
- Modular AI application design

---

## ⚠️ Current Limitations

- Target detection currently uses heuristic-based selection
- Dataset reasoning is summary-driven rather than row-level reasoning
- Visualization support is currently limited to static plots
- The system is optimized for small-to-medium tabular datasets

---

## 📌 Future Improvements

- [ ] XGBoost integration
- [ ] SHAP explainability
- [ ] Multi-dataset comparison
- [ ] Improved target detection heuristics
- [ ] Interactive dashboard UI
- [ ] Hybrid retrieval strategies
- [ ] Advanced evaluation metrics
- [ ] Dataset-specific recommendations

---

## 🎯 Key Concepts Demonstrated

**AI Engineering** — RAG systems, prompt engineering, LLM orchestration, conversational AI

**Data Science** — EDA, statistical analysis, feature engineering, ML workflows, model evaluation

**Software Engineering** — Modular architecture, multi-pipeline orchestration, interactive UI design

---

## 👨‍💻 Author

**Keshav Agarwal**
AI & ML Engineering Student — interested in AI Systems, Machine Learning, Data Science, and Intelligent Applications.
