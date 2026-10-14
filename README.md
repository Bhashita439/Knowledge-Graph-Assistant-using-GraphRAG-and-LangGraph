# Graphy: GraphRAG with LangGraph

Graphy is a powerful Graph Retrieval-Augmented Generation (GraphRAG) application built with **LangGraph**, **Groq**, and **Neo4j**. It allows users to upload PDF documents, extract structured knowledge into a graph database, and perform complex queries using natural language.

## 🚀 Features

- **PDF Knowledge Extraction**: Upload PDF documents and automatically extract entities and relationships.
- **GraphRAG Engine**: Combines the power of Knowledge Graphs with LLMs for more accurate and context-aware answers.
- **Natural Language Querying**: Ask questions in plain English; the app converts them into Cypher queries to retrieve data from Neo4j.
- **Powered by Groq**: Utilizes the lightning-fast inference of Groq with support for multiple models (**Llama 3.3**, **Llama 3.1**, **Mixtral**, **Gemma**) to ensure reliability and performance.
- **LangGraph Orchestration**: Uses a stateful graph workflow to manage the query generation, execution, and response formatting pipeline.

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **LLM**: Llama 3.3, Llama 3.1, Mixtral, Gemma 2 (via Groq API)
- **Graph Database**: Neo4j
- **Orchestration**: LangGraph
- **Framework**: LangChain
- **Embeddings**: HuggingFace (`sentence-transformers/all-MiniLM-L6-v2`)
- **PDF Processing**: PyMuPDF

## 📋 Prerequisites

Before running the application, ensure you have the following:

1.  **Python 3.9+** installed.
2.  A **Neo4j Database** (AuraDB Free tier works great).
3.  A **Groq API Key**.

## ⚙️ Installation

1.  **Clone the repository:**

    ```bash
    git clone <repository-url>
    cd <repository-folder>
    ```

2.  **Install dependencies:**

    ```bash
    pip install streamlit python-dotenv langchain-community langchain-core langchain-groq langchain-huggingface langchain-neo4j langchain-experimental langgraph pymupdf neo4j
    ```

3.  **Set up Environment Variables:**
    Create a `.env` file in the root directory and add your credentials:
    ```env
    GROQ_API_KEY=your_groq_api_key
    NEO4J_URI=neo4j+s://your-instance.databases.neo4j.io
    NEO4J_USERNAME=neo4j
    NEO4J_PASSWORD=your_password
    ```

## 🏃‍♂️ Usage

1.  **Run the Streamlit app:**

    ```bash
    python -m streamlit run main.py
    ```

2.  **Configure & Connect:**

    - Enter your **Groq API Key** and **Neo4j credentials** in the sidebar (or use `.env`).
    - Select your preferred **LLM Model** (e.g., Llama 3.3, Mixtral) from the dropdown to handle rate limits or optimize performance.
    - Click **Connect & Initialize**.

3.  **Upload a PDF:**

    - Use the file uploader to select a PDF document.
    - The app will process the file, split the text, and extract a knowledge graph.

4.  **Ask Questions:**
    - Type your question in the input box (e.g., "What symptoms does the patient have?").
    - The app will generate a Cypher query, execute it against the graph, and provide a natural language answer.

## 🧠 How It Works

1.  **Ingestion**: The PDF is loaded using `PyMuPDFLoader` and split into chunks.
2.  **Extraction**: `LLMGraphTransformer` (powered by the selected Groq model) analyzes the text to identify nodes (e.g., Patient, Disease, Medication) and relationships.
3.  **Storage**: The extracted graph data is stored in Neo4j.
4.  **Retrieval (GraphRAG)**:
    - **Generate Cypher**: The LLM translates the user's question into a Cypher query based on the graph schema.
    - **Execute**: The query is run against the Neo4j database.
    - **Answer**: The results are formatted and synthesized into a final answer by the LLM.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
