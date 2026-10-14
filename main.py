import os
import streamlit as st
from dotenv import load_dotenv
from utils import process_pdf
from graph_agent import GraphRAGAgent

# Load environment variables
load_dotenv()

# Page Configuration
st.set_page_config(
    layout="wide",
    page_title="Graphy AI",
    page_icon="🧠",
    initial_sidebar_state="expanded"
)

# Load Custom CSS
def load_css():
    with open("styles.css", "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

if "agent" not in st.session_state:
    st.session_state.agent = None

# Sidebar Configuration
with st.sidebar:
    st.title("⚙️ Configuration")
    
    # API Keys Section
    with st.expander("🔑 API Keys", expanded=True):
        groq_key = st.text_input(
            "Groq API Key", 
            type="password",
            value=os.getenv("GROQ_API_KEY", ""),
            help="Get your key from console.groq.com"
        )
    
    # Database Section
    with st.expander("🗄️ Database", expanded=True):
        neo4j_url = st.text_input(
            "Neo4j URL",
            value=os.getenv("NEO4J_URI", "neo4j+s://your-instance.databases.neo4j.io")
        )
        neo4j_user = st.text_input(
            "Username",
            value=os.getenv("NEO4J_USERNAME", "neo4j")
        )
        neo4j_password = st.text_input(
            "Password",
            type="password",
            value=os.getenv("NEO4J_PASSWORD", "")
        )

    # Model Section
    with st.expander("🤖 Model Settings", expanded=True):
        selected_model = st.selectbox(
            "Choose Model",
            options=[
                "llama-3.3-70b-versatile",
                "llama-3.1-70b-versatile",
                "llama-3.1-8b-instant",
                "mixtral-8x7b-32768",
                "gemma2-9b-it"
            ],
            index=0,
            help="Switch models if you hit rate limits."
        )
        
        if st.button("🔌 Connect & Initialize"):
            if groq_key and neo4j_password:
                try:
                    with st.spinner("Connecting to Graph Database..."):
                        agent = GraphRAGAgent(
                            neo4j_url=neo4j_url,
                            neo4j_username=neo4j_user,
                            neo4j_password=neo4j_password,
                            groq_api_key=groq_key,
                            model_name=selected_model
                        )
                        st.session_state.agent = agent
                        st.success(f"Connected with {selected_model}!")
                except Exception as e:
                    st.error(f"Connection failed: {str(e)}")
            else:
                st.warning("Please provide all credentials.")

    # File Upload Section
    st.markdown("### 📄 Knowledge Base")
    uploaded_file = st.file_uploader("Upload Medical Records (PDF)", type="pdf")
    
    if uploaded_file and st.session_state.agent:
        if st.button("🧠 Process Document"):
            with st.spinner("Extracting Knowledge Graph..."):
                try:
                    docs = process_pdf(uploaded_file)
                    st.session_state.agent.ingest_documents(docs)
                    st.success(f"Processed {len(docs)} chunks into graph!")
                except Exception as e:
                    st.error(f"Processing failed: {str(e)}")
    elif uploaded_file and not st.session_state.agent:
        st.info("Please connect to database first.")

    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown("""
    **Graphy AI** uses **GraphRAG** to provide accurate answers from your documents.
    
    - **LLM**: Llama 3.3 70B
    - **Graph**: Neo4j
    - **Orchestration**: LangGraph
    """)

# Main Chat Interface
st.title("🧠 Graphy AI Assistant")
st.markdown("Ask questions about your uploaded medical records.")

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input
if prompt := st.chat_input("Ask a question (e.g., 'What symptoms does the patient have?')"):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    if st.session_state.agent:
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.agent.query(prompt)
                    
                    # Extract info
                    answer = response.get("result", "No answer found.")
                    cypher = response.get("cypher_query", "")
                    
                    # Display answer
                    full_response = f"{answer}\n\n"
                    if cypher:
                        full_response += f"```cypher\n{cypher}\n```"
                    
                    message_placeholder.markdown(full_response)
                    
                    # Add to history
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                    
                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    message_placeholder.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
    else:
        with st.chat_message("assistant"):
            st.warning("Please connect to the database in the sidebar to start chatting.")