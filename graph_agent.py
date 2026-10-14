import os
from typing import TypedDict, List, Any
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_neo4j import Neo4jGraph, Neo4jVector
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_core.documents import Document

# Define the state for our graph
class GraphState(TypedDict):
    question: str
    cypher_query: str
    result: str
    error: str
    schema: str

class GraphRAGAgent:
    def __init__(self, neo4j_url, neo4j_username, neo4j_password, groq_api_key, model_name="llama-3.3-70b-versatile"):
        self.neo4j_url = neo4j_url
        self.neo4j_username = neo4j_username
        self.neo4j_password = neo4j_password
        self.groq_api_key = groq_api_key
        self.model_name = model_name
        
        # Initialize LLM
        self.llm = ChatGroq(
            model=model_name,
            temperature=0,
            groq_api_key=groq_api_key
        )
        
        # Initialize Embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Initialize Graph
        self.graph = Neo4jGraph(
            url=neo4j_url,
            username=neo4j_username,
            password=neo4j_password
        )
        
        self.workflow = self._create_workflow()

    def ingest_documents(self, documents: List[Document]):
        """
        Ingest documents into Neo4j: clear DB, extract graph, create vector index.
        """
        # Clear graph database
        self.graph.query("MATCH (n) DETACH DELETE n;")

        # Define schema
        allowed_nodes = ["Patient", "Disease", "Medication", "Test", "Symptom", "Doctor"]
        allowed_relationships = [
            "HAS_DISEASE", "TAKES_MEDICATION", "UNDERWENT_TEST",
            "HAS_SYMPTOM", "TREATED_BY"
        ]

        # Transform to graph
        transformer = LLMGraphTransformer(
            llm=self.llm,
            allowed_nodes=allowed_nodes,
            allowed_relationships=allowed_relationships,
            node_properties=True, 
            relationship_properties=False
        )

        graph_documents = transformer.convert_to_graph_documents(documents)
        self.graph.add_graph_documents(graph_documents, include_source=True)

        # Create vector index
        Neo4jVector.from_existing_graph(
            embedding=self.embeddings,
            url=self.neo4j_url,
            username=self.neo4j_username,
            password=self.neo4j_password,
            database="neo4j",
            node_label="Patient",
            text_node_properties=["id", "text", "name"],
            embedding_node_property="embedding",
            index_name="vector_index",
            keyword_index_name="entity_index",
            search_type="hybrid"
        )

    def _create_workflow(self):
        """Create a LangGraph workflow for Cypher query generation and execution"""
        
        def generate_cypher(state: GraphState) -> GraphState:
            """Generate Cypher query from natural language question"""
            # Sanitize schema to remove embedding properties
            schema = state['schema']
            if isinstance(schema, str):
                # Simple string replacement to hide embedding property from LLM
                schema = schema.replace("embedding: LIST", "")
                schema = schema.replace("embedding", "")

            prompt = f"""Task: Generate a Cypher statement to query the graph database.

Instructions:
- Use only relationship types and properties provided in schema.
- Do not use other relationship types or properties that are not provided.
- Return ONLY the Cypher query, no explanations or markdown formatting.
- Do NOT return the 'embedding' property in your query.
- IMPORTANT: The 'id' property of a node usually contains its Name or primary Text. Use 'id' to filter by name (e.g. WHERE n.id = 'John').
- If a 'name' property is not in the schema, return 'id' as the name.

Schema:
{schema}

Question: {state['question']}

Cypher Query:"""
            
            try:
                response = self.llm.invoke(prompt)
                cypher_query = response.content.strip()
                
                # Clean up the query
                cypher_query = cypher_query.replace("```cypher", "").replace("```", "").strip()
                
                return {
                    **state,
                    "cypher_query": cypher_query,
                    "error": ""
                }
            except Exception as e:
                return {
                    **state,
                    "error": f"Error generating Cypher: {str(e)}"
                }
        
        def execute_cypher(state: GraphState) -> GraphState:
            """Execute the generated Cypher query"""
            if state.get("error"):
                return state
                
            try:
                result = self.graph.query(state["cypher_query"])
                
                # Format result
                if result:
                    formatted_result = self._format_graph_result(result)
                else:
                    formatted_result = "No results found."
                    
                return {
                    **state,
                    "result": formatted_result,
                    "error": ""
                }
            except Exception as e:
                return {
                    **state,
                    "error": f"Error executing Cypher: {str(e)}"
                }
        
        def format_result(state: GraphState) -> GraphState:
            """Format the final result for presentation"""
            if state.get("error"):
                return state
            
            # If result needs additional formatting with LLM
            if state.get("result") and not state.get("error"):
                prompt = f"""Given the following query result, provide a clear and concise answer to the question.

Question: {state['question']}

Query Result: {state['result']}

Answer:"""
                
                try:
                    response = self.llm.invoke(prompt)
                    return {
                        **state,
                        "result": response.content.strip()
                    }
                except Exception as e:
                    # If formatting fails, return original result
                    return state
            
            return state
        
        # Build the graph
        workflow = StateGraph(GraphState)
        
        # Add nodes
        workflow.add_node("generate_cypher", generate_cypher)
        workflow.add_node("execute_cypher", execute_cypher)
        workflow.add_node("format_result", format_result)
        
        # Add edges
        workflow.set_entry_point("generate_cypher")
        workflow.add_edge("generate_cypher", "execute_cypher")
        workflow.add_edge("execute_cypher", "format_result")
        workflow.add_edge("format_result", END)
        
        return workflow.compile()

    def _format_graph_result(self, result):
        """Format graph query results for display"""
        if not result:
            return "No results found."
        
        formatted = []
        for record in result:
            if isinstance(record, dict):
                # Filter out embeddings from result display just in case
                clean_record = {k: v for k, v in record.items() if k != 'embedding'}
                formatted.append(", ".join(f"{k}: {v}" for k, v in clean_record.items()))
            else:
                formatted.append(str(record))
        
        return "\n".join(formatted)

    def query(self, question: str):
        """Run the workflow for a question"""
        initial_state = {
            "question": question,
            "schema": self.graph.get_schema,
            "cypher_query": "",
            "result": "",
            "error": ""
        }
        return self.workflow.invoke(initial_state)
