# Plan: Advanced Agentic RAG with LangGraph and Graphiti

This document outlines a detailed plan to implement a state-of-the-art Agentic RAG system. This system will leverage **LangGraph** for sophisticated control flow and **Graphiti** with a **Neo4j** backend for dynamic, temporally-aware knowledge graph retrieval.

## 1. System Architecture

The new architecture will be a cyclical graph managed by LangGraph. The agent will sit at the center, routing queries to different tools, including our new Graphiti-powered knowledge graph.

### High-Level Architecture

```
                                 ┌──────────────────────────────────┐
                                 │          LangGraph State         │
                                 │ (query, context, agent_outcome)  │
                                 └─────────────────┬────────────────┘
                                                   │
                                                   ▼
                                 ┌──────────────────────────────────┐
                                 │          Agent (Router)          │
                                 │ (Decides which tool to use next) │
                                 └─────────────────┬────────────────┘
                                                   │
                         ┌─────────────┬─────────────┬─────────────┐
                         │             │             │             │
                         ▼             ▼             ▼             ▼
             ┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
             │   Vector Search  │ │  Knowledge Graph │ │   Web Search   │
             │      (Tool 1)    │ │  (Graphiti)      │ │    (Tavily)    │
             └──────────────────┘ └──────────────────┘ └──────────────────┘
                         ▲             ▲             ▲             ▲
                         │             │             │             │
                         └─────────────┼─────────────┼─────────────┘
                                       │             │
                                       ▼             ▼
                                 ┌──────────────────────────────────┐
                                 │      Update LangGraph State      │
                                 └─────────────────┬────────────────┘
                                                   │
                                                   ▼
                                 ┌──────────────────────────────────┐
                                 │    Is the answer complete? (Y/N) │
                                 └─────────────────┬────────────────┘
                                                   │
                                     ┌─────────────┴─────────────┐
                                     │ (No)                        │ (Yes)
                                     ▼                             ▼
                           (Return to Agent)           ┌──────────────────┐
                                                       │  Generate Final  │
                                                       │      Answer      │
                                                       └──────────────────┘
```

## 2. Detailed Implementation Steps

### Step 1: Update Dependencies and Docker Compose

*   **`requirements/prod.txt`:**
    *   Add `langgraph`, `graphiti-core`, `neo4j`, and `tavily-python`.
*   **`docker-compose.yml`:**
    *   Add a service for **Neo4j**.
    *   Configure a persistent volume for the Neo4j data.
    *   Ensure the `whisperx-service` depends on both `chroma` and `neo4j`.

### Step 2: Knowledge Graph and Vector Store Population

*   **Create `scripts/populate_datastores.py`:**
    *   This script will handle the offline data ingestion for both the vector store and the knowledge graph.
    *   **Vector Store:** It will load documents, split them into chunks, and embed them into the ChromaDB vector store.
    *   **Knowledge Graph (Graphiti):** It will use `Graphiti` to extract entities and relationships from the documents and populate the Neo4j database. `Graphiti` will handle the complexity of entity extraction and relationship linking.

### Step 3: Implement the LangGraph Agent

*   **Create `app/langgraph_agent.py`:**
    *   **Define the Graph State:** A `State` class to manage the flow of information.
    *   **Define the Tools:**
        *   **Vector Search Tool:** To query the ChromaDB vector store.
        *   **Graphiti Knowledge Graph Tool:** To query the Neo4j database via `Graphiti`.
        *   **Tavily Web Search Tool:** For real-time information.
    *   **Define the Agent Node:** A router that decides which tool to use based on the query and the current state.
    *   **Define the Tool Nodes:** Nodes to execute the tools and return the results.
    *   **Define the Edges:** Conditional edges to control the flow of the graph.
    *   **Compile the Graph:** Create a runnable LangGraph application.

### Step 4: Update the RAG Router and Services

*   **`app/rag_services.py`:**
    *   This file will now create and return the compiled LangGraph application.
*   **`app/routers/rag.py`:**
    *   The `/rag/chat` endpoint will invoke the LangGraph agent.
    *   A new endpoint, `/rag/populate-datastores`, will trigger the `populate_datastores.py` script as a background task.

### Step 5: Implement the Full Speech-To-Text-To-RAG-ChatBot Pipeline

*   **Create `app/routers/chatbot.py`:**
    *   This will house the new endpoint for the full pipeline.
*   **Create the `/chatbot/speech-to-text` endpoint:**
    *   Accepts an audio file.
    *   Uses `whisperx` to transcribe the audio.
    *   Passes the transcribed text to the LangGraph agent.
    *   Returns the final answer.

## 3. File-by-File Update Summary

*   **`requirements/prod.txt`:** Add `langgraph`, `graphiti-core`, `neo4j`, `tavily-python`.
*   **`docker-compose.yml`:** Add a `neo4j` service.
*   **`scripts/populate_datastores.py` (New File):** To populate both ChromaDB and Neo4j.
*   **`app/langgraph_agent.py` (New File):** To define the LangGraph agent.
*   **`app/rag_services.py`:** To create the LangGraph application.
*   **`app/routers/rag.py`:** To expose the LangGraph agent and the data population endpoint.
*   **`app/routers/chatbot.py` (New File):** For the end-to-end speech-to-text-to-RAG pipeline.

This updated plan provides a clear and comprehensive roadmap for building a truly state-of-the-art Agentic RAG system.

## 4. References

*   **Graphiti:** [https://github.com/getzep/graphiti](https://github.com/getzep/graphiti)
*   **Ottomator Agents (Agentic RAG with Knowledge Graph):** [https://github.com/coleam00/ottomator-agents/tree/main/agentic-rag-knowledge-graph](https://github.com/coleam00/ottomator-agents/tree/main/agentic-rag-knowledge-graph)
*   **LangGraph:** [https://python.langchain.com/docs/langgraph](https://python.langchain.com/docs/langgraph)
*   **Neo4j:** [https://neo4j.com/](https://neo4j.com/)
