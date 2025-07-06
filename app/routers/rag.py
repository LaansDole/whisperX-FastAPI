"""This module contains the FastAPI routes for the RAG chatbot."""

from fastapi import APIRouter

rag_router = APIRouter()

@rag_router.post("/rag/chat", tags=["RAG Chatbot"])
async def chat_with_rag():
    return {"message": "This is the RAG chatbot endpoint."}
