from typing import List, Dict, Optional
from config.settings import settings
from app.rag.embeddings import get_embedding_model
from app.rag.vector_store import get_vector_store
from app.models.schemas import SourceDocument, QueryResponse
from app.utils.logger import get_logger
import anthropic
import openai

logger = get_logger(__name__)


class QueryEngine:
    def __init__(self):
        self.embedding_model = get_embedding_model()
        self.vector_store = get_vector_store()
        self.llm_provider = settings.LLM_PROVIDER
        
        # Initialize LLM client
        if self.llm_provider == "anthropic":
            self.llm_client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
        elif self.llm_provider == "openai":
            openai.api_key = settings.OPENAI_API_KEY
            self.llm_client = openai
        
        logger.info(f"QueryEngine initialized with {self.llm_provider}")
    
    def retrieve_documents(self, question: str, top_k: int = None) -> List[SourceDocument]:
        """Retrieve relevant documents from vector store"""
        if top_k is None:
            top_k = settings.TOP_K_RESULTS
        
        # Generate embedding for question
        query_embedding = self.embedding_model.encode_single(question)
        
        # Query vector store
        results = self.vector_store.query(
            query_embedding=query_embedding.tolist(),
            top_k=top_k
        )
        
        # Parse results
        source_docs = []
        for i, doc in enumerate(results['documents'][0]):
            distance = results['distances'][0][i]
            similarity = 1 - distance  # Convert distance to similarity
            
            # Filter by threshold
            if similarity >= settings.SIMILARITY_THRESHOLD:
                source_docs.append(SourceDocument(
                    content=doc,
                    metadata=results['metadatas'][0][i] if results['metadatas'][0] else {},
                    similarity_score=round(similarity, 3)
                ))
        
        logger.info(f"Retrieved {len(source_docs)} relevant documents")
        return source_docs
    
    def generate_answer(self, question: str, context_docs: List[SourceDocument]) -> str:
        """Generate answer using LLM"""
        if not context_docs:
            return "I don't have enough information to answer that question. Please contact our clinic directly for assistance."
        
        # Build context
        context = "\n\n".join([
            f"Document {i+1} (Relevance: {doc.similarity_score}):\n{doc.content}"
            for i, doc in enumerate(context_docs)
        ])
        
        # Build prompt
        prompt = f"""You are a helpful medical clinic assistant for {settings.CLINIC_NAME}.

Context from our knowledge base:
{context}

User question: {question}

Instructions:
- Provide a helpful, accurate answer based ONLY on the context provided above
- Be friendly, professional, and empathetic
- If the context doesn't contain the answer, politely say you don't know and suggest contacting the clinic
- Keep the response concise but complete
- Use natural, conversational language

Answer:"""
        
        try:
            if self.llm_provider == "anthropic":
                response = self.llm_client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=500,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                answer = response.content[0].text
            
            elif self.llm_provider == "openai":
                response = self.llm_client.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": f"You are a helpful medical clinic assistant for {settings.CLINIC_NAME}."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=500,
                    temperature=0.7
                )
                answer = response.choices[0].message.content
            
            logger.info("Generated answer successfully")
            return answer.strip()
        
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return f"I apologize, but I'm having trouble generating a response right now. Please call us at {settings.CLINIC_PHONE} for assistance."
    
    def query(self, question: str) -> QueryResponse:
        """Main query method - retrieve and generate"""
        logger.info(f"Processing query: {question}")
        
        # Retrieve relevant documents
        source_docs = self.retrieve_documents(question)
        
        # Generate answer
        answer = self.generate_answer(question, source_docs)
        
        # Calculate confidence (average of similarity scores)
        confidence = sum(doc.similarity_score for doc in source_docs) / len(source_docs) if source_docs else 0.0
        
        return QueryResponse(
            answer=answer,
            sources=source_docs,
            confidence=round(confidence, 3),
            question=question
        )


# Singleton instance
query_engine = None


def get_query_engine() -> QueryEngine:
    """Get or create query engine instance"""
    global query_engine
    if query_engine is None:
        query_engine = QueryEngine()
    return query_engine