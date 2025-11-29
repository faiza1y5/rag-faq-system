from typing import List, Dict, Optional
import requests
from config.settings import settings
from app.rag.embeddings import get_embedding_model
from app.rag.vector_store import get_vector_store
from app.models.schemas import SourceDocument, QueryResponse
from app.utils.logger import get_logger

logger = get_logger(__name__)


class QueryEngine:
    def __init__(self):
        self.embedding_model = get_embedding_model()
        self.vector_store = get_vector_store()
        self.llm_provider = settings.LLM_PROVIDER
        
        # Initialize LLM client based on provider
        if self.llm_provider == "anthropic":
            try:
                import anthropic
                self.llm_client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
                logger.info("Initialized Anthropic Claude client")
            except ImportError:
                logger.error("Anthropic package not installed. Run: pip install anthropic")
                raise
            except Exception as e:
                logger.error(f"Failed to initialize Anthropic client: {e}")
                raise
        
        elif self.llm_provider == "openai":
            try:
                import openai
                openai.api_key = settings.OPENAI_API_KEY
                self.llm_client = openai
                logger.info("Initialized OpenAI client")
            except ImportError:
                logger.error("OpenAI package not installed. Run: pip install openai")
                raise
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {e}")
                raise
        
        elif self.llm_provider == "ollama":
            self.ollama_url = settings.OLLAMA_URL
            self.ollama_model = settings.OLLAMA_MODEL
            # Test Ollama connection
            try:
                response = requests.get(f"{self.ollama_url.replace('/api/generate', '')}/api/tags", timeout=5)
                if response.status_code == 200:
                    models = response.json().get('models', [])
                    model_names = [m['name'] for m in models]
                    if self.ollama_model not in model_names:
                        logger.warning(f"Model {self.ollama_model} not found. Available: {model_names}")
                        logger.warning(f"Run: ollama pull {self.ollama_model}")
                    else:
                        logger.info(f"Initialized Ollama client with model: {self.ollama_model}")
                else:
                    logger.warning("Ollama server not responding. Make sure Ollama is running.")
            except requests.exceptions.RequestException as e:
                logger.warning(f"Cannot connect to Ollama. Make sure it's running: {e}")
                logger.warning("Download from: https://ollama.com/download")
        
        else:
            logger.error(f"Unknown LLM provider: {self.llm_provider}")
            raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")
        
        logger.info(f"QueryEngine initialized with provider: {self.llm_provider}")
    
    def retrieve_documents(self, question: str, top_k: int = None) -> List[SourceDocument]:
        """Retrieve relevant documents from vector store"""
        if top_k is None:
            top_k = settings.TOP_K_RESULTS
        
        try:
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
            
            logger.info(f"Retrieved {len(source_docs)} relevant documents (threshold: {settings.SIMILARITY_THRESHOLD})")
            return source_docs
        
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return []
    
    def generate_answer(self, question: str, context_docs: List[SourceDocument]) -> str:
        """Generate answer using LLM"""
        if not context_docs:
            return (
                f"I don't have enough information to answer that question. "
                f"Please contact our clinic directly at {settings.CLINIC_PHONE} for assistance."
            )
        
        # Build context
        context = "\n\n".join([
            f"Document {i+1} (Relevance: {doc.similarity_score}):\n{doc.content}"
            for i, doc in enumerate(context_docs)
        ])
        
        # Build prompt
        system_message = f"You are a helpful medical clinic assistant for {settings.CLINIC_NAME}."
        
        user_prompt = f"""Context from our knowledge base:
{context}

User question: {question}

Instructions:
- Provide a helpful, accurate answer based ONLY on the context provided above
- Be friendly, professional, and empathetic
- If the context doesn't contain the answer, politely say you don't know and suggest contacting the clinic
- Keep the response concise but complete (2-4 sentences)
- Use natural, conversational language

Answer:"""
        
        try:
            # Generate answer based on provider
            if self.llm_provider == "anthropic":
                answer = self._generate_anthropic(user_prompt)
            
            elif self.llm_provider == "openai":
                answer = self._generate_openai(system_message, user_prompt)
            
            elif self.llm_provider == "ollama":
                answer = self._generate_ollama(system_message, user_prompt)
            
            else:
                answer = "Error: Unknown LLM provider"
            
            logger.info("Generated answer successfully")
            return answer.strip()
        
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return (
                f"I apologize, but I'm having trouble generating a response right now. "
                f"Please call us at {settings.CLINIC_PHONE} for assistance."
            )
    
    def _generate_anthropic(self, prompt: str) -> str:
        """Generate answer using Anthropic Claude"""
        try:
            response = self.llm_client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=500,
                temperature=0.7,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            raise
    
    def _generate_openai(self, system_message: str, user_prompt: str) -> str:
        """Generate answer using OpenAI"""
        try:
            response = self.llm_client.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=500,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise
    
    def _generate_ollama(self, system_message: str, user_prompt: str) -> str:
        """Generate answer using Ollama (local LLM)"""
        try:
            # Combine system message and user prompt
            full_prompt = f"{system_message}\n\n{user_prompt}"
            
            # Call Ollama API
            response = requests.post(
                self.ollama_url,
                json={
                    "model": self.ollama_model,
                    "prompt": full_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "num_predict": 500,
                    }
                },
                timeout=60  # Ollama can be slower
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', 'No response generated')
            else:
                logger.error(f"Ollama API error: {response.status_code} - {response.text}")
                raise Exception(f"Ollama API returned status {response.status_code}")
        
        except requests.exceptions.Timeout:
            logger.error("Ollama request timed out (60s)")
            raise Exception("Ollama took too long to respond. Try a smaller model.")
        
        except requests.exceptions.ConnectionError:
            logger.error("Cannot connect to Ollama. Is it running?")
            raise Exception(
                "Cannot connect to Ollama. Make sure Ollama is running. "
                "Start it with: ollama serve"
            )
        
        except Exception as e:
            logger.error(f"Ollama error: {e}")
            raise
    
    def query(self, question: str) -> QueryResponse:
        """Main query method - retrieve and generate"""
        logger.info(f"Processing query: {question}")
        
        # Retrieve relevant documents
        source_docs = self.retrieve_documents(question)
        
        # Generate answer
        answer = self.generate_answer(question, source_docs)
        
        # Calculate confidence (average of similarity scores)
        confidence = (
            sum(doc.similarity_score for doc in source_docs) / len(source_docs) 
            if source_docs else 0.0
        )
        
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