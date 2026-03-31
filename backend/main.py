from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import os
from dotenv import load_dotenv
import logging

from app.services.rag_service import RAGService
from app.services.llm_service import LLMService
from app.services.speech_service import SpeechService
from app.models.query_models import QueryRequest, QueryResponse, VoiceQueryRequest

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Public Service Navigation Assistant",
    description="Voice-enabled AI system for public service navigation",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
rag_service = RAGService()
llm_service = LLMService()
speech_service = SpeechService()

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("Initializing Public Service Navigation Assistant...")
    await rag_service.initialize()
    await llm_service.initialize()
    await speech_service.initialize()
    logger.info("All services initialized successfully!")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Public Service Navigation Assistant API",
        "status": "healthy",
        "version": "1.0.0"
    }

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """
    Process a text query using RAG and LLM
    """
    try:
        logger.info(f"Processing query: {request.query}")
        
        # Retrieve relevant documents
        relevant_docs = await rag_service.retrieve_documents(request.query)
        
        # Generate response using LLM
        response = await llm_service.generate_response(
            query=request.query,
            context_docs=relevant_docs,
            user_context=request.user_context
        )
        
        return QueryResponse(
            response=response,
            sources=relevant_docs,
            confidence=0.95  # Placeholder confidence score
        )
    
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/voice/transcribe")
async def transcribe_audio(audio_file: UploadFile = File(...)):
    """
    Transcribe audio to text using Whisper
    """
    try:
        logger.info(f"Transcribing audio file: {audio_file.filename}")
        
        # Save uploaded file temporarily
        temp_path = f"/tmp/{audio_file.filename}"
        with open(temp_path, "wb") as buffer:
            content = await audio_file.read()
            buffer.write(content)
        
        # Transcribe audio
        transcription = await speech_service.transcribe_audio(temp_path)
        
        # Clean up temp file
        os.remove(temp_path)
        
        return {"transcription": transcription}
    
    except Exception as e:
        logger.error(f"Error transcribing audio: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/voice/synthesize")
async def synthesize_speech(request: VoiceQueryRequest):
    """
    Convert text to speech using TTS
    """
    try:
        logger.info(f"Synthesizing speech for text: {request.text[:50]}...")
        
        # Generate speech
        audio_data = await speech_service.synthesize_speech(
            text=request.text,
            voice=request.voice,
            speed=request.speed
        )
        
        return {
            "audio_data": audio_data,
            "format": "mp3",
            "duration": len(audio_data)  # Placeholder duration
        }
    
    except Exception as e:
        logger.error(f"Error synthesizing speech: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/voice/process")
async def process_voice_query(request: VoiceQueryRequest):
    """
    Complete voice processing pipeline: query → RAG → LLM → TTS
    """
    try:
        logger.info("Processing complete voice query pipeline")
        
        # Step 1: Process the query through RAG and LLM
        query_response = await process_query(QueryRequest(
            query=request.text,
            user_context=request.user_context
        ))
        
        # Step 2: Synthesize the response to speech
        audio_data = await speech_service.synthesize_speech(
            text=query_response.response,
            voice=request.voice,
            speed=request.speed
        )
        
        return {
            "text_response": query_response.response,
            "audio_data": audio_data,
            "sources": query_response.sources,
            "confidence": query_response.confidence
        }
    
    except Exception as e:
        logger.error(f"Error in voice processing pipeline: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Detailed health check for all services"""
    try:
        rag_status = await rag_service.health_check()
        llm_status = await llm_service.health_check()
        speech_status = await speech_service.health_check()
        
        return {
            "status": "healthy",
            "services": {
                "rag_service": rag_status,
                "llm_service": llm_status,
                "speech_service": speech_status
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 