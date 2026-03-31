from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum

class VoiceType(str, Enum):
    """Available voice types for TTS"""
    MALE = "male"
    FEMALE = "female"
    NEUTRAL = "neutral"

class QueryRequest(BaseModel):
    """Request model for text queries"""
    query: str = Field(..., description="The user's query text")
    user_context: Optional[List[Dict[str, Any]]] = Field(
        default=None, 
        description="Conversation history or additional user context"
    )

class QueryResponse(BaseModel):
    """Response model for processed queries"""
    response: str = Field(..., description="The generated response")
    sources: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Source documents used for the response"
    )
    confidence: float = Field(
        ..., 
        ge=0.0, 
        le=1.0,
        description="Confidence score of the response"
    )

class VoiceQueryRequest(BaseModel):
    """Request model for voice processing"""
    text: str = Field(..., description="Text to process or synthesize")
    voice: VoiceType = Field(
        default=VoiceType.NEUTRAL,
        description="Voice type for speech synthesis"
    )
    speed: float = Field(
        default=1.0,
        ge=0.5,
        le=2.0,
        description="Speech speed multiplier"
    )
    user_context: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional user context"
    )

class DocumentSource(BaseModel):
    """Model for document sources"""
    title: str = Field(..., description="Document title")
    content: str = Field(..., description="Relevant content excerpt")
    source_url: Optional[str] = Field(None, description="Source URL if available")
    confidence: float = Field(..., description="Relevance confidence score")

class HealthStatus(BaseModel):
    """Model for service health status"""
    service: str = Field(..., description="Service name")
    status: str = Field(..., description="Health status")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional details")

class TranscriptionResponse(BaseModel):
    """Response model for audio transcription"""
    transcription: str = Field(..., description="Transcribed text")
    confidence: Optional[float] = Field(None, description="Transcription confidence")
    language: Optional[str] = Field(None, description="Detected language")

class SynthesisResponse(BaseModel):
    """Response model for speech synthesis"""
    audio_data: bytes = Field(..., description="Audio data in bytes")
    format: str = Field(default="mp3", description="Audio format")
    duration: float = Field(..., description="Audio duration in seconds")
    word_count: int = Field(..., description="Number of words synthesized") 