from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from enum import Enum

class VoiceType(str, Enum):
    """Available voice types for TTS"""
    MALE = "male"
    FEMALE = "female"
    NEUTRAL = "neutral"


class ExecutionMode(str, Enum):
    """Execution mode for query processing"""
    CLASSIC = "classic"
    AGENTIC = "agentic"

class QueryRequest(BaseModel):
    """Request model for text queries"""
    query: str = Field(..., min_length=1, description="The user's query text")
    user_context: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = Field(
        default=None, 
        description="Conversation history or additional user context"
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Session identifier for multi-turn context in agentic mode"
    )
    mode: ExecutionMode = Field(
        default=ExecutionMode.CLASSIC,
        description="Execution mode: classic pipeline or agentic orchestration"
    )
    max_steps: int = Field(
        default=4,
        ge=1,
        le=8,
        description="Maximum number of tool/planning steps in agentic mode"
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
    session_id: Optional[str] = Field(
        default=None,
        description="Session identifier used by the backend"
    )
    execution_mode: ExecutionMode = Field(
        default=ExecutionMode.CLASSIC,
        description="Execution mode used to generate this response"
    )
    agent_trace: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Optional execution trace for agentic decisions and tool usage"
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