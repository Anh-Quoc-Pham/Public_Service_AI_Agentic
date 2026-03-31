import os
import logging
import asyncio
import tempfile
from typing import Optional, Dict, Any
import whisper
from gtts import gTTS
from gtts.lang import tts_langs
import openai
import io
import base64

logger = logging.getLogger(__name__)

class SpeechService:
    """Service for speech recognition and synthesis"""
    
    def __init__(self):
        self.whisper_model = None
        self.is_initialized = False
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.default_language = "en"
        
    async def initialize(self):
        """Initialize the speech service"""
        try:
            logger.info("Initializing speech service...")
            
            # Initialize Whisper model for speech recognition
            try:
                self.whisper_model = whisper.load_model("base")
                logger.info("Whisper model loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load Whisper model: {str(e)}")
                self.whisper_model = None
            
            # Test OpenAI API if available
            if self.openai_api_key:
                try:
                    openai.api_key = self.openai_api_key
                    logger.info("OpenAI API configured for speech processing")
                except Exception as e:
                    logger.warning(f"Failed to configure OpenAI API: {str(e)}")
            
            self.is_initialized = True
            logger.info("Speech service initialized successfully!")
            
        except Exception as e:
            logger.error(f"Failed to initialize speech service: {str(e)}")
            raise
    
    async def transcribe_audio(self, audio_path: str) -> str:
        """
        Transcribe audio to text using Whisper or OpenAI
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Transcribed text
        """
        try:
            if not self.is_initialized:
                raise RuntimeError("Speech service not initialized")
            
            # Try OpenAI Whisper API first (better quality)
            if self.openai_api_key:
                try:
                    return await self._transcribe_with_openai(audio_path)
                except Exception as e:
                    logger.warning(f"OpenAI transcription failed: {str(e)}")
            
            # Fallback to local Whisper model
            if self.whisper_model:
                return await self._transcribe_with_whisper(audio_path)
            
            # If neither works, return a placeholder
            logger.warning("No transcription service available")
            return "I'm sorry, I couldn't understand the audio. Please try speaking more clearly."
            
        except Exception as e:
            logger.error(f"Error transcribing audio: {str(e)}")
            return "I'm sorry, there was an error processing your audio."
    
    async def _transcribe_with_openai(self, audio_path: str) -> str:
        """Transcribe using OpenAI Whisper API"""
        try:
            with open(audio_path, "rb") as audio_file:
                response = await asyncio.to_thread(
                    openai.Audio.transcribe,
                    "whisper-1",
                    audio_file,
                    language="en"
                )
            
            transcription = response.get("text", "")
            logger.info(f"OpenAI transcription successful: {transcription[:50]}...")
            return transcription
            
        except Exception as e:
            logger.error(f"OpenAI transcription error: {str(e)}")
            raise
    
    async def _transcribe_with_whisper(self, audio_path: str) -> str:
        """Transcribe using local Whisper model"""
        try:
            result = await asyncio.to_thread(
                self.whisper_model.transcribe,
                audio_path,
                language="en"
            )
            
            transcription = result.get("text", "")
            logger.info(f"Whisper transcription successful: {transcription[:50]}...")
            return transcription
            
        except Exception as e:
            logger.error(f"Whisper transcription error: {str(e)}")
            raise
    
    async def synthesize_speech(
        self, 
        text: str, 
        voice: str = "neutral",
        speed: float = 1.0,
        language: str = "en"
    ) -> bytes:
        """
        Convert text to speech using gTTS with SSML support
        
        Args:
            text: Text to synthesize
            voice: Voice type (male, female, neutral)
            speed: Speech speed multiplier
            language: Language code
            
        Returns:
            Audio data as bytes
        """
        try:
            if not self.is_initialized:
                raise RuntimeError("Speech service not initialized")
            
            # Process text with SSML if needed
            processed_text = self._process_ssml(text, voice, speed)
            
            # Generate speech using gTTS
            audio_data = await self._synthesize_with_gtts(processed_text, language)
            
            logger.info(f"Speech synthesis successful for text: {text[:50]}...")
            return audio_data
            
        except Exception as e:
            logger.error(f"Error synthesizing speech: {str(e)}")
            # Return a simple error message as audio
            return await self._synthesize_with_gtts("I'm sorry, there was an error processing your request.", language)
    
    def _process_ssml(self, text: str, voice: str, speed: float) -> str:
        """Process text with SSML markup for better speech quality"""
        
        # Simple SSML processing - in a full implementation, you'd use a proper SSML library
        # For now, we'll do basic text processing
        
        # Add pauses for better speech flow
        text = text.replace(". ", ". <break time='0.5s'/> ")
        text = text.replace("? ", "? <break time='0.5s'/> ")
        text = text.replace("! ", "! <break time='0.5s'/> ")
        text = text.replace(", ", ", <break time='0.2s'/> ")
        
        # Add emphasis to important words
        important_words = ["SNAP", "Medicaid", "Medicare", "Section 8", "housing", "benefits", "eligibility"]
        for word in important_words:
            if word.lower() in text.lower():
                text = text.replace(word, f"<emphasis>{word}</emphasis>")
        
        # Add prosody for speed control
        if speed != 1.0:
            rate = "slow" if speed < 0.8 else "fast" if speed > 1.2 else "medium"
            text = f"<prosody rate='{rate}'>{text}</prosody>"
        
        # Wrap in SSML
        ssml_text = f"""<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="en-US">
            {text}
        </speak>"""
        
        # For gTTS, we'll strip SSML and use basic text processing
        # In a production system, you'd use a proper SSML-capable TTS engine
        return text.replace("<break time='0.5s'/>", " ").replace("<break time='0.2s'/>", " ").replace("<emphasis>", "").replace("</emphasis>", "").replace("<prosody rate='slow'>", "").replace("<prosody rate='fast'>", "").replace("<prosody rate='medium'>", "").replace("</prosody>", "")
    
    async def _synthesize_with_gtts(self, text: str, language: str) -> bytes:
        """Synthesize speech using gTTS"""
        try:
            # Create gTTS object
            tts = gTTS(text=text, lang=language, slow=False)
            
            # Save to bytes buffer
            audio_buffer = io.BytesIO()
            tts.write_to_fp(audio_buffer)
            audio_buffer.seek(0)
            
            return audio_buffer.getvalue()
            
        except Exception as e:
            logger.error(f"gTTS synthesis error: {str(e)}")
            raise
    
    async def get_available_voices(self) -> Dict[str, Any]:
        """Get available voices and languages"""
        try:
            languages = tts_langs()
            
            return {
                "languages": languages,
                "default_language": self.default_language,
                "voice_types": ["male", "female", "neutral"],
                "speed_range": {"min": 0.5, "max": 2.0, "default": 1.0}
            }
            
        except Exception as e:
            logger.error(f"Error getting available voices: {str(e)}")
            return {
                "languages": {"en": "English"},
                "default_language": "en",
                "voice_types": ["neutral"],
                "speed_range": {"min": 0.5, "max": 2.0, "default": 1.0}
            }
    
    async def health_check(self) -> Dict[str, Any]:
        """Check the health of the speech service"""
        try:
            status = {
                "initialized": self.is_initialized,
                "whisper_model_loaded": self.whisper_model is not None,
                "openai_available": bool(self.openai_api_key),
                "default_language": self.default_language
            }
            
            if self.is_initialized:
                # Test speech synthesis
                try:
                    test_audio = await self.synthesize_speech("Test", "neutral", 1.0, "en")
                    status["synthesis_test_successful"] = len(test_audio) > 0
                    status["synthesis_test_size"] = len(test_audio)
                except Exception as e:
                    status["synthesis_test_successful"] = False
                    status["synthesis_test_error"] = str(e)
            
            return status
            
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return {"error": str(e)}
    
    async def transcribe_audio_file(self, audio_file: bytes, file_format: str = "wav") -> str:
        """
        Transcribe audio from file bytes
        
        Args:
            audio_file: Audio file as bytes
            file_format: Audio file format
            
        Returns:
            Transcribed text
        """
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(suffix=f".{file_format}", delete=False) as temp_file:
                temp_file.write(audio_file)
                temp_path = temp_file.name
            
            try:
                # Transcribe the temporary file
                transcription = await self.transcribe_audio(temp_path)
                return transcription
            finally:
                # Clean up temporary file
                os.unlink(temp_path)
                
        except Exception as e:
            logger.error(f"Error transcribing audio file: {str(e)}")
            return "I'm sorry, there was an error processing your audio file." 