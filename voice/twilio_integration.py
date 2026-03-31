import os
import logging
import requests
import json
from typing import Dict, Any, Optional
from flask import Flask, request, Response
from twilio.twiml.voice_response import VoiceResponse
from twilio.rest import Client
import asyncio
import aiohttp
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

app = Flask(__name__)

# Twilio configuration
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")

# Backend services
RASA_WEBHOOK_URL = os.getenv("RASA_WEBHOOK_URL", "http://localhost:5005/webhooks/rest/webhook")
BACKEND_API_URL = os.getenv("BACKEND_API_URL", "http://localhost:8000")

# Initialize Twilio client
twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN) if TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN else None

class TwilioVoiceHandler:
    """Handles Twilio voice call interactions"""
    
    def __init__(self):
        self.session_data = {}
        
    def handle_incoming_call(self) -> str:
        """Handle incoming voice call"""
        try:
            response = VoiceResponse()
            
            # Welcome message
            response.say(
                "Hello! Welcome to the Public Service Navigation Assistant. I can help you with SNAP benefits, housing assistance, healthcare programs, and more. Please speak after the beep.",
                voice="alice",
                language="en-US"
            )
            
            # Record user's question
            response.record(
                action="/process_audio",
                method="POST",
                maxLength="30",
                playBeep=True,
                trim="trim-silence"
            )
            
            # Fallback if no recording
            response.say(
                "I didn't hear anything. Please call back and try again.",
                voice="alice",
                language="en-US"
            )
            
            return str(response)
            
        except Exception as e:
            logger.error(f"Error handling incoming call: {str(e)}")
            return self._create_error_response("I'm sorry, there was an error processing your call. Please try again.")
    
    def handle_audio_processing(self, recording_url: str, call_sid: str) -> str:
        """Process recorded audio and generate response"""
        try:
            # Download the recording
            recording_data = self._download_recording(recording_url)
            if not recording_data:
                return self._create_error_response("I couldn't access your recording. Please try again.")
            
            # Transcribe audio
            transcription = self._transcribe_audio(recording_data)
            if not transcription:
                return self._create_error_response("I couldn't understand what you said. Please speak clearly and try again.")
            
            # Process with Rasa
            rasa_response = self._process_with_rasa(transcription, call_sid)
            if not rasa_response:
                return self._create_error_response("I'm having trouble processing your request. Please try again.")
            
            # Create TwiML response with text-to-speech
            response = VoiceResponse()
            response.say(rasa_response, voice="alice", language="en-US")
            
            # Add follow-up options
            response.say(
                "Press 1 to repeat the information, press 2 to ask another question, or press 3 to speak with a human representative.",
                voice="alice",
                language="en-US"
            )
            
            # Gather DTMF input
            gather = response.gather(
                action="/handle_dtmf",
                method="POST",
                numDigits="1",
                timeout="10"
            )
            
            # Fallback if no input
            response.say(
                "Thank you for calling. Have a great day!",
                voice="alice",
                language="en-US"
            )
            
            return str(response)
            
        except Exception as e:
            logger.error(f"Error processing audio: {str(e)}")
            return self._create_error_response("I'm sorry, there was an error processing your request. Please try again.")
    
    def handle_dtmf_input(self, digits: str, call_sid: str) -> str:
        """Handle DTMF input from user"""
        try:
            response = VoiceResponse()
            
            if digits == "1":
                # Repeat last response
                last_response = self.session_data.get(call_sid, {}).get("last_response", "")
                if last_response:
                    audio_url = self._synthesize_speech(last_response)
                    if audio_url:
                        response.play(audio_url)
                    else:
                        response.say(last_response, voice="alice", language="en-US")
                else:
                    response.say("I don't have a previous response to repeat.", voice="alice", language="en-US")
                    
            elif digits == "2":
                # Ask another question
                response.say(
                    "Please ask your next question after the beep.",
                    voice="alice",
                    language="en-US"
                )
                response.record(
                    action="/process_audio",
                    method="POST",
                    maxLength="30",
                    playBeep=True,
                    trim="trim-silence"
                )
                
            elif digits == "3":
                # Connect to human
                response.say(
                    "I'm transferring you to a human representative. Please hold.",
                    voice="alice",
                    language="en-US"
                )
                # In a real implementation, you would transfer the call here
                response.say(
                    "For immediate assistance, please call 2-1-1 or visit your local public services office.",
                    voice="alice",
                    language="en-US"
                )
                
            else:
                response.say(
                    "I didn't understand your selection. Please try again.",
                    voice="alice",
                    language="en-US"
                )
            
            return str(response)
            
        except Exception as e:
            logger.error(f"Error handling DTMF input: {str(e)}")
            return self._create_error_response("I'm sorry, there was an error processing your selection.")
    
    def _download_recording(self, recording_url: str) -> Optional[bytes]:
        """Download recording from Twilio"""
        try:
            # Use Twilio client to get the recording with proper authentication
            if twilio_client:
                # Extract recording SID from URL
                recording_sid = recording_url.split('/')[-1].split('.')[0]
                logger.info(f"Extracted recording SID: {recording_sid}")
                
                # Use the direct Twilio API to get the recording file
                # The recording URL format is: https://api.twilio.com/2010-04-01/Accounts/{AccountSid}/Recordings/{RecordingSid}.wav
                direct_url = f"https://api.twilio.com/2010-04-01/Accounts/{TWILIO_ACCOUNT_SID}/Recordings/{recording_sid}.wav"
                logger.info(f"Direct download URL: {direct_url}")
                
                # Download the recording with Basic Auth
                response = requests.get(direct_url, auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN), timeout=30)
                if response.status_code == 200:
                    logger.info(f"Successfully downloaded recording, size: {len(response.content)} bytes")
                    return response.content
                else:
                    logger.error(f"Failed to download recording: {response.status_code}")
                    logger.error(f"Response content: {response.text}")
                    return None
            else:
                logger.error("Twilio client not initialized")
                return None
                
        except Exception as e:
            logger.error(f"Error downloading recording: {str(e)}")
            return None
    
    def _transcribe_audio(self, audio_data: bytes) -> Optional[str]:
        """Transcribe audio using backend service"""
        try:
            # Send audio to backend for transcription
            files = {"audio_file": ("recording.wav", audio_data, "audio/wav")}
            response = requests.post(
                f"{BACKEND_API_URL}/voice/transcribe",
                files=files,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("transcription", "")
            else:
                logger.error(f"Transcription failed: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Error transcribing audio: {str(e)}")
            return None
    
    def _process_with_rasa(self, text: str, call_sid: str) -> Optional[str]:
        """Process text with backend RAG service"""
        try:
            # Send query to backend RAG service
            payload = {
                "query": text
            }
            
            response = requests.post(
                f"{BACKEND_API_URL}/query",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                rag_response = result.get("response", "")
                
                # Store in session for potential repetition
                if call_sid not in self.session_data:
                    self.session_data[call_sid] = {}
                self.session_data[call_sid]["last_response"] = rag_response
                
                return rag_response
            else:
                logger.error(f"Backend processing failed: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Error processing with backend: {str(e)}")
            return None
    
    def _synthesize_speech(self, text: str) -> Optional[str]:
        """Synthesize speech using backend service"""
        try:
            # Send text to backend for synthesis
            payload = {
                "text": text,
                "voice": "neutral",
                "speed": 1.0
            }
            
            response = requests.post(
                f"{BACKEND_API_URL}/voice/synthesize",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                audio_data = result.get("audio_data")
                
                if audio_data:
                    # In a real implementation, you would save this to a file server
                    # and return the URL. For now, we'll return None and use TTS
                    return None
                else:
                    logger.warning("No audio data in response")
                    return None
            else:
                logger.error(f"Speech synthesis failed: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Error synthesizing speech: {str(e)}")
            return None
    
    def _create_error_response(self, message: str) -> str:
        """Create error response"""
        response = VoiceResponse()
        response.say(message, voice="alice", language="en-US")
        response.say("Thank you for calling. Goodbye.", voice="alice", language="en-US")
        return str(response)

# Initialize handler
voice_handler = TwilioVoiceHandler()

@app.route("/", methods=["POST"])
def incoming_call():
    """Handle incoming voice calls"""
    return voice_handler.handle_incoming_call()

@app.route("/process_audio", methods=["POST"])
def process_audio():
    """Process recorded audio"""
    recording_url = request.form.get("RecordingUrl")
    call_sid = request.form.get("CallSid")
    
    # Debug logging
    logger.info(f"Received recording URL: {recording_url}")
    logger.info(f"Call SID: {call_sid}")
    
    if not recording_url:
        logger.error("No recording URL received")
        return voice_handler._create_error_response("No recording received.")
    
    return voice_handler.handle_audio_processing(recording_url, call_sid)

@app.route("/handle_dtmf", methods=["POST"])
def handle_dtmf():
    """Handle DTMF input"""
    digits = request.form.get("Digits")
    call_sid = request.form.get("CallSid")
    
    if not digits:
        return voice_handler._create_error_response("No input received.")
    
    return voice_handler.handle_dtmf_input(digits, call_sid)

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "twilio_voice_handler"}

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True) 