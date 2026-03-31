import logging
import requests
import json
from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet
import os
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

class ActionFallbackToRAG(Action):
    """Custom action to fallback to RAG system for detailed responses"""
    
    def name(self) -> Text:
        return "action_fallback_to_rag"
    
    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        try:
            # Get the user's message
            user_message = tracker.latest_message.get('text', '')
            
            if not user_message:
                dispatcher.utter_message(text="I'm sorry, I didn't catch that. Could you please repeat your question?")
                return []
            
            # Prepare user context from slots
            user_context = {}
            if tracker.get_slot('program'):
                user_context['program'] = tracker.get_slot('program')
            if tracker.get_slot('location'):
                user_context['location'] = tracker.get_slot('location')
            if tracker.get_slot('income_level'):
                user_context['income_level'] = tracker.get_slot('income_level')
            if tracker.get_slot('family_size'):
                user_context['family_size'] = tracker.get_slot('family_size')
            
            # Call the RAG backend
            backend_url = os.getenv("RAG_BACKEND_URL", "http://localhost:8000")
            
            payload = {
                "query": user_message,
                "user_context": user_context
            }
            
            response = requests.post(
                f"{backend_url}/query",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                response_text = result.get('response', '')
                
                if response_text:
                    dispatcher.utter_message(text=response_text)
                else:
                    dispatcher.utter_message(text="I'm sorry, I couldn't find specific information about that. Let me help you with general information about public services.")
            else:
                logger.error(f"RAG backend error: {response.status_code}")
                dispatcher.utter_message(text="I'm having trouble accessing detailed information right now. Let me provide you with general guidance about public services.")
                
        except Exception as e:
            logger.error(f"Error in action_fallback_to_rag: {str(e)}")
            dispatcher.utter_message(text="I'm sorry, I'm experiencing technical difficulties. Please try asking your question again or contact your local public services office for assistance.")
        
        return []

class ActionProvideDetailedInfo(Action):
    """Custom action to provide detailed information about specific programs"""
    
    def name(self) -> Text:
        return "action_provide_detailed_info"
    
    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        try:
            # Get the program from slots
            program = tracker.get_slot('program')
            
            if not program:
                dispatcher.utter_message(text="Which program would you like detailed information about? I can help with SNAP, housing assistance, or healthcare benefits.")
                return []
            
            # Create a specific query for the program
            query = f"Tell me detailed information about {program} including eligibility, application process, and benefits."
            
            # Call the RAG backend
            backend_url = os.getenv("RAG_BACKEND_URL", "http://localhost:8000")
            
            payload = {
                "query": query,
                "user_context": {"program": program}
            }
            
            response = requests.post(
                f"{backend_url}/query",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                response_text = result.get('response', '')
                
                if response_text:
                    dispatcher.utter_message(text=response_text)
                else:
                    dispatcher.utter_message(text=f"I don't have detailed information about {program} at the moment. Please contact your local office for specific details.")
            else:
                logger.error(f"RAG backend error: {response.status_code}")
                dispatcher.utter_message(text="I'm having trouble accessing detailed information right now. Please contact your local public services office.")
                
        except Exception as e:
            logger.error(f"Error in action_provide_detailed_info: {str(e)}")
            dispatcher.utter_message(text="I'm sorry, I'm experiencing technical difficulties. Please try again or contact your local office for assistance.")
        
        return []

class ActionConnectToHuman(Action):
    """Custom action to connect user to human assistance"""
    
    def name(self) -> Text:
        return "action_connect_to_human"
    
    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        try:
            # Get user context for transfer
            user_context = {
                "conversation_summary": self._get_conversation_summary(tracker),
                "user_question": tracker.latest_message.get('text', ''),
                "slots": {
                    "program": tracker.get_slot('program'),
                    "location": tracker.get_slot('location'),
                    "income_level": tracker.get_slot('income_level'),
                    "family_size": tracker.get_slot('family_size')
                }
            }
            
            # Log the transfer request
            logger.info(f"Transferring to human agent. Context: {json.dumps(user_context)}")
            
            # Provide transfer message
            transfer_message = """I understand you'd like to speak with a human representative. 

For immediate assistance, you can:
- Call 2-1-1 for information and referrals
- Contact your local Department of Human Services
- Visit your local public services office

Your conversation summary has been saved to help the next representative assist you more efficiently.

Thank you for using our service!"""
            
            dispatcher.utter_message(text=transfer_message)
            
        except Exception as e:
            logger.error(f"Error in action_connect_to_human: {str(e)}")
            dispatcher.utter_message(text="I'm sorry, I'm having trouble processing your request. Please call 2-1-1 for immediate assistance.")
        
        return []
    
    def _get_conversation_summary(self, tracker: Tracker) -> str:
        """Generate a summary of the conversation for human transfer"""
        try:
            messages = []
            for event in tracker.events:
                if event.get('event') == 'user':
                    messages.append(f"User: {event.get('text', '')}")
                elif event.get('event') == 'bot':
                    messages.append(f"Bot: {event.get('text', '')}")
            
            # Take last 10 messages for summary
            recent_messages = messages[-10:] if len(messages) > 10 else messages
            return "\n".join(recent_messages)
            
        except Exception as e:
            logger.error(f"Error generating conversation summary: {str(e)}")
            return "Unable to generate conversation summary"

class ActionSetUserContext(Action):
    """Custom action to set user context based on conversation"""
    
    def name(self) -> Text:
        return "action_set_user_context"
    
    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        try:
            # Extract entities and set context
            entities = tracker.latest_message.get('entities', [])
            
            slots_to_set = []
            
            for entity in entities:
                entity_type = entity.get('entity')
                entity_value = entity.get('value')
                
                if entity_type == 'program':
                    slots_to_set.append(SlotSet('program', entity_value))
                elif entity_type == 'location':
                    slots_to_set.append(SlotSet('location', entity_value))
                elif entity_type == 'income_level':
                    slots_to_set.append(SlotSet('income_level', entity_value))
                elif entity_type == 'family_size':
                    slots_to_set.append(SlotSet('family_size', entity_value))
            
            return slots_to_set
            
        except Exception as e:
            logger.error(f"Error in action_set_user_context: {str(e)}")
            return [] 