import os
import logging
from typing import List, Dict, Any, Optional
import asyncio
import json
from langchain.llms import Ollama
from langchain.chat_models import ChatOllama
from langchain.schema import HumanMessage, SystemMessage
from langchain.prompts import PromptTemplate
import openai

logger = logging.getLogger(__name__)

class LLMService:
    """Service for Large Language Model interactions"""
    
    def __init__(self):
        self.llm = None
        self.chat_model = None
        self.is_initialized = False
        self.model_name = os.getenv("LLM_MODEL", "mixtral-8x7b")
        self.api_base = os.getenv("LLM_API_BASE", "http://localhost:11434")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        
    async def initialize(self):
        """Initialize the LLM service"""
        try:
            logger.info(f"Initializing LLM service with model: {self.model_name}")
            
            # Try to initialize Ollama first (preferred for open-source models)
            try:
                self.chat_model = ChatOllama(
                    model=self.model_name,
                    base_url=self.api_base,
                    temperature=0.7
                )
                
                # Test the connection
                test_message = HumanMessage(content="Hello, this is a test.")
                response = await asyncio.to_thread(self.chat_model.invoke, [test_message])
                
                if response and response.content:
                    logger.info("Ollama LLM initialized successfully!")
                    self.is_initialized = True
                    return
                    
            except Exception as e:
                logger.warning(f"Failed to initialize Ollama: {str(e)}")
            
            # Fallback to OpenAI if available
            if self.openai_api_key:
                try:
                    from openai import OpenAI
                    client = OpenAI(api_key=self.openai_api_key)
                    # Test the connection with a simple request
                    test_response = await asyncio.to_thread(
                        client.chat.completions.create,
                        model="gpt-3.5-turbo",
                        messages=[{"role": "user", "content": "Hello"}],
                        max_tokens=10
                    )
                    self.chat_model = "openai"  # Use OpenAI API directly
                    logger.info("Using OpenAI as fallback LLM")
                    self.is_initialized = True
                    return
                except Exception as e:
                    logger.warning(f"Failed to initialize OpenAI: {str(e)}")
            
            # If neither works, create a mock LLM for development
            logger.warning("No LLM available, using mock responses for development")
            self.chat_model = "mock"
            self.is_initialized = True
            
        except Exception as e:
            logger.error(f"Failed to initialize LLM service: {str(e)}")
            raise
    
    async def generate_response(
        self, 
        query: str, 
        context_docs: List[Dict[str, Any]], 
        user_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate a response using the LLM with retrieved context
        
        Args:
            query: The user's query
            context_docs: Retrieved relevant documents
            user_context: Additional user context
            
        Returns:
            Generated response text
        """
        try:
            if not self.is_initialized:
                raise RuntimeError("LLM service not initialized")
            
            # Create the prompt with context
            prompt = self._create_prompt(query, context_docs, user_context)
            
            if self.chat_model == "mock":
                return await self._generate_mock_response(query, context_docs)
            elif self.chat_model == "openai":
                return await self._generate_openai_response(prompt)
            else:
                return await self._generate_ollama_response(prompt)
                
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return self._generate_fallback_response(query)
    
    def _create_prompt(
        self, 
        query: str, 
        context_docs: List[Dict[str, Any]], 
        user_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a prompt with context for the LLM"""
        
        # Build context from retrieved documents
        context_text = ""
        if context_docs:
            context_text = "Relevant information:\n\n"
            for i, doc in enumerate(context_docs[:3], 1):  # Use top 3 documents
                context_text += f"{i}. {doc.get('content', '')[:500]}...\n\n"
        
        # Build user context
        user_context_text = ""
        if user_context:
            user_context_text = f"User context: {json.dumps(user_context)}\n\n"
        
        # Create the full prompt
        prompt = f"""You are a helpful public service navigation assistant. Your role is to help users understand and access government benefits and services like SNAP, housing assistance, and healthcare programs.

{user_context_text}{context_text}

User Question: {query}

Please provide a clear, helpful, and accurate response based on the information provided. If the information is not sufficient to answer the question completely, acknowledge what you can help with and suggest where they might find more information.

Response:"""
        
        return prompt
    
    async def _generate_ollama_response(self, prompt: str) -> str:
        """Generate response using Ollama"""
        try:
            messages = [
                SystemMessage(content="You are a helpful public service navigation assistant."),
                HumanMessage(content=prompt)
            ]
            
            response = await asyncio.to_thread(self.chat_model.invoke, messages)
            return response.content if response else "I'm sorry, I couldn't generate a response at this time."
            
        except Exception as e:
            logger.error(f"Error generating Ollama response: {str(e)}")
            raise
    
    async def _generate_openai_response(self, prompt: str) -> str:
        """Generate response using OpenAI API"""
        try:
            from openai import OpenAI
            
            client = OpenAI(api_key=self.openai_api_key)
            response = await asyncio.to_thread(
                client.chat.completions.create,
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful public service navigation assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.7
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error generating OpenAI response: {str(e)}")
            raise
    
    async def _generate_mock_response(self, query: str, context_docs: List[Dict[str, Any]]) -> str:
        """Generate a mock response for development/testing"""
        
        # Simple keyword-based responses for development
        query_lower = query.lower()
        
        if "snap" in query_lower or "food" in query_lower or "nutrition" in query_lower:
            return """Based on the information I have, SNAP (Supplemental Nutrition Assistance Program) provides nutrition benefits to help families purchase healthy food. 

To apply for SNAP:
1. Contact your local SNAP office
2. Complete an application form
3. Provide required documentation including proof of income
4. Attend an interview
5. You'll receive a decision within 30 days

Eligibility is based on household income (must be at or below 130% of the federal poverty level) and other factors. Benefits are provided on an EBT card that works like a debit card at authorized retailers.

Would you like me to help you find your local SNAP office or provide more specific information about eligibility requirements?"""
        
        elif "housing" in query_lower or "section 8" in query_lower or "rent" in query_lower:
            return """I can help you with housing assistance programs. There are several options available:

Section 8 Housing Choice Voucher Program:
- Helps low-income families afford decent housing
- You pay 30% of your income toward rent
- The government pays the difference to your landlord

Public Housing:
- Government-owned housing units for low-income families
- Rent is based on your income (usually 30% of adjusted gross income)

To apply for housing assistance:
1. Contact your local Public Housing Authority (PHA)
2. Complete an application with required documentation
3. You'll be placed on a waiting list
4. Attend an interview when contacted

Would you like help finding your local PHA or learning more about specific programs?"""
        
        elif "health" in query_lower or "medicaid" in query_lower or "medicare" in query_lower:
            return """I can help you understand healthcare benefits and programs:

Medicaid:
- Provides health coverage to low-income individuals and families
- Covers doctor visits, hospital stays, prescription drugs, and more
- Eligibility varies by state and income level

Medicare:
- Federal health insurance for people 65 and older
- Also covers some younger people with disabilities
- Includes Part A (hospital insurance) and Part B (medical insurance)

Affordable Care Act (ACA) Marketplace:
- Health insurance marketplace for individuals and families
- Subsidies available based on income
- Open enrollment typically November-December

To apply for healthcare benefits:
1. Visit Healthcare.gov or your state's marketplace
2. Complete an application with income and household information
3. Compare plans and select coverage
4. Enroll in your chosen plan

Would you like help finding specific information about any of these programs?"""
        
        else:
            return """I'm here to help you navigate public services and government benefits. I can provide information about:

- SNAP (food assistance) benefits
- Housing assistance programs like Section 8
- Healthcare benefits including Medicaid and Medicare
- General navigation help for finding local offices and required documents

What specific program or service would you like to learn more about? I can help you understand eligibility requirements, application processes, and where to get started."""
    
    def _generate_fallback_response(self, query: str) -> str:
        """Generate a fallback response when LLM is unavailable"""
        return f"""I'm sorry, I'm having trouble processing your request right now. You asked about: "{query}"

For immediate help with public services, you can:
- Call 2-1-1 for information and referrals
- Visit Benefits.gov to find programs
- Contact your local Department of Human Services
- Visit your local library or community center for assistance

Please try again in a moment, or contact one of these resources for immediate help."""
    
    async def health_check(self) -> Dict[str, Any]:
        """Check the health of the LLM service"""
        try:
            status = {
                "initialized": self.is_initialized,
                "model_name": self.model_name,
                "api_base": self.api_base,
                "provider": "ollama" if self.chat_model and self.chat_model != "openai" and self.chat_model != "mock" else str(self.chat_model)
            }
            
            if self.is_initialized:
                # Test a simple query
                try:
                    test_response = await self.generate_response(
                        "Hello, this is a test.",
                        [],
                        None
                    )
                    status["test_query_successful"] = bool(test_response)
                    status["response_length"] = len(test_response)
                except Exception as e:
                    status["test_query_successful"] = False
                    status["test_error"] = str(e)
            
            return status
            
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return {"error": str(e)} 