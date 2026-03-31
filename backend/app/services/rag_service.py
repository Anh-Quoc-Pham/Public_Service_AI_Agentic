import os
import logging
from typing import List, Dict, Any, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS, Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.schema import Document
import asyncio
from pathlib import Path

logger = logging.getLogger(__name__)

class RAGService:
    """Service for Retrieval-Augmented Generation using document search"""
    
    def __init__(self):
        self.vectorstore = None
        self.embeddings = None
        self.text_splitter = None
        self.documents = []
        self.is_initialized = False
        
    async def initialize(self):
        """Initialize the RAG service with embeddings and vector store"""
        try:
            logger.info("Initializing RAG service...")
            
            # Initialize embeddings model - using a simpler approach for now
            try:
                self.embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2",
                    model_kwargs={'device': 'cpu'}
                )
            except ImportError:
                # Fallback to a basic approach without sentence-transformers
                logger.warning("sentence-transformers not available, using basic text processing")
                self.embeddings = None
            
            # Initialize text splitter
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
            
            # Load and process documents
            await self._load_documents()
            
            # Create vector store
            await self._create_vectorstore()
            
            self.is_initialized = True
            logger.info("RAG service initialized successfully!")
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG service: {str(e)}")
            raise
    
    async def _load_documents(self):
        """Load documents from the data directory"""
        try:
            data_dir = Path(__file__).parent.parent.parent / "data"
            
            if not data_dir.exists():
                logger.warning("Data directory not found, creating sample documents...")
                await self._create_sample_documents()
                return
            
            # Load documents from various formats
            loaders = [
                DirectoryLoader(str(data_dir), glob="**/*.txt", loader_cls=TextLoader),
                DirectoryLoader(str(data_dir), glob="**/*.md", loader_cls=TextLoader),
            ]
            
            for loader in loaders:
                try:
                    docs = loader.load()
                    self.documents.extend(docs)
                except Exception as e:
                    logger.warning(f"Failed to load documents with loader {loader}: {e}")
            
            logger.info(f"Loaded {len(self.documents)} documents")
            
        except Exception as e:
            logger.error(f"Error loading documents: {str(e)}")
            raise
    
    async def _create_sample_documents(self):
        """Create sample documents for public service information"""
        sample_docs = [
            Document(
                page_content="""
                SNAP Benefits (Supplemental Nutrition Assistance Program)
                
                SNAP provides nutrition benefits to supplement the food budget of needy families so they can purchase healthy food and move towards self-sufficiency.
                
                Eligibility Requirements:
                - Household income must be at or below 130% of the federal poverty level
                - Must be a U.S. citizen or legal resident
                - Must meet work requirements (unless exempt)
                
                Application Process:
                1. Contact your local SNAP office
                2. Complete an application form
                3. Provide required documentation
                4. Attend an interview
                5. Receive decision within 30 days
                
                Benefits are provided on an EBT card that works like a debit card at authorized retailers.
                """,
                metadata={"source": "snap_benefits.txt", "category": "nutrition", "title": "SNAP Benefits Guide"}
            ),
            Document(
                page_content="""
                Housing Assistance Programs
                
                Section 8 Housing Choice Voucher Program:
                - Helps low-income families afford decent, safe, and sanitary housing
                - Participants pay 30% of their income toward rent
                - Government pays the difference to the landlord
                
                Public Housing:
                - Government-owned housing units for low-income families
                - Rent is based on income (usually 30% of adjusted gross income)
                - Managed by local Public Housing Authorities (PHAs)
                
                Emergency Shelter:
                - Temporary housing for homeless individuals and families
                - Available through local shelters and organizations
                - Often requires referral from social services
                
                Application Process:
                1. Contact your local Public Housing Authority
                2. Complete application with required documentation
                3. Wait for placement on waiting list
                4. Attend interview when contacted
                """,
                metadata={"source": "housing_assistance.txt", "category": "housing", "title": "Housing Assistance Programs"}
            ),
            Document(
                page_content="""
                Healthcare Benefits and Programs
                
                Medicaid:
                - Provides health coverage to low-income individuals and families
                - Covers doctor visits, hospital stays, prescription drugs, and more
                - Eligibility varies by state and income level
                
                Medicare:
                - Federal health insurance for people 65 and older
                - Also covers some younger people with disabilities
                - Part A (hospital insurance) and Part B (medical insurance)
                
                Affordable Care Act (ACA) Marketplace:
                - Health insurance marketplace for individuals and families
                - Subsidies available based on income
                - Open enrollment period typically November-December
                
                Children's Health Insurance Program (CHIP):
                - Provides health coverage to children in families that earn too much for Medicaid
                - Low-cost health coverage for children
                
                Application Process:
                1. Visit Healthcare.gov or your state's marketplace
                2. Complete application with income and household information
                3. Compare plans and select coverage
                4. Enroll in chosen plan
                """,
                metadata={"source": "healthcare_benefits.txt", "category": "healthcare", "title": "Healthcare Benefits Guide"}
            ),
            Document(
                page_content="""
                General Public Service Navigation
                
                Finding Local Offices:
                - Use Benefits.gov to find programs and local offices
                - Contact your state's Department of Human Services
                - Visit local community centers and libraries
                
                Required Documents:
                - Government-issued photo ID
                - Social Security cards for all household members
                - Proof of income (pay stubs, tax returns)
                - Proof of expenses (rent receipts, utility bills)
                - Birth certificates for children
                
                Getting Help:
                - Call 2-1-1 for information and referrals
                - Visit local social services offices
                - Contact nonprofit organizations in your area
                - Use online resources like Benefits.gov
                
                Tips for Success:
                - Keep copies of all documents
                - Follow up on applications
                - Ask questions if you don't understand
                - Appeal decisions if you disagree
                """,
                metadata={"source": "general_navigation.txt", "category": "general", "title": "Public Service Navigation Guide"}
            )
        ]
        
        self.documents = sample_docs
        logger.info(f"Created {len(self.documents)} sample documents")
    
    async def _create_vectorstore(self):
        """Create and populate the vector store"""
        try:
            if not self.documents:
                logger.warning("No documents to process")
                return
            
            # Split documents into chunks
            texts = self.text_splitter.split_documents(self.documents)
            logger.info(f"Split documents into {len(texts)} chunks")
            
            # Create vector store (using FAISS for better performance)
            if self.embeddings:
                self.vectorstore = FAISS.from_documents(texts, self.embeddings)
            else:
                # Fallback: store documents without embeddings for now
                logger.warning("Using basic document storage without embeddings")
                self.vectorstore = None
            
            # Save the vector store for future use
            if self.vectorstore:
                vectorstore_path = Path(__file__).parent.parent.parent / "vectorstore"
                vectorstore_path.mkdir(exist_ok=True)
                self.vectorstore.save_local(str(vectorstore_path))
            
            logger.info("Vector store created and saved successfully")
            
        except Exception as e:
            logger.error(f"Error creating vector store: {str(e)}")
            raise
    
    async def retrieve_documents(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a given query
        
        Args:
            query: The search query
            k: Number of documents to retrieve
            
        Returns:
            List of relevant documents with metadata
        """
        try:
            if not self.is_initialized:
                raise RuntimeError("RAG service not initialized")
            
            if self.vectorstore:
                # Use vector store for semantic search
                docs = self.vectorstore.similarity_search_with_score(query, k=k)
                
                # Format results
                results = []
                for doc, score in docs:
                    results.append({
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "score": float(score),
                        "relevance": 1.0 - float(score)  # Convert distance to relevance
                    })
                
                logger.info(f"Retrieved {len(results)} documents using vector store for query: {query[:50]}...")
                return results
            else:
                # Fallback to simple keyword search
                logger.info("Using keyword-based search fallback")
                return await self._keyword_search(query, k)
            
        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}")
            return []
    
    async def add_document(self, content: str, metadata: Dict[str, Any]):
        """Add a new document to the knowledge base"""
        try:
            if not self.is_initialized:
                raise RuntimeError("RAG service not initialized")
            
            # Create document
            doc = Document(page_content=content, metadata=metadata)
            
            # Split into chunks
            chunks = self.text_splitter.split_documents([doc])
            
            # Add to vector store
            self.vectorstore.add_documents(chunks)
            
            # Save updated vector store
            vectorstore_path = Path(__file__).parent.parent.parent / "vectorstore"
            self.vectorstore.save_local(str(vectorstore_path))
            
            logger.info(f"Added new document: {metadata.get('title', 'Untitled')}")
            
        except Exception as e:
            logger.error(f"Error adding document: {str(e)}")
            raise
    
    async def _keyword_search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Simple keyword-based search fallback"""
        try:
            query_lower = query.lower()
            query_words = set(query_lower.split())
            
            # Score documents based on keyword matches
            scored_docs = []
            for doc in self.documents:
                content_lower = doc.page_content.lower()
                matches = sum(1 for word in query_words if word in content_lower)
                if matches > 0:
                    relevance = matches / len(query_words)
                    scored_docs.append({
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "score": 1.0 - relevance,  # Invert for consistency with vector search
                        "relevance": relevance
                    })
            
            # Sort by relevance and return top k
            scored_docs.sort(key=lambda x: x["relevance"], reverse=True)
            results = scored_docs[:k]
            
            logger.info(f"Keyword search found {len(results)} relevant documents")
            return results
            
        except Exception as e:
            logger.error(f"Error in keyword search: {str(e)}")
            return []

    async def health_check(self) -> Dict[str, Any]:
        """Check the health of the RAG service"""
        try:
            status = {
                "initialized": self.is_initialized,
                "vectorstore_available": self.vectorstore is not None,
                "embeddings_available": self.embeddings is not None,
                "document_count": len(self.documents) if self.documents else 0
            }
            
            if self.is_initialized:
                # Test a simple query
                test_results = await self.retrieve_documents("test query", k=1)
                status["test_query_successful"] = len(test_results) > 0
            
            return status
            
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return {"error": str(e)} 