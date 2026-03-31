#!/usr/bin/env python3
"""
Test script for the Public Service Navigation Assistant
"""

import requests
import json
import time

def test_backend_health():
    """Test the backend health endpoint"""
    try:
        response = requests.get("http://localhost:8000/health")
        if response.status_code == 200:
            data = response.json()
            print("✅ Backend Health Check:")
            print(f"   Status: {data['status']}")
            print(f"   RAG Service: {'✅' if data['services']['rag_service']['initialized'] else '❌'}")
            print(f"   LLM Service: {'✅' if data['services']['llm_service']['initialized'] else '❌'}")
            print(f"   Speech Service: {'✅' if data['services']['speech_service']['initialized'] else '❌'}")
            return True
        else:
            print(f"❌ Backend health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Backend health check error: {e}")
        return False

def test_voice_handler():
    """Test the voice handler health endpoint"""
    try:
        response = requests.get("http://localhost:5001/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Voice Handler: {data['status']}")
            return True
        else:
            print(f"❌ Voice handler health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Voice handler health check error: {e}")
        return False

def test_query():
    """Test a sample query to the RAG system"""
    try:
        query_data = {
            "query": "What are SNAP benefits and how do I apply?"
        }
        response = requests.post(
            "http://localhost:8000/query",
            headers={"Content-Type": "application/json"},
            json=query_data
        )
        if response.status_code == 200:
            data = response.json()
            print("✅ Query Test:")
            print(f"   Query: {query_data['query']}")
            print(f"   Response: {data['response'][:200]}...")
            print(f"   Confidence: {data['confidence']}")
            return True
        else:
            print(f"❌ Query test failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Query test error: {e}")
        return False

def test_services():
    """Test all available services"""
    print("🚀 Testing Public Service Navigation Assistant...")
    print("=" * 50)
    
    # Test backend
    backend_ok = test_backend_health()
    print()
    
    # Test voice handler
    voice_ok = test_voice_handler()
    print()
    
    # Test query
    query_ok = test_query()
    print()
    
    # Summary
    print("=" * 50)
    print("📊 Test Summary:")
    print(f"   Backend: {'✅ Working' if backend_ok else '❌ Failed'}")
    print(f"   Voice Handler: {'✅ Working' if voice_ok else '❌ Failed'}")
    print(f"   Query System: {'✅ Working' if query_ok else '❌ Failed'}")
    
    if all([backend_ok, voice_ok, query_ok]):
        print("\n🎉 All core services are working!")
        print("\n📋 Available Endpoints:")
        print("   - Backend API: http://localhost:8000")
        print("   - Voice Handler: http://localhost:5001")
        print("   - Nginx Proxy: http://localhost:80")
        print("   - Ollama LLM: http://localhost:11434")
        print("   - PostgreSQL: localhost:5433")
        print("   - Redis: localhost:6379")
        
        print("\n💡 Next Steps:")
        print("   1. Configure Twilio credentials in .env file for voice calls")
        print("   2. Add your OpenAI API key for speech recognition")
        print("   3. Test voice calls through Twilio")
        print("   4. Add more documents to the knowledge base")
    else:
        print("\n⚠️  Some services need attention. Check the logs above.")

if __name__ == "__main__":
    test_services() 