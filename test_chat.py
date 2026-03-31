#!/usr/bin/env python3
"""
Test script for the Public Service Navigation Assistant Chat
"""

import requests
import json
import time

def test_chat_functionality():
    """Test the chat API endpoints"""
    
    base_url = "http://localhost:8000"
    
    print("🤖 Testing Public Service Navigation Assistant Chat")
    print("=" * 50)
    
    # Test 1: Health Check
    print("\n1. Testing Health Check...")
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            print("✅ Health check passed")
            health_data = response.json()
            print(f"   Status: {health_data.get('status')}")
            print(f"   RAG Service: {health_data.get('services', {}).get('rag_service', {}).get('initialized', False)}")
            print(f"   LLM Service: {health_data.get('services', {}).get('llm_service', {}).get('initialized', False)}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False
    
    # Test 2: Basic Query
    print("\n2. Testing Basic Query...")
    test_queries = [
        "What housing assistance programs are available?",
        "How do I apply for SNAP benefits?",
        "What healthcare programs can I qualify for?",
        "What documents do I need for applications?"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n   Test {i}: {query}")
        try:
            response = requests.post(
                f"{base_url}/query",
                headers={"Content-Type": "application/json"},
                json={"query": query}
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"   ✅ Response received")
                print(f"   📝 Response length: {len(data.get('response', ''))} characters")
                print(f"   📚 Sources found: {len(data.get('sources', []))}")
                print(f"   🎯 Confidence: {data.get('confidence', 0)}")
                
                # Show first 100 characters of response
                response_text = data.get('response', '')
                preview = response_text[:100] + "..." if len(response_text) > 100 else response_text
                print(f"   💬 Preview: {preview}")
                
            else:
                print(f"   ❌ Query failed: {response.status_code}")
                print(f"   Error: {response.text}")
                
        except Exception as e:
            print(f"   ❌ Query error: {e}")
    
    # Test 3: Conversation Context
    print("\n3. Testing Conversation Context...")
    try:
        conversation = [
            {"role": "user", "content": "I need help with housing"},
            {"role": "assistant", "content": "I can help you with housing assistance programs. What specific information do you need?"}
        ]
        
        response = requests.post(
            f"{base_url}/query",
            headers={"Content-Type": "application/json"},
            json={
                "query": "What documents do I need?",
                "user_context": conversation
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Conversation context test passed")
            print(f"   📝 Response: {data.get('response', '')[:100]}...")
        else:
            print(f"❌ Conversation context test failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Conversation context error: {e}")
    
    # Test 4: Error Handling
    print("\n4. Testing Error Handling...")
    try:
        response = requests.post(
            f"{base_url}/query",
            headers={"Content-Type": "application/json"},
            json={"query": ""}  # Empty query
        )
        
        if response.status_code == 422:  # Validation error
            print("✅ Error handling test passed (empty query rejected)")
        else:
            print(f"⚠️  Unexpected response for empty query: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error handling test error: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 Chat functionality test completed!")
    print("\n📱 To test the web interface:")
    print("   1. Open http://localhost:8080/chat_interface.html")
    print("   2. Try the quick action buttons")
    print("   3. Type your own questions")
    print("\n📞 To test voice calls:")
    print("   1. Call your Twilio number")
    print("   2. Ask questions about public services")
    print("   3. Follow the voice prompts")
    
    return True

if __name__ == "__main__":
    test_chat_functionality() 