#!/usr/bin/env python3
"""
Public Service Navigation Assistant API Test Script
Tests all endpoints and functionality of the system
"""

import requests
import json
import time
import sys
from typing import Dict, Any, List

class APITester:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
        self.test_results = []
        self.agent_session_id = f"api-test-{int(time.time())}"
        
    def log_test(self, test_name: str, success: bool, message: str = "", response: Dict = None):
        """Log test results"""
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}: {message}")
        
        self.test_results.append({
            "test": test_name,
            "success": success,
            "message": message,
            "response": response
        })
        
        return success
    
    def test_health_check(self) -> bool:
        """Test health check endpoint"""
        try:
            response = self.session.get(f"{self.base_url}/health")
            if response.status_code == 200:
                data = response.json()
                return self.log_test(
                    "Health Check",
                    True,
                    f"All services healthy: {data.get('status', 'unknown')}",
                    data
                )
            else:
                return self.log_test(
                    "Health Check",
                    False,
                    f"Status code: {response.status_code}"
                )
        except Exception as e:
            return self.log_test(
                "Health Check",
                False,
                f"Exception: {str(e)}"
            )
    
    def test_root_endpoint(self) -> bool:
        """Test root endpoint"""
        try:
            response = self.session.get(f"{self.base_url}/")
            if response.status_code == 200:
                data = response.json()
                return self.log_test(
                    "Root Endpoint",
                    True,
                    f"Service: {data.get('message', 'unknown')}",
                    data
                )
            else:
                return self.log_test(
                    "Root Endpoint",
                    False,
                    f"Status code: {response.status_code}"
                )
        except Exception as e:
            return self.log_test(
                "Root Endpoint",
                False,
                f"Exception: {str(e)}"
            )
    
    def test_query_endpoint(self) -> bool:
        """Test query endpoint with RAG and LLM"""
        test_queries = [
            "What is SNAP?",
            "How do I apply for housing assistance?",
            "What are the eligibility requirements for Medicaid?",
            "What documents do I need for benefits?"
        ]
        
        all_success = True
        
        for i, query in enumerate(test_queries, 1):
            try:
                payload = {
                    "query": query,
                    "user_context": {"test": True}
                }
                
                response = self.session.post(
                    f"{self.base_url}/query",
                    json=payload,
                    timeout=30
                )
                
                if response.status_code == 200:
                    data = response.json()
                    response_text = data.get('response', '')
                    
                    if response_text and len(response_text) > 10:
                        self.log_test(
                            f"Query Test {i}",
                            True,
                            f"Query: '{query[:30]}...' - Response length: {len(response_text)} chars",
                            {"query": query, "response_length": len(response_text)}
                        )
                    else:
                        all_success = False
                        self.log_test(
                            f"Query Test {i}",
                            False,
                            f"Query: '{query[:30]}...' - Empty or too short response"
                        )
                else:
                    all_success = False
                    self.log_test(
                        f"Query Test {i}",
                        False,
                        f"Query: '{query[:30]}...' - Status code: {response.status_code}"
                    )
                    
            except Exception as e:
                all_success = False
                self.log_test(
                    f"Query Test {i}",
                    False,
                    f"Query: '{query[:30]}...' - Exception: {str(e)}"
                )
        
        return all_success
    
    def test_voice_synthesis(self) -> bool:
        """Test voice synthesis endpoint"""
        try:
            payload = {
                "text": "Hello, this is a test of the voice synthesis system.",
                "voice": "neutral",
                "speed": 1.0
            }
            
            response = self.session.post(
                f"{self.base_url}/voice/synthesize",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                audio_data = data.get('audio_data')
                
                if audio_data and len(audio_data) > 0:
                    return self.log_test(
                        "Voice Synthesis",
                        True,
                        f"Generated audio data: {len(audio_data)} bytes",
                        {"audio_size": len(audio_data)}
                    )
                else:
                    return self.log_test(
                        "Voice Synthesis",
                        False,
                        "No audio data generated"
                    )
            else:
                return self.log_test(
                    "Voice Synthesis",
                    False,
                    f"Status code: {response.status_code}"
                )
                
        except Exception as e:
            return self.log_test(
                "Voice Synthesis",
                False,
                f"Exception: {str(e)}"
            )
    
    def test_voice_processing_pipeline(self) -> bool:
        """Test complete voice processing pipeline"""
        try:
            payload = {
                "text": "What are SNAP benefits?",
                "voice": "neutral",
                "speed": 1.0,
                "user_context": {"test": True}
            }
            
            response = self.session.post(
                f"{self.base_url}/voice/process",
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                data = response.json()
                text_response = data.get('text_response', '')
                audio_data = data.get('audio_data')
                
                if text_response and audio_data:
                    return self.log_test(
                        "Voice Processing Pipeline",
                        True,
                        f"Text response: {len(text_response)} chars, Audio: {len(audio_data)} bytes",
                        {"text_length": len(text_response), "audio_size": len(audio_data)}
                    )
                else:
                    return self.log_test(
                        "Voice Processing Pipeline",
                        False,
                        "Missing text response or audio data"
                    )
            else:
                return self.log_test(
                    "Voice Processing Pipeline",
                    False,
                    f"Status code: {response.status_code}"
                )
                
        except Exception as e:
            return self.log_test(
                "Voice Processing Pipeline",
                False,
                f"Exception: {str(e)}"
            )
    
    def test_rasa_integration(self) -> bool:
        """Test Rasa integration"""
        try:
            rasa_url = "http://localhost:5005/webhooks/rest/webhook"
            
            payload = {
                "sender": "test_user",
                "message": "What is SNAP?"
            }
            
            response = self.session.post(
                rasa_url,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                if data and len(data) > 0:
                    rasa_response = data[0].get('text', '')
                    return self.log_test(
                        "Rasa Integration",
                        True,
                        f"Rasa response: {len(rasa_response)} chars",
                        {"response_length": len(rasa_response)}
                    )
                else:
                    return self.log_test(
                        "Rasa Integration",
                        False,
                        "No response from Rasa"
                    )
            else:
                return self.log_test(
                    "Rasa Integration",
                    False,
                    f"Status code: {response.status_code}"
                )
                
        except Exception as e:
            return self.log_test(
                "Rasa Integration",
                False,
                f"Exception: {str(e)}"
            )

    def test_agentic_query_endpoint(self) -> bool:
        """Test agentic query execution via /query."""
        try:
            payload = {
                "query": "I need help with SNAP and housing. Which one should I apply for first?",
                "mode": "agentic",
                "session_id": self.agent_session_id,
                "max_steps": 4,
            }

            response = self.session.post(
                f"{self.base_url}/query",
                json=payload,
                timeout=45,
            )

            if response.status_code != 200:
                return self.log_test(
                    "Agentic Query",
                    False,
                    f"Status code: {response.status_code}",
                )

            data = response.json()
            response_text = data.get("response", "")
            execution_mode = data.get("execution_mode")

            if not response_text:
                return self.log_test(
                    "Agentic Query",
                    False,
                    "Response text is empty",
                    data,
                )

            if execution_mode != "agentic":
                return self.log_test(
                    "Agentic Query",
                    False,
                    f"Expected execution_mode=agentic, got {execution_mode}",
                    data,
                )

            return self.log_test(
                "Agentic Query",
                True,
                (
                    f"Session: {data.get('session_id')} | "
                    f"Trace steps: {len(data.get('agent_trace') or [])}"
                ),
                {
                    "session_id": data.get("session_id"),
                    "execution_mode": execution_mode,
                    "response_length": len(response_text),
                },
            )

        except Exception as e:
            return self.log_test(
                "Agentic Query",
                False,
                f"Exception: {str(e)}",
            )

    def test_agentic_session_lifecycle(self) -> bool:
        """Test agentic session inspect and delete endpoints."""
        try:
            get_response = self.session.get(
                f"{self.base_url}/agent/sessions/{self.agent_session_id}",
                timeout=20,
            )

            if get_response.status_code != 200:
                return self.log_test(
                    "Agentic Session Lifecycle",
                    False,
                    f"Session fetch failed with status: {get_response.status_code}",
                )

            session_data = get_response.json()
            delete_response = self.session.delete(
                f"{self.base_url}/agent/sessions/{self.agent_session_id}",
                timeout=20,
            )

            if delete_response.status_code != 200:
                return self.log_test(
                    "Agentic Session Lifecycle",
                    False,
                    f"Session delete failed with status: {delete_response.status_code}",
                )

            return self.log_test(
                "Agentic Session Lifecycle",
                True,
                f"Turn count before delete: {session_data.get('turn_count', 0)}",
                {
                    "session_id": session_data.get("session_id"),
                    "turn_count": session_data.get("turn_count", 0),
                },
            )

        except Exception as e:
            return self.log_test(
                "Agentic Session Lifecycle",
                False,
                f"Exception: {str(e)}",
            )
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all tests and return results"""
        print("🚀 Starting Public Service Navigation Assistant API Tests")
        print("=" * 60)
        
        tests = [
            ("Health Check", self.test_health_check),
            ("Root Endpoint", self.test_root_endpoint),
            ("Query Endpoint", self.test_query_endpoint),
            ("Agentic Query", self.test_agentic_query_endpoint),
            ("Agentic Session Lifecycle", self.test_agentic_session_lifecycle),
            ("Voice Synthesis", self.test_voice_synthesis),
            ("Voice Processing Pipeline", self.test_voice_processing_pipeline),
            ("Rasa Integration", self.test_rasa_integration)
        ]
        
        passed = 0
        total = len(tests)
        
        for test_name, test_func in tests:
            print(f"\n📋 Running {test_name}...")
            if test_func():
                passed += 1
            time.sleep(1)  # Brief pause between tests
        
        # Summary
        print("\n" + "=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)
        print(f"Total Tests: {total}")
        print(f"Passed: {passed}")
        print(f"Failed: {total - passed}")
        print(f"Success Rate: {(passed/total)*100:.1f}%")
        
        if passed == total:
            print("\n🎉 All tests passed! Your Public Service Navigation Assistant is working correctly.")
        else:
            print(f"\n⚠️  {total - passed} test(s) failed. Please check the logs and configuration.")
        
        return {
            "total": total,
            "passed": passed,
            "failed": total - passed,
            "success_rate": (passed/total)*100,
            "results": self.test_results
        }

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Public Service Navigation Assistant API")
    parser.add_argument("--url", default="http://localhost:8000", help="Base URL for API")
    parser.add_argument("--output", help="Output results to JSON file")
    
    args = parser.parse_args()
    
    # Create tester
    tester = APITester(args.url)
    
    try:
        # Run tests
        results = tester.run_all_tests()
        
        # Save results if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n💾 Results saved to {args.output}")
        
        # Exit with appropriate code
        sys.exit(0 if results["passed"] == results["total"] else 1)
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Testing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 