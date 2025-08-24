#!/usr/bin/env python3
"""
Test script for Hugging Face API integration
"""

import requests
import json
import os
import time

# Configuration
AI_DETECTION_MODEL = "SuperAnnotate/ai-detector"
HUGGINGFACE_API_URL = f"https://api-inference.huggingface.co/models/{AI_DETECTION_MODEL}"

def test_huggingface_api():
    """Test the Hugging Face API directly"""
    print("🧪 Testing Hugging Face API Integration...")
    print(f"Model: {AI_DETECTION_MODEL}")
    print(f"API URL: {HUGGINGFACE_API_URL}")
    
    # Check for API token
    token = os.environ.get("HUGGINGFACE_API_TOKEN")
    if token:
        print("✅ Hugging Face API token found")
        headers = {"Authorization": f"Bearer {token}"}
    else:
        print("⚠️  No Hugging Face API token set (will be rate-limited)")
        print("   Set HUGGINGFACE_API_TOKEN environment variable for better performance")
        headers = {}
    
    # Test texts
    test_texts = [
        "This is a simple human-written text about machine learning and its applications.",
        "The utilization of artificial intelligence in contemporary educational paradigms facilitates enhanced pedagogical methodologies and optimizes learning outcomes through data-driven insights.",
        "I love pizza and ice cream. My favorite color is blue."
    ]
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n📝 Test {i}: {text[:50]}...")
        
        try:
            payload = {"inputs": text}
            
            response = requests.post(
                HUGGINGFACE_API_URL,
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 503:
                print("   ⏳ Model is loading, waiting 20 seconds...")
                time.sleep(20)
                
                response = requests.post(
                    HUGGINGFACE_API_URL,
                    headers=headers,
                    json=payload,
                    timeout=30
                )
            
            if response.status_code == 200:
                result = response.json()
                print(f"   ✅ Success! Response: {json.dumps(result, indent=2)}")
                
                # Parse results
                if isinstance(result, list) and len(result) > 0:
                    for pred in result:
                        if isinstance(pred, dict) and 'label' in pred and 'score' in pred:
                            label = pred['label']
                            score = pred['score']
                            print(f"   📊 {label}: {score:.3f}")
            else:
                print(f"   ❌ API Error: {response.status_code} - {response.text}")
                
        except requests.exceptions.RequestException as e:
            print(f"   ❌ Request error: {e}")
        except Exception as e:
            print(f"   ❌ Unexpected error: {e}")

def test_local_api():
    """Test the local AI detection API"""
    print("\n" + "="*50)
    print("🔍 Testing Local AI Detection API...")
    
    try:
        # Test health endpoint
        response = requests.get("http://localhost:5002/health", timeout=10)
        if response.status_code == 200:
            health = response.json()
            print("✅ Health check passed:")
            print(f"   Status: {health.get('status')}")
            print(f"   API Available: {health.get('api_available')}")
            print(f"   Has Token: {health.get('has_token')}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return
            
    except requests.exceptions.RequestException:
        print("❌ Local API not running. Start with: python ai_detection_api.py")
        return
    
    # Test text analysis
    test_payload = {
        "text": "The implementation of machine learning algorithms necessitates comprehensive data preprocessing and feature engineering methodologies.",
        "student_name": "Test Student",
        "assignment_id": "test_assignment"
    }
    
    try:
        response = requests.post(
            "http://localhost:5002/ai-detection/text",
            json=test_payload,
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Text analysis successful:")
            print(f"   AI Score: {result.get('ai_percentage', 0):.2f}%")
            print(f"   Chunks Analyzed: {result.get('chunks_analyzed', 0)}")
            print(f"   Interpretation: {result.get('interpretation')}")
        else:
            print(f"❌ Text analysis failed: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"❌ Error testing text analysis: {e}")

if __name__ == "__main__":
    print("🚀 Hugging Face API Integration Test")
    print("=" * 50)
    
    # Test direct API
    test_huggingface_api()
    
    # Test local API
    test_local_api()
    
    print("\n" + "=" * 50)
    print("🎉 Test completed!")
    print("\n💡 Tips:")
    print("1. Set HUGGINGFACE_API_TOKEN for better API performance")
    print("2. First API calls may be slow as the model loads")
    print("3. Free tier has rate limits - consider upgrading for production")
    print("4. Start local API with: python ai_detection_api.py")
