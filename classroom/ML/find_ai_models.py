#!/usr/bin/env python3
"""
Test different AI detection models available on Hugging Face
"""

import requests
import json
import os
import time

# List of potential AI detection models to test
AI_MODELS = [
    "Hello-SimpleAI/chatgpt-detector-roberta",
    "roberta-base-openai-detector", 
    "openai-detector",
    "martin-ha/toxic-comment-model",
    "unitary/toxic-bert",
    "huggingface/CodeBERTa-small-v1",
    # Add more models to test
]

def test_model(model_name, test_text="This is a test sentence for AI detection."):
    """Test a specific model with the Hugging Face Inference API"""
    api_url = f"https://api-inference.huggingface.co/models/{model_name}"
    
    # Check for API token
    headers = {}
    token = os.environ.get("HUGGINGFACE_API_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    
    print(f"\n🧪 Testing model: {model_name}")
    print(f"   API URL: {api_url}")
    
    try:
        payload = {"inputs": test_text}
        
        response = requests.post(
            api_url,
            headers=headers,
            json=payload,
            timeout=30
        )
        
        print(f"   Status Code: {response.status_code}")
        
        if response.status_code == 503:
            print("   ⏳ Model is loading, waiting 20 seconds...")
            time.sleep(20)
            
            response = requests.post(
                api_url,
                headers=headers,
                json=payload,
                timeout=30
            )
            print(f"   Retry Status Code: {response.status_code}")
        
        if response.status_code == 200:
            try:
                result = response.json()
                print(f"   ✅ SUCCESS! Response:")
                print(f"      {json.dumps(result, indent=4)}")
                return True, result
            except json.JSONDecodeError:
                print(f"   ❌ Invalid JSON response: {response.text}")
                return False, None
        else:
            print(f"   ❌ FAILED: {response.status_code}")
            print(f"      Error: {response.text}")
            return False, None
            
    except requests.exceptions.RequestException as e:
        print(f"   ❌ REQUEST ERROR: {e}")
        return False, None
    except Exception as e:
        print(f"   ❌ UNEXPECTED ERROR: {e}")
        return False, None

def find_best_ai_detection_model():
    """Test multiple models to find the best working one"""
    print("🔍 Finding the best AI detection model...")
    print("=" * 60)
    
    working_models = []
    
    # Test texts with different characteristics
    test_cases = [
        {
            "name": "Simple human text",
            "text": "I love pizza and my dog is cute."
        },
        {
            "name": "Academic/AI-like text", 
            "text": "The implementation of machine learning algorithms necessitates comprehensive understanding of mathematical foundations and statistical principles."
        }
    ]
    
    for model in AI_MODELS:
        print(f"\n{'='*60}")
        success_count = 0
        
        for test_case in test_cases:
            print(f"\n📝 Test case: {test_case['name']}")
            print(f"   Text: {test_case['text'][:50]}...")
            
            success, result = test_model(model, test_case['text'])
            if success:
                success_count += 1
                
                # Analyze the result format
                if isinstance(result, list) and len(result) > 0:
                    print("   📊 Labels found:")
                    for item in result:
                        if isinstance(item, dict) and 'label' in item:
                            print(f"      • {item['label']}: {item.get('score', 'N/A')}")
        
        if success_count > 0:
            working_models.append({
                "model": model,
                "success_rate": success_count / len(test_cases),
                "total_tests": len(test_cases)
            })
            print(f"\n   ✅ Model works! Success rate: {success_count}/{len(test_cases)}")
        else:
            print(f"\n   ❌ Model failed all tests")
    
    print(f"\n{'='*60}")
    print("📋 SUMMARY")
    print("=" * 60)
    
    if working_models:
        print("✅ Working models found:")
        for model_info in sorted(working_models, key=lambda x: x['success_rate'], reverse=True):
            print(f"   • {model_info['model']} - Success: {model_info['success_rate']:.1%}")
        
        best_model = working_models[0]['model']
        print(f"\n🏆 RECOMMENDED MODEL: {best_model}")
        
        # Generate code snippet
        print(f"\n📝 Update your ai_detection_api.py:")
        print(f'   AI_DETECTION_MODEL = "{best_model}"')
        
    else:
        print("❌ No working models found!")
        print("\n💡 Troubleshooting tips:")
        print("   1. Check your internet connection")
        print("   2. Set HUGGINGFACE_API_TOKEN environment variable")
        print("   3. Try again later (some models may be temporarily unavailable)")

if __name__ == "__main__":
    print("🚀 AI Detection Model Finder")
    print("This script will test multiple AI detection models on Hugging Face")
    
    # Check for API token
    if os.environ.get("HUGGINGFACE_API_TOKEN"):
        print("✅ HUGGINGFACE_API_TOKEN is set")
    else:
        print("⚠️  HUGGINGFACE_API_TOKEN not set - may encounter rate limits")
    
    print("")
    find_best_ai_detection_model()
    
    print(f"\n🎉 Testing completed!")
    print("\nNext steps:")
    print("1. Update AI_DETECTION_MODEL in ai_detection_api.py")
    print("2. Restart your services")
    print("3. Test with python test_unified_api.py")
