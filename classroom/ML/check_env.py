#!/usr/bin/env python3
"""
Test environment configuration for ML services
"""

import os

def check_environment():
    """Check if environment variables are set correctly"""
    print("🔧 Environment Configuration Check")
    print("=" * 50)
    
    # Load environment variables from .env file
    try:
        from dotenv import load_dotenv
        load_dotenv()
        print("✅ Successfully loaded .env file")
    except ImportError:
        print("❌ python-dotenv not installed!")
        print("   Install with: pip install python-dotenv")
        return False
    except Exception as e:
        print(f"⚠️  Could not load .env file: {e}")
        print("   Make sure .env file exists in the ML directory")
    
    # Check Hugging Face API token
    hf_token = os.environ.get("HUGGINGFACE_API_TOKEN")
    if hf_token:
        print(f"✅ HUGGINGFACE_API_TOKEN is set")
        print(f"   Token preview: {hf_token[:10]}...")
        print(f"   Token length: {len(hf_token)} characters")
        
        # Validate token format (should start with hf_)
        if hf_token.startswith("hf_") and len(hf_token) > 20:
            print("✅ Token format appears correct")
        else:
            print("⚠️  Token format may be incorrect")
            print("   Hugging Face tokens should start with 'hf_'")
    else:
        print("❌ HUGGINGFACE_API_TOKEN not set!")
        print("   Get a free token from: https://huggingface.co/settings/tokens")
        print("   Add it to the .env file: HUGGINGFACE_API_TOKEN=your_token_here")
        return False
    
    # Test API connectivity
    print("\n🌐 Testing Hugging Face API connectivity...")
    try:
        import requests
        
        # Test with a simple model
        test_url = "https://huggingface.co/SuperAnnotate/ai-detector"
        headers = {"Authorization": f"Bearer {hf_token}"}
        payload = {"inputs": "Hello world!"}
        
        response = requests.post(test_url, headers=headers, json=payload, timeout=10)
        
        if response.status_code == 200:
            print("✅ API connectivity successful!")
            result = response.json()
            print(f"   Test response: {result}")
        elif response.status_code == 503:
            print("⏳ Model is loading (this is normal)")
            print("   API connectivity is working")
        elif response.status_code == 401:
            print("❌ API authentication failed")
            print("   Check your HUGGINGFACE_API_TOKEN")
            return False
        else:
            print(f"⚠️  API returned status {response.status_code}: {response.text}")
    
    except requests.exceptions.RequestException as e:
        print(f"❌ Network error: {e}")
        print("   Check your internet connection")
        return False
    except Exception as e:
        print(f"❌ Error testing API: {e}")
    
    # Check current directory
    print(f"\n📁 Current directory: {os.getcwd()}")
    print(f"📄 .env file exists: {os.path.exists('.env')}")
    
    if os.path.exists('.env'):
        print("📄 .env file contents:")
        try:
            with open('.env', 'r') as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        key = line.split('=')[0]
                        print(f"   {key}=***")
        except Exception as e:
            print(f"   Error reading .env: {e}")
    
    print("\n🎉 Environment check completed!")
    return True

if __name__ == "__main__":
    success = check_environment()
    
    if success:
        print("\n✅ Environment is configured correctly!")
        print("\nNext steps:")
        print("1. Start AI detection service: python ai_detection_api.py")
        print("2. Start unified service: python unified_grading_api.py") 
        print("3. Test with: python test_unified_api.py")
    else:
        print("\n❌ Environment setup needs attention!")
        print("\nTroubleshooting:")
        print("1. Make sure .env file exists in ML directory")
        print("2. Get Hugging Face token: https://huggingface.co/settings/tokens")
        print("3. Install dependencies: pip install -r requirements.txt")
