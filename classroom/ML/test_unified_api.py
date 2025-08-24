#!/usr/bin/env python3
"""
Test script for unified grading API
"""

import requests
import json
import os
import tempfile
from pathlib import Path

def create_test_file(content, filename="test_submission.txt"):
    """Create a temporary test file"""
    temp_dir = tempfile.gettempdir()
    file_path = os.path.join(temp_dir, filename)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    return file_path

def test_unified_api():
    """Test the unified grading API"""
    print("🧪 Testing Unified Grading API...")
    print("=" * 50)
    
    base_url = "http://localhost:5000"
    
    # Test health endpoint
    print("1. Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        if response.status_code == 200:
            health = response.json()
            print("   ✅ Health check passed:")
            print(f"   • Service: {health.get('service')}")
            print(f"   • AI API Available: {health.get('ai_api_available')}")
            print(f"   • HF Token Set: {health.get('huggingface_token_set')}")
        else:
            print(f"   ❌ Health check failed: {response.status_code}")
            return
    except requests.exceptions.RequestException as e:
        print(f"   ❌ Cannot connect to unified API: {e}")
        print("   💡 Make sure to start: python unified_grading_api.py")
        return
    
    print("\n2. Testing submission analysis...")
    
    # Create test files with different characteristics
    test_cases = [
        {
            "name": "Human-like content",
            "content": "I really enjoyed learning about machine learning in this class. The concepts were challenging but the professor explained them well. My favorite part was working on the final project where I got to apply what I learned.",
            "expected_ai": "low"
        },
        {
            "name": "AI-like content", 
            "content": "The implementation of machine learning algorithms necessitates comprehensive understanding of mathematical foundations and statistical principles. The optimization of neural network architectures requires careful consideration of hyperparameter tuning and regularization techniques to mitigate overfitting phenomena.",
            "expected_ai": "high"
        },
        {
            "name": "Mixed content",
            "content": "Machine learning is cool. The algorithms work by finding patterns in data. I think neural networks are particularly interesting because they can learn complex relationships.",
            "expected_ai": "medium"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n   Test {i}: {test_case['name']}")
        print(f"   Content preview: {test_case['content'][:60]}...")
        
        # Create test file
        file_path = create_test_file(test_case['content'], f"test_{i}.txt")
        
        try:
            payload = {
                "student_id": f"test_student_{i}",
                "student_name": f"Test Student {i}",
                "file_path": file_path,
                "assignment_id": "test_assignment",
                "classroom_name": "Test Classroom"
            }
            
            response = requests.post(
                f"{base_url}/grade/analyze",
                json=payload,
                timeout=120  # 2 minutes for API calls
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"   ✅ Analysis completed:")
                
                # Similarity analysis
                sim_analysis = result.get('similarity_analysis', {})
                if sim_analysis:
                    print(f"   • Similarity Score: {sim_analysis.get('similarity_score', 0):.2f}%")
                    print(f"   • Comparisons: {sim_analysis.get('total_comparisons', 0)}")
                
                # AI analysis
                ai_analysis = result.get('ai_analysis', {})
                if ai_analysis:
                    print(f"   • AI Score: {ai_analysis.get('ai_percentage', 0):.2f}%")
                    print(f"   • Chunks Analyzed: {ai_analysis.get('chunks_analyzed', 0)}")
                    print(f"   • Interpretation: {ai_analysis.get('interpretation', 'N/A')}")
                
                # Overall status
                print(f"   • Status: {result.get('status', 'unknown')}")
                
                if result.get('errors'):
                    print(f"   ⚠️  Warnings: {result['errors']}")
                
            else:
                print(f"   ❌ Analysis failed: {response.status_code} - {response.text}")
                
        except requests.exceptions.Timeout:
            print("   ⏰ Request timed out (this is normal for first API calls)")
        except Exception as e:
            print(f"   ❌ Error: {e}")
        finally:
            # Clean up test file
            try:
                os.remove(file_path)
            except:
                pass
    
    print("\n3. Testing statistics endpoint...")
    try:
        response = requests.get(f"{base_url}/grade/stats", timeout=10)
        if response.status_code == 200:
            stats = response.json()
            print("   ✅ Statistics retrieved:")
            
            sim_stats = stats.get('similarity_stats', {})
            ai_stats = stats.get('ai_detection_stats', {})
            
            print(f"   • Total Students: {sim_stats.get('total_students', 0)}")
            print(f"   • Total Submissions: {sim_stats.get('total_submissions', 0)}")
            print(f"   • AI Analyses: {ai_stats.get('total_analyses', 0)}")
            
        else:
            print(f"   ❌ Stats failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Stats error: {e}")

if __name__ == "__main__":
    print("🚀 Unified Grading API Test")
    print("This will test the complete workflow including Hugging Face API calls")
    print("")
    
    # Check for HF token
    if os.environ.get("HUGGINGFACE_API_TOKEN"):
        print("✅ HUGGINGFACE_API_TOKEN is set")
    else:
        print("⚠️  HUGGINGFACE_API_TOKEN not set - API calls may be rate-limited")
    
    print("")
    test_unified_api()
    
    print("\n" + "=" * 50)
    print("🎉 Test completed!")
    print("\n💡 Tips:")
    print("• First API calls to Hugging Face may be slow (model loading)")
    print("• Set HUGGINGFACE_API_TOKEN for better performance") 
    print("• Check the generated reports in the reports/ directory")
