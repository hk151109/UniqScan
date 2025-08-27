import requests
import json

# Prepare request data
payload = {
    "student_id": "64a1b2c3d4e5f6789012345a",
    "student_name": "John Doe",
    "file_url": "http://localhost:4000/uploads/homeworks/ff.txt",
    "assignment_id": "64a1b2c3d4e5f6789012345b",
    "classroom_name": "Computer Science 101"
}


try:
    # Send request to API
    print("Sending request to plagiarism API...")
    response = requests.post(
        "http://localhost:5000/grade/analyze",
        json=payload,
        timeout=300  # 5 minutes timeout
    )

    # Process response
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Analysis completed successfully!")
        print(f"📊 Similarity Score: {result['similarity_analysis']['similarity_score']}%")
        print(f"🤖 AI Detection: {result['ai_analysis']['ai_percentage']}%")
        print(f"📈 Total Comparisons: {result['similarity_analysis']['total_comparisons']}")
        
        # Save HTML report with proper encoding
        with open("report.html", "w", encoding='utf-8') as f:
            f.write(result["report_html"])
        
        print(f"📝 HTML report saved to: report.html")
        
        # Print detailed results if any
        if result['similarity_analysis'].get('detailed_results'):
            print(f"\n🔍 Found {len(result['similarity_analysis']['detailed_results'])} matches:")
            for i, match in enumerate(result['similarity_analysis']['detailed_results'], 1):
                print(f"  {i}. {match['source']} - {match['similarity']}% similar")
        
    else:
        print(f"❌ Error: {response.status_code}")
        print("Response:", response.text)

except requests.exceptions.RequestException as e:
    print(f"❌ Network error: {e}")
    print("💡 Make sure the Flask API is running on http://localhost:5000")

except Exception as e:
    print(f"❌ Unexpected error: {e}")