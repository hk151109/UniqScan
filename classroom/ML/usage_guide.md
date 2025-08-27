# Plagiarism Detection API Usage Guide

## 🚀 Quick Start

### Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the API:
```bash
python app.py
```

The API will start on `http://localhost:5000`

## 📚 API Endpoints

### Health Check
```http
GET /health
```
**Response:**
```json
{
  "status": "healthy",
  "service": "plagiarism_detector",
  "timestamp": "2024-01-15T10:30:00"
}
```

### Document Analysis (Main Endpoint)
```http
POST /grade/analyze
Content-Type: application/json
```

**Request Body:**
```json
{
  "student_id": "64a1b2c3d4e5f6789012345a",
  "student_name": "John Doe",
  "file_url": "http://localhost:4000/uploads/homeworks/John_Doe_Math101_64a1b2c3d4e5f6789012345b.pdf",
  "assignment_id": "64a1b2c3d4e5f6789012345b",
  "classroom_name": "Mathematics 101"
}
```

**Response:**
```json
{
  "status": "completed",
  "similarity_analysis": {
    "similarity_score": 25.5,
    "total_comparisons": 150,
    "detailed_results": [
      {
        "source": "previous_document.pdf",
        "similarity": 85.2,
        "matched_text": "sample text..."
      }
    ]
  },
  "ai_analysis": {
    "ai_percentage": 15.2,
    "interpretation": "Low AI probability",
    "chunks_analyzed": 45,
    "confidence": 0.87
  },
  "report_html": "<!DOCTYPE html>...",
  "errors": []
}
```

### File Serving
```http
GET /uploads/homeworks/<filename>
```
Serves uploaded homework files statically.

### Database Management

#### List All Files
```http
GET /database/files
```

#### Database Statistics
```http
GET /database/stats
```

**Response:**
```json
{
  "total_files": 42,
  "unique_students": 15,
  "unique_assignments": 8,
  "average_similarity_score": 12.34,
  "average_ai_score": 23.45
}
```

#### Clear Database (Testing Only)
```http
DELETE /database/clear
```

## 🔧 Configuration

### Environment Variables
- `PORT`: Server port (default: 5000)
- `FLASK_DEBUG`: Enable debug mode (default: False)

### File Settings
- **Supported formats**: txt, pdf, doc, docx, py, java, cpp, c, js, html, css, md
- **Max file size**: 16MB
- **Storage location**: `homework/` folder
- **Database**: `plagiarism_database.json`

## 📊 How It Works

1. **File Download**: Downloads file from provided URL
2. **Text Extraction**: Extracts text content based on file type
3. **N-gram Analysis**: Uses trigrams (3-word sequences) for comparison
4. **Similarity Detection**: Compares against all files in database
5. **AI Detection**: Generates mock AI probability score
6. **Report Generation**: Creates comprehensive HTML report

### Similarity Algorithm
- Uses Lancaster Stemmer for word normalization
- Removes English stopwords
- Applies sequence matching with configurable thresholds
- Extends matches using edit distance
- Heals neighboring matches within minimum distance

### Mock AI Detection
Currently generates random AI scores with realistic distributions:
- **0-20%**: Very Low AI probability
- **20-40%**: Low AI probability  
- **40-60%**: Medium AI probability
- **60-80%**: High AI probability
- **80-100%**: Very High AI probability

## 🛠️ Error Handling

The API handles various error conditions:
- Missing required fields → 400 Bad Request
- Unsupported file types → 400 Bad Request
- File download failures → 400 Bad Request
- File too large → 413 Request Entity Too Large
- Internal errors → 500 Internal Server Error

All error responses follow the same format:
```json
{
  "status": "failed",
  "similarity_analysis": {"similarity_score": 0},
  "ai_analysis": {"ai_percentage": 0},
  "report_html": "<html><body><h1>Error Message</h1></body></html>",
  "errors": ["Detailed error description"]
}
```

## 📝 Integration Example

```python
import requests
import json

# Prepare request data
payload = {
    "student_id": "64a1b2c3d4e5f6789012345a",
    "student_name": "John Doe",
    "file_url": "http://localhost:4000/uploads/homeworks/assignment.pdf",
    "assignment_id": "64a1b2c3d4e5f6789012345b",
    "classroom_name": "Computer Science 101"
}

# Send request to API
response = requests.post(
    "http://localhost:5000/grade/analyze",
    json=payload,
    timeout=300  # 5 minutes timeout
)

# Process response
if response.status_code == 200:
    result = response.json()
    print(f"Similarity Score: {result['similarity_analysis']['similarity_score']}%")
    print(f"AI Detection: {result['ai_analysis']['ai_percentage']}%")
    
    # Save HTML report
    with open("report.html", "w") as f:
        f.write(result["report_html"])
else:
    print(f"Error: {response.status_code}")
    print(response.json())
```

## 🧪 Testing

Use the provided endpoints to test functionality:

1. **Health Check**: Verify API is running
2. **Upload Test File**: Submit a document for analysis
3. **Check Database**: View stored files and statistics
4. **Clear Database**: Reset for fresh testing

## 🔒 Security Considerations

- File downloads are limited by timeout (30 seconds)
- Filenames are sanitized using `secure_filename()`
- File size limits prevent abuse
- Text extraction handles encoding errors gracefully
- Database operations are logged for audit trails