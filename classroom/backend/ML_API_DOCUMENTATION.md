# ML API Integration Documentation

## Overview

The UniqScan system integrates with an external Machine Learning API to perform academic integrity analysis on student submissions. This document details the complete integration specification.

## API Configuration

### Environment Variables
```bash
ML_API_BASE_URL=http://localhost:5000    # Base URL of the ML API service
ML_API_TIMEOUT=300000                    # Request timeout in milliseconds (5 minutes default)
```

### Endpoints Used
- **Analysis**: `POST {ML_API_BASE_URL}/grade/analyze`
- **Health Check**: `GET {ML_API_BASE_URL}/health`

## Data Sent to ML API

### Request Details
- **Method**: `POST`
- **Endpoint**: `/grade/analyze`
- **Content-Type**: `application/json`
- **Timeout**: 5 minutes (300,000ms)

### Request Payload Structure
```json
{
  "student_id": "string",           // MongoDB ObjectId of the student
  "student_name": "string",         // Full name (firstname lastname)  
  "file_url": "string",             // HTTP URL where ML API can download the file
  "assignment_id": "string",        // MongoDB ObjectId of the homework/assignment
  "classroom_name": "string"        // Name of the classroom
}
```

### Real Example Payload
```json
{
  "student_id": "64a1b2c3d4e5f6789012345a",
  "student_name": "John Doe", 
  "file_url": "http://localhost:4000/uploads/homeworks/John_Doe_Math101_64a1b2c3d4e5f6789012345b.pdf",
  "assignment_id": "64a1b2c3d4e5f6789012345b",
  "classroom_name": "Mathematics 101"
}
```

### File URL Format
The `file_url` is constructed using the centralized URL detection utility:
- **Local Development**: `http://localhost:{PORT}/uploads/homeworks/{filename}`
- **Production**: `https://{domain}/uploads/homeworks/{filename}`
- **File Access**: Publicly accessible via express.static middleware
- **Filename Pattern**: `{studentName}_{classroomTitle}_{homeworkId}.{extension}`

## Expected ML API Response Format

### Complete Response Structure
```json
{
  "status": "completed" | "partial" | "failed",
  "similarity_analysis": {
    "similarity_score": 25.5,        // Float: Percentage (0-100)
    "total_comparisons": 150,         // Integer: Number of comparisons made
    "detailed_results": [             // Array: Detailed match information
      {
        "source": "document_name.pdf",
        "similarity": 85.2,
        "matched_text": "sample text..."
      }
    ]
  },
  "ai_analysis": {
    "ai_percentage": 15.2,           // Float: Percentage (0-100)
    "interpretation": "Low AI probability", // String: Human-readable interpretation
    "chunks_analyzed": 45,           // Integer: Number of text chunks processed
    "confidence": 0.87               // Float: Confidence level (0-1)
  },
  "plagiarism_analysis": {
    "plagiarism_score": 32.8,       // Float: Overall plagiarism percentage (0-100)
    "risk_level": "moderate",        // String: "low", "moderate", "high", "critical"
    "contributing_factors": [        // Array: Factors contributing to plagiarism score
      "high_similarity_matches",
      "ai_generated_content", 
      "improper_citations"
    ]
  },
  "report_html": "<!DOCTYPE html>...",  // String: Complete HTML report (REQUIRED)
  "errors": [                          // Array: Optional error messages
    "Warning: Limited database coverage for this file type"
  ]
}
```

### Minimum Required Response
```json
{
  "status": "completed",
  "similarity_analysis": {
    "similarity_score": 0
  },
  "ai_analysis": {
    "ai_percentage": 0
  },
  "plagiarism_analysis": {
    "plagiarism_score": 0
  },
  "report_html": "<!DOCTYPE html><html><head><title>Report</title></head><body><h1>Analysis Complete</h1></body></html>"
}
```

### Status Field Values
- **`"completed"`**: Analysis fully completed, all scores available
- **`"partial"`**: Analysis partially completed, some scores may be missing
- **`"failed"`**: Analysis failed, fallback template will be used
