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

## Report Generation Process

### 1. HTML Content Priority (in order)
1. **ML API HTML**: `response.report_html` field from ML API response
2. **Fallback Template**: Professional auto-generated template using available scores
3. **Error Template**: Service unavailable template (when ML API fails entirely)

### 2. File Creation Process
- **File Location**: `/public/uploads/processed_docs/report_{userName}_{projectId}.html`
- **Public URL**: `{backend_url}/uploads/processed_docs/report_{fileName}.html`
- **Encoding**: UTF-8
- **Directory**: Auto-created if doesn't exist
- **Error Handling**: Graceful degradation if file save fails

### 3. Database Storage
- **Project Model Field**: `reportPath` stores the public URL path
- **Example**: `/uploads/processed_docs/report_JohnDoe_64a1b2c3d4e5f6789012345a.html`
- **Frontend Access**: Direct URL access or iframe embedding

## Template Features

### Professional Fallback Template
- **📱 Responsive Design**: Mobile-first CSS Grid layout
- **🎨 Modern Styling**: Gradients, shadows, hover effects
- **📊 Color-Coded Scores**: 
  - Green (0-39%): Low risk
  - Orange (40-69%): Medium risk  
  - Red (70-100%): High risk
- **📋 Detailed Sections**: Analysis breakdowns with icons
- **💡 Smart Recommendations**: Context-aware suggestions
- **🔍 Interactive Elements**: Hover effects and animations

### Error Template
- **⚠️ Clear Messaging**: Student-friendly language about service issues
- **📝 Action Items**: Clear next steps for students
- **🎯 Professional Look**: Maintains system credibility
- **📞 Support Guidance**: Contact information suggestions
- **🛠️ Technical Details**: Collapsible error information for debugging

## Error Handling & Fallbacks

### Connection Errors (ECONNREFUSED)
- **Message**: "ML API service is not available"
- **Action**: Show professional error template
- **Scores**: Set to 0 with error indication

### Timeout Errors (ECONNABORTED) 
- **Message**: "ML API request timed out"
- **Action**: Show processing delay template
- **Scores**: Set to 0 with timeout indication

### Invalid/Missing Response
- **Missing `report_html`**: Use fallback template with available scores
- **Invalid JSON**: Use error template
- **Missing scores**: Default to 0 for missing values

### File Access Issues
- **Invalid file_url**: ML API should return error in response
- **File download failure**: ML API should handle gracefully
- **Unsupported format**: ML API should return appropriate error

## Health Check Endpoint

### Request
```http
GET {ML_API_BASE_URL}/health
```

### Expected Response
```json
{
  "status": "healthy",
  "timestamp": "2025-08-27T10:30:00Z",
  "version": "1.0.0",
  "services": {
    "plagiarism_detector": "online",
    "ai_detector": "online",
    "database": "connected"
  }
}
```

## Security Considerations

### File Access
- **Public URLs**: Files are publicly accessible via express.static
- **No Authentication**: Current implementation doesn't require auth for file access
- **Recommendation**: Consider implementing signed URLs or authenticated endpoints

### API Security
- **No Authentication**: Current ML API calls don't include authentication
- **Recommendation**: Add API key or bearer token authentication
- **Network**: Consider VPC/internal network for production

### Data Privacy
- **File Content**: ML API has full access to student submission content
- **Student Data**: Names and IDs are transmitted to ML API
- **Recommendations**: 
  - Implement data retention policies
  - Add encryption for sensitive data
  - Consider data anonymization

## Testing

### Manual Testing
1. Start ML API service on configured port
2. Submit a homework via frontend
3. Check backend logs for API call details
4. Verify report generation and file creation
5. Access report URL in browser

### Health Check Testing
```bash
curl http://localhost:5000/health
```

### API Call Testing
```bash
curl -X POST http://localhost:5000/grade/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "student_id": "test123",
    "student_name": "Test Student",
    "file_url": "http://localhost:4000/uploads/homeworks/test.pdf",
    "assignment_id": "assignment123", 
    "classroom_name": "Test Class"
  }'
```

## Implementation Notes

### File URL Construction
- Uses `urlDetection.js` for environment-aware URL generation
- Supports multiple deployment platforms (localhost, Cloud Run, Heroku, etc.)
- Automatically detects backend URL based on environment variables

### Processing Flow
1. Student submits homework → File saved locally
2. Grading job queued → `gradingQueue.js`
3. ML API called with file URL → `gradingService.js`  
4. Response processed → HTML extracted or fallback generated
5. Report file saved → `/public/uploads/processed_docs/`
6. Database updated → Project model with scores and report path
7. Frontend displays results → Via report URL

### Performance Considerations
- **Timeout**: 5-minute timeout for ML API calls
- **Queue**: In-memory job queue for processing submissions
- **File Size**: No explicit limits (handled by multer middleware)
- **Concurrent Jobs**: Sequential processing (one at a time)

## Report Generation Process

### 1. HTML Content Source Priority:
1. **Primary**: `response.report_html` field from ML API
2. **Fallback**: Auto-generated professional template using scores
3. **Error**: Professional error template if ML API fails

### 2. File Creation:
- HTML content is always saved to: `/public/uploads/processed_docs/report_{userName}_{projectId}.html`
- File is accessible via: `http://backend-url/uploads/processed_docs/report_filename.html`
- Directory is created automatically if it doesn't exist

### 3. Frontend Access:
- Report URL is stored in Project model `reportPath` field
- Frontend can display report in iframe or new tab
- Reports are publicly accessible via express.static serving

## Template Features

### Professional Fallback Template:
- **Responsive design** with mobile support
- **Modern styling** using CSS Grid and Flexbox
- **Color-coded scoring** (green=low, orange=medium, red=high)  
- **Interactive elements** with hover effects
- **Detailed explanations** for each score type
- **Professional branding** with UniqScan styling

### Error Template:
- **Clear communication** about service unavailability
- **Student-friendly messaging** (no technical jargon)
- **Next steps guidance** 
- **Contact information** suggestions
- **Professional appearance** maintaining system credibility

## File Input Format

The ML API receives files via **HTTP URL download**, not direct file upload:
- Backend uploads files to local storage
- Backend constructs public HTTP URLs using `urlDetection.js`
- ML API downloads files using provided `file_url`
- Supports any file type that students can upload (PDF, DOC, TXT, etc.)

## Error Handling

1. **Connection Errors**: Professional template with service unavailable message
2. **Timeout Errors**: Template indicating analysis is taking longer than expected  
3. **Invalid Response**: Fallback template with available scores
4. **Missing HTML**: Auto-generated template using numerical scores
5. **File Save Errors**: Graceful degradation, scores still saved to database

## Security Notes

- File URLs are publicly accessible (consider authentication in production)
- Reports are publicly accessible once generated
- ML API should validate file URLs and handle download errors
- Consider implementing signed URLs for enhanced security
