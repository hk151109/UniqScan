# UniqScan ML APIs Documentation

## Overview

This system provides comprehensive academic integrity analysis through three Flask APIs using **Hugging Face Inference API** for AI detection (no local model downloads required):

1. **Unified Grading API** (Port 5000) - Main service combining similarity and AI detection
2. **Similarity Detection API** (Port 5001) - Standalone plagiarism detection 
3. **AI Detection API** (Port 5002) - Standalone AI content detection via Hugging Face API

## 🚀 Quick Setup

### 1. Install Dependencies
```bash
# Windows
setup_huggingface_api.ps1

# Or manually
pip install -r requirements.txt
```

### 2. Hugging Face API Token (Optional but Recommended)
Get a free token from [Hugging Face](https://huggingface.co/settings/tokens) for better performance:

```bash
# Windows
$env:HUGGINGFACE_API_TOKEN = "your_token_here"

# Linux/Mac  
export HUGGINGFACE_API_TOKEN="your_token_here"
```

### 3. Start Services
```bash
# Windows
start_services.bat

# Linux/Mac
./start_services.sh
```

## 🧪 Test the Setup

```bash
python test_huggingface_api.py
```

## API Endpoints

### Unified Grading API (http://localhost:5000)

#### POST /grade/analyze
Analyze a single submission for both similarity and AI content.

**Request:**
```json
{
  "student_id": "student123",
  "student_name": "John Doe", 
  "file_path": "/path/to/submission.txt",
  "assignment_id": "assignment1",
  "classroom_name": "CS101"
}
```

**Response:**
```json
{
  "student_id": "student123",
  "student_name": "John Doe",
  "status": "completed",
  "similarity_analysis": {
    "similarity_score": 25.5,
    "total_comparisons": 12,
    "detailed_results": [...],
    "report_path": "/reports/similarity_report.html"
  },
  "ai_analysis": {
    "ai_score": 0.15,
    "ai_percentage": 15.0,
    "interpretation": "Very likely human-written content",
    "report_path": "/reports/ai_report.html"
  },
  "unified_report_path": "/reports/unified_report.html"
}
```

#### POST /grade/batch
Process multiple submissions simultaneously.

**Request:**
```json
{
  "submissions": [
    {
      "student_id": "student1",
      "student_name": "Student One",
      "file_path": "/path/to/file1.txt", 
      "assignment_id": "assignment1"
    },
    {
      "student_id": "student2",
      "student_name": "Student Two",
      "file_path": "/path/to/file2.txt",
      "assignment_id": "assignment1" 
    }
  ]
}
```

#### GET /stats
Get comprehensive statistics about all analyses.

#### GET /health
Check service health and model status.

### Similarity Detection API (http://localhost:5001)

#### POST /similarity/analyze
Perform similarity analysis on a submission.

#### GET /similarity/database/stats
Get statistics about the similarity database.

#### POST /similarity/reset
Reset the similarity database (use with caution).

### AI Detection API (http://localhost:5002)

#### POST /ai-detection/analyze
Analyze file for AI-generated content.

#### POST /ai-detection/text
Analyze raw text content directly.

#### GET /ai-detection/stats
Get AI detection statistics.

## Data Flow

### For Node.js Integration

1. **Student submits homework** → Node.js backend receives file
2. **Node.js calls ML API:**
   ```javascript
   const response = await fetch('http://localhost:5000/grade/analyze', {
     method: 'POST',
     headers: { 'Content-Type': 'application/json' },
     body: JSON.stringify({
       student_id: studentId,
       student_name: studentName, 
       file_path: absoluteFilePath,
       assignment_id: homeworkId
     })
   });
   ```
3. **ML API processes** → Returns scores and report paths
4. **Node.js updates database** → Stores scores in Project model
5. **Frontend displays results** → Shows scores and report links

### Required Data from Node.js

- **student_id**: Unique identifier for the student
- **student_name**: Full name for reports
- **file_path**: Absolute path to the uploaded file
- **assignment_id**: Homework/assignment identifier  
- **classroom_name**: Optional classroom context

### Returned Data to Node.js

- **similarity_score**: Float (0-100) - plagiarism percentage
- **ai_score**: Float (0-1) - AI probability (multiply by 100 for percentage)
- **report_path**: String - path to HTML report file
- **status**: String - "completed", "partial", "failed", "error"

## File Storage

The ML APIs expect:
- **Submission files** in readable text format
- **File paths** to be absolute and accessible
- **Reports** will be generated in `reports/` directory

## Database Management

### Similarity Database
- Tracks all submissions per assignment
- Compares new submissions against existing ones
- Builds comparison history over time

### AI Detection Database  
- Stores AI analysis results
- Maintains analysis history
- Provides statistical insights

## Performance Considerations

- **First AI analysis** may be slow (model loading)
- **Subsequent analyses** are faster (model cached)
- **Batch processing** uses thread pool for concurrency
- **Large files** are chunked for AI analysis

## Report Generation

### Similarity Report
- Shows matching text segments
- Identifies source documents
- Highlights suspicious passages

### AI Detection Report
- Breaks text into analyzable chunks
- Shows AI probability per section
- Provides interpretation guidance

### Unified Report
- Combines both analyses
- Calculates overall risk score
- Provides actionable recommendations

## Error Handling

All APIs return standard error responses:
```json
{
  "error": "Description of what went wrong",
  "status": "error"
}
```

Common errors:
- File not found
- Unable to read file
- Model loading failed
- Processing timeout

## Integration Examples

### Node.js Backend Integration

```javascript
// In your Node.js homework controller
const gradingService = require('./gradingService');

async function processHomeworkSubmission(studentId, homeworkId, filePath) {
  try {
    // Get student and homework details
    const student = await User.findById(studentId);
    const homework = await Homework.findById(homeworkId);
    
    // Call ML API
    const response = await fetch('http://localhost:5000/grade/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        student_id: studentId,
        student_name: `${student.name} ${student.lastname}`,
        file_path: filePath,
        assignment_id: homeworkId,
        classroom_name: homework.classroom.name
      })
    });
    
    const results = await response.json();
    
    if (results.status === 'completed') {
      // Update database with scores
      const project = await Projects.findOne({ 
        student: studentId, 
        homework: homeworkId 
      });
      
      if (project) {
        project.similarityScore = results.similarity_analysis.similarity_score;
        project.aiGeneratedScore = results.ai_analysis.ai_percentage;
        project.reportPath = results.unified_report_path;
        await project.save();
      }
    }
    
    return results;
  } catch (error) {
    console.error('Error processing submission:', error);
    throw error;
  }
}
```

## Security Considerations

- APIs run on localhost by default
- No authentication implemented (add as needed)
- File paths should be validated
- Reports contain sensitive information

## Troubleshooting

### Common Issues

1. **Model loading fails**
   - Check internet connection
   - Verify transformers library version
   - Clear model cache if needed

2. **File reading errors**
   - Check file permissions
   - Verify file path exists
   - Try different text encodings

3. **Memory issues**
   - Reduce MAX_CHUNK_SIZE
   - Process smaller batches
   - Monitor system resources

### Logs

Check console output for detailed error messages. Enable debug logging by setting `debug=True` in Flask apps.

### Health Checks

Use health endpoints to verify service status:
```bash
curl http://localhost:5000/health
curl http://localhost:5001/health  
curl http://localhost:5002/health
```

## Configuration

Edit `config.py` to adjust:
- Port numbers
- Model settings
- File paths
- Processing parameters
- Risk thresholds
