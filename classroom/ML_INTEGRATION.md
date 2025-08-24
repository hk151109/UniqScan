# ML API Integration

The backend has been updated to use actual Machine Learning APIs for academic integrity analysis instead of the previous mock implementation.

## What Changed

### Backend Services Updated

1. **gradingService.js** - Now makes HTTP requests to Python ML APIs
2. **gradingQueue.js** - Enhanced with better error handling and context passing
3. **homeworkController.js** - Added ML API health check endpoint
4. **Project model** - Added grading completion tracking fields

### New Features

- **Real AI Detection**: Uses HuggingFace models for AI content detection
- **Similarity Analysis**: Compares submissions against each other
- **Comprehensive Reports**: Detailed HTML reports with analysis results
- **Health Monitoring**: Check ML API status via `/api/homeworks/ml-health`
- **Error Handling**: Graceful fallback when ML services are unavailable

## Setup Instructions

### 1. Install Dependencies

```powershell
# Run the setup script
.\setup-ml-integration.ps1

# Or manually install axios
cd backend
npm install axios
```

### 2. Start ML Services

```bash
cd ML
python unified_grading_api.py
```

This starts all three ML services:
- Unified Grading API (port 5000)
- Similarity Detection API (port 5001) 
- AI Detection API (port 5002)

### 3. Start Backend

```bash
cd backend
npm run dev
```

## Testing the Integration

### Automated Test

```bash
cd backend
node test-ml-integration.js
```

### Manual Test

1. Login as a teacher
2. Create a classroom and homework assignment
3. Login as a student and submit a homework file
4. Check the grading results in the homework page

### Health Check

Visit: `GET /api/homeworks/ml-health` (requires teacher authentication)

## API Flow

1. **File Upload**: Student submits homework → File saved with proper naming
2. **Queue Job**: Grading job added to processing queue
3. **ML Analysis**: 
   - File sent to unified ML API at `http://localhost:5000/grade/analyze`
   - API performs similarity check and AI detection
   - Returns scores and detailed report
4. **Results Storage**: Scores and report saved to database
5. **Frontend Display**: Teacher sees updated grading columns

## Configuration

Environment variables (optional):
- `ML_API_BASE_URL` - Base URL for ML services (default: http://localhost:5000)
- `ML_API_TIMEOUT` - Request timeout in ms (default: 300000 = 5 minutes)

## Error Handling

If ML services are unavailable:
- System falls back to zero scores
- Error message stored in database
- Teacher sees error indication in UI
- System continues to function normally

## File Requirements

The ML APIs can process:
- PDF files (.pdf)
- Word documents (.docx, .doc)
- Text files (.txt)
- Images with text (.jpg, .png) - via OCR

## Performance Notes

- Analysis can take 30 seconds to 5 minutes depending on file size
- Large files (>10MB) may timeout - consider file size limits
- Multiple submissions are processed sequentially
- Progress is shown in real-time via polling

## Troubleshooting

### "ML API service is not available"
- Ensure Python ML services are running: `cd ML && python unified_grading_api.py`
- Check that all Python dependencies are installed
- Verify ports 5000, 5001, 5002 are not blocked

### "Analysis taking too long" 
- Large files take more time to process
- Check ML service logs for processing status
- Consider increasing `ML_API_TIMEOUT`

### "File not found" errors
- Verify file upload completed successfully
- Check file permissions
- Ensure uploads directory exists and is writable

## API Endpoints

- `POST /grade/analyze` - Main grading endpoint
- `GET /health` - Health check for ML services
- `GET /api/homeworks/ml-health` - Backend health check (auth required)
- `GET /api/homeworks/grading-status/:homeworkID` - Check grading progress

## Integration Status

✅ **COMPLETED**: Backend now uses actual ML APIs instead of mock data
✅ **COMPLETED**: Comprehensive error handling and fallback mechanisms  
✅ **COMPLETED**: Real-time grading progress tracking
✅ **COMPLETED**: Detailed HTML reports with analysis results
✅ **COMPLETED**: Health monitoring and diagnostics
