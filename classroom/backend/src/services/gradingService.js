const axios = require('axios');
const path = require('path');
const fs = require('fs');

// ML API Configuration
const ML_API_BASE_URL = process.env.ML_API_BASE_URL || 'http://localhost:5000';
const ML_API_TIMEOUT = process.env.ML_API_TIMEOUT || 300000; // 5 minutes

const getGradingScores = async (filePath, studentInfo, homeworkInfo, classroomInfo) => {
  try {
    // Validate file exists
    if (!fs.existsSync(filePath)) {
      throw new Error(`File not found: ${filePath}`);
    }

    // Get absolute path for ML API
    const absoluteFilePath = path.resolve(filePath);

    // Prepare request payload
    const requestPayload = {
      student_id: studentInfo.studentId,
      student_name: `${studentInfo.name} ${studentInfo.lastname}`,
      file_path: absoluteFilePath,
      assignment_id: homeworkInfo.homeworkId,
      classroom_name: classroomInfo ? classroomInfo.name : 'Unknown Classroom'
    };

    console.log('Calling ML API with payload:', JSON.stringify(requestPayload, null, 2));

    // Call the unified ML API
    const response = await axios.post(
      `${ML_API_BASE_URL}/grade/analyze`,
      requestPayload,
      {
        timeout: ML_API_TIMEOUT,
        headers: {
          'Content-Type': 'application/json'
        }
      }
    );

    const results = response.data;
    console.log('ML API response status:', results.status);

    // Extract scores from ML API response
    let similarityScore = 0;
    let aiGeneratedScore = 0;
    let plagiarismScore = 0;
    let reportPath = null;

    if (results.status === 'completed' || results.status === 'partial') {
      // Extract similarity score
      if (results.similarity_analysis) {
        similarityScore = results.similarity_analysis.similarity_score || 0;
        // Use similarity score as plagiarism score for backward compatibility
        plagiarismScore = similarityScore;
      }

      // Extract AI score
      if (results.ai_analysis) {
        aiGeneratedScore = results.ai_analysis.ai_percentage || 0;
      }

      // Get unified report path
      reportPath = results.unified_report_path;
    }

    // Generate fallback HTML report if needed
    let reportHtml = '';
    if (reportPath && fs.existsSync(reportPath)) {
      try {
        reportHtml = fs.readFileSync(reportPath, 'utf8');
      } catch (error) {
        console.error('Error reading report file:', error);
        reportHtml = generateFallbackReport(similarityScore, aiGeneratedScore, plagiarismScore, results);
      }
    } else {
      reportHtml = generateFallbackReport(similarityScore, aiGeneratedScore, plagiarismScore, results);
    }

    console.log(`Grading completed - Similarity: ${similarityScore}%, AI: ${aiGeneratedScore}%, Plagiarism: ${plagiarismScore}%`);

    return {
      similarityScore: parseFloat(similarityScore.toFixed(2)),
      aiGeneratedScore: parseFloat(aiGeneratedScore.toFixed(2)),
      plagiarismScore: parseFloat(plagiarismScore.toFixed(2)),
      reportHtml,
      reportPath,
      mlApiResponse: results // Include full response for debugging
    };

  } catch (error) {
    console.error('Error in ML API call:', error);

    // Handle specific error types
    if (error.code === 'ECONNREFUSED') {
      throw new Error('ML API service is not available. Please ensure the Python ML services are running on port 5000.');
    }

    if (error.code === 'ECONNABORTED') {
      throw new Error('ML API request timed out. The analysis is taking longer than expected.');
    }

    // For other errors, return mock data with error indication
    console.log('Falling back to mock data due to ML API error');
    return {
      similarityScore: 0,
      aiGeneratedScore: 0,
      plagiarismScore: 0,
      reportHtml: generateErrorReport(error.message),
      reportPath: null,
      error: error.message
    };
  }
};

const generateFallbackReport = (similarityScore, aiGeneratedScore, plagiarismScore, mlResults) => {
  const timestamp = new Date().toLocaleString();
  
  return `
    <!DOCTYPE html>
    <html>
    <head>
        <title>Academic Integrity Analysis Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .header { background-color: #f8f9fa; padding: 20px; border-radius: 5px; margin-bottom: 20px; }
            .score { font-size: 24px; font-weight: bold; margin: 10px 0; }
            .high { color: #e74c3c; }
            .medium { color: #f39c12; }
            .low { color: #27ae60; }
            .section { margin: 20px 0; padding: 15px; border-left: 3px solid #3498db; }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Academic Integrity Analysis Report</h1>
            <p><strong>Generated:</strong> ${timestamp}</p>
            <p><strong>Status:</strong> ${mlResults.status || 'Analysis Completed'}</p>
        </div>
        
        <div class="section">
            <h2>Summary Scores</h2>
            <p>Similarity Score: <span class="score ${getScoreClass(similarityScore)}">${similarityScore}%</span></p>
            <p>AI Generated Score: <span class="score ${getScoreClass(aiGeneratedScore)}">${aiGeneratedScore}%</span></p>
            <p>Plagiarism Score: <span class="score ${getScoreClass(plagiarismScore)}">${plagiarismScore}%</span></p>
        </div>
        
        ${mlResults.similarity_analysis ? `
        <div class="section">
            <h2>Similarity Analysis</h2>
            <p><strong>Total Comparisons:</strong> ${mlResults.similarity_analysis.total_comparisons || 0}</p>
            <p><strong>Matches Found:</strong> ${mlResults.similarity_analysis.detailed_results ? mlResults.similarity_analysis.detailed_results.length : 0}</p>
        </div>
        ` : ''}
        
        ${mlResults.ai_analysis ? `
        <div class="section">
            <h2>AI Content Analysis</h2>
            <p><strong>Interpretation:</strong> ${mlResults.ai_analysis.interpretation || 'N/A'}</p>
            <p><strong>Chunks Analyzed:</strong> ${mlResults.ai_analysis.chunks_analyzed || 0}</p>
        </div>
        ` : ''}
        
        ${mlResults.errors && mlResults.errors.length > 0 ? `
        <div class="section" style="border-left-color: #e74c3c;">
            <h2>Errors</h2>
            <ul>
            ${mlResults.errors.map(error => `<li>${error}</li>`).join('')}
            </ul>
        </div>
        ` : ''}
        
        <div class="section">
            <h2>Recommendations</h2>
            ${generateRecommendations(similarityScore, aiGeneratedScore)}
        </div>
    </body>
    </html>
  `;
};

const generateErrorReport = (errorMessage) => {
  const timestamp = new Date().toLocaleString();
  
  return `
    <!DOCTYPE html>
    <html>
    <head>
        <title>Analysis Error Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .error { background-color: #f8d7da; border: 1px solid #f5c6cb; color: #721c24; padding: 15px; border-radius: 5px; }
        </style>
    </head>
    <body>
        <h1>Analysis Error Report</h1>
        <p><strong>Generated:</strong> ${timestamp}</p>
        
        <div class="error">
            <h2>Error Details</h2>
            <p>${errorMessage}</p>
            <h3>Possible Solutions:</h3>
            <ul>
                <li>Ensure the ML API services are running (run start_services.bat in the ML directory)</li>
                <li>Check that Python dependencies are installed</li>
                <li>Verify the file path is accessible</li>
                <li>Check network connectivity to ML services</li>
            </ul>
        </div>
        
        <p><em>Please contact your system administrator if this error persists.</em></p>
    </body>
    </html>
  `;
};

const getScoreClass = (score) => {
  if (score >= 70) return 'high';
  if (score >= 40) return 'medium';
  return 'low';
};

const generateRecommendations = (similarityScore, aiGeneratedScore) => {
  const recommendations = [];
  
  if (similarityScore >= 70) {
    recommendations.push('🚨 High similarity detected - immediate review required');
  } else if (similarityScore >= 40) {
    recommendations.push('⚠️ Moderate similarity - manual review recommended');
  } else {
    recommendations.push('✅ Low similarity - appears acceptable');
  }
  
  if (aiGeneratedScore >= 70) {
    recommendations.push('🤖 High AI content probability - investigate further');
  } else if (aiGeneratedScore >= 40) {
    recommendations.push('🔍 Possible AI assistance - consider student interview');
  } else {
    recommendations.push('👤 Likely human-written content');
  }
  
  return `<ul>${recommendations.map(rec => `<li>${rec}</li>`).join('')}</ul>`;
};

// Health check function to verify ML API availability
const checkMLAPIHealth = async () => {
  try {
    const response = await axios.get(`${ML_API_BASE_URL}/health`, { timeout: 5000 });
    return {
      status: 'healthy',
      data: response.data
    };
  } catch (error) {
    return {
      status: 'unhealthy',
      error: error.message
    };
  }
};

module.exports = { 
  getGradingScores, 
  checkMLAPIHealth 
};
