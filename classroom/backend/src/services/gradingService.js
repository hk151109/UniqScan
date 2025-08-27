const axios = require('axios');
const path = require('path');
const fs = require('fs');
const { getBackendUrl, getFileUrl } = require('../utils/urlDetection');

// ML API Configuration
const ML_API_BASE_URL = process.env.ML_API_BASE_URL || 'http://localhost:5000';
const ML_API_TIMEOUT = process.env.ML_API_TIMEOUT || 300000; // 5 minutes

const getGradingScores = async (filePath, studentInfo, homeworkInfo, classroomInfo) => {
  try {
    // Validate file exists
    if (!fs.existsSync(filePath)) {
      throw new Error(`File not found: ${filePath}`);
    } 


  // Convert file path to HTTP URL for ML API access (centralized util)
  const backendUrl = getBackendUrl();
  const fileUrl = getFileUrl(filePath);

    // Prepare request payload with HTTP URL for ML API
    const requestPayload = {
      student_id: studentInfo.studentId,
      student_name: `${studentInfo.name} ${studentInfo.lastname}`,
      file_url: fileUrl,
      assignment_id: homeworkInfo.homeworkId,
      classroom_name: classroomInfo ? classroomInfo.name : 'Unknown Classroom'
    };

    // console.log('Backend URL detected:', backendUrl);
    // console.log('Calling ML API with payload:', JSON.stringify(requestPayload, null, 2));

    // Call the unified ML API with retry logic for connection issues
    let response;
    let attempt = 0;
    const maxRetries = 2;
    
    while (attempt <= maxRetries) {
      try {
        response = await axios.post(
          `${ML_API_BASE_URL}/grade/analyze`,
          requestPayload,
          {
            timeout: ML_API_TIMEOUT,
            headers: {
              'Content-Type': 'application/json',
              'Connection': 'close'  // Force connection close after request
            }
          }
        );
        break; // Success, exit retry loop
      } catch (error) {
        attempt++;
        if (error.code === 'ECONNRESET' && attempt <= maxRetries) {
          console.log(`Connection reset, retrying attempt ${attempt}/${maxRetries}`);
          // Wait briefly before retry
          await new Promise(resolve => setTimeout(resolve, 1000 * attempt));
          continue;
        }
        throw error; // Re-throw if not a connection reset or max retries exceeded
      }
    }

    const results = response.data;
    // console.log('ML API response status:', results.status);

    // Extract scores from ML API response
    let similarityScore = 0;
    let aiGeneratedScore = 0;
    let plagiarismScore = 0;
    let reportHtml = '';

    if (results.status === 'completed' || results.status === 'partial') {
      // Extract similarity score
      if (results.similarity_analysis) {
        similarityScore = results.similarity_analysis.similarity_score || 0;
      }

      // Extract AI score
      if (results.ai_analysis) {
        aiGeneratedScore = results.ai_analysis.ai_percentage || 0;
      }

      // Extract plagiarism score (separate from similarity)
      if (results.plagiarism_analysis) {
        plagiarismScore = results.plagiarism_analysis.plagiarism_score || 0;
      } else {
        // Fallback: Use similarity score as plagiarism score if not provided separately
        plagiarismScore = similarityScore;
      }

      // Extract HTML report content from ML API response
      if (results.report_html && typeof results.report_html === 'string') {
        reportHtml = results.report_html;
        // console.log('Using HTML report from ML API response');
      } else {
        // console.log('No HTML report in ML API response, generating fallback');
        reportHtml = generateFallbackReport(similarityScore, aiGeneratedScore, plagiarismScore, results);
      }
    } else {
      // ML API didn't complete successfully, use fallback
      // console.log('ML API analysis incomplete, generating fallback report');
      reportHtml = generateFallbackReport(similarityScore, aiGeneratedScore, plagiarismScore, results);
    }

    // console.log(`Grading completed - Similarity: ${similarityScore}%, AI: ${aiGeneratedScore}%, Plagiarism: ${plagiarismScore}%`);

    return {
      similarityScore: parseFloat(similarityScore.toFixed(2)),
      aiGeneratedScore: parseFloat(aiGeneratedScore.toFixed(2)),
      plagiarismScore: parseFloat(plagiarismScore.toFixed(2)),
      reportHtml,
      reportPath: null, // We're using HTML content directly, no file path needed from ML API
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
    // console.log('Falling back to mock data due to ML API error');
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
    <html lang="en">
    <head>
        <title>Academic Integrity Analysis Report</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            body { 
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
                color: #333;
                background-color: #f8f9fa;
                margin: 0;
                padding: 20px;
            }
            .container {
                max-width: 900px;
                margin: 0 auto;
                background: white;
                border-radius: 12px;
                box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
                overflow: hidden;
            }
            .header {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 30px;
                text-align: center;
            }
            .header h1 {
                font-size: 2.2em;
                margin-bottom: 10px;
                font-weight: 600;
            }
            .header-info {
                display: flex;
                justify-content: space-between;
                margin-top: 20px;
                flex-wrap: wrap;
            }
            .header-info div {
                text-align: center;
                flex: 1;
                min-width: 150px;
                margin: 5px 0;
            }
            .header-info strong {
                display: block;
                font-size: 0.9em;
                opacity: 0.8;
            }
            .content {
                padding: 40px;
            }
            .score-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
                margin-bottom: 40px;
            }
            .score-card {
                background: white;
                border-radius: 12px;
                padding: 25px;
                text-align: center;
                box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
                border-top: 4px solid;
                transition: transform 0.3s ease;
            }
            .score-card:hover {
                transform: translateY(-2px);
            }
            .score-card.low { border-top-color: #27ae60; }
            .score-card.medium { border-top-color: #f39c12; }
            .score-card.high { border-top-color: #e74c3c; }
            .score-value {
                font-size: 2.5em;
                font-weight: 700;
                margin: 10px 0;
            }
            .score-value.low { color: #27ae60; }
            .score-value.medium { color: #f39c12; }
            .score-value.high { color: #e74c3c; }
            .score-label {
                font-size: 1.1em;
                font-weight: 600;
                color: #2c3e50;
                margin-bottom: 8px;
            }
            .score-description {
                font-size: 0.9em;
                color: #7f8c8d;
                line-height: 1.4;
            }
            .analysis-section {
                background-color: #f8f9fa;
                border-radius: 8px;
                padding: 30px;
                margin: 30px 0;
                border-left: 4px solid #3498db;
            }
            .section-title {
                color: #2c3e50;
                font-size: 1.4em;
                font-weight: 600;
                margin-bottom: 20px;
                display: flex;
                align-items: center;
            }
            .section-icon {
                margin-right: 10px;
                font-size: 1.2em;
            }
            .analysis-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin-bottom: 20px;
            }
            .metric {
                background: white;
                padding: 15px;
                border-radius: 6px;
                border: 1px solid #dee2e6;
            }
            .metric-label {
                font-weight: 600;
                color: #495057;
                font-size: 0.9em;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }
            .metric-value {
                font-size: 1.2em;
                color: #2c3e50;
                margin-top: 5px;
            }
            .recommendations {
                background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
                color: white;
                border-radius: 8px;
                padding: 30px;
                margin: 30px 0;
            }
            .recommendations h2 {
                font-size: 1.4em;
                margin-bottom: 20px;
                display: flex;
                align-items: center;
            }
            .recommendations ul {
                list-style: none;
                padding: 0;
            }
            .recommendations li {
                padding: 10px 0;
                display: flex;
                align-items: flex-start;
                border-bottom: 1px solid rgba(255,255,255,0.2);
            }
            .recommendations li:last-child {
                border-bottom: none;
            }
            .recommendations li::before {
                content: "▸";
                color: #fdcb6e;
                font-weight: bold;
                margin-right: 10px;
                margin-top: 2px;
            }
            .footer {
                background-color: #2d3436;
                color: #b2bec3;
                padding: 20px;
                text-align: center;
            }
            .status-badge {
                display: inline-block;
                padding: 8px 16px;
                border-radius: 20px;
                font-size: 0.8em;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 0.5px;
                margin-top: 10px;
            }
            .status-completed {
                background-color: #d4edda;
                color: #155724;
            }
            .status-partial {
                background-color: #fff3cd;
                color: #856404;
            }
            @media (max-width: 600px) {
                body {
                    padding: 10px;
                }
                .header {
                    padding: 20px;
                }
                .header h1 {
                    font-size: 1.8em;
                }
                .content {
                    padding: 20px;
                }
                .score-grid {
                    grid-template-columns: 1fr;
                }
                .header-info {
                    flex-direction: column;
                }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>📊 Academic Integrity Analysis Report</h1>
                <div class="header-info">
                    <div>
                        <strong>Generated</strong>
                        ${timestamp}
                    </div>
                    <div>
                        <strong>Status</strong>
                        <span class="status-badge ${mlResults.status === 'completed' ? 'status-completed' : 'status-partial'}">
                            ${mlResults.status || 'Analysis Completed'}
                        </span>
                    </div>
                </div>
            </div>
            
            <div class="content">
                <div class="score-grid">
                    <div class="score-card ${getScoreClass(similarityScore)}">
                        <div class="score-label">📋 Similarity Score</div>
                        <div class="score-value ${getScoreClass(similarityScore)}">${similarityScore}%</div>
                        <div class="score-description">
                            ${getSimilarityDescription(similarityScore)}
                        </div>
                    </div>
                    
                    <div class="score-card ${getScoreClass(aiGeneratedScore)}">
                        <div class="score-label">🤖 AI Generated Content</div>
                        <div class="score-value ${getScoreClass(aiGeneratedScore)}">${aiGeneratedScore}%</div>
                        <div class="score-description">
                            ${getAiDescription(aiGeneratedScore)}
                        </div>
                    </div>
                    
                    <div class="score-card ${getScoreClass(plagiarismScore)}">
                        <div class="score-label">⚠️ Plagiarism Risk</div>
                        <div class="score-value ${getScoreClass(plagiarismScore)}">${plagiarismScore}%</div>
                        <div class="score-description">
                            ${getPlagiarismDescription(plagiarismScore)}
                        </div>
                    </div>
                </div>
                
                ${mlResults.similarity_analysis ? `
                <div class="analysis-section">
                    <h2 class="section-title">
                        <span class="section-icon">🔍</span>
                        Similarity Analysis Details
                    </h2>
                    <div class="analysis-grid">
                        <div class="metric">
                            <div class="metric-label">Total Comparisons</div>
                            <div class="metric-value">${mlResults.similarity_analysis.total_comparisons || 0}</div>
                        </div>
                        <div class="metric">
                            <div class="metric-label">Matches Found</div>
                            <div class="metric-value">${mlResults.similarity_analysis.detailed_results ? mlResults.similarity_analysis.detailed_results.length : 0}</div>
                        </div>
                    </div>
                </div>
                ` : ''}
                
                ${mlResults.ai_analysis ? `
                <div class="analysis-section">
                    <h2 class="section-title">
                        <span class="section-icon">🧠</span>
                        AI Content Analysis
                    </h2>
                    <div class="analysis-grid">
                        <div class="metric">
                            <div class="metric-label">Interpretation</div>
                            <div class="metric-value">${mlResults.ai_analysis.interpretation || 'Standard Analysis'}</div>
                        </div>
                        <div class="metric">
                            <div class="metric-label">Chunks Analyzed</div>
                            <div class="metric-value">${mlResults.ai_analysis.chunks_analyzed || 0}</div>
                        </div>
                    </div>
                </div>
                ` : ''}
                
                ${mlResults.errors && mlResults.errors.length > 0 ? `
                <div class="analysis-section" style="border-left-color: #e74c3c; background-color: #fdf2f2;">
                    <h2 class="section-title" style="color: #e74c3c;">
                        <span class="section-icon">⚠️</span>
                        Analysis Notices
                    </h2>
                    <ul style="list-style: none; padding: 0;">
                    ${mlResults.errors.map(error => `<li style="padding: 8px 0; color: #721c24;">• ${error}</li>`).join('')}
                    </ul>
                </div>
                ` : ''}
                
                <div class="recommendations">
                    <h2>💡 Analysis Summary & Recommendations</h2>
                    ${generateRecommendations(similarityScore, aiGeneratedScore)}
                </div>
            </div>

            <div class="footer">
                <p><strong>UniqScan Academic Integrity System</strong> | Automated Analysis Report</p>
                <p style="font-size: 0.8em; margin-top: 10px; opacity: 0.8;">
                    This report is generated automatically. For questions about the analysis, contact your instructor.
                </p>
            </div>
        </div>
    </body>
    </html>
  `;
};

const generateErrorReport = (errorMessage) => {
  const timestamp = new Date().toLocaleString();
  
  return `
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <title>Analysis Report - Service Unavailable</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            body { 
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
                color: #333;
                background-color: #f8f9fa;
                margin: 0;
                padding: 20px;
            }
            .container {
                max-width: 800px;
                margin: 0 auto;
                background: white;
                border-radius: 12px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                overflow: hidden;
            }
            .header {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 30px;
                text-align: center;
            }
            .header h1 {
                font-size: 2.2em;
                margin-bottom: 10px;
                font-weight: 600;
            }
            .header p {
                font-size: 1.1em;
                opacity: 0.9;
            }
            .content {
                padding: 40px;
            }
            .status-card {
                background-color: #fff3cd;
                border: 1px solid #ffeaa7;
                border-left: 5px solid #fdcb6e;
                border-radius: 8px;
                padding: 25px;
                margin-bottom: 30px;
                text-align: center;
            }
            .status-icon {
                font-size: 3em;
                color: #e17055;
                margin-bottom: 15px;
            }
            .status-title {
                font-size: 1.4em;
                font-weight: 600;
                color: #2d3436;
                margin-bottom: 10px;
            }
            .status-message {
                color: #636e72;
                font-size: 1.1em;
                line-height: 1.5;
            }
            .info-section {
                background-color: #f8f9fa;
                border-radius: 8px;
                padding: 25px;
                margin: 20px 0;
            }
            .info-title {
                color: #2d3436;
                font-size: 1.2em;
                font-weight: 600;
                margin-bottom: 15px;
                display: flex;
                align-items: center;
            }
            .info-title::before {
                content: "ℹ️";
                margin-right: 10px;
                font-size: 1.2em;
            }
            .info-list {
                list-style: none;
                padding: 0;
            }
            .info-list li {
                padding: 8px 0;
                border-bottom: 1px solid #e9ecef;
                display: flex;
                align-items: center;
            }
            .info-list li:last-child {
                border-bottom: none;
            }
            .info-list li::before {
                content: "▸";
                color: #6c5ce7;
                font-weight: bold;
                margin-right: 10px;
            }
            .contact-section {
                background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
                color: white;
                border-radius: 8px;
                padding: 25px;
                text-align: center;
                margin-top: 30px;
            }
            .contact-title {
                font-size: 1.3em;
                font-weight: 600;
                margin-bottom: 10px;
            }
            .footer {
                background-color: #2d3436;
                color: #b2bec3;
                padding: 20px;
                text-align: center;
                font-size: 0.9em;
            }
            .error-details {
                background-color: #ffe6e6;
                border: 1px solid #ffb3b3;
                border-radius: 6px;
                padding: 15px;
                margin: 20px 0;
                font-family: 'Courier New', monospace;
                font-size: 0.9em;
                color: #721c24;
                word-break: break-word;
            }
            @media (max-width: 600px) {
                body {
                    padding: 10px;
                }
                .header {
                    padding: 20px;
                }
                .header h1 {
                    font-size: 1.8em;
                }
                .content {
                    padding: 20px;
                }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Academic Integrity Analysis</h1>
                <p>Report Generated: ${timestamp}</p>
            </div>
            
            <div class="content">
                <div class="status-card">
                    <div class="status-icon">⚠️</div>
                    <div class="status-title">Analysis Service Temporarily Unavailable</div>
                    <div class="status-message">
                        We're unable to complete the academic integrity analysis at this time due to a service interruption. 
                        Your submission has been received and will be analyzed once the service is restored.
                    </div>
                </div>

                <div class="info-section">
                    <div class="info-title">What This Means</div>
                    <ul class="info-list">
                        <li>Your homework submission has been successfully received and stored</li>
                        <li>The automated analysis tools are temporarily offline</li>
                        <li>Your instructor has been notified of the service status</li>
                        <li>Manual review processes remain available if needed</li>
                    </ul>
                </div>

                <div class="info-section">
                    <div class="info-title">Next Steps</div>
                    <ul class="info-list">
                        <li>No action is required from you at this time</li>
                        <li>Analysis will resume automatically when service is restored</li>
                        <li>You will be notified once results are available</li>
                        <li>Contact your instructor if you have urgent concerns</li>
                    </ul>
                </div>

                <div class="error-details">
                    <strong>Technical Details:</strong><br>
                    ${errorMessage || 'Service connectivity issue detected'}
                </div>

                <div class="contact-section">
                    <div class="contact-title">Need Assistance?</div>
                    <p>If you have questions about your submission or need immediate assistance, 
                    please contact your instructor or system administrator.</p>
                </div>
            </div>

            <div class="footer">
                <p>UniqScan Academic Integrity System | Generated automatically</p>
            </div>
        </div>
    </body>
    </html>
  `;
};

const getScoreClass = (score) => {
  if (score >= 70) return 'high';
  if (score >= 40) return 'medium';
  return 'low';
};

const getSimilarityDescription = (score) => {
  if (score >= 70) return 'High similarity detected - requires immediate review';
  if (score >= 40) return 'Moderate similarity - manual review recommended';
  return 'Low similarity - appears acceptable';
};

const getAiDescription = (score) => {
  if (score >= 70) return 'High probability of AI-generated content';
  if (score >= 40) return 'Possible AI assistance detected';
  return 'Likely human-authored content';
};

const getPlagiarismDescription = (score) => {
  if (score >= 70) return 'High risk - immediate investigation required';
  if (score >= 40) return 'Moderate risk - closer examination needed';
  return 'Low risk - within acceptable parameters';
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
