const path = require('path');
const fs = require('fs');
const Project = require('../models/Projects');
const { getGradingScores } = require('./gradingService');

// Simple in-memory queue for development
const jobQueue = [];
let isProcessing = false;

const addGradingJob = async (jobData) => {
  jobQueue.push(jobData);
  if (!isProcessing) {
    processQueue();
  }
};

const processQueue = async () => {
  if (jobQueue.length === 0) {
    isProcessing = false;
    return;
  }
  
  isProcessing = true;
  const job = jobQueue.shift();
  
  try {
    const { projectId, homeworkId, filePath, userName } = job;
    
    // Add some delay to simulate processing
    setTimeout(async () => {
      try {
        const grading = await getGradingScores(filePath);
        const reportFileName = `report_${userName}_${projectId}.html`;
        const reportPath = path.join(__dirname, '../../../public/uploads/processed_docs', reportFileName);
        
        // Ensure the processed_docs directory exists
        const reportDir = path.dirname(reportPath);
        if (!fs.existsSync(reportDir)) {
          fs.mkdirSync(reportDir, { recursive: true });
        }
        
        fs.writeFileSync(reportPath, grading.reportHtml);
        
        await Project.findByIdAndUpdate(projectId, {
          similarityScore: grading.similarityScore,
          aiGeneratedScore: grading.aiGeneratedScore,
          plagiarismScore: grading.plagiarismScore,
          reportPath: `/public/uploads/processed_docs/${reportFileName}`,
        });
        
        console.log(`Grading completed for project ${projectId}`);
      } catch (error) {
        console.error(`Grading failed for project ${projectId}:`, error);
      }
      
      // Process next job
      processQueue();
    }, 3000); // 3 second delay to simulate processing
    
  } catch (error) {
    console.error('Job processing error:', error);
    processQueue();
  }
};

module.exports = { addGradingJob };
