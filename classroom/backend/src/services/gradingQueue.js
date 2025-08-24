const path = require('path');
const fs = require('fs');
const Project = require('../models/Projects');
const Homework = require('../models/Homework');
const Classroom = require('../models/Classroom');
const User = require('../models/User');
const { getGradingScores } = require('./gradingService');

// Simple in-memory queue for development
const jobQueue = [];
let isProcessing = false;

const addGradingJob = async (jobData) => {
  jobQueue.push(jobData);
  console.log(`Added grading job to queue. Queue length: ${jobQueue.length}`);
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
  console.log(`Processing grading job. Remaining in queue: ${jobQueue.length}`);
  
  try {
    const { projectId, homeworkId, filePath, userName } = job;
    
    // Fetch related data for context
    const project = await Project.findById(projectId).populate('user');
    const homework = await Homework.findById(homeworkId);
    const classroom = homework ? await Classroom.findById(homework.classroom) : null;
    
    if (!project) {
      throw new Error(`Project not found: ${projectId}`);
    }
    
    const studentInfo = {
      studentId: project.user._id,
      name: project.user.name || 'Unknown',
      lastname: project.user.lastname || ''
    };
    
    const homeworkInfo = {
      homeworkId: homeworkId,
      title: homework ? homework.title : 'Unknown Assignment'
    };
    
    const classroomInfo = classroom ? {
      name: classroom.title || classroom.accessCode || 'Unknown Classroom'
    } : null;
    
    // Add some delay to simulate processing
    setTimeout(async () => {
      try {
        console.log(`Starting ML grading analysis for student: ${studentInfo.name} ${studentInfo.lastname}`);
        
        const grading = await getGradingScores(filePath, studentInfo, homeworkInfo, classroomInfo);
        
        let finalReportPath = null;
        
        // Handle report generation
        if (grading.reportHtml) {
          const reportFileName = `report_${userName}_${projectId}.html`;
          const reportPath = path.join(__dirname, '../../public/uploads/processed_docs', reportFileName);
          
          // Ensure the processed_docs directory exists
          const reportDir = path.dirname(reportPath);
          if (!fs.existsSync(reportDir)) {
            fs.mkdirSync(reportDir, { recursive: true });
          }
          
          fs.writeFileSync(reportPath, grading.reportHtml);
          finalReportPath = `/uploads/processed_docs/${reportFileName}`;
          console.log(`Report generated: ${finalReportPath}`);
        } else if (grading.reportPath && fs.existsSync(grading.reportPath)) {
          // Use the ML API generated report
          const reportFileName = `report_${userName}_${projectId}.html`;
          const publicReportPath = path.join(__dirname, '../../public/uploads/processed_docs', reportFileName);
          
          const reportDir = path.dirname(publicReportPath);
          if (!fs.existsSync(reportDir)) {
            fs.mkdirSync(reportDir, { recursive: true });
          }
          
          fs.copyFileSync(grading.reportPath, publicReportPath);
          finalReportPath = `/uploads/processed_docs/${reportFileName}`;
          console.log(`ML report copied: ${finalReportPath}`);
        }
        
        // Update project with grading results
        const updateData = {
          similarityScore: grading.similarityScore,
          aiGeneratedScore: grading.aiGeneratedScore,
          plagiarismScore: grading.plagiarismScore,
          reportPath: finalReportPath,
          gradingCompleted: true,
          gradingError: grading.error || null
        };
        
        await Project.findByIdAndUpdate(projectId, updateData);
        
        console.log(`Grading completed for project ${projectId} - Similarity: ${grading.similarityScore}%, AI: ${grading.aiGeneratedScore}%, Plagiarism: ${grading.plagiarismScore}%`);
        
        if (grading.error) {
          console.warn(`Grading completed with errors: ${grading.error}`);
        }
        
      } catch (error) {
        console.error(`Grading failed for project ${projectId}:`, error);
        
        // Update project with error status
        await Project.findByIdAndUpdate(projectId, {
          similarityScore: 0,
          aiGeneratedScore: 0,
          plagiarismScore: 0,
          reportPath: null,
          gradingCompleted: true,
          gradingError: error.message
        });
      }
      
      // Process next job
      processQueue();
    }, 2000); // 2 second delay
    
  } catch (error) {
    console.error('Job processing error:', error);
    processQueue();
  }
};

module.exports = { addGradingJob };
