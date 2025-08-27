const asyncHandler = require("express-async-handler");
const Homework = require("../models/Homework");
const Classroom = require("../models/Classroom");
const Project = require("../models/Projects");
const CustomError = require("../helpers/errors/CustomError");
const excelCreater = require("../helpers/excel/excelCreater");
const path = require("path");

const addHomework = asyncHandler(async (req, res, next) => {
  const { title, content, endTime } = req.body;
  const classroom = req.classroom;
  const homework = new Homework({ title, content, endTime });
  homework.teacher = req.user.id;
  homework.classroom = classroom._id;
  homework.appointedStudents = classroom.students;
  homework.save();
  await classroom.homeworks.push(homework._id);
  await classroom.save();
  return res.status(200).json({ success: true, data: homework });
});

const submitHomework = asyncHandler(async (req, res, next) => {
  const homework = req.homework;
  // Get classroom from req.classroom (set by middleware) or find it through homework
  let classroom = req.classroom;
  if (!classroom && homework.classroom) {
    classroom = await Classroom.findById(homework.classroom);
  }
  
  const user = req.user;
  // Fetch full user details including name
  const fullUser = await require('../models/User').findById(user.id);
  
  // Check if user has already submitted this homework
  const existingSubmission = await Project.findOne({ 
    user: user.id, 
    homework: homework._id 
  });
  
  // Ensure name and title are present
  const studentName = fullUser && fullUser.name ? fullUser.name.replace(/[^a-zA-Z0-9]/g, '_') : `student_${user && user.id ? user.id : 'unknown'}`;
  const classroomTitle = classroom && classroom.title ? classroom.title.replace(/[^a-zA-Z0-9]/g, '_') : `class_${classroom && classroom._id ? classroom._id : 'unknown'}`;
  const ext = req.file.originalname.split('.').pop();
  
  // For resubmissions, use the same filename as the original submission
  // For new submissions, create a new filename
  let filename;
  if (existingSubmission && existingSubmission.file) {
    // For resubmissions, keep the same base name but update extension if needed
    const existingFilename = existingSubmission.file;
    const existingBaseName = existingFilename.substring(0, existingFilename.lastIndexOf('.'));
    const newExt = req.file.originalname.split('.').pop();
    
    // Check if extension changed
    const existingExt = existingFilename.split('.').pop();
    if (existingExt.toLowerCase() !== newExt.toLowerCase()) {
      // Extension changed, need to delete old file and use new filename
      filename = `${existingBaseName}.${newExt}`;
      
      const fs = require('fs');
      // Delete the old file with different extension
      const oldFilePath = path.join(path.dirname(req.file.path), existingFilename);
      if (fs.existsSync(oldFilePath)) {
        try {
          fs.unlinkSync(oldFilePath);
          // console.log(`Deleted old file with different extension: ${oldFilePath}`);
        } catch (error) {
          console.error(`Could not delete old file: ${error.message}`);
        }
      }
    } else {
      // Same extension, use the same filename (will overwrite)
      filename = existingFilename;
    }
    // console.log(`Resubmission: Using filename: ${filename}`);
  } else {
    // New submission: create new filename
    filename = `${studentName}_${classroomTitle}_${homework._id}.${ext}`;
    // console.log(`New submission: Created filename: ${filename}`);
  }
  
  const fs = require('fs');
  const oldPath = req.file.path;
  const newPath = path.join(path.dirname(oldPath), filename);
  
  // console.log(`Original uploaded file path: ${oldPath}`);
  // console.log(`Target file path: ${newPath}`);
  // console.log(`File directory: ${path.dirname(oldPath)}`);
  
  // Check if the uploaded file exists
  if (!fs.existsSync(oldPath)) {
    return next(new CustomError(`Uploaded file not found: ${oldPath}`, 400));
  }
  
  // Rename the file to our desired filename
  try {
    fs.renameSync(oldPath, newPath);
    // console.log(`Successfully renamed file to: ${newPath}`);
  } catch (error) {
    console.error(`Failed to rename file: ${error.message}`);
    return next(new CustomError(`Failed to rename file: ${error.message}`, 500));
  }

  let project;
  let isResubmission = false;

  if (existingSubmission) {
    // This is a resubmission - update existing project
    isResubmission = true;
    
    // For resubmissions, we're using the same filename, so the new file will overwrite the old one
    // No need to delete the old file since it has the same name
    // console.log(`Resubmission: Will overwrite existing file: ${filename}`);
    
    // Update existing project
    existingSubmission.file = filename; // This should be the same as before
    existingSubmission.originalFileName = req.file.originalname;
    existingSubmission.updatedAt = new Date();
    existingSubmission.isResubmission = true;
    existingSubmission.submissionVersion += 1;
    existingSubmission.score = null;
    existingSubmission.similarityScore = null;
    existingSubmission.aiGeneratedScore = null;
    existingSubmission.plagiarismScore = null;
    existingSubmission.reportPath = null;
    existingSubmission.gradingCompleted = false;
    existingSubmission.gradingError = null;
    
    // console.log(`Updated project filename: ${filename}`);
    
    project = existingSubmission;
  } else {
    // Create new project for first submission
    project = new Project({ 
      user: user.id, 
      homework: homework._id,
      file: filename,
      originalFileName: req.file.originalname,
      isResubmission: false,
      submissionVersion: 1
    });
  }

  if (project.createdAt > homework.endTime && !isResubmission) {
    return next(new CustomError("Sorry, the deadline for homework is late.", 400));
  }
  
  // Save the project (whether new or updated)
  await project.save();
  
  // console.log(`Project saved with filename: ${project.file}`);
  // console.log(`Project ID: ${project._id}`);
  // console.log(`Is resubmission: ${isResubmission}`);

  // Ensure the project is in the homework's submitters list
  if (!homework.submitters.includes(project._id)) {
    homework.submitters.push(project._id);
  }
  
  // For new submissions, remove from appointedStudents
  if (!isResubmission && homework.appointedStudents.includes(user.id)) {
    homework.appointedStudents.splice(
      homework.appointedStudents.indexOf(user.id),
      1
    );
  }
  
  // Update homework's updatedAt timestamp so teachers know there's a change
  homework.updatedAt = new Date();
  
  // Save homework with updated submitters list
  await homework.save();

  // Enqueue grading job
  const { addGradingJob } = require('../services/gradingQueue');
  await addGradingJob({
    projectId: project._id.toString(),
    homeworkId: homework._id.toString(),
    filePath: newPath,
    userName: studentName,
  });
  
  return res.status(200).json({ 
    success: true, 
    data: homework,
    project: project,
    isResubmission: isResubmission,
    message: isResubmission ? 'Homework resubmitted successfully' : 'Homework submitted successfully'
  });
});

const getHomework = asyncHandler(async (req, res, next) => {
  const { homeworkID } = req.params;
  // console.log(`Fetching homework details for ID: ${homeworkID}`);
  
  const homework = await Homework.findById(homeworkID)
    .populate({
      path: "submitters",
      // populate: { path: "user", select: "-password" },
      populate: { path: "user", select: "name lastname" },
    })
    .populate({
      path: "appointedStudents",
      select: "name lastname",
    })
    .populate({
      path: "teacher",
      select: "name lastname",
    });
  
  // console.log(`Found homework with ${homework?.submitters?.length || 0} submitters`);
  if (homework?.submitters) {
    homework.submitters.forEach((submitter, index) => {
      // console.log(`Submitter ${index + 1}: ${submitter.user?.name}, File: ${submitter.file}, Updated: ${submitter.updatedAt}`);
    });
  }
  
  return res.status(200).json({ success: true, homework });
});

const updateHomework = asyncHandler(async (req, res, next) => {
  const homework = req.homework;
  const { title, content, endTime, score } = req.body;
  title ? (homework.title = title) : null;
  content ? (homework.title = content) : null;
  endTime ? (homework.title = endTime) : null;
  homework.save();
  return res.status(200).json({ success: true, data: homework });
});

const rateProject = asyncHandler(async (req, res, next) => {
  const { projectID } = req.params;
  const { score } = req.body;
  if (score < 0 && score > 100) {
    return next(new CustomError("Score must be between 0 and 100", 400));
  }

  const project = await Project.findById(projectID);
  if (!project) return next(new CustomError("Project not found", 400));
  project.score = score;
  project.save();
  return res.status(200).json({ success: true, data: project });
});

const exportScores = asyncHandler(async (req, res, next) => {
  const { classroomID, homeworkID } = req.params;
  const classroom = await Classroom.findById(classroomID);
  const homework = await Homework.findById(homeworkID)
    .populate({
      path: "submitters",
      populate: { path: "user", select: "name lastname" },
    })
    .populate({
      path: "appointedStudents",
      select: "name lastname",
    });

  let projects = [];
  homework.submitters.forEach((project) => {
    let { user, score } = project;
    let data = {};
    data["name"] = user.name;
    data["lastname"] = user.lastname;
    data["score"] = score;
    projects.push(data);
  });

  const excelFile = await excelCreater(classroom.title, homework.title, projects);
  const appPath = path.resolve();
  const filePath = "/public/uploads/excels";
  const myPath = path.join(appPath, filePath, excelFile);
  return res.status(200).sendFile(myPath);
});

const sendHomeworkFile = asyncHandler(async (req, res, next) => {
  const appPath = path.resolve();
  const filePath = "/public/uploads/homeworks";
  const { filename } = req.params;
  const myPath = path.join(appPath, filePath, filename);
  
  // console.log(`Attempting to send file: ${myPath}`);
  // console.log(`File exists: ${require('fs').existsSync(myPath)}`);
  
  if (!require('fs').existsSync(myPath)) {
    // console.log(`File not found: ${myPath}`);
    return next(new CustomError(`File not found: ${filename}`, 404));
  }
  
  return res.status(200).sendFile(myPath);
});

const deleteHomework = asyncHandler(async (req, res, next) => {
  const classroom = req.classroom;
  const homework = req.homework;
  if (req.user.id !== homework.teacher.toString()) {
    return next(new CustomError("You are not authorized", 400));
  }
  classroom.homeworks.splice(classroom.homeworks.indexOf(homework._id), 1);
  await classroom.save();
  await Homework.findByIdAndDelete(homework._id);
  return res.status(200).json({ success: true });
});

// Endpoint to get grading status
const getGradingStatus = asyncHandler(async (req, res, next) => {
  const { homeworkID } = req.params;
  const homework = await Homework.findById(homeworkID);
  if (!homework) return next(new CustomError('Homework not found', 404));
  const isReady = homework.similarityScore !== null && homework.aiGeneratedScore !== null && homework.plagiarismScore !== null && homework.reportPath !== null;
  res.json({ ready: isReady, similarityScore: homework.similarityScore, aiGeneratedScore: homework.aiGeneratedScore, plagiarismScore: homework.plagiarismScore, reportPath: homework.reportPath });
});

// Endpoint to check ML API health
const checkMLAPIHealth = asyncHandler(async (req, res, next) => {
  const { checkMLAPIHealth } = require('../services/gradingService');
  const health = await checkMLAPIHealth();
  res.json(health);
});

// Get student's submission status for a homework
const getMySubmission = asyncHandler(async (req, res, next) => {
  const { homeworkID } = req.params;
  const user = req.user;
  
  const submission = await Project.findOne({ 
    user: user.id, 
    homework: homeworkID 
  }).populate('homework', 'title endTime');
  
  if (!submission) {
    return res.status(200).json({ 
      success: true, 
      submitted: false, 
      message: 'No submission found' 
    });
  }
  
  return res.status(200).json({ 
    success: true, 
    submitted: true,
    submission: {
      _id: submission._id,
      file: submission.file,
      originalFileName: submission.originalFileName,
      createdAt: submission.createdAt,
      updatedAt: submission.updatedAt,
      score: submission.score,
      similarityScore: submission.similarityScore,
      aiGeneratedScore: submission.aiGeneratedScore,
      plagiarismScore: submission.plagiarismScore,
      gradingCompleted: submission.gradingCompleted,
      isResubmission: submission.isResubmission,
      submissionVersion: submission.submissionVersion,
      homework: submission.homework,
      // Status logic: only graded when teacher has given a manual score
      isGraded: submission.score !== null && submission.score !== undefined,
      teacherScore: submission.score
    }
  });
});

// Get all submissions by current student across all homeworks
const getMySubmissions = asyncHandler(async (req, res, next) => {
  const user = req.user;
  
  const submissions = await Project.find({ user: user.id })
    .populate('homework', 'title endTime classroom')
    .populate({
      path: 'homework',
      populate: {
        path: 'classroom',
        select: 'title accessCode'
      }
    })
    .sort({ updatedAt: -1 });
  
  return res.status(200).json({ 
    success: true, 
    submissions: submissions.map(sub => ({
      _id: sub._id,
      file: sub.file,
      originalFileName: sub.originalFileName,
      createdAt: sub.createdAt,
      updatedAt: sub.updatedAt,
      score: sub.score,
      similarityScore: sub.similarityScore,
      aiGeneratedScore: sub.aiGeneratedScore,
      plagiarismScore: sub.plagiarismScore,
      gradingCompleted: sub.gradingCompleted,
      isResubmission: sub.isResubmission,
      submissionVersion: sub.submissionVersion,
      homework: sub.homework,
      // Status logic: only graded when teacher has given a manual score
      isGraded: sub.score !== null && sub.score !== undefined,
      teacherScore: sub.score
    }))
  });
});

module.exports = {
  addHomework,
  submitHomework,
  getHomework,
  updateHomework,
  rateProject,
  exportScores,
  sendHomeworkFile,
  deleteHomework,
  getGradingStatus,
  checkMLAPIHealth,
  getMySubmission,
  getMySubmissions,
};
