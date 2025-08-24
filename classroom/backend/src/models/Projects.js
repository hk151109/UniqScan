const mongoose = require("mongoose");
const Schema = mongoose.Schema;

ProjectSchema = new Schema({
  user: {
    type: mongoose.Types.ObjectId,
    ref: "User",
  },
  homework: {
    type: mongoose.Types.ObjectId,
    ref: "Homework",
    required: true,
  },
  createdAt: {
    type: Date,
    default: Date.now,
  },
  updatedAt: {
    type: Date,
    default: Date.now,
  },
  file: String,
  originalFileName: String,
  score: Number,
  similarityScore: { type: Number, default: null },
  aiGeneratedScore: { type: Number, default: null },
  plagiarismScore: { type: Number, default: null },
  reportPath: { type: String, default: null },
  gradingCompleted: { type: Boolean, default: false },
  gradingError: { type: String, default: null },
  isResubmission: { type: Boolean, default: false },
  submissionVersion: { type: Number, default: 1 },
});

const Project = mongoose.model("Project", ProjectSchema);

module.exports = Project;
