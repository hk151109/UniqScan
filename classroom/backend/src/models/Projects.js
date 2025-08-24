const mongoose = require("mongoose");
const Schema = mongoose.Schema;

ProjectSchema = new Schema({
  user: {
    type: mongoose.Types.ObjectId,
    ref: "User",
  },
  createdAt: {
    type: Date,
    default: Date.now,
  },
  file: String,
  score: Number,
  similarityScore: { type: Number, default: null },
  aiGeneratedScore: { type: Number, default: null },
  plagiarismScore: { type: Number, default: null },
  reportPath: { type: String, default: null },
  gradingCompleted: { type: Boolean, default: false },
  gradingError: { type: String, default: null },
});

const Project = mongoose.model("Project", ProjectSchema);

module.exports = Project;
