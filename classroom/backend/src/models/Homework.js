const mongoose = require("mongoose");
const Schema = mongoose.Schema;

HomeworkSchema = new Schema({
  title: {
    type: String,
    required: true,
  },
  content: String,
  endTime: Date,
  createdAt: {
    type: Date,
    default: Date.now,
  },
  updatedAt: {
    type: Date,
    default: Date.now,
  },
  teacher: {
    type: mongoose.Types.ObjectId,
    ref: "User",
  },
  classroom: {
    type: mongoose.Types.ObjectId,
    ref: "Classroom",
  },
  submitters: [
    {
      type: mongoose.Types.ObjectId,
      ref: "Project",
    },
  ],
  appointedStudents: [
    {
      type: mongoose.Types.ObjectId,
      ref: "User",
    },
  ],
  scoreTable: String,
});

// Update the updatedAt field before saving
HomeworkSchema.pre('save', function(next) {
  if (this.isModified() && !this.isNew) {
    this.updatedAt = new Date();
  }
  next();
});

const Homework = mongoose.model("Homework", HomeworkSchema);

module.exports = Homework;
