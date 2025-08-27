const router = require("express").Router();
const homeworkController = require("../controllers/homeworkController");
const { isAuth, isTeacher, isStudent } = require("../middlewares/auth/auth");
const {
  classroomCheck,
  homeworkCheck,
} = require("../middlewares/database/checkExist");
const uploadFile = require("../middlewares/assets/uploadFile");

// /api/homeworks

router.post(
  "/:classroomID",
  isAuth,
  isTeacher,
  classroomCheck,
  homeworkController.addHomework
);

router.post(
  "/submit/:homeworkID",
  isAuth,
  isStudent,
  homeworkCheck,
  uploadFile.single("homework"),
  homeworkController.submitHomework
);

// Student submission status endpoints - Put these BEFORE the /:homeworkID route
router.get(
  "/my-submissions/all",
  isAuth,
  isStudent,
  homeworkController.getMySubmissions
);

router.get(
  "/my-submission/:homeworkID",
  isAuth,
  isStudent,
  homeworkCheck,
  homeworkController.getMySubmission
);

router.get(
  "/:homeworkID",
  isAuth,
  homeworkCheck,
  homeworkController.getHomework
);

router.patch(
  "/:homeworkID",
  isAuth,
  homeworkCheck,
  homeworkController.updateHomework
);

router.delete(
  "/:classroomID/:homeworkID",
  isAuth,
  isTeacher,
  classroomCheck,
  homeworkCheck,
  homeworkController.deleteHomework
);

router.patch(
  "/project/:projectID",
  isAuth,
  isTeacher,
  homeworkController.rateProject
);

router.get("/download/:filename", homeworkController.sendHomeworkFile);

router.get("/score/:classroomID/:homeworkID", homeworkController.exportScores);

router.get(
  "/grading-status/:homeworkID",
  homeworkController.getGradingStatus
);

router.get(
  "/ml-health",
  isAuth,
  isTeacher,
  homeworkController.checkMLAPIHealth
);

module.exports = router;
