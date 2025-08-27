import { useState, useEffect } from "react";
import { Offcanvas, Button, Form, Alert, Toast, Card, Badge } from "react-bootstrap";
import { useFormik } from "formik";
import { BsFillCapslockFill, BsCheckCircleFill, BsClockFill } from "react-icons/bs";
import { fetchSubmitHomework, fetchMySubmission } from "../../api/homeworkApi";

const SubmitHomeworkOffCanvas = ({ homeworkID }) => {
  const [show, setShow] = useState(false);
  const [toastShow, setToastShow] = useState(false);
  const [toastMessage, setToastMessage] = useState("");
  const [submissionStatus, setSubmissionStatus] = useState(null);
  const [loading, setLoading] = useState(true);

  const handleClose = () => setShow(false);
  const handleShow = () => setShow(true);

  // Fetch submission status when component mounts
  useEffect(() => {
    const fetchSubmissionStatus = async () => {
      try {
        setLoading(true);
        const response = await fetchMySubmission(homeworkID);
        setSubmissionStatus(response.data);
      } catch (error) {
        console.error("Error fetching submission status:", error);
        setSubmissionStatus({ submitted: false });
      } finally {
        setLoading(false);
      }
    };

    fetchSubmissionStatus();
  }, [homeworkID, toastShow]); // Refetch when toast closes (after submission)

  const formik = useFormik({
    initialValues: {
      homework: null,
    },
    onSubmit: async (values, bag) => {
      try {
        const formData = new FormData(); // Create FormData inside the submit handler
        formData.append("homework", values.homework);
        const response = await fetchSubmitHomework(homeworkID, formData);
        setShow(false);
        
        // Set appropriate message based on whether it's a resubmission
        if (response.data.isResubmission) {
          setToastMessage("Homework resubmitted successfully!");
        } else {
          setToastMessage("Homework submitted successfully!");
        }
        setToastShow(true);
        
        // Reset form
        formik.resetForm();
      } catch (e) {
        bag.setErrors({ general: e.response?.data.message });
      }
    },
  });

  const handleChangeFile = (e) => {
    formik.setFieldValue("homework", e.target.files[0]);
  };

  const formatDate = (dateString) => {
    return new Date(dateString).toLocaleString();
  };

  const getSubmissionStatusBadge = () => {
    if (loading) return <Badge bg="secondary">Loading...</Badge>;
    
    if (!submissionStatus?.submitted) {
      return <Badge bg="warning">Not Submitted</Badge>;
    }
    
    const submission = submissionStatus.submission;
    // Check if teacher has given a manual score
    if (submission.isGraded && submission.teacherScore !== null && submission.teacherScore !== undefined) {
      return (
        <div className="d-flex align-items-center gap-2">
          <Badge bg="success">Graded</Badge>
          <Badge bg="primary">{submission.teacherScore}/100</Badge>
        </div>
      );
    } else if (submission.gradingCompleted) {
      return <Badge bg="info">Submitted - AI Analysis Complete</Badge>;
    } else {
      return <Badge bg="info">Submitted - Analysis in Progress</Badge>;
    }
  };

  return (
    <>
      <div className="d-flex align-items-center gap-2">
        <Button size="sm" onClick={handleShow}>
          <BsFillCapslockFill className="me-2" />
          {submissionStatus?.submitted ? "View/Resubmit" : "Submit Homework"}
        </Button>
        {getSubmissionStatusBadge()}
      </div>

      <Offcanvas
        show={show}
        onHide={handleClose}
        className="w-75 h-75 mx-auto p-5"
        placement="top"
      >
        <Offcanvas.Header closeButton={true}>
          <Offcanvas.Title>
            {submissionStatus?.submitted ? "Homework Submission Status" : "Homework Upload"}
          </Offcanvas.Title>
        </Offcanvas.Header>
        <Offcanvas.Body className="mt-2">
          {/* Show current submission status */}
          {submissionStatus?.submitted && (
            <Card className="mb-4">
              <Card.Header className="d-flex align-items-center justify-content-between">
                <h6 className="mb-0">Current Submission</h6>
                <div className="d-flex align-items-center gap-2">
                  <BsCheckCircleFill className="text-success" />
                  {submissionStatus.submission.isResubmission && (
                    <Badge bg="info">Version {submissionStatus.submission.submissionVersion}</Badge>
                  )}
                </div>
              </Card.Header>
              <Card.Body>
                <div className="row">
                  <div className="col-md-6">
                    <p><strong>File:</strong> {submissionStatus.submission.originalFileName}</p>
                    <p><strong>Submitted:</strong> {formatDate(submissionStatus.submission.createdAt)}</p>
                    {submissionStatus.submission.updatedAt !== submissionStatus.submission.createdAt && (
                      <p><strong>Last Updated:</strong> {formatDate(submissionStatus.submission.updatedAt)}</p>
                    )}
                  </div>
                  <div className="col-md-6">
                    <p>
                      <strong>Status:</strong> {" "}
                      {submissionStatus.submission.isGraded && submissionStatus.submission.teacherScore !== null ? (
                        <div className="d-flex align-items-center gap-2">
                          <Badge bg="success">Graded</Badge>
                          <Badge bg="primary">{submissionStatus.submission.teacherScore}/100</Badge>
                        </div>
                      ) : submissionStatus.submission.gradingCompleted ? (
                        <Badge bg="info">Submitted - AI Analysis Complete</Badge>
                      ) : (
                        <div className="d-flex align-items-center gap-1">
                          <BsClockFill />
                          <Badge bg="info">Submitted - Analysis in Progress</Badge>
                        </div>
                      )}
                    </p>
                    {submissionStatus.submission.isGraded && submissionStatus.submission.teacherScore !== null && (
                      <Alert variant="success" className="py-2 mb-2">
                        <i className="fas fa-user-graduate me-2"></i>
                        Teacher has reviewed and scored your submission!
                      </Alert>
                    )}
                    {submissionStatus.submission.gradingCompleted && !submissionStatus.submission.isGraded && (
                      <Alert variant="info" className="py-2 mb-2">
                        <i className="fas fa-robot me-2"></i>
                        AI analysis complete. Awaiting teacher review and scoring.
                      </Alert>
                    )}
                    {submissionStatus.submission.gradingCompleted && (
                      <>
                        {/* Show teacher score prominently if available */}
                        {submissionStatus.submission.teacherScore !== null && submissionStatus.submission.teacherScore !== undefined && (
                          <div className="mb-3">
                            <h6 className="text-success">Teacher Score</h6>
                            <div className="d-flex align-items-center gap-2">
                              <Badge bg="primary" className="fs-6">{submissionStatus.submission.teacherScore}/100</Badge>
                              {submissionStatus.submission.teacherScore >= 80 && <span className="text-success">🎉 Excellent!</span>}
                              {submissionStatus.submission.teacherScore >= 60 && submissionStatus.submission.teacherScore < 80 && <span className="text-info">👍 Good work!</span>}
                              {submissionStatus.submission.teacherScore < 60 && <span className="text-warning">📚 Keep improving!</span>}
                            </div>
                          </div>
                        )}
                        
                        {/* AI Analysis Results */}
                        <h6 className="text-muted">AI Analysis Results</h6>
                        {submissionStatus.submission.score !== null && (
                          <p><strong>Overall Score:</strong> {submissionStatus.submission.score}/100</p>
                        )}
                        {submissionStatus.submission.similarityScore !== null && (
                          <p><strong>Similarity:</strong> {submissionStatus.submission.similarityScore}%</p>
                        )}
                        {submissionStatus.submission.aiGeneratedScore !== null && (
                          <p><strong>AI Detection:</strong> {submissionStatus.submission.aiGeneratedScore}%</p>
                        )}
                        {submissionStatus.submission.plagiarismScore !== null && (
                          <p><strong>Plagiarism:</strong> {submissionStatus.submission.plagiarismScore}%</p>
                        )}
                      </>
                    )}
                  </div>
                </div>
              </Card.Body>
            </Card>
          )}

          {/* Submission/Resubmission Form */}
          <Card>
            <Card.Header>
              <h6 className="mb-0">
                {submissionStatus?.submitted ? "Resubmit Homework" : "Submit Homework"}
              </h6>
            </Card.Header>
            <Card.Body>
              {submissionStatus?.submitted && (
                <Alert variant="info">
                  <strong>Note:</strong> Uploading a new file will replace your current submission. 
                  Your previous file will be permanently deleted.
                </Alert>
              )}
              
              {formik.errors.general && (
                <Alert variant="danger">{formik.errors.general}</Alert>
              )}
              
              <Form onSubmit={formik.handleSubmit} encType="multipart/form-data">
                <Form.Group controlId="homework" className="mb-3 mt-2">
                  <Form.Label>
                    {submissionStatus?.submitted ? "Select New File" : "Select File to Upload"}
                  </Form.Label>
                  <Form.Control
                    onChange={handleChangeFile}
                    type="file"
                    name="homework"
                    aria-label="Upload"
                    required
                  />
                  <Form.Text className="text-muted">
                    Supported formats: PDF, DOC, DOCX, TXT
                  </Form.Text>
                </Form.Group>
                <div className="d-grid gap-2">
                  <Button type="submit" className="mt-2" disabled={!formik.values.homework}>
                    {submissionStatus?.submitted ? "Resubmit" : "Submit"}
                  </Button>
                </div>
              </Form>
            </Card.Body>
          </Card>
        </Offcanvas.Body>
      </Offcanvas>
      
      <Toast
        onClose={() => setToastShow(false)}
        show={toastShow}
        delay={3000}
        autohide
        className="position-fixed top-0 end-0 m-3"
        style={{ zIndex: 1050 }}
      >
        <Toast.Header>
          <strong className="me-auto text-success">Success</strong>
        </Toast.Header>
        <Toast.Body>{toastMessage}</Toast.Body>
      </Toast>
    </>
  );
};

export default SubmitHomeworkOffCanvas;
