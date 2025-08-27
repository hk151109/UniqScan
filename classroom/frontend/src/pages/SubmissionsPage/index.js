import React, { useState, useEffect, useContext } from "react";
import { 
  Container, 
  Row, 
  Col, 
  Card, 
  Badge, 
  Button, 
  Alert, 
  Spinner,
  Table
} from "react-bootstrap";
import { Link, useNavigate } from "react-router-dom";
import { BsArrowLeft, BsFileEarmarkText, BsClockFill, BsCheckCircleFill } from "react-icons/bs";
import { fetchMySubmissions } from "../../api/homeworkApi";
import { AuthContext } from "../../contexts/authContext";

const SubmissionsPage = () => {
  const [submissions, setSubmissions] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const navigate = useNavigate();
  const { user } = useContext(AuthContext);

  useEffect(() => {
    const loadSubmissions = async () => {
      try {
        setLoading(true);
        const response = await fetchMySubmissions();
        setSubmissions(response.data.submissions);
      } catch (err) {
        setError("Failed to load submissions");
        console.error("Error loading submissions:", err);
      } finally {
        setLoading(false);
      }
    };

    loadSubmissions();
  }, []);

  const formatDate = (dateString) => {
    return new Date(dateString).toLocaleDateString("en-US", {
      year: "numeric",
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  };

  const getStatusBadge = (submission) => {
    if (submission.isGraded && submission.teacherScore !== null && submission.teacherScore !== undefined) {
      return <Badge bg="success"><BsCheckCircleFill className="me-1" />Graded</Badge>;
    }
    if (submission.gradingCompleted) {
      return <Badge bg="info"><BsClockFill className="me-1" />Submitted - AI Analysis Complete</Badge>;
    }
    return <Badge bg="info"><BsClockFill className="me-1" />Submitted - Analysis in Progress</Badge>;
  };

  const getScoreBadge = (submission) => {
    // Prioritize teacher score if available
    if (submission.teacherScore !== null && submission.teacherScore !== undefined) {
      let variant = "secondary";
      if (submission.teacherScore >= 80) variant = "success";
      else if (submission.teacherScore >= 60) variant = "warning";
      else variant = "danger";
      
      return (
        <div className="d-flex align-items-center gap-1">
          <Badge bg={variant}>{submission.teacherScore}/100</Badge>
          <small className="text-muted">(Teacher)</small>
        </div>
      );
    }
    
    // Fall back to AI score if no teacher score
    if (submission.score !== null && submission.score !== undefined) {
      let variant = "secondary";
      if (submission.score >= 80) variant = "success";
      else if (submission.score >= 60) variant = "warning";
      else variant = "danger";
      
      return (
        <div className="d-flex align-items-center gap-1">
          <Badge bg={variant} className="opacity-75">{submission.score}/100</Badge>
          <small className="text-muted">(AI)</small>
        </div>
      );
    }
    
    return null;
  };

  if (loading) {
    return (
      <Container className="mt-5 text-center">
        <Spinner animation="border" role="status">
          <span className="visually-hidden">Loading...</span>
        </Spinner>
        <p className="mt-3">Loading your submissions...</p>
      </Container>
    );
  }

  return (
    <Container className="mt-4">
      <div className="d-flex align-items-center mb-4">
        <Button variant="outline-secondary" onClick={() => navigate(-1)} className="me-3">
          <BsArrowLeft className="me-2" />
          Back
        </Button>
        <div>
          <h2 className="mb-0">My Submissions</h2>
          <p className="text-muted mb-0">View all your homework submissions and grades</p>
        </div>
      </div>

      {error && (
        <Alert variant="danger">
          {error}
        </Alert>
      )}

      {submissions.length === 0 ? (
        <Card className="text-center p-5">
          <Card.Body>
            <BsFileEarmarkText size={60} className="text-muted mb-3" />
            <h4>No Submissions Yet</h4>
            <p className="text-muted">You haven't submitted any homework assignments yet.</p>
            <Link to="/home">
              <Button variant="primary">Go to Classrooms</Button>
            </Link>
          </Card.Body>
        </Card>
      ) : (
        <Row>
          <Col>
            <Card>
              <Card.Header>
                <h5 className="mb-0">Submission History ({submissions.length})</h5>
              </Card.Header>
              <Card.Body>
                <Table responsive striped hover>
                  <thead>
                    <tr>
                      <th>Homework</th>
                      <th>Classroom</th>
                      <th>File</th>
                      <th>Submitted</th>
                      <th>Status</th>
                      <th>Score</th>
                      <th>Details</th>
                      <th>Actions</th>
                    </tr>
                  </thead>
                  <tbody>
                    {submissions.map((submission) => (
                      <tr key={submission._id}>
                        <td>
                          <div>
                            <strong>{submission.homework.title}</strong>
                            {submission.isResubmission && (
                              <div>
                                <Badge bg="info" className="mt-1">
                                  Version {submission.submissionVersion}
                                </Badge>
                              </div>
                            )}
                          </div>
                        </td>
                        <td>
                          <div>
                            <div className="fw-bold">{submission.homework.classroom.title}</div>
                            <small className="text-muted">{submission.homework.classroom.accessCode}</small>
                          </div>
                        </td>
                        <td>
                          <div className="d-flex align-items-center">
                            <BsFileEarmarkText className="me-2" />
                            <span className="text-truncate" style={{ maxWidth: "150px" }}>
                              {submission.originalFileName}
                            </span>
                          </div>
                        </td>
                        <td>
                          <div>
                            <div>{formatDate(submission.createdAt)}</div>
                            {submission.updatedAt !== submission.createdAt && (
                              <small className="text-muted">
                                Updated: {formatDate(submission.updatedAt)}
                              </small>
                            )}
                          </div>
                        </td>
                        <td>{getStatusBadge(submission)}</td>
                        <td>{getScoreBadge(submission)}</td>
                        <td>
                          {submission.gradingCompleted && (
                            <div>
                              {submission.similarityScore !== null && (
                                <div><small>Similarity: {submission.similarityScore}%</small></div>
                              )}
                              {submission.aiGeneratedScore !== null && (
                                <div><small>AI: {submission.aiGeneratedScore}%</small></div>
                              )}
                              {submission.plagiarismScore !== null && (
                                <div><small>Plagiarism: {submission.plagiarismScore}%</small></div>
                              )}
                            </div>
                          )}
                        </td>
                        <td>
                          <Link to={`/homework/${submission.homework._id}`}>
                            <Button variant="outline-primary" size="sm">
                              View Homework
                            </Button>
                          </Link>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </Table>
              </Card.Body>
            </Card>
          </Col>
        </Row>
      )}
    </Container>
  );
};

export default SubmissionsPage;
