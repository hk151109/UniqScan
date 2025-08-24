import moment from "moment";
import React, { useContext, useEffect, useState } from "react";
import { Button, Col, Container, Row, Spinner, Table } from "react-bootstrap";
import { useParams, useNavigate } from "react-router-dom";
import {
  fetchDownloadExcelFile,
  fetchDownloadHomeworkFile,
  fetchHomeworkDetail,
  fetchGradingStatus,
} from "../../api/homeworkApi";
import { FaDownload, FaEye } from "react-icons/fa";
import { SiMicrosoftexcel } from "react-icons/si";
import { saveAs } from "file-saver";
import RateProjectOffCanvas from "../../components/MyOffCanvas/RateProjectOffCanvas";
import DocumentViewer from "../../components/DocumentViewer";
import { AuthContext } from "../../contexts/authContext";

const HomeworkPage = () => {
  const { homeworkID } = useParams();
  const navigate = useNavigate();
  const [homework, setHomework] = useState({});
  const [showModals, setShowModals] = useState({}); // Object to track modal state per student
  const [lock, setLock] = useState(false);
  const [gradingStatus, setGradingStatus] = useState({ inProgress: false, processedCount: 0, totalSubmissions: 0 });
  const [documentViewer, setDocumentViewer] = useState({ show: false, fileUrl: '', fileName: '', fileType: '' });
  const { classroom } = useContext(AuthContext);

  useEffect(() => {
    const getHomeworkDetail = async () => {
      const { data } = await fetchHomeworkDetail(homeworkID);
      setHomework({ ...data.homework });
    };
    getHomeworkDetail();
  }, [homeworkID, showModals]); // Update when modals change

  // Polling effect for grading status
  useEffect(() => {
    let intervalId;
    
    const checkGradingStatus = async () => {
      try {
        const response = await fetchGradingStatus(homeworkID);
        const { gradingInProgress, processedCount, totalSubmissions } = response.data;
        
        setGradingStatus({
          inProgress: gradingInProgress,
          processedCount: processedCount,
          totalSubmissions: totalSubmissions
        });

        if (!gradingInProgress) {
          // Refresh homework data when grading is complete
          const { data } = await fetchHomeworkDetail(homeworkID);
          setHomework({ ...data.homework });
        }
      } catch (error) {
        console.error('Error checking grading status:', error);
      }
    };

    // Start polling if homework exists
    if (homeworkID) {
      checkGradingStatus();
      intervalId = setInterval(checkGradingStatus, 5000); // Check every 5 seconds
    }

    return () => {
      if (intervalId) {
        clearInterval(intervalId);
      }
    };
  }, [homeworkID]);

  const downloadFile = async (filename, userName, userLastname) => {
    const homeworkTitle = homework?.title ? homework.title.replace(/[^a-zA-Z0-9]/g, '_') : 'Homework';
    const downloadFileName = userName && userLastname 
      ? `${userName}_${userLastname}_${homeworkTitle}`
      : `Project_${homeworkTitle}`;
    saveAs(fetchDownloadHomeworkFile(filename), downloadFileName);
  };

  const viewDocument = (filename, userName, userLastname) => {
    const homeworkTitle = homework?.title ? homework.title.replace(/[^a-zA-Z0-9]/g, '_') : 'Homework';
    const displayFileName = userName && userLastname 
      ? `${userName}_${userLastname}_${homeworkTitle}`
      : `Project_${homeworkTitle}`;
    
    const fileUrl = fetchDownloadHomeworkFile(filename);
    const fileExtension = filename.split('.').pop()?.toLowerCase();
    
    // Determine file type based on extension
    let fileType = '';
    if (['pdf'].includes(fileExtension)) {
      fileType = 'application/pdf';
    } else if (['jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp'].includes(fileExtension)) {
      fileType = 'image/' + (fileExtension === 'jpg' ? 'jpeg' : fileExtension);
    } else if (['txt', 'md'].includes(fileExtension)) {
      fileType = 'text/plain';
    } else if (fileExtension === 'json') {
      fileType = 'application/json';
    }

    setDocumentViewer({
      show: true,
      fileUrl: fileUrl,
      fileName: displayFileName + (fileExtension ? `.${fileExtension}` : ''),
      fileType: fileType
    });
  };

  const downloadExcelFile = async (homeworkID) => {
    setLock(true);
    const homeworkTitle = homework?.title ? homework.title.replace(/[^a-zA-Z0-9]/g, '_') : 'Homework';
    const classroomName = classroom?.name ? classroom.name.replace(/[^a-zA-Z0-9]/g, '_') : 'Classroom';
    const excelFileName = `${classroomName}_${homeworkTitle}_Grades`;
    saveAs(fetchDownloadExcelFile(classroom._id, homeworkID), excelFileName);
    setTimeout(() => {
      setLock(false);
    }, 500);
  };

  const handleShowModal = (projectId) => {
    setShowModals(prev => ({ ...prev, [projectId]: true }));
  };

  const handleCloseModal = (projectId) => {
    setShowModals(prev => ({ ...prev, [projectId]: false }));
  };

  const handleCloseDocumentViewer = () => {
    setDocumentViewer({ show: false, fileUrl: '', fileName: '', fileType: '' });
  };

  return (
    <Container className="mt-4">
      <Button variant="outline-primary" className="mb-3" onClick={() => navigate(-1)}>
        &larr; Back
      </Button>
      {/* header */}
      <Row>
        <Col>
          <h1>{homework.title}</h1>
          {gradingStatus.inProgress && (
            <div className="alert alert-info">
              <strong>Grading in Progress:</strong> {gradingStatus.processedCount}/{gradingStatus.totalSubmissions} submissions processed
            </div>
          )}
        </Col>
        <Col className="text-end">
          THE LAST DAY : {moment(homework.endTime).format("DD.MM.YYYY")}
          <br />
          <Button
            size="sm"
            variant="success"
            onClick={() => downloadExcelFile(homeworkID)}
            disabled={lock}
          >
            {lock ? (
              <>
                <Spinner
                  as="span"
                  animation="grow"
                  size="sm"
                  role="status"
                  aria-hidden="true"
                  className="me-2"
                />
                Loading...
              </>
            ) : (
              <>
                <SiMicrosoftexcel className="me-2" /> Download student grades
              </>
            )}
          </Button>
        </Col>
      </Row>
      <hr className="bg-primary" />
      {/* body */}
      <p className="lead">{homework.content}</p>

      {homework?.submitters?.length > 0 && (
        <>
          <p className="text-center text-uppercase border border-success border-2">
            Those who do their homework
          </p>
          <Table striped bordered hover size="sm">
            <thead>
              <tr>
                <th>Name</th>
                <th>Lastname</th>
                <th>'s Project</th>
                <th>Score</th>
                <th>Similarity Score</th>
                <th>AI Generated Score</th>
                <th>Plagiarism Score</th>
                <th>Report</th>
                <th>Rate it</th>
              </tr>
            </thead>
            <tbody>
              {homework?.submitters?.map((submitter) => (
                <tr key={submitter._id}>
                  <td>{submitter.user.name}</td>
                  <td>{submitter.user.lastname}</td>
                  <td>
                    <div className="d-flex gap-2">
                      <Button
                        size="sm"
                        variant="primary"
                        onClick={() => viewDocument(submitter?.file, submitter?.user?.name, submitter?.user?.lastname)}
                      >
                        <FaEye className="me-1" />
                        View
                      </Button>
                      <Button
                        size="sm"
                        variant="secondary"
                        onClick={() => downloadFile(submitter?.file, submitter?.user?.name, submitter?.user?.lastname)}
                      >
                        <FaDownload className="me-1" />
                        Download
                      </Button>
                    </div>
                  </td>
                  <td>{submitter.score ? submitter.score : "-"}</td>
                  <td>{submitter.similarityScore !== undefined && submitter.similarityScore !== null ? submitter.similarityScore : '-'}</td>
                  <td>{submitter.aiGeneratedScore !== undefined && submitter.aiGeneratedScore !== null ? submitter.aiGeneratedScore : '-'}</td>
                  <td>{submitter.plagiarismScore !== undefined && submitter.plagiarismScore !== null ? submitter.plagiarismScore : '-'}</td>
                  <td>
                    {submitter.reportPath ? (
                      <a href={submitter.reportPath} target="_blank" rel="noopener noreferrer">View Report</a>
                    ) : '-'}
                  </td>
                  <td>
                    <RateProjectOffCanvas
                      name={submitter.user.name}
                      lastname={submitter.user.lastname}
                      projectID={submitter._id}
                      show={showModals[submitter._id] || false}
                      onShow={() => handleShowModal(submitter._id)}
                      onClose={() => handleCloseModal(submitter._id)}
                    />
                  </td>
                </tr>
              ))}
            </tbody>
          </Table>
        </>
      )}
      {homework?.appointedStudents?.length > 0 && (
        <>
          <p className="text-center text-uppercase border border-warning border-2 ">
            Those who do not do their homework
          </p>

          <Table striped bordered hover size="sm">
            <thead>
              <tr>
                <th>Name</th>
                <th>Lastname</th>
              </tr>
            </thead>
            <tbody>
              {homework?.appointedStudents?.map((student) => (
                <tr key={student._id}>
                  <td>{student.name}</td>
                  <td>{student.lastname}</td>
                </tr>
              ))}
            </tbody>
          </Table>
        </>
      )}
      
      <DocumentViewer
        show={documentViewer.show}
        onHide={handleCloseDocumentViewer}
        fileUrl={documentViewer.fileUrl}
        fileName={documentViewer.fileName}
        fileType={documentViewer.fileType}
      />
    </Container>
  );
};

export default HomeworkPage;
