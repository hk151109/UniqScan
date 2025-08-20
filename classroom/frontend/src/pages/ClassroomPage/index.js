import React, { useContext, useEffect } from "react";
import { Container, Spinner, Button } from "react-bootstrap";
import { useParams, useNavigate } from "react-router-dom";
import { fetchClassroomDetail } from "../../api/classroomApi";
import { AuthContext } from "../../contexts/authContext";
import ClassroomBody from "./ClassroomBody";
import ClassroomHeader from "./ClassroomHeader";

const ClassroomPage = () => {
  const { classroomID } = useParams();
  const navigate = useNavigate();
  const { classroom, setClassroom } = useContext(AuthContext);

  useEffect(() => {
    const getClassroomDetail = async (id) => {
      const { data } = await fetchClassroomDetail(id);
      setClassroom({ ...data.data });
    };
    getClassroomDetail(classroomID);
  }, [classroomID, setClassroom]);

  if (!classroom) {
    return <Spinner />;
  }

  return (
    <Container>
      <Button variant="outline-primary" className="mb-3" onClick={() => navigate(-1)}>
        &larr; Back
      </Button>
      <ClassroomHeader classroom={classroom} />
      <ClassroomBody classroom={classroom} />
    </Container>
  );
};

export default ClassroomPage;
