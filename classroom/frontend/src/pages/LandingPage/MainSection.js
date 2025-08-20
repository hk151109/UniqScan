import React from "react";
import { Button, Col, Container, Image, Row } from "react-bootstrap";
import { LinkContainer } from "react-router-bootstrap";
import { Main } from "./assets";

const MainSection = () => {
  return (
    <section className="bg-light p-5">
      <Container>
        <Row className="justify-content-center align-items-center">
          <Col className="text-center me-5">
            <h2 className="mb-4">
              UniqScan: Redefining Education with Integrity and Innovation
            </h2>
            <p className="lead">
              Empower your classroom with <span className="text-primary fw-bold">UniqScan</span> the ultimate platform for professors and students to connect, collaborate, and ensure originality. Manage your classroom effortlessly, share resources, assign work, and detect plagiarism in a single, intuitive space. Let’s make learning smarter and fairer!
            </p>
            <LinkContainer to="/register">
              <Button>Join Us</Button>
            </LinkContainer>
          </Col>
          <Col className="d-none d-md-block">
            <Image fluid src={Main} alt="Online Education | Classroom App" />
          </Col>
        </Row>
      </Container>
    </section>
  );
};

export default MainSection;
