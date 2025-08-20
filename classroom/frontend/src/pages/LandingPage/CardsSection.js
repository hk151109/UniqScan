import {
  giveNote,
  lectureNotes,
  shareYourHomework,
  downloadExcel,
} from "./assets";
import MyCard from "./Card";
import { Container, Row } from "react-bootstrap";

const CardsSection = () => {
  return (
    <Container className="p-2">
      <Row className="g-4 mt-3 mx-3" xs={1} md={2} sm={2} lg={4}>
        <MyCard
          img={lectureNotes}
          title="Share Lecture Notes"
          text="Professors can upload and share lecture notes, presentations, and other resources with students, ensuring easy access to all essential materials in one place."
        />
        <MyCard
          img={shareYourHomework}
          title="Share Assignments"
          text="Assign tasks, projects, or essays directly to students through the platform. Students can view, download, and complete assignments within the classroom."
        />
        <MyCard
          img={giveNote}
          title="Plagiarism Detection"
          text="Students can submit their work and instantly see their plagiarism percentage compared to submissions by peers in the same classroom. Professors also receive detailed plagiarism reports to evaluate originality."
        />
        <MyCard
          img={giveNote}
          title="Grading by Teachers"
          text="Professors can review submitted work, assign grades, and provide feedback seamlessly, streamlining the evaluation process."
        />
        <MyCard
          img={downloadExcel}
          title="Generate Results as Excel"
          text="Professors can generate comprehensive results as Excel files, including grades, plagiarism reports, and performance insights, and share them with students or other stakeholders effortlessly."
        />
      </Row>
    </Container>
  );
};

export default CardsSection;
