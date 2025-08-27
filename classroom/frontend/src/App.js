import { Navigation } from "./components";
import { Routes, Route } from "react-router-dom";
import {
  LandingPage,
  LoginPage,
  RegisterPage,
  HomePage,
  HomeworkPage,
  ClassroomPage,
  SubmissionsPage,
  ErrorPage,
} from "./pages/index";
import { useContext } from "react";
import { AuthContext } from "./contexts/authContext";
import { PrivateRoutes, NonPrivateRoutes } from "./configs/routeLayout";

function App() {
  const { isLoggin } = useContext(AuthContext);

  return (
    <div className="App">
      <Navigation />
      <Routes>
        <Route path="/" element={isLoggin ? <HomePage /> : <LandingPage />} />
        {/* NonPrivateRoutes */}
        <Route element={<NonPrivateRoutes isLoggin={isLoggin} />}>
          <Route path="/login" element={<LoginPage />} />
          <Route path="/register" element={<RegisterPage />} />
        </Route>
        {/* End of NonPrivateRoutes */}
        {/* PrivateRoutes */}
        <Route element={<PrivateRoutes isLoggin={isLoggin} />}>
          <Route path="/home" element={<HomePage />} />
          <Route path="/classroom/:classroomID" element={<ClassroomPage />} />
          <Route path="/homework/:homeworkID" element={<HomeworkPage />} />
          <Route path="/my-submissions" element={<SubmissionsPage />} />
        </Route>
        {/* End of PrivateRoutes */}
        <Route path="*" element={<ErrorPage />} />
      </Routes>
    </div>
  );
}

export default App;
