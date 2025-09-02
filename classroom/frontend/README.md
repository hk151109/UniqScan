# Classroom Frontend (React)

React app for a Google Classroom–style UI. Works with the Node backend under `../backend`.

## Environmentw

Create `.env` from the example:
```bash
REACT_APP_BASE_URL=http://localhost:4000
```

Install and run
```bash
npm install
npm start
```

## Features and flow

- Auth and classroom management; students join with codes.
- Teachers create posts and homeworks; students submit files.
- On submission, the backend triggers ML analysis; frontend displays similarity/AI/plagiarism scores and renders the HTML report returned by the ML API.

Notes
- Ensure backend `CLIENT_URL` matches this origin, and CORS is enabled in backend `app.js`.
