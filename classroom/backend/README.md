# Classroom Backend (Node/Express)

Google Classroom–style backend with users, classrooms, posts, homeworks, and ML grading integration.

Entry: `app.js`

## Environment

Create `backend/.env`:
```bash
MONGO_DB_URL=mongodb+srv://...
CLIENT_URL=http://localhost:3000

# JWT
SECRET_ACCESS_TOKEN=... 
ACCESS_TOKEN_EXPIRE=1d
SECRET_REFRESH_TOKEN=...
REFRESH_TOKEN_EXPIRE=7d

# ML integration
ML_API_BASE_URL=http://localhost:5000
ML_API_TIMEOUT=300000

# Optional deployment hints
BACKEND_URL=
```

Install and run
```bash
npm install
npm run dev
```

## Flow and features

- Auth, users, classrooms, posts, homeworks via routers under `src/routers/*`.
- Static serve `public/uploads` at `/uploads` so ML service can fetch files by URL.
- Grading flow:
  1. Student uploads homework (file saved under `public/uploads/homeworks`).
  2. Backend builds a public file URL via `src/utils/urlDetection.getFileUrl()`.
  3. Backend calls ML API `POST {ML_API_BASE_URL}/grade/analyze` with student and file metadata (`src/services/gradingService.js`).
  4. Receives `similarityScore`, `aiGeneratedScore`, `plagiarismScore`, and `reportHtml` to show/store.

### Key files
- `src/configs/dbConnection.js` — connects to MongoDB with `MONGO_DB_URL`.
- `src/routers/*.js` — feature routes; mounted under `/api`.
- `src/services/gradingService.js` — ML API client and report handling.
- `src/utils/urlDetection.js` — detects backend base URL and builds static file URLs.

## API contract with ML service

See `ML_API_DOCUMENTATION.md` for payloads and example responses. Core endpoint: `POST /grade/analyze`.
