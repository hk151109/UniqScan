@echo off
echo 🚀 Starting UniqScan ML Services...

REM Create necessary directories
if not exist "reports" mkdir reports
if not exist "submissions" mkdir submissions
if not exist "logs" mkdir logs

echo 📦 Setting up Python environment...

REM Install requirements
pip install -r requirements.txt

REM Download NLTK data
python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('stopwords', quiet=True); print('NLTK data ready')"

echo 🔧 Configuration complete!

echo Starting services...

REM Start services in background
echo 🌐 Starting Unified Grading API on port 5000...
start "Unified API" python unified_grading_api.py

timeout /t 2 > nul

echo 🔍 Starting Similarity API on port 5001...
start "Similarity API" python similarity_api.py

timeout /t 2 > nul

echo 🤖 Starting AI Detection API on port 5002...
start "AI Detection API" python ai_detection_api.py

echo ✅ All services started successfully!
echo.
echo 📍 Service URLs:
echo    Unified API: http://localhost:5000
echo    Similarity API: http://localhost:5001
echo    AI Detection API: http://localhost:5002
echo.
echo 📊 Health Check:
echo    curl http://localhost:5000/health
echo.
echo ⏹️ To stop services: run stop_services.bat

pause
