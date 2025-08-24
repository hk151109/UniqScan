#!/bin/bash

# UniqScan ML APIs Startup Script

echo "🚀 Starting UniqScan ML Services..."

# Create necessary directories
mkdir -p reports
mkdir -p submissions
mkdir -p logs

# Set up Python environment
echo "📦 Setting up Python environment..."

# Install requirements if needed
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install requirements
echo "Installing dependencies..."
pip install -r requirements.txt

# Download NLTK data if needed
python3 -c "
import nltk
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    print('NLTK data already downloaded')
except LookupError:
    print('Downloading NLTK data...')
    nltk.download('punkt')
    nltk.download('stopwords')
    print('NLTK data download complete')
"

echo "🔧 Configuration complete!"

echo "Starting services..."

# Start the unified API service
echo "🌐 Starting Unified Grading API on port 5000..."
python3 unified_grading_api.py &
UNIFIED_PID=$!

# Start individual services if needed
echo "🔍 Starting Similarity API on port 5001..."
python3 similarity_api.py &
SIMILARITY_PID=$!

echo "🤖 Starting AI Detection API on port 5002..."
python3 ai_detection_api.py &
AI_DETECTION_PID=$!

# Save process IDs
echo $UNIFIED_PID > unified_api.pid
echo $SIMILARITY_PID > similarity_api.pid
echo $AI_DETECTION_PID > ai_detection_api.pid

echo "✅ All services started successfully!"
echo ""
echo "📍 Service URLs:"
echo "   Unified API: http://localhost:5000"
echo "   Similarity API: http://localhost:5001" 
echo "   AI Detection API: http://localhost:5002"
echo ""
echo "📊 Health Check URLs:"
echo "   curl http://localhost:5000/health"
echo "   curl http://localhost:5001/health"
echo "   curl http://localhost:5002/health"
echo ""
echo "⏹️  To stop services: ./stop_services.sh"

# Keep script running
wait
