#!/bin/bash

# Stop all ML API services

echo "🛑 Stopping UniqScan ML Services..."

# Kill processes using PID files
if [ -f "unified_api.pid" ]; then
    PID=$(cat unified_api.pid)
    kill $PID 2>/dev/null
    echo "Stopped Unified API (PID: $PID)"
    rm unified_api.pid
fi

if [ -f "similarity_api.pid" ]; then
    PID=$(cat similarity_api.pid)
    kill $PID 2>/dev/null
    echo "Stopped Similarity API (PID: $PID)"
    rm similarity_api.pid
fi

if [ -f "ai_detection_api.pid" ]; then
    PID=$(cat ai_detection_api.pid)
    kill $PID 2>/dev/null
    echo "Stopped AI Detection API (PID: $PID)"
    rm ai_detection_api.pid
fi

# Alternative: Kill by port
pkill -f "unified_grading_api.py"
pkill -f "similarity_api.py" 
pkill -f "ai_detection_api.py"

echo "✅ All services stopped!"
