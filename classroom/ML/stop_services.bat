@echo off
echo 🛑 Stopping UniqScan ML Services...

REM Kill Python processes running the APIs
taskkill /f /im python.exe 2>nul
taskkill /f /im python3.exe 2>nul

REM More specific killing by window title
taskkill /fi "WindowTitle eq Unified API*" /f 2>nul
taskkill /fi "WindowTitle eq Similarity API*" /f 2>nul  
taskkill /fi "WindowTitle eq AI Detection API*" /f 2>nul

echo ✅ All services stopped!
pause
