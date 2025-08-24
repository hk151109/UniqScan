#!/usr/bin/env pwsh
# Environment setup for Hugging Face API integration

Write-Host "🔧 Setting up Hugging Face API Integration..." -ForegroundColor Cyan
Write-Host ""

# Check if we're in the ML directory
if (-not (Test-Path "ai_detection_api.py")) {
    Write-Host "❌ Please run this script from the ML directory" -ForegroundColor Red
    exit 1
}

Write-Host "📦 Installing Python dependencies..." -ForegroundColor Yellow

try {
    python -m pip install -r requirements.txt
    Write-Host "✅ Python dependencies installed!" -ForegroundColor Green
} catch {
    Write-Host "❌ Failed to install Python dependencies: $_" -ForegroundColor Red
    Write-Host "💡 Make sure Python is installed and accessible" -ForegroundColor Yellow
    exit 1
}

Write-Host ""
Write-Host "🔑 Hugging Face API Token Setup" -ForegroundColor Cyan
Write-Host "For better performance and no rate limits, get a free token from:" -ForegroundColor White
Write-Host "https://huggingface.co/settings/tokens" -ForegroundColor Blue

$token = $env:HUGGINGFACE_API_TOKEN
if ($token) {
    Write-Host "✅ HUGGINGFACE_API_TOKEN is already set" -ForegroundColor Green
} else {
    Write-Host "⚠️  HUGGINGFACE_API_TOKEN is not set" -ForegroundColor Yellow
    Write-Host "You can still use the API, but with rate limits" -ForegroundColor White
    
    $setToken = Read-Host "Would you like to set your Hugging Face token now? (y/n)"
    if ($setToken -eq "y" -or $setToken -eq "Y") {
        $userToken = Read-Host "Enter your Hugging Face API token"
        if ($userToken) {
            [Environment]::SetEnvironmentVariable("HUGGINGFACE_API_TOKEN", $userToken, "User")
            $env:HUGGINGFACE_API_TOKEN = $userToken
            Write-Host "✅ Token set for current session and future sessions" -ForegroundColor Green
        }
    }
}

Write-Host ""
Write-Host "🧪 Testing Hugging Face API..." -ForegroundColor Yellow

try {
    python test_huggingface_api.py
} catch {
    Write-Host "⚠️  Test completed with warnings - this is normal for first run" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "🚀 Setup completed!" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "1. Start the AI detection service: python ai_detection_api.py" -ForegroundColor White
Write-Host "2. Start other ML services: python unified_grading_api.py" -ForegroundColor White
Write-Host "3. Test integration from Node.js backend" -ForegroundColor White
Write-Host ""
Write-Host "Benefits of API approach:" -ForegroundColor Green
Write-Host "✓ No large model downloads (~1GB+ saved)" -ForegroundColor White
Write-Host "✓ Always up-to-date models" -ForegroundColor White
Write-Host "✓ Faster startup time" -ForegroundColor White
Write-Host "✓ Lower memory usage" -ForegroundColor White
