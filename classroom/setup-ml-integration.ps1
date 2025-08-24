#!/usr/bin/env pwsh
# Setup script for ML API integration

Write-Host "🔧 Setting up ML API Integration..." -ForegroundColor Cyan
Write-Host ""

# Change to backend directory
Push-Location -Path (Join-Path $PSScriptRoot "backend")

Write-Host "📦 Installing axios dependency..." -ForegroundColor Yellow
try {
    npm install axios
    Write-Host "✅ Axios installed successfully!" -ForegroundColor Green
} catch {
    Write-Host "❌ Failed to install axios: $_" -ForegroundColor Red
    Pop-Location
    exit 1
}

Write-Host ""
Write-Host "🧪 Running ML Integration Test..." -ForegroundColor Yellow

try {
    node test-ml-integration.js
} catch {
    Write-Host "⚠️  Test completed with errors - this is normal if ML services aren't running yet" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "🚀 Setup completed!" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "1. Start the ML services: cd ML && python unified_grading_api.py" -ForegroundColor White
Write-Host "2. Start the Node.js backend: npm run dev" -ForegroundColor White
Write-Host "3. Test the integration by uploading a homework file" -ForegroundColor White
Write-Host ""
Write-Host "💡 You can check ML API health at: http://localhost:3000/api/homeworks/ml-health" -ForegroundColor White

Pop-Location
