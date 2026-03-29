@echo off
echo.
echo GoCalma Redact
echo ==============================
echo.

REM Check if Docker is running
docker info >nul 2>&1
if %errorlevel% equ 0 (
    echo Docker detected.
    echo.
    echo Start with local LLM verification (Ollama)?
    echo   1^) No  - regex + NER only (faster startup)
    echo   2^) Yes - adds Ollama with qwen2.5:1.5b (~1.5 GB download on first run)
    echo.
    set /p choice="Choice [1]: "
    if not defined choice set choice=1

    if "%choice%"=="2" (
        echo.
        echo Starting with Ollama profile...
        docker compose --profile ollama up --build
    ) else (
        echo.
        echo Starting GoCalma (NER only)...
        docker compose up --build
    )
) else (
    echo Docker not found or not running.
    echo.
    echo To run manually:
    echo.
    echo   1. Create a virtual environment:
    echo      python -m venv .venv
    echo      .venv\Scripts\activate
    echo.
    echo   2. Install dependencies:
    echo      pip install -r requirements.txt
    echo.
    echo   3. Install Tesseract OCR (for scanned PDFs):
    echo      choco install tesseract
    echo      or download from: https://github.com/UB-Mannheim/tesseract/wiki
    echo.
    echo   4. Run the app:
    echo      streamlit run app.py
    echo.
    echo   5. (Optional) For LLM verification:
    echo      Download Ollama from https://ollama.ai
    echo      ollama serve
    echo      ollama pull qwen2.5:1.5b
    echo.
    exit /b 1
)
