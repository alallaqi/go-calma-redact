#!/usr/bin/env bash
set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo ""
echo -e "${GREEN}GoCalma Redact${NC}"
echo "=============================="
echo ""

# Check if Docker is running
if command -v docker &>/dev/null && docker info &>/dev/null 2>&1; then
    echo -e "${GREEN}Docker detected.${NC}"
    echo ""

    # Ask about LLM support
    echo "Start with local LLM verification (Ollama)?"
    echo "  1) No  — regex + NER only (faster startup)"
    echo "  2) Yes — adds Ollama with qwen2.5:1.5b (~1.5 GB download on first run)"
    echo ""
    read -rp "Choice [1]: " choice
    choice="${choice:-1}"

    if [ "$choice" = "2" ]; then
        echo ""
        echo -e "${YELLOW}Starting with Ollama profile...${NC}"
        docker compose --profile ollama up --build
    else
        echo ""
        echo -e "${GREEN}Starting GoCalma (NER only)...${NC}"
        docker compose up --build
    fi
else
    if command -v docker &>/dev/null; then
        echo -e "${YELLOW}Docker is installed but not running.${NC}"
        echo -e "Open ${GREEN}Docker Desktop${NC} and wait for it to start, then re-run this script."
        echo ""
    else
        echo -e "${YELLOW}Docker not found.${NC}"
        echo -e "Install it from: ${GREEN}https://docker.com/products/docker-desktop/${NC}"
        echo ""
    fi
    echo "Or run manually without Docker:"
    echo ""
    echo -e "  ${GREEN}1.${NC} Create a virtual environment:"
    echo "     python3 -m venv .venv"
    echo "     source .venv/bin/activate"
    echo ""
    echo -e "  ${GREEN}2.${NC} Install dependencies:"
    echo "     pip install -r requirements.txt"
    echo ""
    echo -e "  ${GREEN}3.${NC} Install Tesseract OCR (for scanned PDFs):"
    echo "     brew install tesseract tesseract-lang        # macOS"
    echo "     sudo apt install tesseract-ocr tesseract-ocr-deu  # Linux"
    echo ""
    echo -e "  ${GREEN}4.${NC} Run the app:"
    echo "     streamlit run app.py"
    echo ""
    echo -e "  ${GREEN}5.${NC} (Optional) For LLM verification:"
    echo "     brew install ollama && ollama serve"
    echo "     ollama pull qwen2.5:1.5b"
    echo ""
    exit 1
fi
