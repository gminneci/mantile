#!/bin/bash

# Mantile - Run Backend Server

cd "$(dirname "$0")"

echo "🚀 Starting Mantile Backend..."

# Check if mantile venv exists, if not create it
if [ ! -d "mantile" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv mantile
fi

# Activate mantile venv
source mantile/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -q -r backend/requirements.txt

# Kill any process using port 8000
echo "Checking port 8000..."
lsof -ti :8000 | xargs kill -9 2>/dev/null || true

# Load environment variables if .env exists
if [ -f ".env" ]; then
    echo "Loading environment variables from .env"
    export $(cat .env | grep -v '^#' | xargs)
fi

# Run server
echo "Starting FastAPI server on http://localhost:8000"
if [ -n "$ROOT_PATH" ]; then
    echo "  API path prefix: $ROOT_PATH"
fi
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
