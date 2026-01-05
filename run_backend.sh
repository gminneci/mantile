#!/bin/bash

# Mantile - Run Backend Server

cd "$(dirname "$0")"

echo "🚀 Starting Mantile Backend..."

# Check if venv exists, if not create it
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
fi

# Activate venv
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -q -r backend/requirements.txt

# Run server
echo "Starting FastAPI server on http://localhost:8000"
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
