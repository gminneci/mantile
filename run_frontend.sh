#!/bin/bash

# Mantile - Run Frontend Dev Server

cd "$(dirname "$0")/frontend"

echo "🎨 Starting Mantile Frontend..."

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "Installing dependencies..."
    npm install
fi

# Kill any process using port 5173
echo "Checking port 5173..."
lsof -ti :5173 | xargs kill -9 2>/dev/null || true

echo "Starting Vite dev server on http://localhost:5173"
npm run dev
