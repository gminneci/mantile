#!/bin/bash

# Mantile Development Server Startup Script
# Starts both backend and frontend

echo "🚀 Starting Mantile Development Environment"
echo "============================================"

# Kill any existing processes
echo "Cleaning up existing processes..."
lsof -ti :8000 | xargs kill -9 2>/dev/null
lsof -ti :5173 | xargs kill -9 2>/dev/null
sleep 1

# Start backend
echo ""
echo "📦 Starting Backend (port 8000)..."
cd "$(dirname "$0")"
python3 -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000 &
BACKEND_PID=$!
echo "Backend PID: $BACKEND_PID"

# Wait for backend to start
sleep 3

# Test backend
echo "Testing backend..."
if curl -s http://127.0.0.1:8000/hardware > /dev/null; then
    echo "✅ Backend is running"
else
    echo "❌ Backend failed to start"
    kill $BACKEND_PID 2>/dev/null
    exit 1
fi

# Start frontend
echo ""
echo "🎨 Starting Frontend (port 5173)..."
cd frontend
npm run dev &
FRONTEND_PID=$!
echo "Frontend PID: $FRONTEND_PID"

echo ""
echo "============================================"
echo "✅ Development servers started!"
echo ""
echo "Backend:  http://localhost:8000"
echo "Frontend: http://localhost:5173"
echo ""
echo "Press Ctrl+C to stop both servers"
echo "============================================"

# Trap Ctrl+C to kill both processes
trap "echo '\nStopping servers...'; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit" INT

# Wait
wait
