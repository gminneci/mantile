FROM python:3.11-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY backend ./backend

# Expose port (Railway will set $PORT)
EXPOSE $PORT

# Start command
CMD uvicorn backend.main:app --host 0.0.0.0 --port $PORT
