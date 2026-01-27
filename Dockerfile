FROM python:3.11-slim

WORKDIR /app

# Copy backend code and requirements
COPY backend ./backend

# Install dependencies
RUN pip install --no-cache-dir -r backend/requirements.txt

# Expose port (Railway will set $PORT)
EXPOSE $PORT

# Start command
CMD uvicorn backend.main:app --host 0.0.0.0 --port $PORT
