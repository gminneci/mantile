# Deployment Guide

This guide explains how to deploy Mantile with custom path prefixes for reverse proxy scenarios.

## Path Prefix Support

Mantile supports deployment under custom URL paths, which is common when deploying behind a reverse proxy or on platforms with path-based routing.

### Example Deployment Scenario

- **Frontend**: `https://your-domain.com/estimator/`
- **Backend API**: `https://your-domain.com/estimator-api/`

## Configuration

### 1. Environment Variables

Create a `.env` file in the project root (copy from `.env.example`):

```bash
# Backend API path prefix
ROOT_PATH=/estimator-api

# Frontend base path  
VITE_BASE_PATH=/estimator/

# Backend API URL (for frontend to connect)
VITE_API_URL=https://your-domain.com/estimator-api

# CORS allowed origins
CORS_ORIGINS=https://your-domain.com,https://your-domain.com/estimator
```

### 2. Backend Configuration

The backend automatically reads `ROOT_PATH` from environment variables:

```bash
# Development
ROOT_PATH=/estimator-api uvicorn backend.main:app --reload

# Production with root_path
uvicorn backend.main:app --root-path /estimator-api --host 0.0.0.0 --port 8000
```

**How it works:**
- FastAPI's `root_path` parameter ensures all routes are served under the specified prefix
- Routes like `/api/layers` become `/estimator-api/api/layers`
- OpenAPI docs (Swagger) also adjusts to the new path

### 3. Frontend Configuration

The frontend uses Vite's `base` option for path prefixes:

```bash
# Development
VITE_BASE_PATH=/estimator/ npm run dev

# Production build
VITE_BASE_PATH=/estimator/ npm run build
```

**How it works:**
- All static assets (JS, CSS, images) are referenced relative to the base path
- Client-side routing (if added later) respects the base path
- API calls use `VITE_API_URL` to reach the backend

### 4. CORS Configuration

When deploying with custom paths, update `CORS_ORIGINS` to include all valid frontend URLs:

```bash
CORS_ORIGINS=https://your-domain.com,https://your-domain.com/estimator,http://localhost:5173
```

## Reverse Proxy Examples

### Nginx

```nginx
# Frontend (static files)
location /estimator/ {
    alias /var/www/mantile/frontend/dist/;
    try_files $uri $uri/ /estimator/index.html;
}

# Backend API
location /estimator-api/ {
    proxy_pass http://localhost:8000/;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
}
```

### Apache

```apache
# Frontend (static files)
Alias /estimator /var/www/mantile/frontend/dist
<Directory /var/www/mantile/frontend/dist>
    Options -Indexes +FollowSymLinks
    AllowOverride All
    Require all granted
    FallbackResource /estimator/index.html
</Directory>

# Backend API
ProxyPass /estimator-api/ http://localhost:8000/
ProxyPassReverse /estimator-api/ http://localhost:8000/
```

### Caddy

```caddy
# Frontend
handle_path /estimator/* {
    root * /var/www/mantile/frontend/dist
    try_files {path} /index.html
    file_server
}

# Backend API
handle_path /estimator-api/* {
    reverse_proxy localhost:8000
}
```

## Local Development with Path Prefixes

To test path prefix configuration locally:

1. Create `.env` file:
```bash
ROOT_PATH=/estimator-api
VITE_BASE_PATH=/estimator/
VITE_API_URL=http://localhost:8000/estimator-api
CORS_ORIGINS=http://localhost:5173
```

2. Start backend:
```bash
./run_backend.sh
```

3. Start frontend:
```bash
./run_frontend.sh
```

4. Access the app at:
   - Frontend: `http://localhost:5173/estimator/`
   - API Docs: `http://localhost:8000/estimator-api/docs`
   - API Endpoint: `http://localhost:8000/estimator-api/api/layers`

## Production Deployment

### Backend

```bash
# Install dependencies
pip install -r backend/requirements.txt

# Set environment variables
export ROOT_PATH=/estimator-api
export CORS_ORIGINS=https://your-domain.com

# Run with production server
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Frontend

```bash
# Install dependencies
cd frontend
npm install

# Build with base path
VITE_BASE_PATH=/estimator/ VITE_API_URL=https://your-domain.com/estimator-api npm run build

# Output will be in frontend/dist/
# Serve with your web server (nginx, Apache, etc.)
```

## Troubleshooting

### API Requests Fail (404)

**Issue**: Frontend can't reach backend API

**Solution**: Verify `VITE_API_URL` includes the full path prefix:
```bash
VITE_API_URL=https://your-domain.com/estimator-api
```

### CORS Errors

**Issue**: Browser blocks API requests due to CORS

**Solution**: Add frontend URL to `CORS_ORIGINS`:
```bash
CORS_ORIGINS=https://your-domain.com,https://your-domain.com/estimator
```

### Static Assets 404

**Issue**: Frontend loads but CSS/JS files return 404

**Solution**: Ensure `VITE_BASE_PATH` matches your deployment path:
```bash
VITE_BASE_PATH=/estimator/
```

### OpenAPI Docs Not Loading

**Issue**: FastAPI docs at `/docs` don't work

**Solution**: Access docs at the prefixed path:
```
http://localhost:8000/estimator-api/docs
```

## Root Path Deployment

For deployment at the root path (no prefix), simply omit or set empty values:

```bash
ROOT_PATH=
VITE_BASE_PATH=/
VITE_API_URL=http://localhost:8000
```

Or just don't set these variables at all—the defaults work for root deployment.
