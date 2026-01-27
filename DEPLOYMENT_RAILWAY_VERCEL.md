# Railway + Vercel Deployment Guide

This guide covers deploying the Mantile app with the backend on Railway and frontend on Vercel.

## Architecture

- **Backend (Railway)**: FastAPI server serving model/hardware configs and compute estimates
- **Frontend (Vercel)**: Static React app built with Vite

## Prerequisites

- GitHub account with this repository
- [Railway account](https://railway.app) (free tier available)
- [Vercel account](https://vercel.com) (free tier available)

## Step 1: Deploy Backend to Railway

### 1.1 Create New Project

1. Go to [railway.app](https://railway.app) and sign in
2. Click **"New Project"**
3. Select **"Deploy from GitHub repo"**
4. Choose your `Mantile` repository
5. Railway will auto-detect the Python app

### 1.2 Configure Environment Variables

In Railway project settings, add:

```
ALLOWED_ORIGINS=https://your-app.vercel.app
```

> **Note**: After deploying frontend in Step 2, you'll update this with your actual Vercel URL.

### 1.3 Configure Build Settings

Railway should auto-detect Python, but verify:

- **Build Command**: `pip install -r backend/requirements.txt`
- **Start Command**: Automatically uses `Procfile` (already configured)
- **Root Directory**: Leave as `/` (Procfile references backend correctly)

### 1.4 Deploy

1. Click **"Deploy"**
2. Wait for deployment to complete
3. Copy your Railway app URL (e.g., `https://mantile-production.up.railway.app`)

## Step 2: Deploy Frontend to Vercel

### 2.1 Create New Project

1. Go to [vercel.com](https://vercel.com) and sign in
2. Click **"Add New..." → "Project"**
3. Import your `Mantile` repository
4. Vercel will auto-detect Vite

### 2.2 Configure Project Settings

In the project configuration screen:

- **Framework Preset**: Vite (should auto-detect)
- **Root Directory**: `frontend`
- **Build Command**: `npm run build` (or leave default)
- **Output Directory**: `dist` (or leave default)

### 2.3 Configure Environment Variables

Add the following environment variable:

```
VITE_API_URL=https://your-railway-app.up.railway.app
```

Replace with your actual Railway URL from Step 1.4.

### 2.4 Deploy

1. Click **"Deploy"**
2. Wait for deployment to complete
3. Copy your Vercel app URL (e.g., `https://mantile.vercel.app`)

## Step 3: Update CORS Configuration

Now that both apps are deployed, update Railway's CORS settings:

1. Go back to your Railway project
2. Navigate to **Variables**
3. Update `ALLOWED_ORIGINS` with your Vercel URL:

```
ALLOWED_ORIGINS=https://mantile.vercel.app
```

4. Railway will automatically redeploy with the new settings

## Step 4: Verify Deployment

1. Open your Vercel URL in a browser
2. Open browser DevTools (F12) → Network tab
3. Interact with the app (select models/hardware)
4. Verify API calls to Railway backend succeed without CORS errors

## Environment Variables Reference

### Backend (Railway)

| Variable | Required | Description | Example |
|----------|----------|-------------|---------|
| `ALLOWED_ORIGINS` | Yes | Comma-separated list of allowed frontend origins | `https://mantile.vercel.app` |
| `API_PATH_PREFIX` | No | API path prefix for reverse proxy setups | `/estimator-api` |
| `PORT` | Auto-set | Railway provides this automatically | `8000` |

### Frontend (Vercel)

| Variable | Required | Description | Example |
|----------|----------|-------------|---------|
| `VITE_API_URL` | Yes | Backend API URL | `https://mantile-production.up.railway.app` |
| `VITE_BASE_PATH` | No | Frontend base path (for subdirectory hosting) | `/` (default) |

## Custom Domains

### Adding Custom Domain to Vercel

1. In Vercel project settings → **Domains**
2. Add your custom domain (e.g., `estimator.yourdomain.com`)
3. Configure DNS as instructed by Vercel
4. **Update Railway's `ALLOWED_ORIGINS`** to include the custom domain

### Adding Custom Domain to Railway

1. In Railway project settings → **Settings** → **Domains**
2. Add your custom domain (e.g., `api.yourdomain.com`)
3. Configure DNS as instructed by Railway
4. **Update Vercel's `VITE_API_URL`** environment variable with the custom domain
5. Redeploy Vercel to apply changes

## Troubleshooting

### CORS Errors

**Symptoms**: Network requests fail with CORS errors in browser console

**Solutions**:
- Verify `ALLOWED_ORIGINS` in Railway includes your exact Vercel URL
- Check for trailing slashes (URL should NOT have trailing slash)
- Ensure Railway redeployed after updating environment variables

### API Calls Return 404

**Symptoms**: Frontend loads but API calls fail

**Solutions**:
- Verify `VITE_API_URL` in Vercel matches your Railway URL exactly
- Test Railway API directly by visiting `https://your-railway-app.up.railway.app/health`
- Check Railway logs for errors

### Frontend Build Fails

**Symptoms**: Vercel build fails during deployment

**Solutions**:
- Verify Root Directory is set to `frontend` in Vercel
- Check that `frontend/package.json` exists
- Review Vercel build logs for specific errors

### Backend Won't Start

**Symptoms**: Railway deployment fails or crashes

**Solutions**:
- Verify `backend/requirements.txt` exists and is valid
- Check Railway logs for Python errors
- Ensure `Procfile` exists at repository root
- Verify Python version compatibility (3.11+ recommended)

## Monitoring and Logs

### Railway Logs

Access logs in Railway dashboard → **Deployments** → Click deployment → **View Logs**

### Vercel Logs

Access logs in Vercel dashboard → **Deployments** → Click deployment → **View Function Logs**

## Updating the App

Both platforms support automatic deployments:

1. Push changes to your `main` branch
2. Railway and Vercel will automatically detect changes and redeploy
3. No manual intervention needed

To disable auto-deploy, adjust settings in each platform's dashboard.

## Cost Considerations

### Railway Free Tier
- $5 of usage credit per month
- Suitable for development/testing
- May need paid plan for production

### Vercel Free Tier
- Unlimited bandwidth for personal projects
- 100GB bandwidth for commercial
- Sufficient for most use cases

## Security Notes

1. **No HuggingFace Token Needed**: The backend serves pre-generated model configs from JSON files. HF tokens are only required during model config creation (handled separately in `model_builder/`).

2. **No Secrets in Frontend**: Never add sensitive data to `VITE_*` environment variables - they are embedded in the client-side JavaScript bundle.

3. **CORS Configuration**: Keep `ALLOWED_ORIGINS` restrictive - only add trusted frontend domains.
