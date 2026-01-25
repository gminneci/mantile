# Mantile Frontend

React + Vite application for LLM inference performance estimation.

## Development

```bash
# Install dependencies
npm install

# Start dev server (connects to localhost:8000 by default)
npm run dev
```

The app will be available at `http://localhost:5173`

## Environment Variables

Configure via `.env` file (see `.env.example` in project root):

- **`VITE_API_URL`**: Backend API URL
  - Development: Defaults to `http://localhost:8000` (no configuration needed)
  - Production: Set to your backend URL (e.g., `https://api.your-domain.com`)
  - Docker: Set to container name (e.g., `http://backend:8000`)

- **`VITE_BASE_PATH`**: Frontend base path for deployment
  - Default: `/` (root)
  - Reverse proxy: `/estimator/` (must include trailing slash)

## Deployment

```bash
# Build for production
npm run build

# Preview production build locally
npm run preview
```

Set environment variables before building or use runtime configuration.

## Code Structure

```
src/
├── App.jsx                 # Main application component
├── components/             # React components
│   ├── LayerConfigCard.jsx      # Layer configuration UI
│   ├── LayerMetricsDisplay.jsx  # Layer-level metrics visualization
│   └── MetricsDisplay.jsx       # System-level metrics visualization
└── utils/                  # Shared utilities
    ├── constants.js             # CHART_COLORS, MEMORY_COLORS
    └── formatters.js            # formatNumber() helper
```

**Key Design Decisions:**
- Hardware config is the **single source of truth** for available dtypes and max parallelism
- No hardcoded fallbacks - errors are thrown when configs are invalid or missing
- Auto-selects best dtype with priority: nvfp4 > nvfp8 > fp8 > bf16 > fp16 > int8
- Default model: `openai_GPT-OSS-120B`
- Time formatting auto-converts to seconds at 1000ms threshold

---

## React + Vite

This template provides a minimal setup to get React working in Vite with HMR and some ESLint rules.

Currently, two official plugins are available:

- [@vitejs/plugin-react](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react) uses [Babel](https://babeljs.io/) (or [oxc](https://oxc.rs) when used in [rolldown-vite](https://vite.dev/guide/rolldown)) for Fast Refresh
- [@vitejs/plugin-react-swc](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react-swc) uses [SWC](https://swc.rs/) for Fast Refresh

## React Compiler

The React Compiler is not enabled on this template because of its impact on dev & build performances. To add it, see [this documentation](https://react.dev/learn/react-compiler/installation).

## Expanding the ESLint configuration

If you are developing a production application, we recommend using TypeScript with type-aware lint rules enabled. Check out the [TS template](https://github.com/vitejs/vite/tree/main/packages/create-vite/template-react-ts) for information on how to integrate TypeScript and [`typescript-eslint`](https://typescript-eslint.io) in your project.
