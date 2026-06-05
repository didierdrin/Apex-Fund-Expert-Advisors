# Deploy on Render

1. Create a **Web Service** (not Cron) from this repo.
2. Set environment variables:
   - `DATABASE_URL` — Neon PostgreSQL connection string
   - `CORS_ORIGINS` — `https://apex-firm.vercel.app,http://localhost:3000`
3. Start command (from render.yaml):  
   `gunicorn fib_sma_bot:app --bind 0.0.0.0:$PORT --workers 1 --timeout 300`
4. Verify after deploy:
   - `https://YOUR-SERVICE.onrender.com/health` → `OK`
   - `https://YOUR-SERVICE.onrender.com/api/alerts?limit=5` → `[]` or JSON array
5. In **Vercel** (apex-firm project), set:  
   `ALERTS_API_URL=https://YOUR-SERVICE.onrender.com`

Empty DB returns `[]` (HTTP 200), not 404.
