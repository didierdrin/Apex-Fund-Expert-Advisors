# Deploy on Render

Bot matches `fib_sma.pine` (15m chart + 4H Fib 0.618, SMA50/200 trend). Saves every TradingView-style mark: `15m_breakout`, `4h_breakout`, `both_breakout`, and the pullback variants (BUY in bullish / SELL in bearish).

1. Create a **Web Service** (not Cron) from this repo.
2. Set environment variables:
   - `DATABASE_URL` — Neon PostgreSQL connection string
   - `CORS_ORIGINS` — `https://apex-firm.vercel.app,http://localhost:3000`
   - Optional: `TIMEFRAME=15m`, `HTF_MINUTES=240`, `FIB_LEVEL=0.618` (defaults match Pine)
3. Start command (from render.yaml):  
   `gunicorn fib_sma_bot:app --bind 0.0.0.0:$PORT --workers 1 --timeout 300`
4. Verify after deploy:
   - `https://YOUR-SERVICE.onrender.com/health` → `OK`
   - `https://YOUR-SERVICE.onrender.com/status` → `"strategy":"fib_sma"`
   - `https://YOUR-SERVICE.onrender.com/debug?symbol=EURUSD=X` → last-bar Fib/SMA filters
   - `https://YOUR-SERVICE.onrender.com/api/alerts?limit=5` → `[]` or JSON array
5. In **Vercel** (apex-firm project), set:  
   `ALERTS_API_URL=https://YOUR-SERVICE.onrender.com`

Empty DB returns `[]` (HTTP 200), not 404.
