"""
Stationarity Trading Bot — yfinance edition
============================================
Data source : yfinance (free, no API key, no rate limits, no monthly cap)
Strategy    : Pine Script v5 'Stationarity Trading Strategy'
Signals     : entry_long / entry_short / exit_long / exit_short
"""

import time
import os
import sys
import base64
import tempfile
from datetime import datetime
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
import yfinance as yf
import firebase_admin
from firebase_admin import credentials, firestore
from flask import Flask, jsonify
import threading
import socket
from scipy import stats

app = Flask(__name__)

bot_instance    = None
last_check_time = None
total_signals   = 0
_startup_test_written = False


# ──────────────────────────────────────────────
# Startup test (zero external API calls)
# ──────────────────────────────────────────────

def write_startup_test_to_firebase(db, *, bot=None):
    """Write a lightweight config snapshot to Firestore at boot — no data API calls."""
    global _startup_test_written
    if _startup_test_written:
        return False
    if not db:
        print("⚠️  Firebase not initialised, skipping startup test write")
        _startup_test_written = True
        return False
    try:
        now    = datetime.utcnow()
        doc_id = f"{now.strftime('%Y%m%dT%H%M%S')}_{os.getpid()}"
        payload = {
            "type":             "backend_startup_test",
            "timestamp_utc":    now.isoformat() + "Z",
            "hostname":         socket.gethostname(),
            "pid":              os.getpid(),
            "python_version":   sys.version,
            "data_source":      "yfinance (free, no API key)",
            "timeframe":        getattr(bot, "timeframe",        "1h")  if bot else None,
            "htf_timeframe":    getattr(bot, "htf_timeframe",    "4h")  if bot else None,
            "watchlist_count":  len(getattr(bot, "watchlist",    []))   if bot else None,
            "zscore_threshold": getattr(bot, "zscore_threshold", None)  if bot else None,
            "lookback":         getattr(bot, "lookback",         None)  if bot else None,
        }
        db.collection("backend_startup_test").document(doc_id).set(payload)
        _startup_test_written = True
        print("✅ Startup test written to Firebase (zero API calls)")
        return True
    except Exception as e:
        _startup_test_written = True
        print(f"❌ Error writing startup test: {e}")
        return False


# ──────────────────────────────────────────────
# Bot
# ──────────────────────────────────────────────

class FibSMATradingBot:
    """
    Python implementation of Pine Script v5 'Stationarity Trading Strategy'.

    Data layer  : yfinance — free, unlimited, no API key required.
    Signal logic (verbatim from Pine):
        entry_long  = is_stationary AND zscore < -threshold AND htf_trend_bullish AND bullish_valid
        entry_short = is_stationary AND zscore >  threshold AND htf_trend_bearish AND bearish_valid
        exit_long   = zscore >  threshold AND bearish_valid
        exit_short  = zscore < -threshold AND bullish_valid
    where:
        bullish_valid = sma50>sma200 AND dist>=500ticks AND slope<=0.25%
        bearish_valid = sma50<sma200 AND dist>=500ticks AND slope<=0.25%
    """

    def __init__(self):
        self.watchlist = [
            "EURUSD=X", "GBPJPY=X", "AUDJPY=X", "XAUUSD=X", "USDCAD=X",
            "GBPUSD=X", "EURJPY=X", "USDJPY=X", "AUDUSD=X", "NZDUSD=X",
            "USDCHF=X", "EURGBP=X", "EURCAD=X", "GBPCAD=X", "AUDCAD=X",
            "EURAUD=X", "BTC-USD", "ETH-USD",
        ]

        # ── Strategy parameters (env-overridable, matching Pine inputs) ─
        self.lookback               = int(os.environ.get("LOOKBACK",            "50"))
        self.stationarity_threshold = float(os.environ.get("STATIONARITY_THRESH","0.05"))
        self.zscore_threshold       = float(os.environ.get("ZSCORE_THRESH",     "1.5"))
        self.trend_ma_period        = int(os.environ.get("TREND_MA_PERIOD",     "20"))
        self.sma_slope_period       = int(os.environ.get("SMA_SLOPE_PERIOD",    "5"))
        self.max_sma_slope_percent  = float(os.environ.get("MAX_SMA_SLOPE_PCT", "0.25"))
        self.min_sma_distance_ticks = int(os.environ.get("MIN_SMA_DIST_TICKS", "500"))

        # ── Timeframes ────────────────────────────────────────────────
        # yfinance supports: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
        # 4h is NOT native — we download 1h and resample (see fetch_closes)
        self.timeframe     = os.environ.get("TIMEFRAME",     "1h")
        self.htf_timeframe = os.environ.get("HTF_TIMEFRAME", "4h")

        # ── Min tick per symbol (for SMA distance filter) ─────────────
        self.symbol_min_tick = {
            "EURUSD=X": 0.0001, "GBPJPY=X": 0.01,   "AUDJPY=X": 0.01,
            "XAUUSD=X": 0.01,   "USDCAD=X": 0.0001, "GBPUSD=X": 0.0001,
            "EURJPY=X": 0.01,   "USDJPY=X": 0.01,   "AUDUSD=X": 0.0001,
            "NZDUSD=X": 0.0001, "USDCHF=X": 0.0001, "EURGBP=X": 0.0001,
            "EURCAD=X": 0.0001, "GBPCAD=X": 0.0001, "AUDCAD=X": 0.0001,
            "EURAUD=X": 0.0001, "BTC-USD":  0.01,
            "ETH-USD":  0.01,
        }

        # ── Sessions (UTC) ────────────────────────────────────────────
        self.sessions = {
            "asia":    {"open": 23, "close": 8,  "name": "Asian Session"},
            "london":  {"open": 7,  "close": 16, "name": "London Session"},
            "newyork": {"open": 13, "close": 22, "name": "New York Session"},
        }

        # ── Minimum bars needed ───────────────────────────────────────
        self.min_bars     = max(200, self.lookback, 50 + self.sma_slope_period)
        self.min_htf_bars = self.trend_ma_period

        # ── Series cache (avoids re-downloading within same scan cycle) ─
        self._series_cache:   dict = {}
        self._cache_ttl_seconds    = int(os.environ.get("CACHE_TTL_S", str(20 * 60)))

        self.db = self.init_firebase()

    # ── Firebase ──────────────────────────────────────────────────────

    def init_firebase(self):
        try:
            if not firebase_admin._apps:
                print("🔑 Initialising Firebase…")
                if "FIREBASE_SERVICE_ACCOUNT" in os.environ:
                    sa_json = base64.b64decode(os.environ["FIREBASE_SERVICE_ACCOUNT"]).decode()
                    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
                        f.write(sa_json)
                        tmp = f.name
                    firebase_admin.initialize_app(credentials.Certificate(tmp))
                    os.unlink(tmp)
                    print("✅ Firebase initialised from env var")
                else:
                    path = os.path.join(
                        os.path.dirname(os.path.abspath(__file__)), "serviceAccountKey.json"
                    )
                    firebase_admin.initialize_app(credentials.Certificate(path))
                    print("✅ Firebase initialised from local file")
            else:
                print("✅ Firebase already initialised")
            return firestore.client()
        except Exception as e:
            print(f"❌ Firebase init error: {e}")
            return None

    def save_alert_to_firebase(self, alert_data):
        if not self.db:
            print("⚠️  Firebase not initialised, skipping save")
            return False
        try:
            alert_data.setdefault("timestamp", datetime.utcnow().isoformat() + "Z")
            alert_data["session"] = self.get_current_session()
            self.db.collection("trading_alerts").document().set(alert_data)
            print(f"✅ Alert saved: {alert_data['symbol']} — {alert_data['type']} / {alert_data['signal']}")
            return True
        except Exception as e:
            print(f"❌ Firebase save error: {e}")
            return False

    # ── Session helpers ───────────────────────────────────────────────

    def get_current_session(self):
        h = datetime.utcnow().hour
        for sid, s in self.sessions.items():
            o, c = s["open"], s["close"]
            if c < o:
                if h >= o or h < c:
                    return sid
            else:
                if o <= h < c:
                    return sid
        return "off_hours"

    # ── Data fetching via yfinance ────────────────────────────────────

    def _resample_1h_to_4h(self, df: pd.DataFrame) -> list:
        """Resample a 1H yfinance DataFrame to 4H close prices."""
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        close = df["Close"].squeeze()
        resampled = close.resample("4h").last().dropna()
        return resampled.tolist()

    def fetch_closes(self, symbol: str, interval: str, min_len: int) -> list:
        """
        Download close-price series using yfinance.

        Key facts about yfinance limits (no API key needed):
          • 1h  data  — up to 730 days back
          • 15m data  — up to 60 days back
          • 4h        — NOT a native interval; we download 1h + resample to 4H
          • No monthly cap, no hard rate limit (be polite with sleep between calls)

        Returns a plain Python list of floats, oldest → newest.
        """
        cache_key = (symbol, interval)
        cached    = self._series_cache.get(cache_key)
        if cached:
            try:
                age = (datetime.utcnow() - datetime.fromisoformat(cached["fetched_at"])).total_seconds()
                if age <= self._cache_ttl_seconds and len(cached["closes"]) >= min_len:
                    print(f"   📦 Cache hit: {symbol} {interval} ({len(cached['closes'])} bars)")
                    return cached["closes"]
            except Exception:
                pass

        closes = []

        # ── 4H: download 1H then resample ────────────────────────────
        if interval == "4h":
            for period in ["60d", "730d"]:
                try:
                    print(f"📊 yfinance: {symbol} 1h→4H resample (period={period})…")
                    df = yf.download(
                        symbol, interval="1h", period=period,
                        auto_adjust=True, progress=False, threads=False
                    )
                    if df.empty:
                        continue
                    closes = self._resample_1h_to_4h(df)
                    if len(closes) >= min_len:
                        break
                except Exception as e:
                    print(f"   ⚠️  yfinance 4h error for {symbol}: {e}")

        # ── Native intervals (1h, 15m, 1d …) ─────────────────────────
        else:
            periods_map = {
                "1h":  ["60d", "730d"],
                "15m": ["60d"],
                "1d":  ["2y", "5y"],
            }
            for period in periods_map.get(interval, ["60d", "730d"]):
                try:
                    print(f"📊 yfinance: {symbol} {interval} (period={period})…")
                    df = yf.download(
                        symbol, interval=interval, period=period,
                        auto_adjust=True, progress=False, threads=False
                    )
                    if df.empty:
                        continue
                    closes = df["Close"].squeeze().dropna().tolist()
                    if len(closes) >= min_len:
                        break
                except Exception as e:
                    print(f"   ⚠️  yfinance error for {symbol} {interval}: {e}")

        self._series_cache[cache_key] = {
            "closes":     closes,
            "fetched_at": datetime.utcnow().isoformat(),
        }
        return closes

    # ── Stationarity check ────────────────────────────────────────────

    def check_stationarity(self, prices: list, verbose: bool = True):
        """
        Multi-method stationarity check.
        Pine Script hardcodes p_value=0.01 (always passes the gate).
        Here we run a real ADF test combined with 3 supporting checks
        so the bot only trades genuinely mean-reverting conditions.
        Returns (final_p_value, is_stationary).
        """
        try:
            if len(prices) < self.lookback:
                return 1.0, False

            arr = np.array(prices[-self.lookback:])

            # 1. Augmented Dickey-Fuller
            try:
                adf_p  = adfuller(arr, autolag="AIC")[1]
                adf_ok = adf_p < self.stationarity_threshold
            except Exception:
                adf_p, adf_ok = 1.0, False

            # 2. Linear trend strength
            x              = np.arange(len(arr))
            slope          = stats.linregress(x, arr).slope
            trend_strength = abs(slope) / (np.std(arr) + 1e-10)
            no_strong_trend = trend_strength <= 0.1

            # 3. Rolling-mean stability
            rolling_mean   = pd.Series(arr).rolling(10).mean()
            mean_variation = rolling_mean.std() / (abs(rolling_mean.mean()) + 1e-10)
            mean_stable    = mean_variation < 0.05

            # 4. Mean-reversion cross rate
            m       = arr.mean()
            crosses = sum(
                1 for i in range(1, len(arr))
                if (arr[i - 1] - m) * (arr[i] - m) < 0
            )
            cross_rate     = crosses / len(arr)
            mean_reverting = cross_rate > 0.15

            is_stationary = adf_ok and no_strong_trend and mean_stable and mean_reverting
            final_p       = max(adf_p, 1.0 - cross_rate)

            if verbose:
                print(f"   Stationarity — ADF p={adf_p:.4f}  trend={trend_strength:.3f}"
                      f"  mean_var={mean_variation:.3f}  cross_rate={cross_rate:.3f}"
                      f"  → stationary={is_stationary}")
            return final_p, is_stationary

        except Exception as e:
            print(f"⚠️  Stationarity error: {e}")
            return 1.0, False

    # ── Core signal logic — direct Pine Script translation ─────────────

    def check_signals(self, symbol: str, prices: list, htf_prices: list,
                      verbose: bool = True) -> list:
        """
        Pine → Python variable map:
            close              prices[-1]
            htf_close          htf_prices[-1]
            is_stationary      check_stationarity()
            sma_lookback       rolling(lookback).mean().iloc[-1]
            stddev_lookback    rolling(lookback).std(ddof=0).iloc[-1]
            zscore             (close - sma_lookback) / stddev_lookback
            htf_sma            rolling(trend_ma_period).mean() on htf series
            htf_trend_bullish  htf_close > htf_sma
            sma50 / sma200     rolling 50/200 on prices
            bullish_trend      sma50 > sma200
            sma50_pct_change   (sma50-sma50[slope_period])/sma50[slope_period]*100
            sma_slope_ok       abs(sma50_pct_change) <= max_sma_slope_percent
            sma_distance_ticks abs(sma50-sma200) / mintick
            sma_distance_ok    sma_distance_ticks >= min_sma_distance_ticks
            bullish_valid      bullish_trend and sma_distance_ok and sma_slope_ok
            bearish_valid      bearish_trend and sma_distance_ok and sma_slope_ok
            entry_long         is_stationary and z<-t and htf_bullish and bullish_valid
            entry_short        is_stationary and z>+t and htf_bearish and bearish_valid
            exit_long          z>+t and bearish_valid
            exit_short         z<-t and bullish_valid
        """
        min_required = max(200, self.lookback, 50 + self.sma_slope_period)
        if len(prices) < min_required:
            if verbose:
                print(f"   ⚠️  Insufficient data: {len(prices)} bars (need {min_required})")
            return []
        if len(htf_prices) < self.trend_ma_period:
            if verbose:
                print(f"   ⚠️  Insufficient HTF data: {len(htf_prices)} bars "
                      f"(need {self.trend_ma_period})")
            return []

        closes     = pd.Series(prices,     dtype="float64")
        htf_closes = pd.Series(htf_prices, dtype="float64")

        # ── Stationarity ──────────────────────────────────────────────
        p_value, is_stationary = self.check_stationarity(prices, verbose=verbose)

        # ── Z-score ───────────────────────────────────────────────────
        sma_lb = closes.rolling(self.lookback).mean().iloc[-1]
        std_lb = closes.rolling(self.lookback).std(ddof=0).iloc[-1]
        if pd.isna(sma_lb) or pd.isna(std_lb) or std_lb == 0:
            if verbose:
                print("   ⚠️  NaN / zero std in z-score calc")
            return []
        current_price = float(closes.iloc[-1])
        zscore        = (current_price - float(sma_lb)) / float(std_lb)

        # ── HTF trend ─────────────────────────────────────────────────
        htf_sma_s     = htf_closes.rolling(self.trend_ma_period).mean()
        htf_sma_val   = htf_sma_s.iloc[-1]
        htf_close_val = float(htf_closes.iloc[-1])
        if pd.isna(htf_sma_val):
            if verbose:
                print("   ⚠️  NaN HTF SMA")
            return []
        htf_trend_bullish = htf_close_val > float(htf_sma_val)
        htf_trend_bearish = htf_close_val < float(htf_sma_val)

        # ── SMA 50 / 200 ──────────────────────────────────────────────
        sma50_s  = closes.rolling(50).mean()
        sma200_s = closes.rolling(200).mean()
        sma50    = float(sma50_s.iloc[-1])
        sma200   = float(sma200_s.iloc[-1])
        if pd.isna(sma50) or pd.isna(sma200):
            if verbose:
                print("   ⚠️  NaN SMA50/200")
            return []

        bullish_trend = sma50 > sma200
        bearish_trend = sma50 < sma200

        # ── SMA slope filter ──────────────────────────────────────────
        prev_sma50 = float(sma50_s.iloc[-1 - self.sma_slope_period]) \
            if len(sma50_s) > self.sma_slope_period else sma50
        sma50_pct_change = ((sma50 - prev_sma50) / prev_sma50 * 100) if prev_sma50 != 0 else 0.0
        sma_slope_ok     = abs(sma50_pct_change) <= self.max_sma_slope_percent

        # ── SMA distance filter ───────────────────────────────────────
        min_tick           = self.symbol_min_tick.get(symbol, 0.0001)
        sma_distance_ticks = abs(sma50 - sma200) / min_tick
        sma_distance_ok    = sma_distance_ticks >= self.min_sma_distance_ticks

        # ── Combined trend validity ───────────────────────────────────
        bullish_valid = bullish_trend and sma_distance_ok and sma_slope_ok
        bearish_valid = bearish_trend and sma_distance_ok and sma_slope_ok

        if verbose:
            print(f"   is_stationary={is_stationary}  p={p_value:.4f}")
            print(f"   Z-score={zscore:.3f}  (threshold ±{self.zscore_threshold})")
            print(f"   SMA50={sma50:.5f}  SMA200={sma200:.5f}  "
                  f"slope={sma50_pct_change:.3f}%  dist={sma_distance_ticks:.0f}tks")
            print(f"   HTF bullish={htf_trend_bullish}  "
                  f"bullish_valid={bullish_valid}  bearish_valid={bearish_valid}")

        # ── Pine Script signal conditions (verbatim) ──────────────────
        entry_long  = is_stationary and zscore < -self.zscore_threshold \
                      and htf_trend_bullish and bullish_valid
        entry_short = is_stationary and zscore >  self.zscore_threshold \
                      and htf_trend_bearish and bearish_valid
        exit_long   = zscore >  self.zscore_threshold and bearish_valid
        exit_short  = zscore < -self.zscore_threshold and bullish_valid

        # ── Build signal payloads ─────────────────────────────────────
        ctx = {
            "price":               current_price,
            "zscore":              float(zscore),
            "stationarity_p":      float(p_value),
            "is_stationary":       bool(is_stationary),
            "sma50":               sma50,
            "sma200":              sma200,
            "sma50_slope_pct":     float(sma50_pct_change),
            "sma_distance_ticks":  float(sma_distance_ticks),
            "htf_sma":             float(htf_sma_val),
            "htf_close":           htf_close_val,
            "htf_trend_bullish":   bool(htf_trend_bullish),
            "bullish_valid":       bool(bullish_valid),
            "bearish_valid":       bool(bearish_valid),
            "timeframe":           self.timeframe,
            "htf_timeframe":       self.htf_timeframe,
        }

        signals = []
        def add(sig_type, sig_name, confidence):
            signals.append({"type": sig_type, "signal": sig_name,
                            "confidence": confidence, **ctx})
            if verbose:
                print(f"   🎯 {sig_type} — {sig_name}")

        if entry_long:   add("BUY",        "stationarity_entry_long",  "HIGH")
        if entry_short:  add("SELL",       "stationarity_entry_short", "HIGH")
        if exit_long:    add("EXIT_LONG",  "stationarity_exit_long",   "MEDIUM")
        if exit_short:   add("EXIT_SHORT", "stationarity_exit_short",  "MEDIUM")

        if not signals and verbose:
            print("   ℹ️  No signal on this bar")

        return signals

    # ── Watchlist scan ────────────────────────────────────────────────

    def scan_watchlist(self):
        global last_check_time, total_signals

        print(f"\n{'='*60}")
        print(f"🚀 Scan at {datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC")
        print(f"   Strategy   : Stationarity + Z-score + HTF SMA + SMA50/200 filters")
        print(f"   Data source: yfinance (free, no API key, no rate limits)")
        print(f"   Chart TF   : {self.timeframe}  |  HTF : {self.htf_timeframe}")
        print(f"{'='*60}")

        session         = self.get_current_session()
        last_check_time = datetime.utcnow().isoformat()
        signals_found   = 0

        for symbol in self.watchlist:
            print(f"\n🔍 {symbol}…")
            try:
                prices     = self.fetch_closes(symbol, self.timeframe,     self.min_bars)
                htf_prices = self.fetch_closes(symbol, self.htf_timeframe, self.min_htf_bars)

                if len(prices) < self.min_bars:
                    print(f"   ⚠️  Skipping: only {len(prices)} {self.timeframe} bars "
                          f"(need {self.min_bars})")
                    continue
                if len(htf_prices) < self.min_htf_bars:
                    print(f"   ⚠️  Skipping: only {len(htf_prices)} {self.htf_timeframe} bars "
                          f"(need {self.min_htf_bars})")
                    continue

                print(f"   📈 {self.timeframe} bars: {len(prices)}"
                      f"  |  {self.htf_timeframe} bars: {len(htf_prices)}")

                signals = self.check_signals(symbol, prices, htf_prices, verbose=True)

                for sig in signals:
                    saved = self.save_alert_to_firebase(
                        {"symbol": symbol, "session": session, **sig}
                    )
                    if saved:
                        signals_found += 1
                        total_signals  += 1

            except Exception as e:
                print(f"   ❌ Error processing {symbol}: {e}")

            # Small polite pause between symbols
            time.sleep(1.0)

        print(f"\n{'='*60}")
        print(f"✅ Scan done — {signals_found} new signal(s)")
        print(f"{'='*60}")


# ──────────────────────────────────────────────
# Flask routes
# ──────────────────────────────────────────────

@app.route("/")
def home():
    return jsonify({
        "status":        "running",
        "data_source":   "yfinance (free, no API key)",
        "last_check":    last_check_time,
        "total_signals": total_signals,
    })

@app.route("/health")
def health():
    return "OK", 200

@app.route("/status")
def status():
    if not bot_instance:
        return jsonify({"error": "Bot not initialised"}), 500
    return jsonify({
        "status":              "running",
        "data_source":         "yfinance (free, no API key)",
        "strategy":            "Stationarity Z-score + HTF SMA + SMA50/200 filters",
        "timeframe":           bot_instance.timeframe,
        "htf_timeframe":       bot_instance.htf_timeframe,
        "lookback":            bot_instance.lookback,
        "zscore_threshold":    bot_instance.zscore_threshold,
        "stationarity_thresh": bot_instance.stationarity_threshold,
        "current_session":     bot_instance.get_current_session(),
        "current_time_utc":    datetime.utcnow().isoformat(),
        "last_check":          last_check_time,
        "total_signals":       total_signals,
        "watchlist":           bot_instance.watchlist,
    })


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────

def run_bot():
    while True:
        try:
            bot_instance.scan_watchlist()
        except Exception as e:
            print(f"❌ Bot scan error: {e}")
        print("💤 Sleeping 10 minutes until next scan…")
        time.sleep(600)


if __name__ == "__main__":
    print("🚀 Initialising Stationarity Trading Bot (yfinance edition)…")
    print(f"Python : {sys.version}")
    print(f"UTC    : {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}")
    print("📡 Data source: yfinance — free, no API key, no rate limits, no monthly cap")

    bot_instance = FibSMATradingBot()          # ← no API key argument needed

    write_startup_test_to_firebase(bot_instance.db, bot=bot_instance)

    threading.Thread(target=run_bot, daemon=True).start()
    print("✅ Bot thread started")

    port = int(os.environ.get("PORT", 10000))
    print(f"🚀 Flask on port {port}")
    app.run(host="0.0.0.0", port=port)