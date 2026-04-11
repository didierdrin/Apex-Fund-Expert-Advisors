import requests
import time
import os
import sys
import base64
import tempfile
from datetime import datetime
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
import firebase_admin
from firebase_admin import credentials, firestore
from flask import Flask, jsonify
import threading
import socket
from scipy import stats

app = Flask(__name__)

bot_instance = None
last_check_time = None
total_signals = 0
_startup_test_written = False


# ──────────────────────────────────────────────
# Startup test
# ──────────────────────────────────────────────

def fetch_h1_prices_snapshot(bot):
    prices, errors = {}, {}
    for symbol in getattr(bot, "watchlist", []):
        try:
            hist = bot.get_historical_data(symbol, interval="1h", range="2d")
            closes = bot.extract_close_prices(hist)
            if closes:
                prices[symbol] = float(closes[-1])
            else:
                errors[symbol] = "no_close_prices"
        except Exception as e:
            errors[symbol] = str(e)
    return prices, errors


def write_startup_test_to_firebase(db, *, bot=None):
    global _startup_test_written
    if _startup_test_written:
        return False
    if not db:
        print("⚠️  Firebase not initialised, skipping startup test write")
        _startup_test_written = True
        return False
    try:
        now = datetime.utcnow()
        doc_id = f"{now.strftime('%Y%m%dT%H%M%S')}_{os.getpid()}"
        payload = {
            "type": "backend_startup_test",
            "timestamp_utc": now.isoformat() + "Z",
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "python_version": sys.version,
        }
        if bot is not None:
            prices, errs = fetch_h1_prices_snapshot(bot)
            payload.update({
                "timeframe": bot.timeframe,
                "h1_prices": prices,
                "h1_price_errors": errs,
                "h1_prices_count": len(prices),
                "h1_price_errors_count": len(errs),
                "rapidapi_requests_used": getattr(bot, "request_count", None),
            })
        db.collection("backend_startup_test").document(doc_id).set(payload)
        _startup_test_written = True
        print("✅ Startup test written to Firebase")
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

    Signal logic (verbatim from Pine):
        entry_long  = is_stationary AND zscore < -threshold AND htf_trend_bullish
                      AND bullish_valid
        entry_short = is_stationary AND zscore >  threshold AND htf_trend_bearish
                      AND bearish_valid
        exit_long   = zscore >  threshold AND bearish_valid
        exit_short  = zscore < -threshold AND bullish_valid

    where:
        bullish_valid = sma50>sma200  AND dist>=500ticks  AND slope<=0.25%
        bearish_valid = sma50<sma200  AND dist>=500ticks  AND slope<=0.25%

    Pine htf = "15" (relative to chart TF). Since this bot uses 1H candles as
    its base chart, the next meaningful HTF is 4H (env: HTF_TIMEFRAME, default 4h).
    """

    def __init__(self, api_key):
        self.api_key  = api_key
        self.base_url = "https://yahoo-finance166.p.rapidapi.com"
        self.headers  = {
            "x-rapidapi-key":  api_key,
            "x-rapidapi-host": "yahoo-finance166.p.rapidapi.com",
        }

        self.watchlist = [
            "EURUSD=X", "GBPJPY=X", "AUDJPY=X", "XAUUSD=X", "USDCAD=X",
            "GBPUSD=X", "EURJPY=X", "USDJPY=X", "AUDUSD=X", "NZDUSD=X",
            "USDCHF=X", "EURGBP=X", "EURCAD=X", "GBPCAD=X", "AUDCAD=X",
            "EURAUD=X", "XAUEUR=X", "BTC-USD",  "ETH-USD",
        ]

        self.request_count = 0
        self.max_requests  = 450

        # ── Strategy parameters (matching Pine Script inputs) ──────────
        self.lookback               = int(os.environ.get("LOOKBACK",             "50"))
        self.stationarity_threshold = float(os.environ.get("STATIONARITY_THRESH","0.05"))
        self.zscore_threshold       = float(os.environ.get("ZSCORE_THRESH",      "1.5"))
        self.trend_ma_period        = int(os.environ.get("TREND_MA_PERIOD",      "20"))
        self.sma_slope_period       = int(os.environ.get("SMA_SLOPE_PERIOD",     "5"))
        self.max_sma_slope_percent  = float(os.environ.get("MAX_SMA_SLOPE_PCT",  "0.25"))
        self.min_sma_distance_ticks = int(os.environ.get("MIN_SMA_DIST_TICKS",  "500"))

        # ── Timeframes ────────────────────────────────────────────────
        self.timeframe      = os.environ.get("TIMEFRAME",     "1h")
        self.htf_timeframe  = os.environ.get("HTF_TIMEFRAME", "4h")
        self.data_range     = os.environ.get("DATA_RANGE",    "1mo")
        self.htf_data_range = os.environ.get("HTF_DATA_RANGE","3mo")

        # ── Min tick per symbol ───────────────────────────────────────
        self.symbol_min_tick = {
            "EURUSD=X": 0.0001, "GBPJPY=X": 0.01,   "AUDJPY=X": 0.01,
            "XAUUSD=X": 0.01,   "USDCAD=X": 0.0001, "GBPUSD=X": 0.0001,
            "EURJPY=X": 0.01,   "USDJPY=X": 0.01,   "AUDUSD=X": 0.0001,
            "NZDUSD=X": 0.0001, "USDCHF=X": 0.0001, "EURGBP=X": 0.0001,
            "EURCAD=X": 0.0001, "GBPCAD=X": 0.0001, "AUDCAD=X": 0.0001,
            "EURAUD=X": 0.0001, "XAUEUR=X": 0.01,   "BTC-USD":  0.01,
            "ETH-USD":  0.01,
        }

        # ── Sessions ──────────────────────────────────────────────────
        self.sessions = {
            "asia":    {"open": 23, "close": 8,  "name": "Asian Session"},
            "london":  {"open": 7,  "close": 16, "name": "London Session"},
            "newyork": {"open": 13, "close": 22, "name": "New York Session"},
        }

        # ── Minimum bars ──────────────────────────────────────────────
        self.min_bars     = max(200, self.lookback, 50 + self.sma_slope_period)
        self.min_htf_bars = self.trend_ma_period

        self._series_cache: dict = {}
        self.db = self.init_firebase()

    # ── Firebase ──────────────────────────────

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
                    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "serviceAccountKey.json")
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

    # ── Session helpers ────────────────────────

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

    # ── Data fetching ──────────────────────────

    def get_historical_data(self, symbol, interval="1h", range="1mo"):
        if self.request_count >= self.max_requests:
            print(f"⚠️  Request limit reached for {symbol}")
            return None
        url = f"{self.base_url}/api/stock/get-chart"
        params = {"symbol": symbol, "interval": interval, "range": range, "region": "US"}
        try:
            print(f"📊 Fetching {interval} data for {symbol} (range={range})…")
            resp = requests.get(url, headers=self.headers, params=params)
            self.request_count += 1
            if resp.status_code == 200:
                return resp.json()
            print(f"⚠️  API error {resp.status_code} for {symbol}")
        except Exception as e:
            print(f"❌ Fetch error for {symbol}: {e}")
        return None

    def extract_close_prices(self, hist_data):
        try:
            quotes = hist_data["chart"]["result"][0]["indicators"]["quote"][0]
            return [p for p in quotes.get("close", []) if p is not None]
        except Exception:
            return []

    def get_close_series(self, symbol, *, interval, preferred_range, min_len,
                         max_age_seconds=1200):
        key = (symbol, interval)
        cached = self._series_cache.get(key)
        if cached:
            try:
                age = (datetime.utcnow() - datetime.fromisoformat(cached["fetched_at"])).total_seconds()
                if age <= max_age_seconds and len(cached["closes"]) >= min_len:
                    return cached["closes"]
            except Exception:
                pass
        ranges = [preferred_range, "3mo", "6mo", "1y", "2y"]
        best = []
        for r in ranges:
            closes = self.extract_close_prices(
                self.get_historical_data(symbol, interval=interval, range=r)
            )
            if len(closes) > len(best):
                best = closes
            if len(closes) >= min_len:
                break
        self._series_cache[key] = {"closes": best, "fetched_at": datetime.utcnow().isoformat()}
        return best

    # ── Stationarity ──────────────────────────

    def check_stationarity(self, prices, verbose=True):
        """
        Multi-method approximation of ADF stationarity test.
        Pine Script hardcodes p_value = 0.01; here we compute it properly.
        Returns (final_p_value, is_stationary).
        """
        try:
            if len(prices) < self.lookback:
                return 1.0, False

            arr = np.array(prices[-self.lookback:])

            # 1. Augmented Dickey-Fuller
            try:
                adf_p = adfuller(arr, autolag="AIC")[1]
                adf_ok = adf_p < self.stationarity_threshold
            except Exception:
                adf_p, adf_ok = 1.0, False

            # 2. Linear trend strength
            x = np.arange(len(arr))
            slope = stats.linregress(x, arr).slope
            trend_strength = abs(slope) / (np.std(arr) + 1e-10)
            no_strong_trend = trend_strength <= 0.1

            # 3. Rolling-mean stability
            rolling_mean = pd.Series(arr).rolling(10).mean()
            mean_variation = rolling_mean.std() / (abs(rolling_mean.mean()) + 1e-10)
            mean_stable = mean_variation < 0.05

            # 4. Mean-reversion cross rate
            m = arr.mean()
            crosses = sum(
                1 for i in range(1, len(arr))
                if (arr[i - 1] - m) * (arr[i] - m) < 0
            )
            cross_rate = crosses / len(arr)
            mean_reverting = cross_rate > 0.15

            is_stationary = adf_ok and no_strong_trend and mean_stable and mean_reverting
            final_p = max(adf_p, 1.0 - cross_rate)

            if verbose:
                print(f"   Stationarity — ADF p={adf_p:.4f}  trend_strength={trend_strength:.3f}"
                      f"  mean_var={mean_variation:.3f}  cross_rate={cross_rate:.3f}"
                      f"  → stationary={is_stationary}")
            return final_p, is_stationary

        except Exception as e:
            print(f"⚠️  Stationarity error: {e}")
            return 1.0, False

    # ── Signal logic — direct Pine translation ─

    def check_signals(self, symbol, prices, htf_prices, verbose=True):
        """
        Pine Script variable → Python variable mapping:

            close               prices[-1]
            htf_close           htf_prices[-1]
            is_stationary       check_stationarity()
            sma_lookback        rolling(lookback).mean().iloc[-1]
            stddev_lookback     rolling(lookback).std(ddof=0).iloc[-1]
            zscore              (close - sma_lookback) / stddev_lookback
            htf_sma             rolling(trend_ma_period).mean() on htf series
            htf_trend_bullish   htf_close > htf_sma
            sma50 / sma200      rolling 50/200 on 1H prices
            bullish_trend       sma50 > sma200
            sma50_pct_change    (sma50 - sma50[slope_period]) / sma50[slope_period] * 100
            sma_slope_ok        abs(sma50_pct_change) <= max_sma_slope_percent
            sma_distance_ticks  abs(sma50 - sma200) / mintick
            sma_distance_ok     sma_distance_ticks >= min_sma_distance_ticks
            bullish_valid       bullish_trend and sma_distance_ok and sma_slope_ok
            bearish_valid       bearish_trend and sma_distance_ok and sma_slope_ok
            entry_long          is_stationary and zscore<-t and htf_bullish and bullish_valid
            entry_short         is_stationary and zscore>+t and htf_bearish and bearish_valid
            exit_long           zscore>+t and bearish_valid
            exit_short          zscore<-t and bullish_valid
        """
        min_required = max(200, self.lookback, 50 + self.sma_slope_period)
        if len(prices) < min_required:
            if verbose:
                print(f"   ⚠️  Insufficient data: {len(prices)} bars (need {min_required})")
            return []
        if len(htf_prices) < self.trend_ma_period:
            if verbose:
                print(f"   ⚠️  Insufficient HTF data: {len(htf_prices)} bars (need {self.trend_ma_period})")
            return []

        closes     = pd.Series(prices,     dtype="float64")
        htf_closes = pd.Series(htf_prices, dtype="float64")

        # ── Stationarity ──────────────────────────────────────────────
        p_value, is_stationary = self.check_stationarity(prices, verbose=verbose)

        # ── Z-score ───────────────────────────────────────────────────
        sma_lb  = closes.rolling(self.lookback).mean().iloc[-1]
        std_lb  = closes.rolling(self.lookback).std(ddof=0).iloc[-1]
        if pd.isna(sma_lb) or pd.isna(std_lb) or std_lb == 0:
            if verbose:
                print("   ⚠️  NaN / zero std in z-score calc")
            return []
        current_price = float(closes.iloc[-1])
        zscore = (current_price - float(sma_lb)) / float(std_lb)

        # ── HTF trend ─────────────────────────────────────────────────
        htf_sma_s   = htf_closes.rolling(self.trend_ma_period).mean()
        htf_sma_val = htf_sma_s.iloc[-1]
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
        # Pine: (sma50 - sma50[sma_slope_period]) / sma50[sma_slope_period] * 100
        prev_sma50 = float(sma50_s.iloc[-1 - self.sma_slope_period]) \
            if len(sma50_s) > self.sma_slope_period else sma50
        sma50_pct_change = ((sma50 - prev_sma50) / prev_sma50 * 100) if prev_sma50 != 0 else 0.0
        sma_slope_ok = abs(sma50_pct_change) <= self.max_sma_slope_percent

        # ── SMA distance filter ───────────────────────────────────────
        # Pine: math.abs(sma50 - sma200) / syminfo.mintick
        min_tick = self.symbol_min_tick.get(symbol, 0.0001)
        sma_distance_ticks = abs(sma50 - sma200) / min_tick
        sma_distance_ok    = sma_distance_ticks >= self.min_sma_distance_ticks

        # ── Combined validity ─────────────────────────────────────────
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

        # ── Build signals ─────────────────────────────────────────────
        ctx = {
            "price":               current_price,
            "zscore":              float(zscore),
            "stationarity_p":      float(p_value),
            "is_stationary":       is_stationary,
            "sma50":               sma50,
            "sma200":              sma200,
            "sma50_slope_pct":     float(sma50_pct_change),
            "sma_distance_ticks":  float(sma_distance_ticks),
            "htf_sma":             float(htf_sma_val),
            "htf_close":           htf_close_val,
            "htf_trend_bullish":   htf_trend_bullish,
            "bullish_valid":       bullish_valid,
            "bearish_valid":       bearish_valid,
            "timeframe":           self.timeframe,
            "htf_timeframe":       self.htf_timeframe,
        }

        signals = []
        def add(sig_type, sig_name, confidence):
            signals.append({"type": sig_type, "signal": sig_name, "confidence": confidence, **ctx})
            if verbose:
                print(f"   🎯 {sig_type} — {sig_name}")

        if entry_long:   add("BUY",        "stationarity_entry_long",  "HIGH")
        if entry_short:  add("SELL",       "stationarity_entry_short", "HIGH")
        if exit_long:    add("EXIT_LONG",  "stationarity_exit_long",   "MEDIUM")
        if exit_short:   add("EXIT_SHORT", "stationarity_exit_short",  "MEDIUM")

        if not signals and verbose:
            print("   ℹ️  No signal on this bar")

        return signals

    # ── Watchlist scan ─────────────────────────

    def scan_watchlist(self):
        global last_check_time, total_signals

        print(f"\n{'='*60}")
        print(f"🚀 Scan at {datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC")
        print(f"   Strategy : Stationarity + Z-score + HTF SMA + SMA50/200 filters")
        print(f"   Chart TF : {self.timeframe}  |  HTF : {self.htf_timeframe}")
        print(f"{'='*60}")

        session = self.get_current_session()
        last_check_time = datetime.utcnow().isoformat()
        signals_found = 0

        for symbol in self.watchlist:
            if self.request_count >= self.max_requests:
                print("⚠️  Monthly request limit reached. Stopping scan.")
                break

            print(f"\n🔍 {symbol}…")
            try:
                prices = self.get_close_series(
                    symbol, interval=self.timeframe,
                    preferred_range=self.data_range, min_len=self.min_bars,
                )
                htf_prices = self.get_close_series(
                    symbol, interval=self.htf_timeframe,
                    preferred_range=self.htf_data_range, min_len=self.min_htf_bars,
                )

                if len(prices) < self.min_bars:
                    print(f"   ⚠️  Skipping: only {len(prices)} {self.timeframe} bars (need {self.min_bars})")
                    continue
                if len(htf_prices) < self.min_htf_bars:
                    print(f"   ⚠️  Skipping: only {len(htf_prices)} {self.htf_timeframe} bars (need {self.min_htf_bars})")
                    continue

                print(f"   📈 {self.timeframe} bars: {len(prices)}  |  {self.htf_timeframe} bars: {len(htf_prices)}")

                signals = self.check_signals(symbol, prices, htf_prices, verbose=True)

                for sig in signals:
                    saved = self.save_alert_to_firebase({"symbol": symbol, "session": session, **sig})
                    if saved:
                        signals_found += 1
                        total_signals  += 1

            except Exception as e:
                print(f"   ❌ Error: {e}")

            time.sleep(1.5)

        print(f"\n{'='*60}")
        print(f"✅ Scan done — {signals_found} new signal(s)  |  "
              f"Requests: {self.request_count}/{self.max_requests}")
        print(f"{'='*60}")


# ──────────────────────────────────────────────
# Flask
# ──────────────────────────────────────────────

@app.route("/")
def home():
    return jsonify({
        "status": "running",
        "last_check": last_check_time,
        "total_signals": total_signals,
        "requests_used": bot_instance.request_count if bot_instance else 0,
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
        "requests_used":       bot_instance.request_count,
        "requests_remaining":  bot_instance.max_requests - bot_instance.request_count,
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
        time.sleep(600)


if __name__ == "__main__":
    print("🚀 Initialising Stationarity Trading Bot…")
    print(f"Python : {sys.version}")
    print(f"UTC    : {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}")

    API_KEY = os.environ.get("RAPIDAPI_KEY")
    if not API_KEY:
        print("❌ RAPIDAPI_KEY not set")
        sys.exit(1)

    bot_instance = FibSMATradingBot(API_KEY)

    write_startup_test_to_firebase(bot_instance.db, bot=bot_instance)

    threading.Thread(target=run_bot, daemon=True).start()
    print("✅ Bot thread started")

    port = int(os.environ.get("PORT", 10000))
    print(f"🚀 Flask on port {port}")
    app.run(host="0.0.0.0", port=port)