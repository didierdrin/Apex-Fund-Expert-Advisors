"""
SR Break + Stationarity Combined — yfinance edition
====================================================
Data   : yfinance (free, no API key)
Pine   : "SR Break + Stationarity Combined" by Didier_drin (@version=5)
Chart  : 1-minute — fires on green/red filtered triangles (BUY / SELL)
HTF    : 15-minute (stationarity trend filter)
"""

import time
import os
import sys
import base64
import tempfile
from datetime import datetime

import pandas as pd
import numpy as np
import yfinance as yf
import firebase_admin
from firebase_admin import credentials, firestore
from flask import Flask, jsonify
import threading
import socket

app = Flask(__name__)

bot_instance = None
last_check_time = None
total_signals = 0
_startup_test_written = False


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
            "data_source": "yfinance (free, no API key)",
            "strategy": "sr_break_stationarity_1m",
            "timeframe": getattr(bot, "timeframe", "1m") if bot else None,
            "htf_timeframe": getattr(bot, "htf_timeframe", "15m") if bot else None,
            "watchlist_count": len(getattr(bot, "watchlist", [])) if bot else None,
        }
        db.collection("backend_startup_test").document(doc_id).set(payload)
        _startup_test_written = True
        print("✅ Startup test written to Firebase")
        return True
    except Exception as e:
        _startup_test_written = True
        print(f"❌ Error writing startup test: {e}")
        return False


def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten yfinance multi-index columns to Open/High/Low/Close/Volume."""
    if df is None or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [str(c[0]).title() for c in df.columns]

    rename = {}
    for col in df.columns:
        low = str(col).lower()
        if low in ("open", "high", "low", "close", "volume"):
            rename[col] = low.capitalize() if low != "volume" else "Volume"
    df = df.rename(columns=rename)

    needed = ["Open", "High", "Low", "Close", "Volume"]
    if not all(c in df.columns for c in needed):
        return pd.DataFrame()

    out = df[needed].copy()
    out = out.apply(pd.to_numeric, errors="coerce").dropna()
    if out.index.tz is None:
        out.index = out.index.tz_localize("UTC")
    else:
        out.index = out.index.tz_convert("UTC")
    return out


def pivothigh_series(high: np.ndarray, left: int, right: int) -> np.ndarray:
    """ta.pivothigh(left, right) — value at pivot bar when confirmed."""
    n = len(high)
    out = np.full(n, np.nan)
    for i in range(left + right, n):
        center = i - right
        window = high[center - left : center + right + 1]
        if len(window) == left + right + 1 and high[center] == np.nanmax(window):
            out[i] = high[center]
    return out


def pivotlow_series(low: np.ndarray, left: int, right: int) -> np.ndarray:
    n = len(low)
    out = np.full(n, np.nan)
    for i in range(left + right, n):
        center = i - right
        window = low[center - left : center + right + 1]
        if len(window) == left + right + 1 and low[center] == np.nanmin(window):
            out[i] = low[center]
    return out


def fixnan_forward(arr: np.ndarray) -> np.ndarray:
    """Pine fixnan — carry last non-NaN value forward."""
    out = arr.copy()
    last = np.nan
    for i in range(len(out)):
        if not np.isnan(out[i]):
            last = out[i]
        elif not np.isnan(last):
            out[i] = last
    return out


def shift_one(arr: np.ndarray) -> np.ndarray:
    out = np.full(len(arr), np.nan)
    if len(arr) > 1:
        out[1:] = arr[:-1]
    return out


def crossover(series_a: np.ndarray, series_b: np.ndarray) -> np.ndarray:
    out = np.zeros(len(series_a), dtype=bool)
    for i in range(1, len(series_a)):
        if np.isnan(series_a[i - 1]) or np.isnan(series_b[i - 1]):
            continue
        out[i] = series_a[i - 1] <= series_b[i - 1] and series_a[i] > series_b[i]
    return out


def crossunder(series_a: np.ndarray, series_b: np.ndarray) -> np.ndarray:
    out = np.zeros(len(series_a), dtype=bool)
    for i in range(1, len(series_a)):
        if np.isnan(series_a[i - 1]) or np.isnan(series_b[i - 1]):
            continue
        out[i] = series_a[i - 1] >= series_b[i - 1] and series_a[i] < series_b[i]
    return out


class SRStationarityBot:
    """
    Pine: SR Break + Stationarity Combined (1m chart).

    Emits Firebase alerts when:
      showBuyTriangle  = isBlueBar and entry_long  and distToResistance <= buffer
      showSellTriangle = isRedBar  and entry_short and distToSupport    <= buffer
    Only on the edge (false → true) on the latest closed bar.
    """

    def __init__(self):
        self.watchlist = [
            "EURUSD=X", "GBPJPY=X", "AUDJPY=X", "XAUUSD=X", "USDCAD=X",
            "GBPUSD=X", "EURJPY=X", "USDJPY=X", "AUDUSD=X", "NZDUSD=X",
            "USDCHF=X", "EURGBP=X", "EURCAD=X", "GBPCAD=X", "AUDCAD=X",
            "EURAUD=X", "BTC-USD", "ETH-USD",
        ]

        # Pine inputs
        self.toggle_breaks = os.environ.get("TOGGLE_BREAKS", "true").lower() in ("1", "true", "yes")
        self.left_bars = int(os.environ.get("LEFT_BARS", "15"))
        self.right_bars = int(os.environ.get("RIGHT_BARS", "15"))
        self.volume_thresh = float(os.environ.get("VOLUME_THRESH", "20"))

        self.lookback = int(os.environ.get("LOOKBACK", "50"))
        self.zscore_threshold = float(os.environ.get("ZSCORE_THRESH", "1.5"))
        self.trend_ma_period = int(os.environ.get("TREND_MA_PERIOD", "20"))
        self.sma_slope_period = int(os.environ.get("SMA_SLOPE_PERIOD", "5"))
        self.max_sma_slope_percent = float(os.environ.get("MAX_SMA_SLOPE_PCT", "0.25"))
        self.min_sma_distance_ticks = int(os.environ.get("MIN_SMA_DIST_TICKS", "500"))
        self.stationarity_threshold = float(os.environ.get("STATIONARITY_THRESH", "0.05"))

        self.timeframe = os.environ.get("TIMEFRAME", "1m")
        self.htf_timeframe = os.environ.get("HTF_TIMEFRAME", "15m")

        # 1m buffer = 6 pips (Pine); 1h would be 60 — we run 1m only
        self.buffer_pips_1m = int(os.environ.get("BUFFER_PIPS_1M", "6"))

        self.symbol_min_tick = {
            "EURUSD=X": 0.0001, "GBPJPY=X": 0.01, "AUDJPY=X": 0.01,
            "XAUUSD=X": 0.01, "USDCAD=X": 0.0001, "GBPUSD=X": 0.0001,
            "EURJPY=X": 0.01, "USDJPY=X": 0.01, "AUDUSD=X": 0.0001,
            "NZDUSD=X": 0.0001, "USDCHF=X": 0.0001, "EURGBP=X": 0.0001,
            "EURCAD=X": 0.0001, "GBPCAD=X": 0.0001, "AUDCAD=X": 0.0001,
            "EURAUD=X": 0.0001, "BTC-USD": 0.01, "ETH-USD": 0.01,
        }

        self.sessions = {
            "asia": {"open": 23, "close": 8, "name": "Asian Session"},
            "london": {"open": 7, "close": 16, "name": "London Session"},
            "newyork": {"open": 13, "close": 22, "name": "New York Session"},
        }

        pivot_warmup = self.left_bars + self.right_bars + 5
        self.min_bars = max(250, self.lookback, 50 + self.sma_slope_period, pivot_warmup + 50)
        self.min_htf_bars = self.trend_ma_period + 5

        self._ohlcv_cache = {}
        self._cache_ttl_seconds = int(os.environ.get("CACHE_TTL_S", "55"))
        self.scan_sleep_seconds = int(os.environ.get("SCAN_INTERVAL_S", "60"))

        self.db = self.init_firebase()

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
                else:
                    path = os.path.join(
                        os.path.dirname(os.path.abspath(__file__)), "serviceAccountKey.json"
                    )
                    firebase_admin.initialize_app(credentials.Certificate(path))
                print("✅ Firebase initialised")
            return firestore.client()
        except Exception as e:
            print(f"❌ Firebase init error: {e}")
            return None

    def save_alert_to_firebase(self, alert_data):
        if not self.db:
            print("⚠️  Firebase not initialised, skipping save")
            return False
        try:
            now_utc = datetime.utcnow()
            alert_data["session"] = self.get_current_session()
            alert_data["timestamp_ms"] = int(now_utc.timestamp() * 1000)
            alert_data["timestamp_iso"] = now_utc.isoformat() + "Z"
            alert_data["timestamp"] = firestore.SERVER_TIMESTAMP
            alert_data["strategy"] = "sr_break_stationarity_1m"

            for key, value in list(alert_data.items()):
                if isinstance(value, (np.bool_, np.integer, np.floating)):
                    alert_data[key] = value.item()

            self.db.collection("trading_alerts").document().set(alert_data)
            print(f"✅ Alert saved: {alert_data['symbol']} — {alert_data['type']} / {alert_data['signal']}")
            return True
        except Exception as e:
            print(f"❌ Firebase save error: {e}")
            return False

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

    def fetch_ohlcv(self, symbol: str, interval: str, min_len: int) -> pd.DataFrame:
        cache_key = (symbol, interval)
        cached = self._ohlcv_cache.get(cache_key)
        if cached is not None:
            try:
                age = (datetime.utcnow() - datetime.fromisoformat(cached["fetched_at"])).total_seconds()
                if age <= self._cache_ttl_seconds and len(cached["df"]) >= min_len:
                    print(f"   📦 Cache hit: {symbol} {interval} ({len(cached['df'])} bars)")
                    return cached["df"].copy()
            except Exception:
                pass

        periods_map = {
            "1m": ["7d", "5d"],
            "2m": ["60d"],
            "5m": ["60d"],
            "15m": ["60d"],
            "30m": ["60d"],
            "1h": ["60d", "730d"],
        }
        periods = periods_map.get(interval, ["60d"])

        best = pd.DataFrame()
        for period in periods:
            try:
                print(f"📊 yfinance: {symbol} {interval} (period={period})…")
                raw = yf.download(
                    symbol,
                    interval=interval,
                    period=period,
                    auto_adjust=True,
                    progress=False,
                    threads=False,
                )
                df = _normalize_ohlcv(raw)
                if len(df) > len(best):
                    best = df
                if len(df) >= min_len:
                    best = df
                    break
            except Exception as e:
                print(f"   ⚠️  yfinance error {symbol} {interval}: {e}")

        self._ohlcv_cache[cache_key] = {
            "df": best,
            "fetched_at": datetime.utcnow().isoformat(),
        }
        return best.copy()

    def _volume_series(self, df: pd.DataFrame) -> pd.Series:
        """Forex often has zero volume on Yahoo — use range as proxy when needed."""
        vol = df["Volume"].astype(float).copy()
        if vol.fillna(0).max() <= 0:
            vol = (df["High"] - df["Low"]).abs()
        return vol.replace(0, np.nan).ffill().fillna(1.0)

    def _align_htf_close(self, m1: pd.DataFrame, htf: pd.DataFrame) -> np.ndarray:
        """request.security(htf, close) — forward-fill last completed HTF close onto 1m index."""
        htf_close = htf["Close"].copy()
        htf_close.index = pd.to_datetime(htf_close.index, utc=True)
        m1_index = pd.to_datetime(m1.index, utc=True)
        aligned = pd.merge_asof(
            pd.DataFrame({"t": m1_index}).sort_values("t"),
            pd.DataFrame({"t": htf_close.index, "htf_close": htf_close.values}).sort_values("t"),
            on="t",
            direction="backward",
        )
        return aligned["htf_close"].to_numpy(dtype=float)

    def check_signals(self, symbol: str, m1: pd.DataFrame, htf: pd.DataFrame, verbose: bool = True) -> list:
        if len(m1) < self.min_bars:
            if verbose:
                print(f"   ⚠️  Need {self.min_bars} M1 bars, got {len(m1)}")
            return []
        if len(htf) < self.min_htf_bars:
            if verbose:
                print(f"   ⚠️  Need {self.min_htf_bars} HTF bars, got {len(htf)}")
            return []

        o = m1["Open"].to_numpy(dtype=float)
        h = m1["High"].to_numpy(dtype=float)
        l = m1["Low"].to_numpy(dtype=float)
        c = m1["Close"].to_numpy(dtype=float)
        n = len(c)

        min_tick = self.symbol_min_tick.get(symbol, 0.0001)
        pip = min_tick * 10
        buffer_price = self.buffer_pips_1m * pip

        # ─── S/R pivots (Pine) ───────────────────────────────────────
        ph = pivothigh_series(h, self.left_bars, self.right_bars)
        pl = pivotlow_series(l, self.left_bars, self.right_bars)
        high_use = fixnan_forward(shift_one(ph))
        low_use = fixnan_forward(shift_one(pl))

        # ─── Volume oscillator ─────────────────────────────────────────
        vol = self._volume_series(m1).to_numpy(dtype=float)
        short_vol = pd.Series(vol).ewm(span=5, adjust=False).mean().to_numpy()
        long_vol = pd.Series(vol).ewm(span=10, adjust=False).mean().to_numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            osc = np.where(long_vol != 0, 100.0 * (short_vol - long_vol) / long_vol, 0.0)

        use_vol_filter = np.nanmax(vol) > 0 or np.nanmax(osc) != 0

        cross_up_res = crossover(c, high_use)
        cross_dn_sup = crossunder(c, low_use)

        body_bear = (o - c) < (h - o)
        body_bull = (o - l) > (c - o)

        vol_ok = osc > self.volume_thresh if use_vol_filter else np.ones(n, dtype=bool)

        is_bear_break = self.toggle_breaks & cross_dn_sup & ~body_bear & vol_ok
        is_bear_wick = self.toggle_breaks & cross_dn_sup & body_bear
        is_bull_break = self.toggle_breaks & cross_up_res & ~body_bull & vol_ok
        is_bull_wick = self.toggle_breaks & cross_up_res & body_bull

        is_red_bar = is_bear_break | is_bear_wick
        is_blue_bar = is_bull_break | is_bull_wick

        # ─── Stationarity (Pine: always true when threshold > 0.01) ──
        is_stationary = 0.01 < self.stationarity_threshold

        closes = pd.Series(c, dtype=float)
        sma_lb = closes.rolling(self.lookback).mean().to_numpy()
        std_lb = closes.rolling(self.lookback).std(ddof=0).to_numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            zscore = np.where(std_lb != 0, (c - sma_lb) / std_lb, np.nan)

        htf_close = self._align_htf_close(m1, htf)
        htf_sma = pd.Series(htf_close).rolling(self.trend_ma_period).mean().to_numpy()
        htf_trend_bullish = htf_close > htf_sma
        htf_trend_bearish = htf_close < htf_sma

        sma50 = closes.rolling(50).mean().to_numpy()
        sma200 = closes.rolling(200).mean().to_numpy()
        sma50_prev = np.roll(sma50, self.sma_slope_period)
        with np.errstate(divide="ignore", invalid="ignore"):
            sma50_pct = np.where(sma50_prev != 0, (sma50 - sma50_prev) / sma50_prev * 100.0, 0.0)
        sma_slope_ok = np.abs(sma50_pct) <= self.max_sma_slope_percent
        sma_dist_ticks = np.abs(sma50 - sma200) / min_tick
        sma_dist_ok = sma_dist_ticks >= self.min_sma_distance_ticks

        bullish_valid = (sma50 > sma200) & sma_dist_ok & sma_slope_ok
        bearish_valid = (sma50 < sma200) & sma_dist_ok & sma_slope_ok

        entry_long = is_stationary & (zscore < -self.zscore_threshold) & htf_trend_bullish & bullish_valid
        entry_short = is_stationary & (zscore > self.zscore_threshold) & htf_trend_bearish & bearish_valid

        dist_support = np.abs(c - low_use)
        dist_resistance = np.abs(c - high_use)

        show_buy = is_blue_bar & entry_long & (dist_resistance <= buffer_price)
        show_sell = is_red_bar & entry_short & (dist_support <= buffer_price)

        # Only trigger on rising edge of latest bar (new triangle this bar)
        i = n - 1
        if i < 1:
            return []

        buy_edge = bool(show_buy[i] and not show_buy[i - 1])
        sell_edge = bool(show_sell[i] and not show_sell[i - 1])

        if verbose:
            print(f"   Bar {m1.index[-1]} close={c[i]:.5f}")
            print(f"   S/R resist={high_use[i]:.5f} support={low_use[i]:.5f}")
            print(f"   z={zscore[i]:.3f} blue={is_blue_bar[i]} red={is_red_bar[i]}")
            print(f"   entry_long={entry_long[i]} entry_short={entry_short[i]}")
            print(f"   dist_res={dist_resistance[i]:.5f} dist_sup={dist_support[i]:.5f} buffer={buffer_price:.5f}")
            print(f"   show_buy={show_buy[i]} show_sell={show_sell[i]} → edge buy={buy_edge} sell={sell_edge}")

        if not buy_edge and not sell_edge:
            if verbose:
                print("   ℹ️  No new triangle signal on latest bar")
            return []

        ctx = {
            "price": float(c[i]),
            "zscore": float(zscore[i]) if not np.isnan(zscore[i]) else None,
            "is_stationary": bool(is_stationary),
            "sma50": float(sma50[i]) if not np.isnan(sma50[i]) else None,
            "sma200": float(sma200[i]) if not np.isnan(sma200[i]) else None,
            "resistance": float(high_use[i]) if not np.isnan(high_use[i]) else None,
            "support": float(low_use[i]) if not np.isnan(low_use[i]) else None,
            "dist_to_resistance": float(dist_resistance[i]),
            "dist_to_support": float(dist_support[i]),
            "buffer_price": float(buffer_price),
            "htf_close": float(htf_close[i]) if not np.isnan(htf_close[i]) else None,
            "htf_sma": float(htf_sma[i]) if not np.isnan(htf_sma[i]) else None,
            "timeframe": self.timeframe,
            "htf_timeframe": self.htf_timeframe,
            "volume_osc": float(osc[i]),
            "is_bull_break": bool(is_bull_break[i]),
            "is_bear_break": bool(is_bear_break[i]),
        }

        signals = []
        if buy_edge:
            signals.append({
                "type": "BUY",
                "signal": "sr_break_stationarity_buy_triangle",
                "confidence": "HIGH",
                **ctx,
            })
            if verbose:
                print("   🎯 BUY — green filtered triangle (S/R break + stationarity)")
        if sell_edge:
            signals.append({
                "type": "SELL",
                "signal": "sr_break_stationarity_sell_triangle",
                "confidence": "HIGH",
                **ctx,
            })
            if verbose:
                print("   🎯 SELL — red filtered triangle (S/R break + stationarity)")

        return signals

    def scan_watchlist(self):
        global last_check_time, total_signals

        self._ohlcv_cache.clear()

        print(f"\n{'='*60}")
        print(f"🚀 Scan at {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
        print("   Strategy: SR Break + Stationarity Combined (Pine port)")
        print(f"   Data: yfinance | Chart: {self.timeframe} | HTF: {self.htf_timeframe}")
        print(f"{'='*60}")

        session = self.get_current_session()
        last_check_time = datetime.utcnow().isoformat() + "Z"
        signals_found = 0

        for symbol in self.watchlist:
            print(f"\n🔍 {symbol}…")
            try:
                m1 = self.fetch_ohlcv(symbol, self.timeframe, self.min_bars)
                htf = self.fetch_ohlcv(symbol, self.htf_timeframe, self.min_htf_bars)

                if len(m1) < self.min_bars:
                    print(f"   ⚠️  Skipping: {len(m1)} {self.timeframe} bars (need {self.min_bars})")
                    continue
                if len(htf) < self.min_htf_bars:
                    print(f"   ⚠️  Skipping: {len(htf)} {self.htf_timeframe} bars (need {self.min_htf_bars})")
                    continue

                print(f"   📈 {self.timeframe}: {len(m1)} bars | {self.htf_timeframe}: {len(htf)} bars")

                signals = self.check_signals(symbol, m1, htf, verbose=True)
                for sig in signals:
                    if self.save_alert_to_firebase({"symbol": symbol, "session": session, **sig}):
                        signals_found += 1
                        total_signals += 1
            except Exception as e:
                print(f"   ❌ Error: {e}")

            time.sleep(0.8)

        print(f"\n{'='*60}")
        print(f"✅ Scan done — {signals_found} new signal(s)")
        print(f"{'='*60}")


# Backwards-compatible alias for Render / imports
FibSMATradingBot = SRStationarityBot


@app.route("/")
def home():
    return jsonify({
        "status": "running",
        "strategy": "sr_break_stationarity_1m",
        "data_source": "yfinance",
        "last_check": last_check_time,
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
        "status": "running",
        "strategy": "SR Break + Stationarity Combined (1m)",
        "data_source": "yfinance",
        "timeframe": bot_instance.timeframe,
        "htf_timeframe": bot_instance.htf_timeframe,
        "left_bars": bot_instance.left_bars,
        "right_bars": bot_instance.right_bars,
        "zscore_threshold": bot_instance.zscore_threshold,
        "buffer_pips_1m": bot_instance.buffer_pips_1m,
        "current_session": bot_instance.get_current_session(),
        "last_check": last_check_time,
        "total_signals": total_signals,
        "watchlist": bot_instance.watchlist,
    })


def run_bot():
    while True:
        try:
            bot_instance.scan_watchlist()
        except Exception as e:
            print(f"❌ Bot scan error: {e}")
        print(f"💤 Sleeping {bot_instance.scan_sleep_seconds}s until next scan…")
        time.sleep(bot_instance.scan_sleep_seconds)


if __name__ == "__main__":
    print("🚀 SR Break + Stationarity Bot (yfinance, 1m)")
    print(f"Python: {sys.version}")
    print(f"UTC: {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}")

    bot_instance = SRStationarityBot()
    write_startup_test_to_firebase(bot_instance.db, bot=bot_instance)

    run_once = os.environ.get("BOT_RUN_ONCE", "").lower() in ("1", "true", "yes")
    if run_once:
        print("🏁 BOT_RUN_ONCE: single scan then exit")
        bot_instance.scan_watchlist()
        sys.exit(0)

    threading.Thread(target=run_bot, daemon=True).start()
    print("✅ Bot thread started")

    port = int(os.environ.get("PORT", 10000))
    print(f"🚀 Flask on port {port}")
    app.run(host="0.0.0.0", port=port)
