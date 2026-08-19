"""
MTF Fibonacci + SMA Trend Direction — matches fib_sma.pine

TradingView marks (saved as alerts):
  BUY  (sma50 > sma200): 15m/4h/both breakout + 15m/4h/both pullback
  SELL (sma50 < sma200): 15m/4h/both breakout + 15m/4h/both pullback

Chart TF = 15m, HTF = 4H, Fib = 0.618 of highest/lowest close over 50 bars.
"""

import os
import sys
import time
import threading
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf
from flask import Flask, jsonify, request
from flask_cors import CORS

import db_alerts

app = Flask(__name__)

_cors_origins_env = os.environ.get("CORS_ORIGINS", "*").strip()
if _cors_origins_env == "*":
    _cors_origins = "*"
else:
    _cors_origins = [o.strip() for o in _cors_origins_env.split(",") if o.strip()]

CORS(
    app,
    resources={r"/*": {"origins": _cors_origins}},
    supports_credentials=False,
    methods=["GET", "HEAD", "OPTIONS"],
    allow_headers=["Content-Type", "Accept", "Authorization"],
    expose_headers=["Content-Type"],
    max_age=86400,
)


@app.after_request
def _apply_cors_headers(response):
    """Ensure CORS headers even on errors (e.g. DB down during boot)."""
    origin = request.headers.get("Origin")
    if not origin:
        return response
    if _cors_origins == "*" or origin in _cors_origins:
        response.headers["Access-Control-Allow-Origin"] = "*" if _cors_origins == "*" else origin
        response.headers["Access-Control-Allow-Methods"] = "GET, HEAD, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Accept, Authorization"
        response.headers["Vary"] = "Origin"
    return response


bot_instance = None
last_check_time = None
total_signals = 0

WATCHLIST = [
    "EURUSD=X", "GBPJPY=X", "AUDJPY=X", "XAUUSD=X", "USDCAD=X",
    "GBPUSD=X", "EURJPY=X", "USDJPY=X", "AUDUSD=X", "NZDUSD=X",
    "USDCHF=X", "EURGBP=X", "EURCAD=X", "GBPCAD=X", "AUDCAD=X",
    "EURAUD=X", "BTC-USD", "ETH-USD",
    "EURCHF=X", "EURNZD=X", "GBPCHF=X", "GBPAUD=X", "GBPNZD=X",
    "AUDCHF=X", "AUDNZD=X", "CADJPY=X", "CHFJPY=X", "NZDJPY=X",
    "NZDCAD=X", "NZDCHF=X", "CADCHF=X", "XAUEUR=X", "XAUGBP=X",
    "XAUJPY=X", "XAGUSD=X",
    "USDMXN=X", "USDZAR=X", "USDTRY=X", "USDSEK=X", "USDNOK=X",
    "USDDKK=X", "USDPLN=X", "USDSGD=X", "USDHKD=X", "USDCNH=X",
    "SOL-USD", "BNB-USD", "XRP-USD", "ADA-USD", "DOGE-USD",
    "DOT-USD", "AVAX-USD", "LINK-USD", "LTC-USD", "BCH-USD",
    "MATIC-USD", "UNI-USD", "ATOM-USD", "XLM-USD", "SHIB-USD", "TRX-USD",
    "^GSPC", "^DJI", "^IXIC", "^RUT", "^VIX", "^FTSE", "^GDAXI",
    "^FCHI", "^N225", "^HSI", "^STOXX50E",
    "GC=F", "SI=F", "CL=F", "BZ=F", "NG=F", "HG=F", "ZC=F", "ZW=F",
    "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "META", "NVDA", "TSLA",
    "AMD", "INTC", "NFLX", "CRM", "ORCL", "IBM", "JPM", "BAC", "WFC",
    "GS", "V", "MA", "JNJ", "UNH", "PFE", "XOM", "CVX", "WMT", "HD",
    "DIS", "KO", "PEP", "NKE", "BA", "COIN", "MSTR",
    "SPY", "QQQ", "DIA", "IWM", "XLF", "XLE", "XLK", "GLD", "SLV", "USO",
]

# Pine plotshape signal keys (must match fib_sma.pine alert JSON "signal" field)
SIGNAL_DEFS = (
    ("15m_breakout", "buy_signal_15m", "sell_signal_15m"),
    ("4h_breakout", "buy_signal_4h", "sell_signal_4h"),
    ("both_breakout", "buy_signal_both", "sell_signal_both"),
    ("15m_pullback", "buy_pullback_15m", "sell_pullback_15m"),
    ("4h_pullback", "buy_pullback_4h", "sell_pullback_4h"),
    ("both_pullback", "buy_pullback_both", "sell_pullback_both"),
)


def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
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
    out = df[needed].copy().apply(pd.to_numeric, errors="coerce").dropna()
    if out.index.tz is None:
        out.index = out.index.tz_localize("UTC")
    else:
        out.index = out.index.tz_convert("UTC")
    return out


def drop_incomplete_bar(df: pd.DataFrame, bar_minutes: int) -> pd.DataFrame:
    """Pine alerts on bar close — drop the still-forming candle."""
    if df.empty:
        return df
    now = pd.Timestamp.now(tz="UTC")
    last = pd.Timestamp(df.index[-1]).tz_convert("UTC")
    bar_start = last.floor(f"{bar_minutes}min")
    if now < bar_start + pd.Timedelta(minutes=bar_minutes):
        return df.iloc[:-1].copy()
    return df


def crossover(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """ta.crossover(a, b): a crosses above b."""
    out = np.zeros(len(a), dtype=bool)
    for i in range(1, len(a)):
        if np.isnan(a[i - 1]) or np.isnan(b[i - 1]) or np.isnan(a[i]) or np.isnan(b[i]):
            continue
        out[i] = a[i - 1] <= b[i - 1] and a[i] > b[i]
    return out


def crossunder(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """ta.crossunder(a, b): a crosses below b."""
    out = np.zeros(len(a), dtype=bool)
    for i in range(1, len(a)):
        if np.isnan(a[i - 1]) or np.isnan(b[i - 1]) or np.isnan(a[i]) or np.isnan(b[i]):
            continue
        out[i] = a[i - 1] >= b[i - 1] and a[i] < b[i]
    return out


def fib_618_series(close: pd.Series, length: int = 50, level: float = 0.618) -> pd.Series:
    """Pine get_fib_levels(close, length) → fib_618."""
    highest = close.rolling(length).max()
    lowest = close.rolling(length).min()
    return lowest + (highest - lowest) * level


_TF_UNIT_MINUTES = {"m": 1, "h": 60, "d": 1440}


def timeframe_to_minutes(tf: str, default: int) -> int:
    """Parse '1m'/'15m'/'4h'/'1d' style strings into minutes; fall back to default."""
    tf = (tf or "").strip().lower()
    if len(tf) >= 2 and tf[-1] in _TF_UNIT_MINUTES and tf[:-1].isdigit():
        return int(tf[:-1]) * _TF_UNIT_MINUTES[tf[-1]]
    return default


def htf_fetch_interval(htf_minutes: int) -> str:
    """
    Native yfinance interval to fetch HTF bars from, independent of whatever
    (possibly very short) base timeframe the LTF series uses. Pine's
    request.security() pulls HTF bars from the exchange directly rather than
    resampling the chart's own series — this mirrors that so a 50-bar HTF Fib
    window isn't bottlenecked by the LTF interval's yfinance history cap
    (e.g. 1m bars are capped at 7d, too short for a 50-bar 4H window).
    """
    return "5m" if htf_minutes < 60 else "1h"


def htf_fib_on_ltf(
    ltf_index: pd.Index,
    htf_close_raw: pd.Series,
    htf_minutes: int = 240,
    fib_length: int = 50,
    fib_level: float = 0.618,
) -> np.ndarray:
    """
    Match request.security(..., htf, get_fib_levels(close, 50)) with lookahead_off:
    use only completed HTF bars, then forward-fill onto LTF timestamps.

    htf_close_raw must be fetched independently (see htf_fetch_interval) so its
    history depth isn't limited by the LTF series' own fetch window.
    """
    if htf_close_raw.empty:
        return np.full(len(ltf_index), np.nan)

    htf_close = htf_close_raw.resample(f"{htf_minutes}min").last().dropna()
    if htf_close.empty:
        return np.full(len(ltf_index), np.nan)

    now = pd.Timestamp.now(tz="UTC")
    last_htf_start = pd.Timestamp(htf_close.index[-1]).tz_convert("UTC")
    if now < last_htf_start + pd.Timedelta(minutes=htf_minutes):
        htf_close = htf_close.iloc[:-1]
    if htf_close.empty:
        return np.full(len(ltf_index), np.nan)

    fib_htf = fib_618_series(htf_close, length=fib_length, level=fib_level)
    aligned = fib_htf.reindex(ltf_index, method="ffill")
    return aligned.to_numpy(dtype=float)


class FibSMATradingBot:
    """Implements fib_sma.pine MTF Fib 0.618 + SMA50/200 trend filters."""

    def __init__(self):
        self.watchlist = list(WATCHLIST)
        self.fib_level = float(os.environ.get("FIB_LEVEL", "0.618"))
        self.fib_length = int(os.environ.get("FIB_LENGTH", "50"))
        self.sma_fast = int(os.environ.get("SMA_FAST", "50"))
        self.sma_slow = int(os.environ.get("SMA_SLOW", "200"))
        # Chart TF defaults to 1m (fastest way to get a verifiable mark; Pine's
        # "current chart" adapts to whatever timeframe you view it on). HTF
        # stays fixed at 4H, matching fib_sma.pine's actual default input —
        # safe at any chart TF since HTF bars are now fetched independently
        # (see fetch_htf_close) instead of resampled from the chart series.
        self.timeframe = os.environ.get("TIMEFRAME", "1m")
        self.htf_timeframe = os.environ.get("HTF_TIMEFRAME", "4h")
        # BAR_MINUTES/HTF_MINUTES are optional overrides — derived from
        # TIMEFRAME/HTF_TIMEFRAME by default so they can't silently drift out
        # of sync with them (e.g. TIMEFRAME=1m deployed without BAR_MINUTES
        # would otherwise fall back to a stale "15" and break bar-close
        # detection, which is exactly what happened to this bot's marking).
        self.bar_minutes = int(
            os.environ.get("BAR_MINUTES") or timeframe_to_minutes(self.timeframe, 15)
        )
        self.htf_minutes = int(
            os.environ.get("HTF_MINUTES") or timeframe_to_minutes(self.htf_timeframe, 240)
        )
        self.lookback_bars = int(os.environ.get("LOOKBACK_BARS", "2000"))  # ~33h of 1m
        self.min_bars = max(self.sma_slow + 5, self.fib_length + 5, 220)
        self.symbol_sleep_seconds = float(os.environ.get("SYMBOL_SLEEP_S", "0.35"))
        self.scan_sleep_seconds = int(os.environ.get("SCAN_INTERVAL_S", "60"))
        self._ohlcv_cache = {}
        self._cache_ttl_seconds = int(os.environ.get("CACHE_TTL_S", "55"))
        self.sessions = {
            "asia": {"open": 23, "close": 8, "name": "Asian Session"},
            "london": {"open": 7, "close": 16, "name": "London Session"},
            "newyork": {"open": 13, "close": 22, "name": "New York Session"},
        }

    def get_current_session(self):
        h = datetime.now(timezone.utc).hour
        for sid, s in self.sessions.items():
            o, c = s["open"], s["close"]
            if c < o:
                if h >= o or h < c:
                    return sid
            else:
                if o <= h < c:
                    return sid
        return "off_hours"

    def save_alert(self, alert_data):
        alert_data["session"] = self.get_current_session()
        alert_data.setdefault("strategy", "fib_sma")
        for key, value in list(alert_data.items()):
            if isinstance(value, (np.bool_, np.integer, np.floating)):
                alert_data[key] = value.item()
        return db_alerts.save_alert(alert_data)

    def fetch_ohlcv(self, symbol: str, interval: str, min_len: int) -> pd.DataFrame:
        cache_key = (symbol, interval)
        cached = self._ohlcv_cache.get(cache_key)
        if cached is not None:
            try:
                age = (datetime.now(timezone.utc) - datetime.fromisoformat(cached["fetched_at"])).total_seconds()
                if age <= self._cache_ttl_seconds and len(cached["df"]) >= min_len:
                    return cached["df"].copy()
            except Exception:
                pass
        periods_map = {
            "15m": ["60d", "30d"],
            "1h": ["730d", "60d"],
            "60m": ["730d", "60d"],
            "1m": ["7d", "5d"],
            "5m": ["60d", "30d"],
        }
        best = pd.DataFrame()
        for period in periods_map.get(interval, ["60d"]):
            try:
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
                print(f"   ⚠️  yfinance {symbol} {interval}: {e}")
        self._ohlcv_cache[cache_key] = {
            "df": best,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        }
        return best.copy()

    def fetch_htf_close(self, symbol: str) -> pd.Series:
        """
        Fetch the HTF close series independently of the LTF fetch (mirrors Pine's
        request.security pulling HTF bars straight from the exchange). Prevents a
        short base TIMEFRAME's yfinance history cap (e.g. 1m → 7d) from starving
        the HTF Fib window of enough completed bars to ever resolve.
        """
        interval = htf_fetch_interval(self.htf_minutes)
        min_len = self.fib_length + 5
        df = self.fetch_ohlcv(symbol, interval, min_len)
        if df.empty:
            return pd.Series(dtype=float)
        return df["Close"].astype(float)

    def compute_signals(self, df_15m: pd.DataFrame, htf_close_raw: pd.Series) -> dict:
        """
        Reproduce fib_sma.pine boolean series on closed 15m bars.
        Returns dict of named bool arrays + context float arrays.
        """
        close = df_15m["Close"].astype(float)
        c = close.to_numpy(dtype=float)
        n = len(c)

        sma50 = close.rolling(self.sma_fast).mean().to_numpy(dtype=float)
        sma200 = close.rolling(self.sma_slow).mean().to_numpy(dtype=float)
        bullish_trend = sma50 > sma200
        bearish_trend = sma50 < sma200

        fib_618_15m = fib_618_series(close, self.fib_length, self.fib_level).to_numpy(dtype=float)
        fib_618_4h = htf_fib_on_ltf(
            close.index,
            htf_close_raw,
            htf_minutes=self.htf_minutes,
            fib_length=self.fib_length,
            fib_level=self.fib_level,
        )

        # Pine: breakout = ta.crossover(close, fib)
        breakout_15m = crossover(c, fib_618_15m)
        breakout_4h = crossover(c, fib_618_4h)

        # Pine: pullback = ta.crossunder(close, fib) and close[1] > fib[1]
        cu_15m = crossunder(c, fib_618_15m)
        cu_4h = crossunder(c, fib_618_4h)
        prev_above_15m = np.zeros(n, dtype=bool)
        prev_above_4h = np.zeros(n, dtype=bool)
        for i in range(1, n):
            if not (np.isnan(c[i - 1]) or np.isnan(fib_618_15m[i - 1])):
                prev_above_15m[i] = c[i - 1] > fib_618_15m[i - 1]
            if not (np.isnan(c[i - 1]) or np.isnan(fib_618_4h[i - 1])):
                prev_above_4h[i] = c[i - 1] > fib_618_4h[i - 1]
        pullback_15m = cu_15m & prev_above_15m
        pullback_4h = cu_4h & prev_above_4h

        return {
            "close": c,
            "sma50": sma50,
            "sma200": sma200,
            "fib_15m": fib_618_15m,
            "fib_4h": fib_618_4h,
            "bullish_trend": bullish_trend,
            "bearish_trend": bearish_trend,
            "breakout_15m": breakout_15m,
            "breakout_4h": breakout_4h,
            "pullback_15m": pullback_15m,
            "pullback_4h": pullback_4h,
            # BUY (only in bullish trend) — matches plotshape titles
            "buy_signal_15m": breakout_15m & bullish_trend,
            "buy_signal_4h": breakout_4h & bullish_trend,
            "buy_signal_both": breakout_15m & breakout_4h & bullish_trend,
            "buy_pullback_15m": pullback_15m & bullish_trend,
            "buy_pullback_4h": pullback_4h & bullish_trend,
            "buy_pullback_both": pullback_15m & pullback_4h & bullish_trend,
            # SELL (only in bearish trend)
            "sell_signal_15m": breakout_15m & bearish_trend,
            "sell_signal_4h": breakout_4h & bearish_trend,
            "sell_signal_both": breakout_15m & breakout_4h & bearish_trend,
            "sell_pullback_15m": pullback_15m & bearish_trend,
            "sell_pullback_4h": pullback_4h & bearish_trend,
            "sell_pullback_both": pullback_15m & pullback_4h & bearish_trend,
        }

    def check_signals(self, symbol: str, df: pd.DataFrame, verbose: bool = True) -> list:
        df = drop_incomplete_bar(df, self.bar_minutes)
        if len(df) < self.min_bars:
            if verbose:
                print(f"   ⚠️  Need {self.min_bars} closed {self.timeframe} bars, got {len(df)}")
            return []

        htf_close_raw = self.fetch_htf_close(symbol)
        calc = self.compute_signals(df, htf_close_raw)
        n = len(calc["close"])
        start_i = max(1, n - self.lookback_bars)
        signals = []

        for i in range(start_i, n):
            bar_ts = pd.Timestamp(df.index[i]).tz_convert("UTC")
            bar_time_iso = bar_ts.isoformat().replace("+00:00", "Z")
            trend = "bullish" if bool(calc["bullish_trend"][i]) else "bearish"
            ctx = {
                "price": float(calc["close"][i]),
                "sma50": float(calc["sma50"][i]) if not np.isnan(calc["sma50"][i]) else None,
                "sma200": float(calc["sma200"][i]) if not np.isnan(calc["sma200"][i]) else None,
                "fib_15m": float(calc["fib_15m"][i]) if not np.isnan(calc["fib_15m"][i]) else None,
                "fib_4h": float(calc["fib_4h"][i]) if not np.isnan(calc["fib_4h"][i]) else None,
                "trend": trend,
                "timeframe": self.timeframe,
                "htf_timeframe": self.htf_timeframe,
                "bar_time": bar_time_iso,
                "strategy": "fib_sma",
            }

            for signal_name, buy_key, sell_key in SIGNAL_DEFS:
                if bool(calc[buy_key][i]):
                    if not db_alerts.alert_exists(symbol, "BUY", bar_time_iso, signal_name):
                        conf = "HIGH" if "both" in signal_name else "MEDIUM"
                        signals.append({
                            "type": "BUY",
                            "signal": signal_name,
                            "confidence": conf,
                            **ctx,
                        })
                        if verbose:
                            print(
                                f"   🎯 BUY  {signal_name} @ {bar_time_iso} "
                                f"price={ctx['price']:.5f} fib15={ctx['fib_15m']} fib4h={ctx['fib_4h']}"
                            )
                if bool(calc[sell_key][i]):
                    if not db_alerts.alert_exists(symbol, "SELL", bar_time_iso, signal_name):
                        conf = "HIGH" if "both" in signal_name else "MEDIUM"
                        signals.append({
                            "type": "SELL",
                            "signal": signal_name,
                            "confidence": conf,
                            **ctx,
                        })
                        if verbose:
                            print(
                                f"   🎯 SELL {signal_name} @ {bar_time_iso} "
                                f"price={ctx['price']:.5f} fib15={ctx['fib_15m']} fib4h={ctx['fib_4h']}"
                            )

        if verbose and not signals:
            i = n - 1
            print(
                f"   Bar {df.index[i]} close={calc['close'][i]:.5f} "
                f"trend={'BULL' if calc['bullish_trend'][i] else 'BEAR'} "
                f"fib15={calc['fib_15m'][i]:.5f} fib4h={calc['fib_4h'][i]:.5f}"
            )
            print(
                f"   bo15={calc['breakout_15m'][i]} bo4h={calc['breakout_4h'][i]} "
                f"pb15={calc['pullback_15m'][i]} pb4h={calc['pullback_4h'][i]}"
            )
            print("   ℹ️  No new Fib SMA marks in lookback window")

        return signals

    def scan_watchlist(self):
        global last_check_time, total_signals
        self._ohlcv_cache.clear()
        print(f"\n{'='*60}")
        print(
            f"🚀 Fib SMA scan {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC "
            f"| {len(self.watchlist)} symbols | {self.timeframe}/{self.htf_timeframe}"
        )
        print(f"{'='*60}")
        session = self.get_current_session()
        last_check_time = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        found = 0
        for symbol in self.watchlist:
            print(f"\n🔍 {symbol}…")
            try:
                df = self.fetch_ohlcv(symbol, self.timeframe, self.min_bars)
                if len(df) < self.min_bars + 1:
                    print(f"   ⚠️  Skipping — only {len(df)} bars")
                    continue
                for sig in self.check_signals(symbol, df, verbose=True):
                    if self.save_alert({"symbol": symbol, "session": session, **sig}):
                        found += 1
                        total_signals += 1
            except Exception as e:
                print(f"   ❌ {e}")
            time.sleep(self.symbol_sleep_seconds)
        print(f"\n✅ Scan done — {found} new alert(s)")
        return found


_scanner_started = False


def ensure_services_started():
    global bot_instance, _scanner_started
    if bot_instance is None:
        db_alerts.init_db()
        bot_instance = FibSMATradingBot()
    if _scanner_started:
        return
    if os.environ.get("BOT_RUN_ONCE", "").lower() in ("1", "true", "yes"):
        return
    threading.Thread(target=run_bot, daemon=True).start()
    _scanner_started = True
    print("✅ Background scanner started")


@app.before_request
def _lazy_start():
    if request.method == "OPTIONS":
        return None
    try:
        ensure_services_started()
    except Exception as exc:
        print(f"⚠️ Service boot: {exc}")
    return None


@app.route("/")
def home():
    return jsonify({
        "status": "running",
        "strategy": "fib_sma",
        "storage": "neon_postgresql",
        "last_check": last_check_time,
        "total_signals": total_signals,
    })


@app.route("/health")
def health():
    return "OK", 200


@app.route("/api/alerts")
def api_alerts():
    try:
        limit = min(int(request.args.get("limit", 10)), 100)
        return jsonify(db_alerts.fetch_alerts(limit=limit))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/debug")
def debug():
    """Run Fib SMA calc on one symbol and return last-bar filter values."""
    try:
        ensure_services_started()
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
    symbol = request.args.get("symbol", "EURUSD=X")
    try:
        df = bot_instance.fetch_ohlcv(symbol, bot_instance.timeframe, bot_instance.min_bars)
        if df.empty or len(df) < bot_instance.min_bars:
            return jsonify({
                "symbol": symbol,
                "error": f"only {len(df)} bars, need {bot_instance.min_bars}",
            })
        dfd = drop_incomplete_bar(df, bot_instance.bar_minutes)
        htf_close_raw = bot_instance.fetch_htf_close(symbol)
        calc = bot_instance.compute_signals(dfd, htf_close_raw)
        i = len(calc["close"]) - 1
        marks = []
        for signal_name, buy_key, sell_key in SIGNAL_DEFS:
            if bool(calc[buy_key][i]):
                marks.append({"type": "BUY", "signal": signal_name})
            if bool(calc[sell_key][i]):
                marks.append({"type": "SELL", "signal": signal_name})
        return jsonify({
            "symbol": symbol,
            "bars": len(calc["close"]),
            "bar_time": pd.Timestamp(dfd.index[i]).tz_convert("UTC").isoformat().replace("+00:00", "Z"),
            "price": float(calc["close"][i]),
            "sma50": float(calc["sma50"][i]) if not np.isnan(calc["sma50"][i]) else None,
            "sma200": float(calc["sma200"][i]) if not np.isnan(calc["sma200"][i]) else None,
            "trend": "bullish" if bool(calc["bullish_trend"][i]) else "bearish",
            "fib_15m": float(calc["fib_15m"][i]) if not np.isnan(calc["fib_15m"][i]) else None,
            "fib_4h": float(calc["fib_4h"][i]) if not np.isnan(calc["fib_4h"][i]) else None,
            "breakout_15m": bool(calc["breakout_15m"][i]),
            "breakout_4h": bool(calc["breakout_4h"][i]),
            "pullback_15m": bool(calc["pullback_15m"][i]),
            "pullback_4h": bool(calc["pullback_4h"][i]),
            "marks_on_last_bar": marks,
            "timeframe": bot_instance.timeframe,
            "htf_timeframe": bot_instance.htf_timeframe,
            "bar_minutes": bot_instance.bar_minutes,
            "htf_minutes": bot_instance.htf_minutes,
            "htf_bars_fetched": int(len(htf_close_raw)),
            "fib_level": bot_instance.fib_level,
            "fib_length": bot_instance.fib_length,
        })
    except Exception as e:
        return jsonify({"symbol": symbol, "error": str(e)}), 500


@app.route("/status")
def status():
    if not bot_instance:
        return jsonify({"error": "Bot not initialised"}), 500
    return jsonify({
        "status": "running",
        "storage": "neon_postgresql",
        "strategy": "fib_sma",
        "pine": "MTF Fibonacci + SMA Trend Direction",
        "timeframe": bot_instance.timeframe,
        "htf_timeframe": bot_instance.htf_timeframe,
        "fib_level": bot_instance.fib_level,
        "fib_length": bot_instance.fib_length,
        "sma_fast": bot_instance.sma_fast,
        "sma_slow": bot_instance.sma_slow,
        "symbols": len(bot_instance.watchlist),
        "last_check": last_check_time,
        "total_signals": total_signals,
        "signal_types": [s[0] for s in SIGNAL_DEFS],
    })


def run_bot():
    while True:
        try:
            bot_instance.scan_watchlist()
        except Exception as e:
            print(f"❌ Scan error: {e}")
        print(f"💤 Sleep {bot_instance.scan_sleep_seconds}s…")
        time.sleep(bot_instance.scan_sleep_seconds)


def bootstrap(start_scanner: bool = True):
    """One-shot scan mode (cron) or prepare bot instance for web server."""
    global bot_instance
    run_once = os.environ.get("BOT_RUN_ONCE", "").lower() in ("1", "true", "yes")
    if run_once:
        # Cron/batch mode: let a DB failure raise so the scheduler sees a
        # nonzero exit instead of silently skipping the scan.
        db_alerts.init_db()
        bot_instance = FibSMATradingBot()
        bot_instance.scan_watchlist()
        return
    if start_scanner:
        # Persistent web server: never let a boot-time DB hiccup (bad
        # DATABASE_URL, Neon cold-start, network blip) crash the whole
        # process before it can bind $PORT — that turns a transient/fixable
        # DB issue into a full failed deploy. Retry lazily per request
        # instead, same as the gunicorn path (see _lazy_start below).
        try:
            ensure_services_started()
        except Exception as exc:
            print(f"⚠️ Deferred service boot (will retry on first request): {exc}")


if __name__ == "__main__":
    print("🚀 Fib SMA Bot | matches fib_sma.pine | yfinance + Neon PostgreSQL")
    run_once = os.environ.get("BOT_RUN_ONCE", "").lower() in ("1", "true", "yes")
    if run_once:
        bootstrap(start_scanner=False)
        sys.exit(0)
    bootstrap(start_scanner=True)
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
