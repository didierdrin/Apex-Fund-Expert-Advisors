"""
SR Break + Stationarity Combined — yfinance + Neon PostgreSQL
Pine: green/red filtered triangles on 1m (showBuyTriangle / showSellTriangle)
"""

import json
import os
import sys
import time
import threading
import socket
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


def syminfo_mintick(symbol: str) -> float:
    """Match TradingView syminfo.mintick (point size, not pip)."""
    if symbol.endswith("=X"):
        if "JPY" in symbol or "XAU" in symbol or "XAG" in symbol:
            if "XAU" in symbol or "XAG" in symbol:
                return 0.01
            return 0.001
        return 0.00001
    if symbol.endswith("-USD"):
        if symbol in ("SHIB-USD",):
            return 0.00000001
        if symbol in ("BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "BCH-USD", "LTC-USD"):
            return 0.01
        return 0.0001
    if symbol.endswith("=F") or symbol.startswith("^"):
        return 0.01
    return 0.01


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


def drop_incomplete_bar(df: pd.DataFrame) -> pd.DataFrame:
    """Pine alerts on bar close — drop the still-forming 1m candle."""
    if df.empty:
        return df
    now = pd.Timestamp.now(tz="UTC")
    last = pd.Timestamp(df.index[-1]).tz_convert("UTC")
    if last.floor("min") >= now.floor("min"):
        return df.iloc[:-1].copy()
    return df


def pivothigh_series(high: np.ndarray, left: int, right: int) -> np.ndarray:
    n = len(high)
    out = np.full(n, np.nan)
    for i in range(left + right, n):
        center = i - right
        window = high[center - left : center + right + 1]
        if len(window) == left + right + 1 and high[center] >= np.nanmax(window):
            out[i] = high[center]
    return out


def pivotlow_series(low: np.ndarray, left: int, right: int) -> np.ndarray:
    n = len(low)
    out = np.full(n, np.nan)
    for i in range(left + right, n):
        center = i - right
        window = low[center - left : center + right + 1]
        if len(window) == left + right + 1 and low[center] <= np.nanmin(window):
            out[i] = low[center]
    return out


def fixnan_forward(arr: np.ndarray) -> np.ndarray:
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


def crossover(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.zeros(len(a), dtype=bool)
    for i in range(1, len(a)):
        if np.isnan(a[i - 1]) or np.isnan(b[i - 1]) or np.isnan(a[i]) or np.isnan(b[i]):
            continue
        out[i] = a[i - 1] <= b[i - 1] and a[i] > b[i]
    return out


def crossunder(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.zeros(len(a), dtype=bool)
    for i in range(1, len(a)):
        if np.isnan(a[i - 1]) or np.isnan(b[i - 1]) or np.isnan(a[i]) or np.isnan(b[i]):
            continue
        out[i] = a[i - 1] >= b[i - 1] and a[i] < b[i]
    return out


def htf_close_from_m1(closes: np.ndarray, index: pd.DatetimeIndex, minutes: int = 15) -> np.ndarray:
    """Pine request.security close: developing HTF close built from 1m bars."""
    s = pd.Series(closes, index=pd.to_datetime(index, utc=True))
    bucket = s.index.floor(f"{minutes}min")
    return s.groupby(bucket).transform("last").to_numpy(dtype=float)


class SRStationarityBot:
    def __init__(self):
        self.watchlist = list(WATCHLIST)
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
        self.htf_minutes = int(os.environ.get("HTF_MINUTES", "15"))
        self.buffer_pips_1m = int(os.environ.get("BUFFER_PIPS_1M", "6"))
        self.lookback_minutes = int(os.environ.get("LOOKBACK_MINUTES", "10"))
        self.symbol_sleep_seconds = float(os.environ.get("SYMBOL_SLEEP_S", "0.35"))
        self.scan_sleep_seconds = int(os.environ.get("SCAN_INTERVAL_S", "60"))
        self._ohlcv_cache = {}
        self._cache_ttl_seconds = int(os.environ.get("CACHE_TTL_S", "55"))
        pivot_warmup = self.left_bars + self.right_bars + 5
        self.min_bars = max(250, self.lookback, 50 + self.sma_slope_period, pivot_warmup + 50)
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
        periods_map = {"1m": ["7d", "5d"], "15m": ["60d"], "1h": ["60d", "730d"]}
        best = pd.DataFrame()
        for period in periods_map.get(interval, ["60d"]):
            try:
                raw = yf.download(symbol, interval=interval, period=period, auto_adjust=True, progress=False, threads=False)
                df = _normalize_ohlcv(raw)
                if len(df) > len(best):
                    best = df
                if len(df) >= min_len:
                    best = df
                    break
            except Exception as e:
                print(f"   ⚠️  yfinance {symbol} {interval}: {e}")
        self._ohlcv_cache[cache_key] = {"df": best, "fetched_at": datetime.now(timezone.utc).isoformat()}
        return best.copy()

    def _volume_series(self, df: pd.DataFrame) -> pd.Series:
        vol = df["Volume"].astype(float).copy()
        if vol.fillna(0).max() <= 0:
            vol = (df["High"] - df["Low"]).abs()
        return vol.replace(0, np.nan).ffill().fillna(1.0)

    def check_signals(self, symbol: str, m1: pd.DataFrame, verbose: bool = True) -> list:
        m1 = drop_incomplete_bar(m1)
        if len(m1) < self.min_bars:
            if verbose:
                print(f"   ⚠️  Need {self.min_bars} closed M1 bars, got {len(m1)}")
            return []

        o = m1["Open"].to_numpy(dtype=float)
        h = m1["High"].to_numpy(dtype=float)
        l = m1["Low"].to_numpy(dtype=float)
        c = m1["Close"].to_numpy(dtype=float)
        n = len(c)
        min_tick = syminfo_mintick(symbol)
        buffer_price = self.buffer_pips_1m * min_tick * 10

        ph = pivothigh_series(h, self.left_bars, self.right_bars)
        pl = pivotlow_series(l, self.left_bars, self.right_bars)
        high_use = fixnan_forward(shift_one(ph))
        low_use = fixnan_forward(shift_one(pl))

        vol = self._volume_series(m1).to_numpy(dtype=float)
        short_vol = pd.Series(vol).ewm(span=5, adjust=False).mean().to_numpy()
        long_vol = pd.Series(vol).ewm(span=10, adjust=False).mean().to_numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            osc = np.where(long_vol != 0, 100.0 * (short_vol - long_vol) / long_vol, 0.0)

        cross_up = crossover(c, high_use)
        cross_dn = crossunder(c, low_use)
        body_bear = (o - c) < (h - o)
        body_bull = (o - l) > (c - o)
        vol_ok = osc > self.volume_thresh

        is_bear_break = self.toggle_breaks & cross_dn & ~body_bear & vol_ok
        is_bear_wick = self.toggle_breaks & cross_dn & body_bear
        is_bull_break = self.toggle_breaks & cross_up & ~body_bull & vol_ok
        is_bull_wick = self.toggle_breaks & cross_up & body_bull
        is_red = is_bear_break | is_bear_wick
        is_blue = is_bull_break | is_bull_wick

        # Stationarity: always True (p_value=0.01 < threshold=0.05), matches Pine
        is_stationary = bool(0.01 < self.stationarity_threshold)

        closes = pd.Series(c, dtype=float)
        sma_lb = closes.rolling(self.lookback).mean().to_numpy()
        std_lb = closes.rolling(self.lookback).std(ddof=0).to_numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            zscore = np.where(std_lb != 0, (c - sma_lb) / std_lb, np.nan)

        htf_close = htf_close_from_m1(c, m1.index, self.htf_minutes)
        htf_sma = pd.Series(htf_close).rolling(self.trend_ma_period).mean().to_numpy()
        htf_bull = htf_close > htf_sma
        htf_bear = htf_close < htf_sma

        sma50 = closes.rolling(50).mean().to_numpy()
        sma200 = closes.rolling(200).mean().to_numpy()

        sma50_prev = np.full(n, np.nan)
        if n > self.sma_slope_period:
            sma50_prev[self.sma_slope_period:] = sma50[: n - self.sma_slope_period]
        with np.errstate(divide="ignore", invalid="ignore"):
            sma50_pct = np.where(
                (~np.isnan(sma50_prev)) & (sma50_prev != 0),
                (sma50 - sma50_prev) / sma50_prev * 100.0,
                0.0,
            )
        sma_slope_ok = np.abs(sma50_pct) <= self.max_sma_slope_percent

        sma_dist_ticks = np.abs(sma50 - sma200) / min_tick

        price_ref = c[n - 1] if not np.isnan(c[n - 1]) else 1.0
        if min_tick <= 0.00001:
            effective_min_dist_ticks = self.min_sma_distance_ticks
        elif min_tick <= 0.001:
            effective_min_dist_ticks = self.min_sma_distance_ticks * (0.00001 / min_tick)
        else:
            effective_min_dist_ticks = (price_ref * 0.0005) / min_tick

        sma_dist_ok = sma_dist_ticks >= effective_min_dist_ticks
        bull_valid = (sma50 > sma200) & sma_dist_ok & sma_slope_ok
        bear_valid = (sma50 < sma200) & sma_dist_ok & sma_slope_ok

        entry_long = is_stationary & (zscore < -self.zscore_threshold) & htf_bull & bull_valid
        entry_short = is_stationary & (zscore > self.zscore_threshold) & htf_bear & bear_valid

        dist_sup = np.abs(c - low_use)
        dist_res = np.abs(c - high_use)
        show_buy = is_blue & entry_long & (dist_res <= buffer_price)
        show_sell = is_red & entry_short & (dist_sup <= buffer_price)

        signals = []
        last_bar_time = pd.Timestamp(m1.index[-1]).tz_convert("UTC")
        cutoff_time = last_bar_time - pd.Timedelta(minutes=self.lookback_minutes)

        start_i = n - 1
        for idx in range(n - 1, 0, -1):
            if pd.Timestamp(m1.index[idx]).tz_convert("UTC") >= cutoff_time:
                start_i = idx
            else:
                break
        start_i = max(1, start_i)

        for i in range(start_i, n):
            buy_edge = bool(show_buy[i] and not show_buy[i - 1])
            sell_edge = bool(show_sell[i] and not show_sell[i - 1])
            if not buy_edge and not sell_edge:
                continue

            bar_ts = pd.Timestamp(m1.index[i]).tz_convert("UTC")
            bar_time_iso = bar_ts.isoformat().replace("+00:00", "Z")

            ctx = {
                "price": float(c[i]),
                "zscore": float(zscore[i]) if not np.isnan(zscore[i]) else None,
                "is_stationary": bool(is_stationary),
                "sma50": float(sma50[i]) if not np.isnan(sma50[i]) else None,
                "sma200": float(sma200[i]) if not np.isnan(sma200[i]) else None,
                "resistance": float(high_use[i]) if not np.isnan(high_use[i]) else None,
                "support": float(low_use[i]) if not np.isnan(low_use[i]) else None,
                "dist_to_resistance": float(dist_res[i]),
                "dist_to_support": float(dist_sup[i]),
                "buffer_price": float(buffer_price),
                "htf_close": float(htf_close[i]) if not np.isnan(htf_close[i]) else None,
                "htf_sma": float(htf_sma[i]) if not np.isnan(htf_sma[i]) else None,
                "timeframe": self.timeframe,
                "htf_timeframe": self.htf_timeframe,
                "volume_osc": float(osc[i]),
                "bar_time": bar_time_iso,
                "sma_dist_ticks": float(sma_dist_ticks[i]) if not np.isnan(sma_dist_ticks[i]) else None,
                "effective_min_dist": float(effective_min_dist_ticks),
                "sma_slope_pct": float(sma50_pct[i]) if not np.isnan(sma50_pct[i]) else None,
            }

            if buy_edge and not db_alerts.alert_exists(symbol, "BUY", bar_time_iso):
                signals.append({
                    "type": "BUY",
                    "signal": "sr_break_stationarity_buy_triangle",
                    "confidence": "HIGH",
                    **ctx,
                })
                if verbose:
                    print(
                        f"   🎯 BUY  @ {bar_time_iso} price={c[i]:.5f} z={zscore[i]:.2f} "
                        f"dist_ok={sma_dist_ok[i]} slope_ok={sma_slope_ok[i]}"
                    )

            if sell_edge and not db_alerts.alert_exists(symbol, "SELL", bar_time_iso):
                signals.append({
                    "type": "SELL",
                    "signal": "sr_break_stationarity_sell_triangle",
                    "confidence": "HIGH",
                    **ctx,
                })
                if verbose:
                    print(
                        f"   🎯 SELL @ {bar_time_iso} price={c[i]:.5f} z={zscore[i]:.2f} "
                        f"dist_ok={sma_dist_ok[i]} slope_ok={sma_slope_ok[i]}"
                    )

        if verbose and not signals:
            i = n - 1
            print(f"   Bar {m1.index[i]} close={c[i]:.5f} mintick={min_tick} buffer={buffer_price:.6f}")
            print(f"   S/R R={high_use[i]:.5f} S={low_use[i]:.5f} | blue={is_blue[i]} red={is_red[i]}")
            print(f"   entry_L={entry_long[i]} entry_S={entry_short[i]} distR={dist_res[i]:.6f} distS={dist_sup[i]:.6f}")
            print(
                f"   sma_dist_ticks={sma_dist_ticks[i]:.0f} "
                f"(eff_min={effective_min_dist_ticks:.0f}) slope_ok={sma_slope_ok[i]}"
            )
            print(f"   bull_valid={bull_valid[i]} bear_valid={bear_valid[i]}")
            print(f"   show_buy={show_buy[i]} show_sell={show_sell[i]}")
            print("   ℹ️  No new triangle alert in lookback window (last {0} min)".format(self.lookback_minutes))

        return signals

    def scan_watchlist(self):
        global last_check_time, total_signals
        self._ohlcv_cache.clear()
        print(f"\n{'='*60}")
        print(f"🚀 Scan {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC | {len(self.watchlist)} symbols")
        print(f"{'='*60}")
        session = self.get_current_session()
        last_check_time = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        found = 0
        for symbol in self.watchlist:
            print(f"\n🔍 {symbol}…")
            try:
                m1 = self.fetch_ohlcv(symbol, self.timeframe, self.min_bars)
                if len(m1) < self.min_bars + 1:
                    print(f"   ⚠️  Skipping — only {len(m1)} bars")
                    continue
                for sig in self.check_signals(symbol, m1, verbose=True):
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
        bot_instance = SRStationarityBot()
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


FibSMATradingBot = SRStationarityBot


@app.route("/")
def home():
    return jsonify({"status": "running", "storage": "neon_postgresql", "last_check": last_check_time, "total_signals": total_signals})


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


@app.route("/status")
def status():
    if not bot_instance:
        return jsonify({"error": "Bot not initialised"}), 500
    return jsonify({
        "status": "running",
        "storage": "neon_postgresql",
        "strategy": "SR Break + Stationarity 1m",
        "timeframe": bot_instance.timeframe,
        "htf_timeframe": bot_instance.htf_timeframe,
        "symbols": len(bot_instance.watchlist),
        "last_check": last_check_time,
        "total_signals": total_signals,
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
    db_alerts.init_db()
    bot_instance = SRStationarityBot()
    run_once = os.environ.get("BOT_RUN_ONCE", "").lower() in ("1", "true", "yes")
    if run_once:
        bot_instance.scan_watchlist()
        return
    if start_scanner:
        ensure_services_started()


if __name__ == "__main__":
    print("🚀 SR Break + Stationarity Bot | yfinance + Neon PostgreSQL")
    run_once = os.environ.get("BOT_RUN_ONCE", "").lower() in ("1", "true", "yes")
    if run_once:
        bootstrap(start_scanner=False)
        sys.exit(0)
    bootstrap(start_scanner=True)
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
