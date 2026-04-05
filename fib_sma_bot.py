import requests
import json
import time
import os
import sys
import base64
import tempfile
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
import firebase_admin
from firebase_admin import credentials, firestore
from flask import Flask, jsonify
import threading
import socket
from scipy import stats

# Initialize Flask app
app = Flask(__name__)

# Global variables for bot status
bot_instance = None
last_check_time = None
total_signals = 0
_startup_test_written = False


def fetch_h1_prices_snapshot(bot):
    """
    Fetch the latest H1 close for each symbol in the bot watchlist.
    Returns (prices_dict, errors_dict).
    """
    prices = {}
    errors = {}

    for symbol in getattr(bot, "watchlist", []):
        try:
            tf = getattr(bot, "timeframe", "1h")
            hist_data = bot.get_historical_data(symbol, interval=tf, range="2d")
            closes = bot.extract_close_prices(hist_data)
            if closes:
                prices[symbol] = float(closes[-1])
            else:
                errors[symbol] = "no_close_prices"
        except Exception as e:
            errors[symbol] = str(e)

    return prices, errors


def write_startup_test_to_firebase(db, *, bot=None):
    """
    Write a single startup marker document to Firestore.

    This runs once per backend start (process lifetime) and is useful to verify
    Firestore writes are working by checking the Firebase console.
    """
    global _startup_test_written

    if _startup_test_written:
        return False

    if not db:
        print("⚠️ Firebase not initialized, skipping startup test write")
        _startup_test_written = True
        return False

    try:
        now_utc = datetime.utcnow()
        doc_id = f"{now_utc.strftime('%Y%m%dT%H%M%S')}_{os.getpid()}"
        payload = {
            "type": "backend_startup_test",
            "timestamp_utc": now_utc.isoformat() + "Z",
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "python_version": sys.version,
        }

        if bot is not None:
            prices, price_errors = fetch_h1_prices_snapshot(bot)
            payload["timeframe"] = getattr(bot, "timeframe", "1h")
            payload["h1_prices"] = prices
            payload["h1_price_errors"] = price_errors
            payload["h1_prices_count"] = len(prices)
            payload["h1_price_errors_count"] = len(price_errors)
            payload["rapidapi_requests_used"] = getattr(bot, "request_count", None)

        db.collection("backend_startup_test").document(doc_id).set(payload)
        _startup_test_written = True
        print("✅ Startup test written to Firebase collection: backend_startup_test")
        return True
    except Exception as e:
        _startup_test_written = True
        print(f"❌ Error writing startup test to Firebase: {e}")
        return False

class FibSMATradingBot:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://yahoo-finance166.p.rapidapi.com"
        self.headers = {
            "x-rapidapi-key": api_key,
            "x-rapidapi-host": "yahoo-finance166.p.rapidapi.com"
        }
        
        # Trading symbols
        self.watchlist = [
            'EURUSD=X',  # EUR/USD
            'GBPJPY=X',  # GBP/JPY
            'AUDJPY=X',  # AUD/JPY
            'XAUUSD=X',  # Gold vs USD
            'USDCAD=X',  # USD/CAD
            'GBPUSD=X',  # GBP/USD
            'EURJPY=X',  # EUR/JPY
            'USDJPY=X',  # USD/JPY
            'AUDUSD=X',  # AUD/USD
            'NZDUSD=X',  # NZD/USD
            'USDCHF=X',  # USD/CHF
            'EURGBP=X',  # EUR/GBP
            'EURCAD=X',  # EUR/CAD
            'GBPCAD=X',  # GBP/CAD
            'AUDCAD=X',  # AUD/CAD
            'EURAUD=X',  # EUR/AUD
            'XAUEUR=X',  # Gold vs EUR
            'BTC-USD',   # Bitcoin vs USD
            'ETH-USD'    # Ethereum vs USD
        ]
        
        self.request_count = 0
        self.max_requests = 450
        
        # Session times (UTC)
        self.sessions = {
            'asia': {'open': 23, 'close': 8, 'name': 'Asian Session'},
            'london': {'open': 7, 'close': 16, 'name': 'London Session'},
            'newyork': {'open': 13, 'close': 22, 'name': 'New York Session'}
        }
        
        # Trading parameters
        self.check_interval = 10  # minutes
        self.session_active_hours = 3
        # TEST MODE: 15m + SMA50/200 cross only (see sma_cross_test_mode). Legacy strategy uses 1h/4h when disabled.
        self.sma_cross_test_mode = os.environ.get("SMA_CROSS_TEST_MODE", "true").lower() in ("1", "true", "yes")
        self.timeframe = "15m" if self.sma_cross_test_mode else "1h"
        self.htf_timeframe = "4h"
        self.data_range = "6mo" if self.sma_cross_test_mode else "1mo"
        self.htf_data_range = "3mo"
        # Throttle repeated Firebase alerts for "cross within 4w" (not live bar)
        self._sma_recent_firebase_ts = {}
        self.sma_recent_firebase_cooldown_s = int(os.environ.get("SMA_RECENT_CROSS_COOLDOWN_S", str(6 * 3600)))

        # Simple in-memory cache for price series (process lifetime)
        # Structure: { (symbol, interval): { "closes": [...], "fetched_at": iso } }
        self._series_cache = {}

        # Stationarity strategy parameters (translated from Pine Script)
        self.lookback = 50
        self.stationarity_threshold = 0.05  # p-value threshold for stationarity
        self.zscore_threshold = 1.5
        self.trend_ma_period = 20
        self.sma_slope_period = 5
        self.max_sma_slope_percent = 0.25
        self.min_sma_distance_ticks = 500

        # Replay window: in test mode = last N weeks on 15m (96 bars/day * 7 * weeks). Else H1 bars.
        if os.environ.get("HISTORICAL_SIGNAL_LOOKBACK") is not None:
            self.historical_signal_lookback_bars = int(os.environ["HISTORICAL_SIGNAL_LOOKBACK"])
        else:
            weeks = int(os.environ.get("HISTORICAL_SIGNAL_WEEKS", "4"))
            if self.sma_cross_test_mode:
                self.historical_signal_lookback_bars = max(1, weeks) * 7 * 24 * 4  # 15m bars per week
            else:
                self.historical_signal_lookback_bars = max(1, weeks) * 7 * 24
        self.historical_signal_step_bars = int(os.environ.get("HISTORICAL_SIGNAL_STEP", "4"))
        self.historical_signal_max_matches = int(os.environ.get("HISTORICAL_SIGNAL_MAX_MATCHES", "10"))
        self._historical_replay_weeks_approx = (
            self.historical_signal_lookback_bars / float(7 * 24 * 4)
            if self.sma_cross_test_mode
            else self.historical_signal_lookback_bars / float(7 * 24)
        )

        # Approximate min tick by symbol for SMA-distance filter
        self.symbol_min_tick = {
            'EURUSD=X': 0.0001,
            'GBPJPY=X': 0.01,
            'AUDJPY=X': 0.01,
            'XAUUSD=X': 0.01,
            'USDCAD=X': 0.0001,
            'GBPUSD=X': 0.0001,
            'EURJPY=X': 0.01,
            'USDJPY=X': 0.01,
            'AUDUSD=X': 0.0001,
            'NZDUSD=X': 0.0001,
            'USDCHF=X': 0.0001,
            'EURGBP=X': 0.0001,
            'EURCAD=X': 0.0001,
            'GBPCAD=X': 0.0001,
            'AUDCAD=X': 0.0001,
            'EURAUD=X': 0.0001,
            'XAUEUR=X': 0.01,
            'BTC-USD': 0.01,
            'ETH-USD': 0.01
        }
        
        # Initialize Firebase (only once)
        self.db = self.init_firebase()
    
    def init_firebase(self):
        """Initialize Firebase connection - only once"""
        try:
            # Check if Firebase is already initialized
            if not firebase_admin._apps:
                print("🔑 Initializing Firebase...")
                
                # Check if we have the service account in environment variable
                if 'FIREBASE_SERVICE_ACCOUNT' in os.environ:
                    # Decode from base64
                    service_account_json = base64.b64decode(
                        os.environ['FIREBASE_SERVICE_ACCOUNT']
                    ).decode('utf-8')
                    
                    # Create a temporary file
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                        f.write(service_account_json)
                        temp_path = f.name
                    
                    cred = credentials.Certificate(temp_path)
                    firebase_admin.initialize_app(cred)
                    
                    # Clean up temp file
                    os.unlink(temp_path)
                    print("✅ Firebase initialized from environment variable")
                    
                else:
                    # Fallback to local file
                    script_dir = os.path.dirname(os.path.abspath(__file__))
                    service_account_path = os.path.join(script_dir, 'serviceAccountKey.json')
                    cred = credentials.Certificate(service_account_path)
                    firebase_admin.initialize_app(cred)
                    print("✅ Firebase initialized from local file")
            else:
                print("✅ Firebase already initialized, reusing existing instance")
            
            return firestore.client()
            
        except Exception as e:
            print(f"❌ Error initializing Firebase: {e}")
            return None
    
    def save_alert_to_firebase(self, alert_data):
        """Save alert to Firebase Firestore"""
        if not self.db:
            print("⚠️ Firebase not initialized, skipping save")
            return False
        
        try:
            # Add timestamp if not present
            if 'timestamp' not in alert_data:
                alert_data['timestamp'] = datetime.now().isoformat()
            
            # Add session info
            current_session = self.get_current_session()
            alert_data['session'] = current_session
            
            # Add to firestore
            doc_ref = self.db.collection('trading_alerts').document()
            doc_ref.set(alert_data)
            print(f"✅ Alert saved to Firebase: {alert_data['symbol']} - {alert_data['type']} ({current_session})")
            return True
        except Exception as e:
            print(f"❌ Error saving to Firebase: {e}")
            return False
    
    def get_current_session(self):
        """Determine which trading session is currently active"""
        now_utc = datetime.utcnow()
        current_hour = now_utc.hour
        
        for session_id, session in self.sessions.items():
            open_hour = session['open']
            close_hour = session['close']
            
            if close_hour < open_hour:
                if current_hour >= open_hour or current_hour < close_hour:
                    return session_id
            else:
                if open_hour <= current_hour < close_hour:
                    return session_id
        
        return 'off_hours'
    
    def is_trading_time(self):
        """Check if current time is within trading window"""
        now_utc = datetime.utcnow()
        current_hour = now_utc.hour
        current_minute = now_utc.minute
        
        for session_id, session in self.sessions.items():
            open_hour = session['open']
            window_end = (open_hour + self.session_active_hours) % 24
            
            if window_end < open_hour:
                if current_hour >= open_hour or current_hour < window_end:
                    if current_minute % self.check_interval < 15:
                        return True, session_id
            else:
                if open_hour <= current_hour < window_end:
                    if current_minute % self.check_interval < 15:
                        return True, session_id
        
        return False, None
    
    def get_historical_data(self, symbol, interval='15m', range='2d'):
        """Get historical data for indicator calculation"""
        if self.request_count >= self.max_requests:
            print(f"⚠️ Monthly request limit approaching for {symbol}")
            return None
            
        url = f"{self.base_url}/api/stock/get-chart"
        querystring = {
            "symbol": symbol,
            "interval": interval,
            "range": range,
            "region": "US"
        }
        
        try:
            print(f"📊 Fetching {interval} data for {symbol}...")
            response = requests.get(url, headers=self.headers, params=querystring)
            self.request_count += 1
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"⚠️ API error for {symbol}: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"❌ Error fetching history for {symbol}: {e}")
        return None
    
    def extract_close_prices(self, hist_data):
        """Extract valid close prices from Yahoo chart response."""
        prices = []
        if hist_data and 'chart' in hist_data:
            if 'result' in hist_data['chart'] and len(hist_data['chart']['result']) > 0:
                result = hist_data['chart']['result'][0]
                if 'indicators' in result and 'quote' in result['indicators']:
                    quotes = result['indicators']['quote'][0]
                    if 'close' in quotes:
                        prices = [p for p in quotes['close'] if p is not None]
        return prices

    def get_close_series(self, symbol, *, interval, preferred_range, min_len, ranges_to_try=None, max_age_seconds=60 * 20):
        """
        Return a list of close prices for a symbol/interval.

        - Reuses a recent cached series when available.
        - If the series is too short, retries with larger ranges to obtain enough candles.
        """
        cache_key = (symbol, interval)
        cached = self._series_cache.get(cache_key)
        if cached:
            try:
                fetched_at = datetime.fromisoformat(cached["fetched_at"])
                age_seconds = (datetime.utcnow() - fetched_at).total_seconds()
                if age_seconds <= max_age_seconds and isinstance(cached.get("closes"), list) and len(cached["closes"]) >= min_len:
                    return cached["closes"]
            except Exception:
                pass

        if ranges_to_try is None:
            ranges_to_try = [preferred_range, '3mo', '6mo', '1y', '2y', '5y']

        best = []
        for r in ranges_to_try:
            hist_data = self.get_historical_data(symbol, interval=interval, range=r)
            closes = self.extract_close_prices(hist_data)
            if len(closes) > len(best):
                best = closes
            if len(closes) >= min_len:
                best = closes
                break

        self._series_cache[cache_key] = {
            "closes": best,
            "fetched_at": datetime.utcnow().isoformat(),
            "range_used": r if 'r' in locals() else preferred_range,
        }
        return best

    def check_stationarity(self, prices, verbose=True):
        """
        Check stationarity using multiple methods to match TradingView's behavior.
        Since TradingView doesn't have a built-in ADF test, we implement several
        stationarity detection methods and combine them.
        """
        try:
            # Use the most recent data for stationarity test
            if len(prices) < self.lookback:
                return 1.0, False  # Not stationary, high p-value
            
            # Use the last 'lookback' periods
            recent_prices = prices[-self.lookback:]
            recent_prices_array = np.array(recent_prices)
            
            # Method 1: Augmented Dickey-Fuller test
            try:
                adf_result = adfuller(recent_prices_array, autolag='AIC')
                adf_p_value = adf_result[1]
                adf_is_stationary = adf_p_value < self.stationarity_threshold
            except:
                adf_p_value = 1.0
                adf_is_stationary = False
            
            # Method 2: Check for trend using linear regression
            # If there's a strong trend, it's likely non-stationary
            x = np.arange(len(recent_prices_array))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, recent_prices_array)
            trend_strength = abs(slope) / (np.std(recent_prices_array) + 1e-10)
            has_strong_trend = trend_strength > 0.1  # If slope > 10% of volatility
            
            # Method 3: Rolling mean stability
            rolling_mean = pd.Series(recent_prices_array).rolling(window=10).mean()
            mean_variation = rolling_mean.std() / (rolling_mean.mean() + 1e-10)
            mean_stable = mean_variation < 0.05  # Less than 5% variation in rolling mean
            
            # Method 4: Check for mean reversion using Hurst exponent approximation
            # Simplified: check if price crosses its mean frequently
            mean_price = np.mean(recent_prices_array)
            cross_count = 0
            for i in range(1, len(recent_prices_array)):
                if (recent_prices_array[i-1] - mean_price) * (recent_prices_array[i] - mean_price) < 0:
                    cross_count += 1
            cross_rate = cross_count / len(recent_prices_array)
            is_mean_reverting = cross_rate > 0.15  # Crosses mean at least 15% of the time
            
            # Combine methods
            # For stationarity, we want: no strong trend, stable mean, mean-reverting behavior, AND low ADF p-value
            is_stationary = (not has_strong_trend) and mean_stable and is_mean_reverting and adf_is_stationary
            
            # Use the most conservative p-value (max) for reporting
            final_p_value = max(adf_p_value, 1.0 - cross_rate)
            
            # Log detailed stationarity check
            if verbose:
                print(f"   Stationarity Details - ADF p-value: {adf_p_value:.4f}, Trend Strength: {trend_strength:.3f}, "
                      f"Mean Variation: {mean_variation:.3f}, Cross Rate: {cross_rate:.3f}")
                print(f"   Stationary: {is_stationary} (Strong trend: {has_strong_trend}, Mean stable: {mean_stable}, Mean reverting: {is_mean_reverting})")
            
            return final_p_value, is_stationary
            
        except Exception as e:
            print(f"⚠️ Error in stationarity test: {e}")
            return 1.0, False  # Return high p-value and not stationary on error

    def _htf_prefix_for_h1_end(self, htf_prices, h1_total_len, end_index):
        """Approximate how much 4H history aligns when H1 series ends at end_index (inclusive)."""
        if not htf_prices or h1_total_len <= 0:
            return []
        ratio = (end_index + 1) / float(h1_total_len)
        htf_len = int(round(ratio * len(htf_prices)))
        htf_len = max(self.trend_ma_period, htf_len)
        htf_len = min(htf_len, len(htf_prices))
        return htf_prices[:htf_len]

    def _sma_cross_at_end(self, prices, end_index_inclusive):
        """
        If a 50/200 SMA cross completes on bar end_index_inclusive, return ('BUY'|'SELL', signal_name).
        Otherwise None. Uses closes up to and including that bar.
        """
        if end_index_inclusive < 199:
            return None
        sub = prices[: end_index_inclusive + 1]
        if len(sub) < 201:
            return None
        s = pd.Series(sub, dtype="float64")
        sma50 = s.rolling(window=50).mean()
        sma200 = s.rolling(window=200).mean()
        p50_p, p200_p = sma50.iloc[-2], sma200.iloc[-2]
        p50_n, p200_n = sma50.iloc[-1], sma200.iloc[-1]
        if any(pd.isna(v) for v in (p50_p, p200_p, p50_n, p200_n)):
            return None
        if p50_p <= p200_p and p50_n > p200_n:
            return ("BUY", "sma50_cross_above_200_15m")
        if p50_p >= p200_p and p50_n < p200_n:
            return ("SELL", "sma50_cross_below_200_15m")
        return None

    def _most_recent_sma_cross_in_window(self, prices, max_bars_ago):
        """Newest cross within last max_bars_ago bars (excluding bar 0 handled separately). bars_ago >= 1."""
        n = len(prices)
        start = max(199, n - 1 - max_bars_ago)
        for end in range(n - 2, start - 1, -1):
            hit = self._sma_cross_at_end(prices, end)
            if hit:
                return hit[0], hit[1], n - 1 - end
        return None

    def _log_legacy_strategy_display_only(self, symbol, prices, htf_prices, verbose=True):
        """
        Former stationarity / z-score / HTF / filter stack — LOG ONLY (no trades) in SMA test mode.
        Uses the same 15m close series as the active test strategy.
        """
        if not verbose:
            return
        min_need = max(200, self.lookback, 50 + self.sma_slope_period)
        if len(prices) < min_need:
            print(f"   [legacy display] insufficient 15m bars ({len(prices)} < {min_need})")
            return

        closes = pd.Series(prices, dtype="float64")
        p_value, is_stationary = self.check_stationarity(prices, verbose=verbose)
        if verbose:
            print(f"   [legacy display] composite p_value={p_value:.4f}, is_stationary={is_stationary}")

        sma_lookback_series = closes.rolling(window=self.lookback).mean()
        stddev_lookback_series = closes.rolling(window=self.lookback).std(ddof=0)
        sma50_series = closes.rolling(window=50).mean()
        sma200_series = closes.rolling(window=200).mean()

        sma_lookback = sma_lookback_series.iloc[-1]
        stddev_lookback = stddev_lookback_series.iloc[-1]
        sma50 = sma50_series.iloc[-1]
        sma200 = sma200_series.iloc[-1]
        if len(sma50_series) > self.sma_slope_period:
            prev_sma50 = sma50_series.iloc[-1 - self.sma_slope_period]
        else:
            prev_sma50 = sma50
        current_price = closes.iloc[-1]

        if any(pd.isna(v) for v in [sma_lookback, stddev_lookback, sma50, sma200]) or stddev_lookback == 0:
            print("   [legacy display] skip (NaN or zero std)")
            return

        zscore = (current_price - sma_lookback) / stddev_lookback
        if prev_sma50 != 0:
            sma50_pct_change = ((sma50 - prev_sma50) / prev_sma50) * 100
        else:
            sma50_pct_change = 0
        sma_slope_ok = abs(sma50_pct_change) <= self.max_sma_slope_percent
        min_tick = self.symbol_min_tick.get(symbol, 0.0001)
        sma_distance_ticks = abs(sma50 - sma200) / min_tick
        sma_distance_ok = sma_distance_ticks >= self.min_sma_distance_ticks
        bullish_trend = sma50 > sma200
        bearish_trend = sma50 < sma200
        bullish_valid = bullish_trend and sma_distance_ok and sma_slope_ok
        bearish_valid = bearish_trend and sma_distance_ok and sma_slope_ok

        htf_trend_bullish = htf_trend_bearish = None
        if htf_prices and len(htf_prices) >= self.trend_ma_period:
            htf_closes = pd.Series(htf_prices, dtype="float64")
            htf_sma_series = htf_closes.rolling(window=self.trend_ma_period).mean()
            htf_close = htf_closes.iloc[-1]
            htf_sma = htf_sma_series.iloc[-1]
            if not pd.isna(htf_sma):
                htf_trend_bullish = htf_close > htf_sma
                htf_trend_bearish = htf_close < htf_sma

        print(
            f"   [legacy display] Z-Score: {zscore:.3f}, SMA50 slope: {sma50_pct_change:.2f}%, "
            f"SMA dist ticks: {sma_distance_ticks:.0f} (ok={sma_distance_ok})"
        )
        if htf_trend_bullish is not None:
            print(
                f"   [legacy display] HTF(4h) bullish={htf_trend_bullish}, bearish={htf_trend_bearish} | "
                f"bull_valid={bullish_valid} bear_valid={bearish_valid} (NOT used for entries in test mode)"
            )
        else:
            print(
                f"   [legacy display] HTF skipped | bull_valid={bullish_valid} bear_valid={bearish_valid} "
                f"(NOT used for entries in test mode)"
            )

    def _build_sma_cross_signals(self, symbol, prices, verbose=True):
        """Entries: live 50/200 cross on latest bar, or most recent cross inside lookback window (throttled)."""
        if len(prices) < 201:
            return []

        live = self._sma_cross_at_end(prices, len(prices) - 1)
        current_price = float(prices[-1])
        s = pd.Series(prices, dtype="float64")
        sma50 = float(s.rolling(50).mean().iloc[-1])
        sma200 = float(s.rolling(200).mean().iloc[-1])

        base_ctx = {
            "sma50": sma50,
            "sma200": sma200,
            "strategy_mode": "sma50_200_cross_15m_test",
        }

        signals = []
        if live:
            sig_type, sig_name = live
            signals.append({
                "type": sig_type,
                "signal": sig_name,
                "price": current_price,
                "confidence": "TEST_LIVE_CROSS",
                "trigger": "live",
                "cross_bars_ago": 0,
                **base_ctx,
            })
            if verbose:
                print(f"   🎯 TEST: Live SMA cross — {sig_type} ({sig_name})")
            return signals

        recent = self._most_recent_sma_cross_in_window(prices, self.historical_signal_lookback_bars)
        if not recent:
            if verbose:
                print("   TEST: No SMA50/200 cross in last bar or lookback window.")
            return []

        _t, sig_name, bars_ago = recent
        sig_type = "BUY" if "above" in sig_name else "SELL"
        now = time.time()
        last_t = self._sma_recent_firebase_ts.get(symbol, 0)
        if now - last_t < self.sma_recent_firebase_cooldown_s:
            if verbose:
                print(
                    f"   TEST: Recent cross ~{bars_ago} bars ago ({sig_type}) — cooldown, no duplicate Firebase"
                )
            return []

        self._sma_recent_firebase_ts[symbol] = now
        cross_idx = len(prices) - 1 - int(bars_ago)
        cross_price = float(prices[cross_idx]) if 0 <= cross_idx < len(prices) else current_price
        signals.append({
            "type": sig_type,
            "signal": sig_name + "_within_lookback",
            "price": cross_price,
            "price_last_bar": current_price,
            "confidence": "TEST_RECENT_CROSS",
            "trigger": "recent_lookback",
            "cross_bars_ago": int(bars_ago),
            **base_ctx,
        })
        if verbose:
            print(f"   🎯 TEST: SMA cross within lookback — {sig_type} ~{bars_ago} bars ago ({sig_name})")
        return signals

    def find_historical_matching_signals(self, symbol, prices, htf_prices):
        """
        Log replay: SMA50/200 crosses in test mode; else legacy stationarity + HTF replay.
        """
        if self.sma_cross_test_mode:
            min_bars = 201
            n = len(prices)
            if n < min_bars:
                return []
            win = min(self.historical_signal_lookback_bars, n - min_bars)
            start = max(min_bars - 1, n - 1 - win)
            step = max(1, self.historical_signal_step_bars)
            matches = []
            for end in range(n - 1, start - 1, -step):
                hit = self._sma_cross_at_end(prices, end)
                if not hit:
                    continue
                sig_type, sig_name = hit
                bars_ago = n - 1 - end
                if len(matches) >= self.historical_signal_max_matches:
                    break
                matches.append({
                    "bars_ago": bars_ago,
                    "h1_end_index": end,
                    "type": sig_type,
                    "signal": sig_name,
                    "price": float(prices[end]),
                    "zscore": None,
                    "is_stationary": None,
                })
            return matches

        min_required = max(200, self.lookback, 50 + self.sma_slope_period)
        h1_total = len(prices)
        if h1_total < min_required or len(htf_prices) < self.trend_ma_period:
            return []

        start = max(min_required - 1, h1_total - self.historical_signal_lookback_bars)
        step = max(1, self.historical_signal_step_bars)
        matches = []

        # Newest → oldest so logs show the most recent bars that matched first
        for end in range(h1_total - 1, start - 1, -step):
            sub_h1 = prices[: end + 1]
            sub_htf = self._htf_prefix_for_h1_end(htf_prices, h1_total, end)
            if len(sub_htf) < self.trend_ma_period:
                continue

            sigs = self._legacy_check_signals_stationarity_htf(symbol, sub_h1, sub_htf, verbose=False)
            if not sigs:
                continue

            bars_ago = h1_total - 1 - end
            for s in sigs:
                if len(matches) >= self.historical_signal_max_matches:
                    return matches
                matches.append({
                    "bars_ago": bars_ago,
                    "h1_end_index": end,
                    "type": s.get("type"),
                    "signal": s.get("signal"),
                    "price": s.get("price"),
                    "zscore": s.get("zscore"),
                    "is_stationary": s.get("is_stationary"),
                })

        return matches

    def check_signals(self, symbol, prices, htf_prices, verbose=True):
        """Dispatch: test mode = 15m SMA50/200 cross only (+ legacy metrics logged). Else full stationarity strategy."""
        if self.sma_cross_test_mode:
            if len(prices) < 201:
                if verbose:
                    print(f"⚠️ Insufficient 15m data: {len(prices)} closes (need >= 201)")
                return []
            self._log_legacy_strategy_display_only(symbol, prices, htf_prices or [], verbose=verbose)
            if verbose:
                print("   --- TEST MODE: Firebase entries only from SMA50 × SMA200 (above); legacy = display only ---")
            return self._build_sma_cross_signals(symbol, prices, verbose=verbose)
        return self._legacy_check_signals_stationarity_htf(symbol, prices, htf_prices, verbose=verbose)

    def _legacy_check_signals_stationarity_htf(self, symbol, prices, htf_prices, verbose=True):
        """Original stationarity + 4H + z-score strategy (disabled when sma_cross_test_mode is True)."""
        min_required = max(200, self.lookback, 50 + self.sma_slope_period)
        if len(prices) < min_required or len(htf_prices) < self.trend_ma_period:
            if verbose:
                print(f"⚠️ Insufficient data: prices={len(prices)}, htf={len(htf_prices)}")
            return []

        closes = pd.Series(prices, dtype='float64')
        htf_closes = pd.Series(htf_prices, dtype='float64')

        # Calculate stationarity (returns tuple of p_value and is_stationary)
        p_value, is_stationary = self.check_stationarity(prices, verbose=verbose)
        
        if verbose:
            print(f"📊 Stationarity check: p_value={p_value:.4f}, is_stationary={is_stationary}")

        # Calculate all indicators
        sma_lookback_series = closes.rolling(window=self.lookback).mean()
        stddev_lookback_series = closes.rolling(window=self.lookback).std(ddof=0)
        sma50_series = closes.rolling(window=50).mean()
        sma200_series = closes.rolling(window=200).mean()
        htf_sma_series = htf_closes.rolling(window=self.trend_ma_period).mean()

        # Get latest values
        sma_lookback = sma_lookback_series.iloc[-1]
        stddev_lookback = stddev_lookback_series.iloc[-1]
        sma50 = sma50_series.iloc[-1]
        sma200 = sma200_series.iloc[-1]
        
        # Get previous SMA50 for slope calculation
        if len(sma50_series) > self.sma_slope_period:
            prev_sma50 = sma50_series.iloc[-1 - self.sma_slope_period]
        else:
            prev_sma50 = sma50
            
        htf_close = htf_closes.iloc[-1]
        htf_sma = htf_sma_series.iloc[-1]
        current_price = closes.iloc[-1]

        # Check for NaN values
        if any(pd.isna(v) for v in [sma_lookback, stddev_lookback, sma50, sma200, htf_sma]):
            if verbose:
                print(f"⚠️ NaN values detected, skipping signal check")
            return []
        
        if stddev_lookback == 0:
            if verbose:
                print(f"⚠️ Standard deviation is zero, skipping")
            return []

        # Calculate Z-score
        zscore = (current_price - sma_lookback) / stddev_lookback

        # Higher timeframe trend
        htf_trend_bullish = htf_close > htf_sma
        htf_trend_bearish = htf_close < htf_sma
        
        # SMA trend direction
        bullish_trend = sma50 > sma200
        bearish_trend = sma50 < sma200

        # SMA slope check
        if prev_sma50 != 0:
            sma50_pct_change = ((sma50 - prev_sma50) / prev_sma50) * 100
        else:
            sma50_pct_change = 0
            
        sma_slope_ok = abs(sma50_pct_change) <= self.max_sma_slope_percent

        # SMA distance in ticks
        min_tick = self.symbol_min_tick.get(symbol, 0.0001)
        sma_distance_ticks = abs(sma50 - sma200) / min_tick
        sma_distance_ok = sma_distance_ticks >= self.min_sma_distance_ticks

        # Combined trend validity (matching Pine Script logic)
        bullish_valid = bullish_trend and sma_distance_ok and sma_slope_ok
        bearish_valid = bearish_trend and sma_distance_ok and sma_slope_ok

        if verbose:
            print(f"📊 Indicators - Z-Score: {zscore:.3f}, SMA50 Slope: {sma50_pct_change:.2f}%, Distance: {sma_distance_ticks:.0f} ticks")
            print(f"📊 Trends - HTF Bullish: {htf_trend_bullish}, Bullish Trend: {bullish_trend}, Bearish Valid: {bearish_valid}")
            print(f"📊 Filters - SMA Slope OK: {sma_slope_ok}, SMA Distance OK: {sma_distance_ok}")

        # Base conditions (matching Pine Script exactly)
        base_entry_long = is_stationary and zscore < -self.zscore_threshold and htf_trend_bullish
        base_entry_short = is_stationary and zscore > self.zscore_threshold and htf_trend_bearish
        base_exit_long = zscore > self.zscore_threshold
        base_exit_short = zscore < -self.zscore_threshold

        # Apply filters (entries in trend direction, exits in opposite trend)
        entry_long = base_entry_long and bullish_valid
        entry_short = base_entry_short and bearish_valid
        exit_long = base_exit_long and bearish_valid    # long exit only in valid downtrend
        exit_short = base_exit_short and bullish_valid   # short exit only in valid uptrend

        signals = []
        
        if entry_long:
            signals.append({
                'type': 'BUY',
                'signal': 'h1_stationarity_entry_long',
                'price': float(current_price),
                'confidence': 'HIGH'
            })
            if verbose:
                print(f"🎯 LONG ENTRY SIGNAL DETECTED!")
            
        if entry_short:
            signals.append({
                'type': 'SELL',
                'signal': 'h1_stationarity_entry_short',
                'price': float(current_price),
                'confidence': 'HIGH'
            })
            if verbose:
                print(f"🎯 SHORT ENTRY SIGNAL DETECTED!")
            
        if exit_long:
            signals.append({
                'type': 'EXIT_LONG',
                'signal': 'h1_stationarity_exit_long',
                'price': float(current_price),
                'confidence': 'MEDIUM'
            })
            if verbose:
                print(f"🎯 LONG EXIT SIGNAL DETECTED!")
            
        if exit_short:
            signals.append({
                'type': 'EXIT_SHORT',
                'signal': 'h1_stationarity_exit_short',
                'price': float(current_price),
                'confidence': 'MEDIUM'
            })
            if verbose:
                print(f"🎯 SHORT EXIT SIGNAL DETECTED!")

        # Add indicator context to signals
        indicator_context = {
            'zscore': float(zscore),
            'stationarity_p_value': float(p_value),
            'is_stationary': is_stationary,
            'sma50': float(sma50),
            'sma200': float(sma200),
            'sma50_slope_pct': float(sma50_pct_change),
            'sma_distance_ticks': float(sma_distance_ticks),
            'htf_sma': float(htf_sma),
            'htf_close': float(htf_close),
            'bullish_valid': bullish_valid,
            'bearish_valid': bearish_valid
        }
        
        for s in signals:
            s.update(indicator_context)

        return signals
    
    def scan_watchlist(self):
        """Run one full watchlist pass (24/7 — not limited to FX session windows)."""
        global last_check_time, total_signals
        
        print(f"\n{'='*60}")
        print(f"🚀 Scan started at {datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC")
        print(f"{'='*60}")
        
        # 24/7 scan — no session window gate. Session name is only stored on alerts for context.
        current_session = self.get_current_session()
        last_check_time = datetime.now().isoformat()

        if current_session in self.sessions:
            sess_label = self.sessions[current_session]['name']
        else:
            sess_label = 'off_hours (UTC)'
        print(f"📈 24/7 scan — all hours (metadata session tag: {sess_label})")
        
        signals_found = 0
        
        for symbol in self.watchlist:
            if self.request_count >= self.max_requests:
                print("⚠️ Monthly limit reached. Stopping scan.")
                break
            
            print(f"\n🔍 Checking {symbol}...")
            try:
                if self.sma_cross_test_mode:
                    min_fetch = max(
                        210,
                        min(self.historical_signal_lookback_bars + 220, 4000),
                    )
                    prices = self.get_close_series(
                        symbol,
                        interval=self.timeframe,
                        preferred_range=self.data_range,
                        min_len=min_fetch,
                    )
                    htf_prices = self.get_close_series(
                        symbol,
                        interval=self.htf_timeframe,
                        preferred_range=self.htf_data_range,
                        min_len=self.trend_ma_period,
                    )
                    htf_for_display = htf_prices if len(htf_prices) >= self.trend_ma_period else []
                    ok = len(prices) >= 201
                else:
                    min_required = max(200, self.lookback, 50 + self.sma_slope_period)
                    prices = self.get_close_series(
                        symbol,
                        interval=self.timeframe,
                        preferred_range=self.data_range,
                        min_len=min_required,
                    )
                    htf_prices = self.get_close_series(
                        symbol,
                        interval=self.htf_timeframe,
                        preferred_range=self.htf_data_range,
                        min_len=self.trend_ma_period,
                    )
                    htf_for_display = htf_prices
                    ok = len(prices) >= min_required and len(htf_prices) >= self.trend_ma_period

                if ok:
                    signals = self.check_signals(symbol, prices, htf_for_display, verbose=True)
                        
                    if signals:
                        for signal in signals:
                            signal['timeframe'] = self.timeframe
                            signal['session'] = current_session

                            # Add to Firebase
                            success = self.save_alert_to_firebase({
                                'symbol': symbol,
                                **signal
                            })

                            if success:
                                signals_found += 1
                                total_signals += 1

                                print(f"\n🎯 SIGNAL FOUND: {symbol}")
                                print(f"   Type: {signal['type']} - {signal['signal']}")
                                print(f"   Price: ${signal['price']:.4f}")
                                zs = signal.get("zscore")
                                print(f"   Z-Score: {zs:.3f}" if zs is not None else "   Z-Score: n/a")
                                print(f"   Confidence: {signal['confidence']}")
                            else:
                                print(f"❌ Failed to save signal to Firebase")
                    else:
                        print(f"ℹ️ No signals for {symbol}")

                    past = self.find_historical_matching_signals(symbol, prices, htf_prices)
                    bar_label = "15m" if self.sma_cross_test_mode else "H1"
                    if past:
                        print(
                            f"   📜 Historical matches (same rules; last {self.historical_signal_lookback_bars} {bar_label} bars "
                            f"≈ {self._historical_replay_weeks_approx:.1f} weeks, step {self.historical_signal_step_bars}):"
                        )
                        for m in past:
                            z = m.get("zscore")
                            z_str = f"{z:.3f}" if z is not None else "n/a"
                            pr = m.get("price")
                            p_str = f"{pr:.4f}" if pr is not None else "n/a"
                            st = m.get("is_stationary")
                            print(
                                f"      ~{m['bars_ago']} bars ago | {m.get('type')} | {m.get('signal')} | "
                                f"z={z_str} | price={p_str} | stationary={st}"
                            )
                    else:
                        print(
                            f"   📜 No historical matches in replay window "
                            f"({self.historical_signal_lookback_bars} {bar_label} ≈ {self._historical_replay_weeks_approx:.1f} weeks, "
                            f"step {self.historical_signal_step_bars})."
                        )
                else:
                    print(
                        f"⚠️ Insufficient data for {symbol} ({len(prices)} {self.timeframe} candles, "
                        f"{len(htf_prices)} {self.htf_timeframe} candles)"
                    )
            except Exception as e:
                print(f"❌ Error processing {symbol}: {e}")
            
            time.sleep(1.5)
        
        print(f"\n{'='*60}")
        print(f"✅ Scan completed! Found {signals_found} signals")
        print(f"📊 API Requests used: {self.request_count}/{self.max_requests}")
        print(f"{'='*60}")

# Flask routes for health checks and status
@app.route('/')
def home():
    return jsonify({
        'status': 'running',
        'last_check': last_check_time,
        'total_signals': total_signals,
        'requests_used': bot_instance.request_count if bot_instance else 0
    })

@app.route('/health')
def health():
    return 'OK', 200

@app.route('/status')
def status():
    """Detailed status endpoint"""
    if not bot_instance:
        return jsonify({'error': 'Bot not initialized'}), 500
    
    current_session = bot_instance.get_current_session()
    is_trading, active_session = bot_instance.is_trading_time()
    
    return jsonify({
        'status': 'running',
        'scan_mode': '24_7',
        'scan_note': 'Watchlist is scanned around the clock; session fields are labels only.',
        'current_time_utc': datetime.utcnow().isoformat(),
        'current_session': current_session,
        'is_trading_time': is_trading,
        'active_session': active_session if is_trading else None,
        'last_check': last_check_time,
        'total_signals': total_signals,
        'requests_used': bot_instance.request_count,
        'requests_remaining': bot_instance.max_requests - bot_instance.request_count,
        'watchlist': bot_instance.watchlist,
        'sma_cross_test_mode': getattr(bot_instance, 'sma_cross_test_mode', False),
        'timeframe': bot_instance.timeframe,
    })

def run_bot():
    """Run the bot in a background thread"""
    global bot_instance
    while True:
        try:
            bot_instance.scan_watchlist()
        except Exception as e:
            print(f"❌ Error in bot scan: {e}")
        
        # Sleep for 10 minutes
        time.sleep(600)

if __name__ == "__main__":
    print("🚀 Initializing Fib SMA Trading Bot...")
    print(f"Python version: {sys.version}")
    print(f"Current UTC time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}")
    
    # Get API key from environment variable
    API_KEY = os.environ.get('RAPIDAPI_KEY')
    if not API_KEY:
        print("❌ RAPIDAPI_KEY environment variable not set!")
        sys.exit(1)
    
    # Create bot instance
    bot_instance = FibSMATradingBot(API_KEY)
    if bot_instance.sma_cross_test_mode:
        print(
            "🧪 SMA_CROSS_TEST_MODE=ON: 15m SMA50/200 cross → Firebase; "
            "stationarity/HTF/z-score logged as [legacy display] only. "
            "Set SMA_CROSS_TEST_MODE=false to restore full strategy."
        )

    # One-time startup test write (verify Firebase console receives data)
    write_startup_test_to_firebase(bot_instance.db, bot=bot_instance)
    
    # Start bot in background thread
    bot_thread = threading.Thread(target=run_bot, daemon=True)
    bot_thread.start()
    print("✅ Bot thread started")
    
    # Get port from environment (Render sets this automatically)
    port = int(os.environ.get('PORT', 10000))
    print(f"🚀 Starting web server on port {port}")
    
    # Run Flask app
    app.run(host='0.0.0.0', port=port)


