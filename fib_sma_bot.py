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
import firebase_admin
from firebase_admin import credentials, firestore
from flask import Flask, jsonify
import threading

# Initialize Flask app
app = Flask(__name__)

# Global variables for bot status
bot_instance = None
last_check_time = None
total_signals = 0

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
        self.check_interval = 30  # minutes
        self.session_active_hours = 3
        self.timeframe = '1h'
        self.htf_timeframe = '4h'
        self.data_range = '1mo'
        self.htf_data_range = '3mo'

        # Stationarity strategy parameters (translated from Pine Script)
        self.lookback = 50
        self.stationarity_threshold = 0.05
        self.zscore_threshold = 1.5
        self.trend_ma_period = 20
        self.sma_slope_period = 5
        self.max_sma_slope_percent = 0.25
        self.min_sma_distance_ticks = 500

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

    def check_signals(self, symbol, prices, htf_prices):
        """Check stationarity strategy signals on H1 timeframe."""
        min_required = max(200, self.lookback, 50 + self.sma_slope_period)
        if len(prices) < min_required or len(htf_prices) < self.trend_ma_period:
            return []

        closes = pd.Series(prices, dtype='float64')
        htf_closes = pd.Series(htf_prices, dtype='float64')

        # Stationarity placeholder from Pine script
        p_value = 0.01
        is_stationary = p_value < self.stationarity_threshold
        if not is_stationary:
            return []

        sma_lookback_series = closes.rolling(window=self.lookback).mean()
        stddev_lookback_series = closes.rolling(window=self.lookback).std(ddof=0)
        sma50_series = closes.rolling(window=50).mean()
        sma200_series = closes.rolling(window=200).mean()
        htf_sma_series = htf_closes.rolling(window=self.trend_ma_period).mean()

        sma_lookback = sma_lookback_series.iloc[-1]
        stddev_lookback = stddev_lookback_series.iloc[-1]
        sma50 = sma50_series.iloc[-1]
        sma200 = sma200_series.iloc[-1]
        prev_sma50 = sma50_series.iloc[-1 - self.sma_slope_period]
        htf_close = htf_closes.iloc[-1]
        htf_sma = htf_sma_series.iloc[-1]
        current_price = closes.iloc[-1]

        if any(pd.isna(v) for v in [sma_lookback, stddev_lookback, sma50, sma200, prev_sma50, htf_sma]):
            return []
        if stddev_lookback == 0 or prev_sma50 == 0:
            return []

        zscore = (current_price - sma_lookback) / stddev_lookback

        htf_trend_bullish = htf_close > htf_sma
        htf_trend_bearish = htf_close < htf_sma
        bullish_trend = sma50 > sma200
        bearish_trend = sma50 < sma200

        sma50_pct_change = ((sma50 - prev_sma50) / prev_sma50) * 100
        sma_slope_ok = abs(sma50_pct_change) <= self.max_sma_slope_percent

        min_tick = self.symbol_min_tick.get(symbol, 0.0001)
        sma_distance_ticks = abs(sma50 - sma200) / min_tick
        sma_distance_ok = sma_distance_ticks >= self.min_sma_distance_ticks

        bullish_valid = bullish_trend and sma_distance_ok and sma_slope_ok
        bearish_valid = bearish_trend and sma_distance_ok and sma_slope_ok

        base_entry_long = is_stationary and zscore < -self.zscore_threshold and htf_trend_bullish
        base_entry_short = is_stationary and zscore > self.zscore_threshold and htf_trend_bearish
        base_exit_long = zscore > self.zscore_threshold
        base_exit_short = zscore < -self.zscore_threshold

        entry_long = base_entry_long and bullish_valid
        entry_short = base_entry_short and bearish_valid
        exit_long = base_exit_long and bearish_valid
        exit_short = base_exit_short and bullish_valid

        signals = []
        if entry_long:
            signals.append({
                'type': 'BUY',
                'signal': 'h1_stationarity_entry_long',
                'price': float(current_price),
                'confidence': 'HIGH'
            })
        if entry_short:
            signals.append({
                'type': 'SELL',
                'signal': 'h1_stationarity_entry_short',
                'price': float(current_price),
                'confidence': 'HIGH'
            })
        if exit_long:
            signals.append({
                'type': 'EXIT_LONG',
                'signal': 'h1_stationarity_exit_long',
                'price': float(current_price),
                'confidence': 'MEDIUM'
            })
        if exit_short:
            signals.append({
                'type': 'EXIT_SHORT',
                'signal': 'h1_stationarity_exit_short',
                'price': float(current_price),
                'confidence': 'MEDIUM'
            })

        indicator_context = {
            'zscore': float(zscore),
            'stationarity_p_value': float(p_value),
            'sma50': float(sma50),
            'sma200': float(sma200),
            'sma50_slope_pct': float(sma50_pct_change),
            'sma_distance_ticks': float(sma_distance_ticks),
            'htf_sma': float(htf_sma),
            'htf_close': float(htf_close)
        }
        for s in signals:
            s.update(indicator_context)

        return signals
    
    def scan_watchlist(self):
        """Main scanning function"""
        global last_check_time, total_signals
        
        print(f"\n{'='*60}")
        print(f"🚀 Scan started at {datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC")
        print(f"{'='*60}")
        
        is_trading_time, current_session = self.is_trading_time()
        last_check_time = datetime.now().isoformat()
        
        if not is_trading_time:
            print(f"⏰ Not in trading window. Next check at {(datetime.utcnow() + timedelta(minutes=30)).strftime('%H:%M')} UTC")
            return
        
        print(f"📈 Trading window active - {self.sessions[current_session]['name']}")
        
        signals_found = 0
        
        for symbol in self.watchlist:
            if self.request_count >= self.max_requests:
                print("⚠️ Monthly limit reached. Stopping scan.")
                break
            
            print(f"\n🔍 Checking {symbol}...")
            hist_data = self.get_historical_data(symbol, interval=self.timeframe, range=self.data_range)
            htf_data = self.get_historical_data(symbol, interval=self.htf_timeframe, range=self.htf_data_range)

            if hist_data and htf_data:
                try:
                    prices = self.extract_close_prices(hist_data)
                    htf_prices = self.extract_close_prices(htf_data)

                    if len(prices) >= 200 and len(htf_prices) >= self.trend_ma_period:
                        signals = self.check_signals(symbol, prices, htf_prices)
                        
                        if signals:
                            for signal in signals:
                                signal['timeframe'] = self.timeframe
                                signal['session'] = current_session
                                
                                self.save_alert_to_firebase({
                                    'symbol': symbol,
                                    **signal
                                })
                                signals_found += 1
                                total_signals += 1
                                
                                print(f"\n🎯 SIGNAL FOUND: {symbol}")
                                print(f"   Type: {signal['type']} - {signal['signal']}")
                                print(f"   Price: ${signal['price']:.4f}")
                                print(f"   Z-Score: {signal['zscore']:.3f}")
                                print(f"   Confidence: {signal['confidence']}")
                        else:
                            print(f"ℹ️ No signals for {symbol}")
                    else:
                        print(f"⚠️ Insufficient data for {symbol} ({len(prices)} {self.timeframe} candles, {len(htf_prices)} {self.htf_timeframe} candles)")
                        
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
        'current_time_utc': datetime.utcnow().isoformat(),
        'current_session': current_session,
        'is_trading_time': is_trading,
        'active_session': active_session if is_trading else None,
        'last_check': last_check_time,
        'total_signals': total_signals,
        'requests_used': bot_instance.request_count,
        'requests_remaining': bot_instance.max_requests - bot_instance.request_count,
        'watchlist': bot_instance.watchlist
    })

def run_bot():
    """Run the bot in a background thread"""
    global bot_instance
    while True:
        try:
            bot_instance.scan_watchlist()
        except Exception as e:
            print(f"❌ Error in bot scan: {e}")
        
        # Sleep for 30 minutes
        time.sleep(1800)

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
    
    # Start bot in background thread
    bot_thread = threading.Thread(target=run_bot, daemon=True)
    bot_thread.start()
    print("✅ Bot thread started")
    
    # Get port from environment (Render sets this automatically)
    port = int(os.environ.get('PORT', 10000))
    print(f"🚀 Starting web server on port {port}")
    
    # Run Flask app
    app.run(host='0.0.0.0', port=port)