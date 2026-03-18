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
            'GBPJPY=X',  # GBP/JPY Forex
            'XAUUSD=X',  # Gold vs USD
            'USDCAD=X',  # USD/CAD Forex
            'BTC-USD',   # Bitcoin vs USD
            'EURUSD=X',  # EUR/USD
            'GBPUSD=X',  # GBP/USD
            'USDJPY=X'   # USD/JPY
        ]
        
        self.request_count = 0
        self.max_requests = 450
        
        # Session times (UTC)
        self.sessions = {
            'asia': {'open': 23, 'close': 8, 'name': 'Asian Session'},      # 11 PM UTC - 8 AM UTC
            'london': {'open': 7, 'close': 16, 'name': 'London Session'},    # 7 AM UTC - 4 PM UTC
            'newyork': {'open': 13, 'close': 22, 'name': 'New York Session'} # 1 PM UTC - 10 PM UTC
        }
        
        # Trading parameters
        self.check_interval = 30  # minutes
        self.session_active_hours = 3  # Check for 3 hours after session open
        self.timeframe = '15m'  # Using 15-minute timeframe
        
        # Initialize Firebase
        self.init_firebase()
    
    def init_firebase(self):
        """Initialize Firebase connection"""
        try:
            # Check if we have the service account in environment variable
            if 'FIREBASE_SERVICE_ACCOUNT' in os.environ:
                print("🔑 Using Firebase service account from environment variable")
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
                
            else:
                # Fallback to local file
                print("📁 Using local serviceAccountKey.json file")
                script_dir = os.path.dirname(os.path.abspath(__file__))
                service_account_path = os.path.join(script_dir, 'serviceAccountKey.json')
                
                if not os.path.exists(service_account_path):
                    print(f"⚠️ Service account file not found at: {service_account_path}")
                    print("Current directory:", os.getcwd())
                    print("Files in directory:", os.listdir(script_dir))
                    
                cred = credentials.Certificate(service_account_path)
                firebase_admin.initialize_app(cred)
            
            self.db = firestore.client()
            print("✅ Firebase initialized successfully")
            
        except Exception as e:
            print(f"❌ Error initializing Firebase: {e}")
            self.db = None
    
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
            
            # Handle overnight sessions
            if close_hour < open_hour:
                if current_hour >= open_hour or current_hour < close_hour:
                    return session_id
            else:
                if open_hour <= current_hour < close_hour:
                    return session_id
        
        return 'off_hours'
    
    def is_trading_time(self):
        """Check if current time is within trading window (3 hours after session open)"""
        now_utc = datetime.utcnow()
        current_hour = now_utc.hour
        current_minute = now_utc.minute
        
        for session_id, session in self.sessions.items():
            open_hour = session['open']
            
            # Calculate the time window (3 hours after session open)
            window_start = open_hour
            window_end = (open_hour + self.session_active_hours) % 24
            
            # Handle day wrap
            if window_end < window_start:
                if current_hour >= window_start or current_hour < window_end:
                    # Check if we're within the 30-minute check interval
                    if current_minute % self.check_interval < 15:  # Check at :00 and :30
                        return True, session_id
            else:
                if window_start <= current_hour < window_end:
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
    
    def calculate_sma(self, prices, period):
        """Calculate Simple Moving Average"""
        if len(prices) < period:
            return None
        return sum(prices[-period:]) / period
    
    def calculate_fib_levels(self, prices, period=50, fib_level=0.618):
        """Calculate Fibonacci levels"""
        if len(prices) < period:
            return None
            
        recent_prices = prices[-period:]
        highest = max(recent_prices)
        lowest = min(recent_prices)
        fib_618 = lowest + (highest - lowest) * fib_level
        
        return {
            'highest': highest,
            'lowest': lowest,
            'fib_618': fib_618,
            'range': highest - lowest
        }
    
    def check_signals(self, symbol, prices):
        """Check for trading signals on 15min timeframe"""
        if len(prices) < 200:  # Need enough data for SMAs (200 periods on 15min = ~2 days)
            return None
        
        # Calculate indicators
        sma50 = self.calculate_sma(prices, 50)    # 50 periods on 15min = ~12.5 hours
        sma200 = self.calculate_sma(prices, 200)  # 200 periods on 15min = ~2 days
        fib_data = self.calculate_fib_levels(prices, 100)  # 100 periods on 15min = 1 day for fib levels
        
        if not all([sma50, sma200, fib_data]):
            return None
            
        current_price = prices[-1]
        prev_price = prices[-2]
        prev_5_prices = prices[-6:-1]  # Last 5 candles before current
        
        # Determine trend
        bullish_trend = sma50 > sma200
        bearish_trend = sma50 < sma200
        
        # Check for strong trend (SMA slope)
        sma50_slope = sma50 - self.calculate_sma(prices, 50) if len(prices) > 51 else 0
        
        signals = []
        
        # Breakout conditions
        fib_breakout = prev_price <= fib_data['fib_618'] and current_price > fib_data['fib_618']
        
        # Pullback conditions (price touched fib level and bounced)
        fib_touched = any(abs(p - fib_data['fib_618']) / fib_data['fib_618'] < 0.001 for p in prev_5_prices)
        fib_pullback = fib_touched and abs(current_price - fib_data['fib_618']) / fib_data['fib_618'] > 0.002
        
        # Volume-like confirmation using price action
        strong_move = abs(current_price - prev_price) / prev_price > 0.001  # 0.1% move
        
        # Generate signals based on trend
        if bullish_trend and strong_move:
            if fib_breakout:
                signals.append({
                    'type': 'BUY',
                    'signal': '15m_breakout',
                    'price': current_price,
                    'fib_level': fib_data['fib_618'],
                    'sma50': sma50,
                    'sma200': sma200,
                    'confidence': 'HIGH' if sma50_slope > 0 else 'MEDIUM'
                })
            elif fib_pullback:
                signals.append({
                    'type': 'BUY',
                    'signal': '15m_pullback',
                    'price': current_price,
                    'fib_level': fib_data['fib_618'],
                    'sma50': sma50,
                    'sma200': sma200,
                    'confidence': 'MEDIUM'
                })
        
        elif bearish_trend and strong_move:
            if fib_breakout:
                signals.append({
                    'type': 'SELL',
                    'signal': '15m_breakout',
                    'price': current_price,
                    'fib_level': fib_data['fib_618'],
                    'sma50': sma50,
                    'sma200': sma200,
                    'confidence': 'HIGH' if sma50_slope < 0 else 'MEDIUM'
                })
            elif fib_pullback:
                signals.append({
                    'type': 'SELL',
                    'signal': '15m_pullback',
                    'price': current_price,
                    'fib_level': fib_data['fib_618'],
                    'sma50': sma50,
                    'sma200': sma200,
                    'confidence': 'MEDIUM'
                })
        
        return signals
    
    def scan_watchlist(self):
        """Main scanning function - runs on schedule"""
        print(f"\n{'='*60}")
        print(f"🚀 Scan started at {datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC")
        print(f"{'='*60}")
        
        # Check if it's trading time
        is_trading_time, current_session = self.is_trading_time()
        
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
            
            # Get 15min historical data
            hist_data = self.get_historical_data(symbol, interval='15m', range='2d')
            
            if hist_data and 'chart' in hist_data:
                try:
                    # Extract closing prices
                    prices = []
                    if 'result' in hist_data['chart'] and len(hist_data['chart']['result']) > 0:
                        result = hist_data['chart']['result'][0]
                        if 'indicators' in result and 'quote' in result['indicators']:
                            quotes = result['indicators']['quote'][0]
                            if 'close' in quotes:
                                prices = [p for p in quotes['close'] if p is not None]
                    
                    if len(prices) >= 200:
                        signals = self.check_signals(symbol, prices)
                        
                        if signals:
                            for signal in signals:
                                # Add session context
                                signal['timeframe'] = '15m'
                                signal['session'] = current_session
                                
                                # Save to Firebase
                                self.save_alert_to_firebase({
                                    'symbol': symbol,
                                    **signal
                                })
                                signals_found += 1
                                
                                # Print to console
                                print(f"\n🎯 SIGNAL FOUND: {symbol}")
                                print(f"   Type: {signal['type']} - {signal['signal']}")
                                print(f"   Price: ${signal['price']:.4f}")
                                print(f"   Confidence: {signal['confidence']}")
                        else:
                            print(f"ℹ️ No signals for {symbol}")
                    else:
                        print(f"⚠️ Insufficient data for {symbol} ({len(prices)} candles)")
                        
                except Exception as e:
                    print(f"❌ Error processing {symbol}: {e}")
            
            # Rate limiting
            time.sleep(1.5)
        
        print(f"\n{'='*60}")
        print(f"✅ Scan completed! Found {signals_found} signals")
        print(f"📊 API Requests used: {self.request_count}/{self.max_requests}")
        print(f"🕒 Next check at {(datetime.utcnow() + timedelta(minutes=30)).strftime('%H:%M')} UTC")
        print(f"{'='*60}")
    
    def run_continuously(self):
        """Run the bot continuously with 30-minute intervals"""
        print("🤖 Starting Fib SMA Trading Bot - Continuous Mode")
        print(f"📈 Timeframe: {self.timeframe}")
        print(f"⏰ Checking every {self.check_interval} minutes during active sessions")
        print(f"🎯 Active trading windows: 3 hours after each major session open")
        print("\nSession Times (UTC):")
        for session_id, session in self.sessions.items():
            print(f"   {session['name']}: {session['open']:02d}:00 - {session['close']:02d}:00 UTC")
        
        print("\n🚀 Bot is now running... Press Ctrl+C to stop\n")
        
        try:
            while True:
                # Run scan
                self.scan_watchlist()
                
                # Calculate next check time
                next_check = datetime.utcnow() + timedelta(minutes=self.check_interval)
                print(f"\n💤 Sleeping until {next_check.strftime('%H:%M')} UTC...\n")
                
                # Sleep for 30 minutes
                time.sleep(self.check_interval * 60)
                
        except KeyboardInterrupt:
            print("\n\n👋 Bot stopped by user")
            print(f"📊 Total API requests used: {self.request_count}")

# Main execution
if __name__ == "__main__":
    print("🚀 Initializing Fib SMA Trading Bot...")
    print(f"Python version: {sys.version}")
    print(f"Current UTC time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}")
    
    # Get API key from environment variable or use default
    API_KEY = os.environ.get('RAPIDAPI_KEY', '93e69d4612mshb9882c64196bebap186e1ajsn4e3c4ca3b804')
    
    # Create bot instance
    bot = FibSMATradingBot(API_KEY)
    
    # Check if running on Render (cron) or locally
    if os.environ.get('RENDER'):  # Running on Render
        print("🎯 Running on Render - Single scan mode")
        bot.scan_watchlist()
    else:  # Running locally
        print("💻 Running locally - Continuous mode")
        bot.run_continuously()
        