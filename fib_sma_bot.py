import requests
import json
import time
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

class FibSMATradingBot:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://yahoo-finance166.p.rapidapi.com"
        self.headers = {
            "x-rapidapi-key": api_key,
            "x-rapidapi-host": "yahoo-finance166.p.rapidapi.com"
        }
        self.watchlist = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']  # Add your symbols
        self.request_count = 0
        self.max_requests = 450  # Buffer to stay under 500
        
    def get_current_price(self, symbol):
        """Get current price for a symbol"""
        if self.request_count >= self.max_requests:
            print("Monthly request limit approaching")
            return None
            
        url = f"{self.base_url}/api/stock/get-price"
        querystring = {"region": "US", "symbol": symbol}
        
        try:
            response = requests.get(url, headers=self.headers, params=querystring)
            self.request_count += 1
            data = response.json()
            
            if 'price' in data:
                return {
                    'symbol': symbol,
                    'price': data['price'],
                    'timestamp': datetime.now().isoformat()
                }
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
        return None
    
    def get_historical_data(self, symbol, interval='1h', range='5d'):
        """Get historical data for indicator calculation"""
        if self.request_count >= self.max_requests:
            return None
            
        url = f"{self.base_url}/api/stock/get-chart"
        querystring = {
            "symbol": symbol,
            "interval": interval,  # 1h for H1 timeframe
            "range": range,
            "region": "US"
        }
        
        try:
            response = requests.get(url, headers=self.headers, params=querystring)
            self.request_count += 1
            return response.json()
        except Exception as e:
            print(f"Error fetching history for {symbol}: {e}")
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
            'fib_618': fib_618
        }
    
    def check_signals(self, symbol, prices):
        """Check for trading signals on H1 timeframe"""
        if len(prices) < 210:  # Need enough data for SMAs
            return None
            
        # Calculate indicators
        sma50 = self.calculate_sma(prices, 50)
        sma200 = self.calculate_sma(prices, 200)
        fib_data = self.calculate_fib_levels(prices, 50)
        
        if not all([sma50, sma200, fib_data]):
            return None
            
        current_price = prices[-1]
        prev_price = prices[-2]
        
        # Determine trend
        bullish_trend = sma50 > sma200
        bearish_trend = sma50 < sma200
        
        signals = []
        
        # Check breakout conditions
        fib_breakout = prev_price <= fib_data['fib_618'] and current_price > fib_data['fib_618']
        fib_pullback = prev_price >= fib_data['fib_618'] and current_price < fib_data['fib_618']
        
        # Generate signals based on trend
        if bullish_trend:
            if fib_breakout:
                signals.append({
                    'type': 'BUY',
                    'signal': 'H1_breakout',
                    'price': current_price,
                    'fib_level': fib_data['fib_618'],
                    'sma50': sma50,
                    'sma200': sma200
                })
            if fib_pullback:
                signals.append({
                    'type': 'BUY',
                    'signal': 'H1_pullback',
                    'price': current_price,
                    'fib_level': fib_data['fib_618'],
                    'sma50': sma50,
                    'sma200': sma200
                })
        elif bearish_trend:
            if fib_breakout:
                signals.append({
                    'type': 'SELL',
                    'signal': 'H1_breakout',
                    'price': current_price,
                    'fib_level': fib_data['fib_618'],
                    'sma50': sma50,
                    'sma200': sma200
                })
            if fib_pullback:
                signals.append({
                    'type': 'SELL',
                    'signal': 'H1_pullback',
                    'price': current_price,
                    'fib_level': fib_data['fib_618'],
                    'sma50': sma50,
                    'sma200': sma200
                })
        
        return signals
    
    def send_alert(self, symbol, signal):
        """Send/display alert"""
        alert = {
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            **signal
        }
        print(json.dumps(alert, indent=2))
        # Here you could also send to webhook, email, etc.
        return alert
    
    def scan_watchlist(self):
        """Main scanning function - runs on H1 timeframe"""
        print(f"\n=== Scan started at {datetime.now().strftime('%Y-%m-%d %H:%M')} ===")
        
        for symbol in self.watchlist:
            if self.request_count >= self.max_requests:
                print("Monthly limit reached. Stopping scan.")
                break
                
            print(f"\nChecking {symbol}...")
            
            # Get H1 historical data
            hist_data = self.get_historical_data(symbol, interval='1h', range='10d')
            
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
                    
                    if len(prices) >= 210:
                        signals = self.check_signals(symbol, prices)
                        
                        if signals:
                            for signal in signals:
                                self.send_alert(symbol, signal)
                        else:
                            print(f"No signals for {symbol}")
                    else:
                        print(f"Insufficient data for {symbol}")
                        
                except Exception as e:
                    print(f"Error processing {symbol}: {e}")
            
            # Respect rate limit
            time.sleep(2)  # 2 second delay between requests
        
        print(f"\nRequests used this month: {self.request_count}/{self.max_requests}")

# Main execution
if __name__ == "__main__":
    API_KEY = "93e69d4612mshb9882c64196bebap186e1ajsn4e3c4ca3b804"
    
    bot = FibSMATradingBot(API_KEY)
    
    # Add your watchlist
    # bot.watchlist = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA']
    
    # Updated watchlist with your requested symbols
    bot.watchlist = [
        'GBPJPY=X',  # GBP/JPY Forex
        'XAUUSD=X',  # Gold vs USD
        'USDCAD=X',  # USD/CAD Forex
        'BTC-USD'    # Bitcoin vs USD
    ]

    # Manual H1 check (run this every hour)
    # For automation, use cron job or scheduler
    bot.scan_watchlist()
    
    # To track usage
    print(f"\nRemaining requests: {bot.max_requests - bot.request_count}")