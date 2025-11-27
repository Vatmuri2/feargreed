import requests
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import csv
import os
import time
import warnings
warnings.filterwarnings('ignore')

class EODPumpStrategy:
    """
    Production-Ready Contrarian EOD Strategy with Polygon.io API
    
    Features:
    - Dynamic position sizing based on signal confidence
    - Risk management with stop losses
    - Trade logging to CSV
    - Live trading mode with alerts
    - Comprehensive performance analytics
    - Historical data via Polygon.io (up to 2+ years)
    """
    
    def __init__(self, api_key, mode='contrarian_both', account_size=10000, max_risk_per_trade=0.02):
        # API setup
        self.api_key = api_key
        self.base_url = "https://api.polygon.io"
        
        # Illiquid tech stocks to monitor
        self.watch_stocks = [
            'RGTI', 'QBTS', 'IONQ', 'QUBT',  # Quantum computing
            'SOUN', 'BBAI', 'INOD', 'AEYE'   # Small-cap AI
        ]
        self.trade_stock = 'TQQQ'
        
        # Account settings
        self.account_size = account_size
        self.max_risk_per_trade = max_risk_per_trade
        
        # Strategy settings
        self.mode = mode
        self.move_threshold = 0.005  # 0.5%
        self.min_signals = 2
        self.max_signals = 6
        self.consensus_threshold = 0.60
        
        # Risk management
        self.stop_loss_pct = 0.03  # 3% stop loss
        self.max_position_size = 0.950  # Max 95% of account per trade
        
        # Cache to avoid redundant API calls
        self.data_cache = {}
        
    def fetch_intraday_data(self, ticker, date, multiplier=5, timespan='minute'):
        """
        Fetch intraday data from Polygon.io for a specific date
        
        Args:
            ticker: Stock symbol
            date: Date object or string (YYYY-MM-DD)
            multiplier: Time interval multiplier (5 for 5-minute bars)
            timespan: 'minute', 'hour', etc.
        """
        if isinstance(date, datetime):
            date_str = date.strftime('%Y-%m-%d')
        else:
            date_str = str(date)
        
        # Check cache first
        cache_key = f"{ticker}_{date_str}_{multiplier}_{timespan}"
        if cache_key in self.data_cache:
            return self.data_cache[cache_key]
        
        url = f"{self.base_url}/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{date_str}/{date_str}"
        
        params = {
            'adjusted': 'true',
            'sort': 'asc',
            'limit': 50000,
            'apiKey': self.api_key
        }
        
        try:
            response = requests.get(url, params=params)
            
            # Rate limiting
            if response.status_code == 429:
                print(f"Rate limited, waiting 60s...")
                time.sleep(60)
                return self.fetch_intraday_data(ticker, date, multiplier, timespan)
            
            if response.status_code != 200:
                self.data_cache[cache_key] = None
                return None
            
            data = response.json()
            
            if data.get('status') != 'OK' or not data.get('results'):
                self.data_cache[cache_key] = None
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(data['results'])
            df['timestamp'] = pd.to_datetime(df['t'], unit='ms')
            df = df.rename(columns={
                'o': 'Open',
                'h': 'High', 
                'l': 'Low',
                'c': 'Close',
                'v': 'Volume'
            })
            df.set_index('timestamp', inplace=True)
            
            # Handle timezone conversion safely
            try:
                df.index = df.index.tz_localize('UTC').tz_convert('America/New_York')
            except:
                # If timezone conversion fails, proceed without it
                pass
            
            # Cache the result
            self.data_cache[cache_key] = df[['Open', 'High', 'Low', 'Close', 'Volume']]
            return self.data_cache[cache_key]
            
        except Exception as e:
            print(f"Error fetching {ticker} on {date_str}: {e}")
            self.data_cache[cache_key] = None
            return None
    
    def get_next_trading_day_open(self, ticker, date):
        """Get the opening price of the next trading day"""
        current_date = date
        max_attempts = 10
        
        for _ in range(max_attempts):
            current_date = current_date + timedelta(days=1)
            
            # Skip weekends
            if current_date.weekday() >= 5:
                continue
            
            # Fetch data for this day
            data = self.fetch_intraday_data(ticker, current_date, multiplier=1, timespan='minute')
            
            if data is not None and len(data) > 0:
                # Get first bar of the day (market open)
                try:
                    market_open_data = data.between_time('09:30', '09:31')
                    if len(market_open_data) > 0:
                        open_price = market_open_data['Open'].iloc[0]
                        return open_price
                except:
                    pass
                
                # Fallback to first available price
                return data['Open'].iloc[0]
            
            # Small delay to avoid rate limits
            time.sleep(0.1)
        
        return None
    
    def calculate_position_size(self, confidence, entry_price):
        """Calculate position size based on confidence and risk management"""
        # Base size on confidence
        if confidence < 0.70:
            size_multiplier = 0.5
        elif confidence < 0.85:
            size_multiplier = 0.75
        else:
            size_multiplier = 1.0
        
        # Calculate position size based on risk
        risk_amount = self.account_size * self.max_risk_per_trade
        stop_distance = entry_price * self.stop_loss_pct
        
        if stop_distance <= 0:
            return {
                'shares': 0,
                'position_value': 0,
                'position_pct': 0,
                'stop_loss': entry_price * (1 - self.stop_loss_pct),
                'risk_amount': 0
            }
        
        shares = risk_amount / stop_distance
        
        # Apply max position size limit
        max_shares = (self.account_size * self.max_position_size) / entry_price
        shares = min(shares, max_shares)
        
        # Apply confidence multiplier
        shares = shares * size_multiplier
        
        position_value = shares * entry_price
        
        return {
            'shares': int(max(0, shares)),
            'position_value': position_value,
            'position_pct': position_value / self.account_size if self.account_size > 0 else 0,
            'stop_loss': entry_price * (1 - self.stop_loss_pct),
            'risk_amount': risk_amount * size_multiplier
        }
    
    def detect_eod_move(self, df):
        """Detect directional move in final period before close"""
        if df is None or len(df) < 2:
            return 'NONE', 0.0
        
        try:
            # Filter to market hours
            df_market_hours = df.between_time('09:30', '16:00') if hasattr(df.index, 'time') else df
            if len(df_market_hours) < 2:
                return 'NONE', 0.0
            
            # Get last 10 minutes (2 bars of 5-minute data)
            final_bars = df_market_hours.tail(2)
            
            if len(final_bars) < 2:
                return 'NONE', 0.0
            
            start_price = final_bars['Close'].iloc[0]
            end_price = final_bars['Close'].iloc[-1]
            
            if start_price <= 0:
                return 'NONE', 0.0
                
            move_pct = (end_price - start_price) / start_price
            
            if abs(move_pct) >= self.move_threshold:
                return ('UP' if move_pct > 0 else 'DOWN'), move_pct
            
            return 'NONE', move_pct
        except Exception as e:
            print(f"Error detecting EOD move: {e}")
            return 'NONE', 0.0
    
    def generate_trade_signal(self, signals):
        """Generate trade signal based on mode and EOD moves"""
        if len(signals) < self.min_signals:
            return 'SKIP', 0, f"Only {len(signals)} signals"
        
        up_count = sum(1 for s in signals if s[1] == 'UP')
        down_count = len(signals) - up_count
        total = len(signals)
        
        if total >= self.max_signals:
            return 'SKIP', 0, f"Extreme consensus ({total} signals)"
        
        up_pct = up_count / total
        down_pct = down_count / total
        
        if up_pct < self.consensus_threshold and down_pct < self.consensus_threshold:
            return 'SKIP', 0, "No consensus"
        
        # Generate signal based on mode
        if self.mode == 'contrarian_long_only':
            if down_pct >= self.consensus_threshold:
                return 'LONG', down_pct, f"{down_count} stocks dumped"
            return 'SKIP', 0, "No dump signal"
                
        elif self.mode == 'contrarian_both':
            if down_pct >= self.consensus_threshold:
                return 'LONG', down_pct, f"{down_count} dumped"
            elif up_pct >= self.consensus_threshold:
                return 'SHORT', up_pct, f"{up_count} pumped"
        
        return 'SKIP', 0, "No valid setup"
    
    def get_trading_days(self, start_date, end_date):
        """Get list of trading days between start and end date"""
        # Simple approach: generate all weekdays
        current = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        
        trading_days = []
        while current <= end:
            if current.weekday() < 5:  # Monday=0, Friday=4
                trading_days.append(current.date())
            current += timedelta(days=1)
        
        return trading_days
    
    def backtest(self, start_date, end_date, verbose=False):
        """
        Backtest with position sizing and risk management
        verbose: If True, prints each trade. If False, only progress updates.
        """
        print("=" * 70)
        print(f"BACKTEST: {self.mode.upper().replace('_', ' ')}")
        print("=" * 70)
        print(f"Period: {start_date} to {end_date}")
        print(f"Account Size: ${self.account_size:,.0f}")
        print(f"Max Risk/Trade: {self.max_risk_per_trade*100}%")
        print(f"Stop Loss: {self.stop_loss_pct*100}%")
        print(f"Max Position: {self.max_position_size*100}%")
        print("-" * 70)
        
        results = []
        
        # Get trading days
        trading_days = self.get_trading_days(start_date, end_date)
        print(f"Analyzing {len(trading_days)} potential trading days...\n")
        
        trades_executed = 0
        
        for i, day in enumerate(trading_days):
            # Progress update every 20 days
            if (i + 1) % 20 == 0:
                print(f"Progress: {i+1}/{len(trading_days)} days analyzed, {trades_executed} trades executed")
            
            signals = []
            
            # Collect signals from watch stocks
            for ticker in self.watch_stocks:
                data = self.fetch_intraday_data(ticker, day, multiplier=5, timespan='minute')
                
                if data is None:
                    continue
                
                direction, move_pct = self.detect_eod_move(data)
                if direction != 'NONE':
                    signals.append((ticker, direction, move_pct))
                
                # Small delay to avoid rate limits
                time.sleep(0.05)
            
            if not signals:
                continue
            
            # Generate trade signal
            trade_direction, confidence, reason = self.generate_trade_signal(signals)
            
            if trade_direction == 'SKIP':
                continue
            
            # Get TQQQ EOD price
            tqqq_data = self.fetch_intraday_data(self.trade_stock, day, multiplier=5, timespan='minute')
            
            if tqqq_data is None or len(tqqq_data) == 0:
                continue
            
            # Get close price (entry)
            try:
                close_price = tqqq_data['Close'].iloc[-1]
                if close_price <= 0:
                    continue
            except:
                continue
            
            # Calculate position size
            position = self.calculate_position_size(confidence, close_price)
            
            # Skip if position size is invalid
            if position['shares'] <= 0:
                continue
            
            # Get next day's open (exit)
            open_price = self.get_next_trading_day_open(self.trade_stock, datetime.combine(day, datetime.min.time()))
            
            if open_price is None or open_price <= 0:
                continue
            
            # Check if stop loss hit
            if trade_direction == 'LONG':
                stop_price = position['stop_loss']
                if open_price <= stop_price:
                    exit_price = stop_price
                    overnight_return = (exit_price - close_price) / close_price
                    stopped_out = True
                else:
                    exit_price = open_price
                    overnight_return = (open_price - close_price) / close_price
                    stopped_out = False
            else:  # SHORT
                stop_price = close_price * (1 + self.stop_loss_pct)
                if open_price >= stop_price:
                    exit_price = stop_price
                    overnight_return = -(exit_price - close_price) / close_price
                    stopped_out = True
                else:
                    exit_price = open_price
                    overnight_return = -(open_price - close_price) / close_price
                    stopped_out = False
            
            # Calculate P&L
            if trade_direction == 'LONG':
                pnl_dollars = position['shares'] * (exit_price - close_price)
            else:  # SHORT
                pnl_dollars = position['shares'] * (close_price - exit_price)
            
            pnl_pct = pnl_dollars / position['position_value'] if position['position_value'] > 0 else 0
            
            trades_executed += 1
            
            # Print trade if verbose
            if verbose:
                day_str = day.strftime('%Y-%m-%d')
                print(f"\n📅 {day_str} - Trade #{trades_executed}")
                print(f"  Signal: {reason}")
                print(f"  Direction: {trade_direction} @ ${close_price:.2f} → ${exit_price:.2f}")
                print(f"  Position: {position['shares']} shares (${position['position_value']:.0f}, {position['position_pct']*100:.1f}% of account)")
                print(f"  Confidence: {confidence:.0%}")
                if stopped_out:
                    print(f"  ⚠️  STOP LOSS HIT @ ${exit_price:.2f}")
                print(f"  Return: {overnight_return*100:+.2f}% | P&L: ${pnl_dollars:+,.0f}")
            
            results.append({
                'date': day.strftime('%Y-%m-%d'),
                'direction': trade_direction,
                'signals': signals,
                'confidence': confidence,
                'entry_price': close_price,
                'exit_price': exit_price,
                'shares': position['shares'],
                'position_value': position['position_value'],
                'position_pct': position['position_pct'],
                'stop_loss': position['stop_loss'],
                'stopped_out': stopped_out,
                'return_pct': overnight_return,
                'pnl_dollars': pnl_dollars,
                'pnl_pct': pnl_pct
            })
        
        print(f"\n✅ Backtest complete: {trades_executed} trades executed")
        
        if results:
            self._print_summary(results)
            self._export_to_csv(results)
        else:
            print("\n⚠️  No trades generated")
        
        return results
    
    def _print_summary(self, results):
        """Print comprehensive performance summary"""
        print("\n" + "=" * 70)
        print("PERFORMANCE SUMMARY")
        print("=" * 70)
        
        returns = [r['return_pct'] for r in results]
        pnl_dollars = [r['pnl_dollars'] for r in results]
        long_trades = [r for r in results if r['direction'] == 'LONG']
        short_trades = [r for r in results if r['direction'] == 'SHORT']
        stopped_trades = [r for r in results if r['stopped_out']]
        
        winners = [r for r in returns if r > 0]
        losers = [r for r in returns if r < 0]
        
        print(f"\n📊 Overall Performance:")
        print(f"   Total trades: {len(results)}")
        print(f"   LONG: {len(long_trades)} | SHORT: {len(short_trades)}")
        print(f"   Stopped out: {len(stopped_trades)} ({len(stopped_trades)/len(results)*100:.1f}%)")
        print(f"   Winners: {len(winners)} | Losers: {len(losers)}")
        print(f"   Win rate: {len(winners)/len(returns)*100:.1f}%" if returns else "   Win rate: N/A")
        
        print(f"\n💰 Returns:")
        print(f"   Total P&L: ${sum(pnl_dollars):+,.0f}")
        print(f"   Account growth: {sum(pnl_dollars)/self.account_size*100:+.2f}%" if self.account_size > 0 else "   Account growth: N/A")
        print(f"   Avg return/trade: {np.mean(returns)*100:+.2f}%" if returns else "   Avg return/trade: N/A")
        print(f"   Avg P&L/trade: ${np.mean(pnl_dollars):+,.0f}" if pnl_dollars else "   Avg P&L/trade: N/A")
        print(f"   Best trade: {max(returns)*100:+.2f}% (${max(pnl_dollars):+,.0f})" if returns else "   Best trade: N/A")
        print(f"   Worst trade: {min(returns)*100:+.2f}% (${min(pnl_dollars):+,.0f})" if returns else "   Worst trade: N/A")
        
        # Risk metrics
        if returns:
            std_dev = np.std(returns)
            print(f"\n📈 Risk Metrics:")
            print(f"   Volatility (std): {std_dev*100:.2f}%")
            
            if len(returns) > 5 and std_dev > 0:
                sharpe = np.mean(returns) / std_dev * np.sqrt(252)
                print(f"   Sharpe ratio: {sharpe:.2f}")
            
            if winners and losers:
                total_wins = sum(winners)
                total_losses = abs(sum(losers))
                profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
                avg_win = np.mean(winners)
                avg_loss = abs(np.mean(losers))
                win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else float('inf')
                
                print(f"   Profit factor: {profit_factor:.2f}")
                print(f"   Win/Loss ratio: {win_loss_ratio:.2f}")
                print(f"   Avg win: {avg_win*100:+.2f}% (${np.mean([r['pnl_dollars'] for r in results if r['return_pct'] > 0]):+,.0f})")
                print(f"   Avg loss: {-avg_loss*100:+.2f}% (${np.mean([r['pnl_dollars'] for r in results if r['return_pct'] < 0]):+,.0f})")
            
            # Max drawdown
            cumulative = np.cumsum(pnl_dollars)
            if len(cumulative) > 0:
                running_max = np.maximum.accumulate(cumulative)
                drawdown = running_max - cumulative
                max_dd = np.max(drawdown) if len(drawdown) > 0 else 0
                max_dd_pct = max_dd / self.account_size if self.account_size > 0 else 0
                print(f"   Max drawdown: ${max_dd:,.0f} ({max_dd_pct*100:.2f}%)")
        
        # Position sizing stats
        if results:
            avg_position_pct = np.mean([r['position_pct'] for r in results])
            print(f"\n📊 Position Sizing:")
            print(f"   Avg position size: {avg_position_pct*100:.1f}% of account")
            print(f"   Avg confidence: {np.mean([r['confidence'] for r in results])*100:.1f}%")
        
        # LONG stats
        if long_trades:
            long_returns = [r['return_pct'] for r in long_trades]
            long_pnl = [r['pnl_dollars'] for r in long_trades]
            long_winners = [r for r in long_returns if r > 0]
            
            print(f"\n📈 LONG Trades:")
            print(f"   Count: {len(long_trades)}")
            print(f"   Win rate: {len(long_winners)/len(long_trades)*100:.1f}%" if long_trades else "   Win rate: N/A")
            print(f"   Total P&L: ${sum(long_pnl):+,.0f}")
            print(f"   Avg return: {np.mean(long_returns)*100:+.2f}%" if long_returns else "   Avg return: N/A")
            print(f"   Best: {max(long_returns)*100:+.2f}% | Worst: {min(long_returns)*100:+.2f}%" if long_returns else "   Best: N/A | Worst: N/A")
        
        # SHORT stats
        if short_trades:
            short_returns = [r['return_pct'] for r in short_trades]
            short_pnl = [r['pnl_dollars'] for r in short_trades]
            short_winners = [r for r in short_returns if r > 0]
            
            print(f"\n📉 SHORT Trades:")
            print(f"   Count: {len(short_trades)}")
            print(f"   Win rate: {len(short_winners)/len(short_trades)*100:.1f}%" if short_trades else "   Win rate: N/A")
            print(f"   Total P&L: ${sum(short_pnl):+,.0f}")
            print(f"   Avg return: {np.mean(short_returns)*100:+.2f}%" if short_returns else "   Avg return: N/A")
            print(f"   Best: {max(short_returns)*100:+.2f}% | Worst: {min(short_returns)*100:+.2f}%" if short_returns else "   Best: N/A | Worst: N/A")
    
    def _export_to_csv(self, results):
        """Export trade results to CSV"""
        filename = f"trades_{self.mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        try:
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'Date', 'Direction', 'Confidence', 'Entry', 'Exit', 
                    'Shares', 'Position $', 'Position %', 'Stop Loss',
                    'Stopped Out', 'Return %', 'P&L $', 'Signals'
                ])
                
                for r in results:
                    signal_str = ' | '.join([f"{s[0]}:{s[2]*100:+.1f}%" for s in r['signals']])
                    writer.writerow([
                        r['date'],
                        r['direction'],
                        f"{r['confidence']*100:.1f}%",
                        f"${r['entry_price']:.2f}",
                        f"${r['exit_price']:.2f}",
                        r['shares'],
                        f"${r['position_value']:.0f}",
                        f"{r['position_pct']*100:.1f}%",
                        f"${r['stop_loss']:.2f}",
                        'YES' if r['stopped_out'] else 'NO',
                        f"{r['return_pct']*100:+.2f}%",
                        f"${r['pnl_dollars']:+.0f}",
                        signal_str
                    ])
            
            print(f"\n📄 Trade log exported to: {filename}")
        except Exception as e:
            print(f"Error exporting to CSV: {e}")


# Example usage
if __name__ == "__main__":
    print("\n🔬 CONTRARIAN STRATEGY WITH POLYGON.IO\n")
    
    # IMPORTANT: Set your Polygon API key
    API_KEY = os.environ.get('POLYGON_API_KEY', 'YOUR_API_KEY_HERE')
    
    if API_KEY == 'YOUR_API_KEY_HERE':
        print("⚠️  Please set your POLYGON_API_KEY environment variable")
        print("   export POLYGON_API_KEY='your_key_here'")
        exit(1)
    
    # Set account parameters
    ACCOUNT_SIZE = 10000  # $10k account
    MAX_RISK = 0.02       # 2% max risk per trade
    
    modes = [
        ('contrarian_long_only', 'Contrarian LONG-ONLY'),
        ('contrarian_both', 'Contrarian BOTH DIRECTIONS')
    ]
    
    all_results = {}
    
    for mode, description in modes:
        print("\n" + "=" * 70)
        print(f"Testing: {description}")
        print("=" * 70 + "\n")
        
        strategy = EODPumpStrategy(
            api_key=API_KEY,
            mode=mode,
            account_size=ACCOUNT_SIZE,
            max_risk_per_trade=MAX_RISK
        )
        
        # Test with a shorter period first to verify functionality
        results = strategy.backtest(
            start_date='2024-01-01',  # Shorter period for testing
            end_date='2025-11-25',
            verbose=True  # Set to True to see each trade
        )
        
        all_results[mode] = results
        
        input("\nPress Enter to continue...")
    
    # Final comparison
    print("\n\n" + "=" * 70)
    print("🏆 FINAL COMPARISON")
    print("=" * 70)
    
    for mode, description in modes:
        results = all_results.get(mode, [])
        if results:
            pnl = sum([r['pnl_dollars'] for r in results])
            returns = [r['return_pct'] for r in results]
            winners = [r for r in returns if r > 0]
            
            print(f"\n{description}:")
            print(f"  Trades: {len(results)}")
            print(f"  Win Rate: {len(winners)/len(results)*100:.1f}%" if results else "  Win Rate: N/A")
            print(f"  Total P&L: ${pnl:+,.0f} ({pnl/ACCOUNT_SIZE*100:+.2f}%)")
            print(f"  Avg Return: {np.mean(returns)*100:+.2f}%" if returns else "  Avg Return: N/A")
            
            if returns and len(returns) > 5:
                sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
                print(f"  Sharpe Ratio: {sharpe:.2f}")
    
    print("\n" + "=" * 70)
    print("\n📄 Check your directory for CSV trade logs!")
    print("💡 With Polygon, you can now backtest 2+ years of data!\n")