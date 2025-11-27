import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import csv
import os

class EODPumpStrategy:
    """
    Production-Ready Contrarian EOD Strategy
    
    Features:
    - Dynamic position sizing based on signal confidence
    - Risk management with stop losses
    - Trade logging to CSV
    - Live trading mode with alerts
    - Comprehensive performance analytics
    """
    
    def __init__(self, mode='contrarian_both', account_size=10000, max_risk_per_trade=0.02):
        # Illiquid tech stocks to monitor
        self.watch_stocks = [
            'RGTI', 'QBTS', 'IONQ', 'QUBT',  # Quantum computing
            'SOUN', 'BBAI', 'INOD', 'AEYE'   # Small-cap AI
        ]
        self.trade_stock = 'TQQQ'
        
        # Account settings
        self.account_size = account_size
        self.max_risk_per_trade = max_risk_per_trade  # 2% max risk per trade
        
        # Strategy settings
        self.mode = mode
        self.move_threshold = 0.005  # 0.5%
        self.min_signals = 2
        self.max_signals = 6
        self.consensus_threshold = 0.60
        
        # Risk management
        self.stop_loss_pct = 0.03  # 3% stop loss
        self.max_position_size = 0.950  # Max 30% of account per trade
        
    def calculate_position_size(self, confidence, entry_price):
        """
        Calculate position size based on confidence and risk management
        
        Confidence-based sizing:
        - 60-70% confidence: 50% of max size
        - 70-85% confidence: 75% of max size
        - 85-100% confidence: 100% of max size
        """
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
        shares = risk_amount / stop_distance
        
        # Apply max position size limit
        max_shares = (self.account_size * self.max_position_size) / entry_price
        shares = min(shares, max_shares)
        
        # Apply confidence multiplier
        shares = shares * size_multiplier
        
        position_value = shares * entry_price
        
        return {
            'shares': int(shares),
            'position_value': position_value,
            'position_pct': position_value / self.account_size,
            'stop_loss': entry_price * (1 - self.stop_loss_pct),
            'risk_amount': risk_amount * size_multiplier
        }
    
    def fetch_intraday_data(self, ticker, period='5d', interval='5m'):
        """Fetch intraday data from yfinance"""
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(period=period, interval=interval)
            return data
        except Exception as e:
            return None
    
    def detect_eod_move(self, df, interval='5m'):
        """Detect directional move in final period before close"""
        if df is None or len(df) < 2:
            return 'NONE', 0.0
        
        df = df.between_time('09:30', '16:00')
        if len(df) < 2:
            return 'NONE', 0.0
        
        window = 6 if interval == '1m' else 2
        final_bars = df.tail(window)
        
        if len(final_bars) < 2:
            return 'NONE', 0.0
        
        start_price = final_bars['Close'].iloc[0]
        end_price = final_bars['Close'].iloc[-1]
        move_pct = (end_price - start_price) / start_price
        
        if abs(move_pct) >= self.move_threshold:
            return ('UP' if move_pct > 0 else 'DOWN'), move_pct
        
        return 'NONE', move_pct
    
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
    
    def backtest(self, start_date, end_date, interval='5m', verbose=False):
        """
        Backtest with position sizing and risk management
        verbose: If True, prints each trade. If False, only progress updates.
        """
        print("=" * 70)
        print(f"BACKTEST: {self.mode.upper().replace('_', ' ')}")
        print("=" * 70)
        print(f"Account Size: ${self.account_size:,.0f}")
        print(f"Max Risk/Trade: {self.max_risk_per_trade*100}%")
        print(f"Stop Loss: {self.stop_loss_pct*100}%")
        print(f"Max Position: {self.max_position_size*100}%")
        print("-" * 70)
        
        # Determine period
        period = '7d' if interval == '1m' else '60d'
        
        results = []
        
        # Fetch TQQQ data
        print("Fetching data...")
        tqqq_data = self.fetch_intraday_data(self.trade_stock, period=period, interval=interval)
        if tqqq_data is None:
            print("Failed to fetch TQQQ data")
            return []
        
        trading_days = sorted(set(tqqq_data.index.date))
        print(f"Analyzing {len(trading_days)} trading days...\n")
        
        trades_executed = 0
        
        for day in trading_days:
            signals = []
            
            # Collect signals
            for ticker in self.watch_stocks:
                data = self.fetch_intraday_data(ticker, period=period, interval=interval)
                if data is None:
                    continue
                
                day_data = data[data.index.date == day]
                if len(day_data) > 0:
                    direction, move_pct = self.detect_eod_move(day_data, interval)
                    if direction != 'NONE':
                        signals.append((ticker, direction, move_pct))
            
            if not signals:
                continue
            
            # Generate trade signal
            trade_direction, confidence, reason = self.generate_trade_signal(signals)
            
            if trade_direction == 'SKIP':
                continue
            
            # Get TQQQ prices
            tqqq_day_data = tqqq_data[tqqq_data.index.date == day]
            if len(tqqq_day_data) == 0:
                continue
            
            close_price = tqqq_day_data['Close'].iloc[-1]
            
            # Calculate position size
            position = self.calculate_position_size(confidence, close_price)
            
            # Find next day's open
            next_day = day + timedelta(days=1)
            next_day_data = tqqq_data[tqqq_data.index.date == next_day]
            
            if len(next_day_data) > 0:
                open_price = next_day_data['Open'].iloc[0]
                
                # Check if stop loss hit (simulate worst case: goes against us first)
                if trade_direction == 'LONG':
                    # Check if gap down hit stop
                    if open_price <= position['stop_loss']:
                        # Stop loss hit
                        exit_price = position['stop_loss']
                        overnight_return = (exit_price - close_price) / close_price
                        stopped_out = True
                    else:
                        exit_price = open_price
                        overnight_return = (open_price - close_price) / close_price
                        stopped_out = False
                else:  # SHORT
                    # For shorts, stop is above entry
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
                pnl_dollars = position['shares'] * (exit_price - close_price)
                if trade_direction == 'SHORT':
                    pnl_dollars = -pnl_dollars
                
                pnl_pct = pnl_dollars / position['position_value']
                
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
        print(f"   Win rate: {len(winners)/len(returns)*100:.1f}%")
        
        print(f"\n💰 Returns:")
        print(f"   Total P&L: ${sum(pnl_dollars):+,.0f}")
        print(f"   Account growth: {sum(pnl_dollars)/self.account_size*100:+.2f}%")
        print(f"   Avg return/trade: {np.mean(returns)*100:+.2f}%")
        print(f"   Avg P&L/trade: ${np.mean(pnl_dollars):+,.0f}")
        print(f"   Best trade: {max(returns)*100:+.2f}% (${max(pnl_dollars):+,.0f})")
        print(f"   Worst trade: {min(returns)*100:+.2f}% (${min(pnl_dollars):+,.0f})")
        
        # Risk metrics
        std_dev = np.std(returns)
        print(f"\n📈 Risk Metrics:")
        print(f"   Volatility (std): {std_dev*100:.2f}%")
        
        if len(returns) > 5:
            sharpe = np.mean(returns) / std_dev * np.sqrt(252)
            print(f"   Sharpe ratio: {sharpe:.2f}")
        
        if winners and losers:
            total_wins = sum(winners)
            total_losses = abs(sum(losers))
            profit_factor = total_wins / total_losses
            avg_win = np.mean(winners)
            avg_loss = abs(np.mean(losers))
            win_loss_ratio = avg_win / avg_loss
            
            print(f"   Profit factor: {profit_factor:.2f}")
            print(f"   Win/Loss ratio: {win_loss_ratio:.2f}")
            print(f"   Avg win: {avg_win*100:+.2f}% (${np.mean([r['pnl_dollars'] for r in results if r['return_pct'] > 0]):+,.0f})")
            print(f"   Avg loss: {-avg_loss*100:+.2f}% (${np.mean([r['pnl_dollars'] for r in results if r['return_pct'] < 0]):+,.0f})")
        
        # Max drawdown
        cumulative = np.cumsum(pnl_dollars)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = running_max - cumulative
        max_dd = np.max(drawdown) if len(drawdown) > 0 else 0
        max_dd_pct = max_dd / self.account_size
        print(f"   Max drawdown: ${max_dd:,.0f} ({max_dd_pct*100:.2f}%)")
        
        # Position sizing stats
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
            print(f"   Win rate: {len(long_winners)/len(long_trades)*100:.1f}%")
            print(f"   Total P&L: ${sum(long_pnl):+,.0f}")
            print(f"   Avg return: {np.mean(long_returns)*100:+.2f}%")
            print(f"   Best: {max(long_returns)*100:+.2f}% | Worst: {min(long_returns)*100:+.2f}%")
        
        # SHORT stats
        if short_trades:
            short_returns = [r['return_pct'] for r in short_trades]
            short_pnl = [r['pnl_dollars'] for r in short_trades]
            short_winners = [r for r in short_returns if r > 0]
            
            print(f"\n📉 SHORT Trades:")
            print(f"   Count: {len(short_trades)}")
            print(f"   Win rate: {len(short_winners)/len(short_trades)*100:.1f}%")
            print(f"   Total P&L: ${sum(short_pnl):+,.0f}")
            print(f"   Avg return: {np.mean(short_returns)*100:+.2f}%")
            print(f"   Best: {max(short_returns)*100:+.2f}% | Worst: {min(short_returns)*100:+.2f}%")
    
    def _export_to_csv(self, results):
        """Export trade results to CSV"""
        filename = f"trades_{self.mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
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
    
    def live_monitor(self, interval='1m'):
        """
        Live monitoring mode with trade alerts
        """
        print("\n" + "=" * 70)
        print("🔴 LIVE MONITORING MODE")
        print("=" * 70)
        print(f"Mode: {self.mode}")
        print(f"Watching: {', '.join(self.watch_stocks)}")
        print(f"Account: ${self.account_size:,.0f}")
        print(f"Checking every 60 seconds...")
        print("\nPress Ctrl+C to stop\n")
        
        last_alert_time = None
        
        while True:
            try:
                current_time = datetime.now()
                
                # Only monitor during final 30 minutes (3:30 PM - 4:00 PM ET)
                if current_time.hour == 15 and current_time.minute >= 30:
                    signals = []
                    
                    for ticker in self.watch_stocks:
                        data = self.fetch_intraday_data(ticker, period='1d', interval=interval)
                        direction, move_pct = self.detect_eod_move(data, interval)
                        
                        if direction != 'NONE':
                            signals.append((ticker, direction, move_pct))
                    
                    if signals:
                        trade_direction, confidence, reason = self.generate_trade_signal(signals)
                        
                        if trade_direction != 'SKIP':
                            # Avoid duplicate alerts within 5 minutes
                            if last_alert_time is None or (current_time - last_alert_time).seconds > 300:
                                # Get current TQQQ price
                                tqqq = yf.Ticker(self.trade_stock)
                                tqqq_price = tqqq.history(period='1d', interval='1m')['Close'].iloc[-1]
                                
                                # Calculate position
                                position = self.calculate_position_size(confidence, tqqq_price)
                                
                                print(f"\n{'='*70}")
                                print(f"🚨 TRADE ALERT - {current_time.strftime('%H:%M:%S')}")
                                print(f"{'='*70}")
                                print(f"Signal: {reason}")
                                print(f"Direction: {trade_direction} {self.trade_stock}")
                                print(f"Current Price: ${tqqq_price:.2f}")
                                print(f"Confidence: {confidence:.0%}")
                                print(f"\n📊 Recommended Position:")
                                print(f"   Shares: {position['shares']}")
                                print(f"   Position Value: ${position['position_value']:,.0f} ({position['position_pct']*100:.1f}% of account)")
                                print(f"   Stop Loss: ${position['stop_loss']:.2f}")
                                print(f"   Risk Amount: ${position['risk_amount']:,.0f}")
                                print(f"\n📈 Signals:")
                                for ticker, direction, move_pct in signals:
                                    arrow = "📈" if direction == 'UP' else "📉"
                                    print(f"   {arrow} {ticker}: {move_pct*100:+.2f}%")
                                print("=" * 70 + "\n")
                                
                                last_alert_time = current_time
                
                # Wait before next check
                import time
                time.sleep(60)
                
            except KeyboardInterrupt:
                print("\n\n🛑 Monitoring stopped")
                break
            except Exception as e:
                print(f"Error: {e}")
                import time
                time.sleep(60)


# Run comparison
if __name__ == "__main__":
    print("\n🔬 CONTRARIAN STRATEGY COMPARISON WITH FULL RISK MANAGEMENT\n")
    
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
            mode=mode,
            account_size=ACCOUNT_SIZE,
            max_risk_per_trade=MAX_RISK
        )
        
        results = strategy.backtest(
            start_date='2024-01-01',
            end_date='2024-12-31',
            interval='5m',
            verbose=False  # Set to True to see each trade
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
            print(f"  Win Rate: {len(winners)/len(results)*100:.1f}%")
            print(f"  Total P&L: ${pnl:+,.0f} ({pnl/ACCOUNT_SIZE*100:+.2f}%)")
            print(f"  Avg Return: {np.mean(returns)*100:+.2f}%")
            
            if len(returns) > 5:
                sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
                print(f"  Sharpe Ratio: {sharpe:.2f}")
    
    print("\n" + "=" * 70)
    print("\n💡 To run LIVE monitoring, uncomment the line below:")
    print("   # strategy.live_monitor(interval='1m')")
    print("\n📄 Check your directory for CSV trade logs!\n")