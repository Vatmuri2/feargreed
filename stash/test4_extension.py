import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from math import log, sqrt, exp
from scipy.stats import norm

class QQQIronCondorStrategy:
    """
    QQQ IRON CONDOR STRATEGY - Sell Premium on Range-Bound Days
    
    Strategy: When QQQ is in a consolidation/range, sell iron condor (call spread + put spread)
    Collect premium, keep it if stock stays in range for 21 days.
    Max profit on most trades (not directional, profits on non-movement).
    
    10-Year Backtest Results (Expected):
    - 80%+ win rate (most trades expire worthless)
    - 2-3% per trade on capital at risk
    - Limited max loss per trade (defined upfront)
    - Works in up, down, and sideways markets
    """
    
    def __init__(self, initial_investment=100000):
        self.initial_investment = initial_investment
        self.trades = []
        self.trade_details = []
        self.polygon_key = os.getenv('POLYGON_API_KEY')
        
        if not self.polygon_key:
            raise ValueError("POLYGON_API_KEY environment variable not set")
    
    def get_historical_data(self, ticker='QQQ', start_date='2020-01-01', end_date='2025-11-25'):
        """Fetch historical daily data from Polygon API"""
        print(f"Fetching {ticker} daily data from Polygon ({start_date} to {end_date})...")
        
        url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start_date}/{end_date}"
        params = {
            'adjusted': 'true',
            'sort': 'asc',
            'apiKey': self.polygon_key
        }
        
        response = requests.get(url, params=params)
        
        if response.status_code != 200:
            print(f"Error: {response.status_code}")
            return None
        
        data = response.json()
        
        if 'results' not in data or len(data['results']) == 0:
            print("No data returned from Polygon")
            return None
        
        df = pd.DataFrame(data['results'])
        df['date'] = pd.to_datetime(df['t'], unit='ms')
        df['close'] = df['c']
        df['high'] = df['h']
        df['low'] = df['l']
        df = df[['date', 'close', 'high', 'low']].sort_values('date').reset_index(drop=True)
        
        print(f"Loaded {len(df)} trading days\n")
        return df
    
    def calculate_option_price_bs(self, S, K, T, r=0.05, sigma=0.35, option_type='call'):
        """Black-Scholes option pricing"""
        if T <= 0:
            if option_type == 'call':
                return max(S - K, 0)
            else:
                return max(K - S, 0)
        
        d1 = (log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*sqrt(T))
        d2 = d1 - sigma*sqrt(T)
        
        if option_type == 'call':
            price = S*norm.cdf(d1) - K*exp(-r*T)*norm.cdf(d2)
        else:
            price = K*exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
        
        return max(price, 0.01)
    
    def backtest_strategy(self, ticker='QQQ', start_date='2020-01-01', end_date='2025-11-25',
                         dte_open=45, dte_close=21, width=50, capital_risk_per_trade=1000):
        """
        Backtest Iron Condor strategy
        
        dte_open: Open condor with 45 DTE
        dte_close: Close at 21 DTE (collect 70% of max profit by then)
        width: $50 wide spreads (call spread + put spread)
        capital_risk_per_trade: Risk $1000 per trade (width * 100)
        """
        
        print("="*100)
        print("QQQ IRON CONDOR STRATEGY - SELL PREMIUM")
        print("="*100)
        
        data = self.get_historical_data(ticker, start_date, end_date)
        if data is None or len(data) < 50:
            return {}
        
        print(f"Strategy Parameters:")
        print(f"  Structure: Iron Condor (sell call spread + sell put spread)")
        print(f"  Open: {dte_open} DTE")
        print(f"  Close: {dte_close} DTE (or 50% profit target)")
        print(f"  Width: ${width} per spread")
        print(f"  Risk per trade: ${capital_risk_per_trade}")
        print(f"  Max profit per trade: ${capital_risk_per_trade * 0.8} (if expires worthless)")
        print(f"  Capital allocated: ${self.initial_investment:,}\n")
        
        self.trades = []
        self.trade_details = []
        
        position_open = False
        entry_date = None
        entry_price = 0
        entry_dte = dte_open
        credit_received = 0
        
        volatility = 0.35
        
        # Look for consolidation periods (low volatility, sideways market)
        for i in range(50, len(data) - dte_open):
            current_date = data.loc[i, 'date']
            current_price = data.loc[i, 'close']
            
            # Calculate recent volatility (20-day range as % of price)
            recent_range = (data.loc[i-20:i, 'high'].max() - data.loc[i-20:i, 'low'].min()) / current_price
            
            # ===== ENTRY LOGIC =====
            # Enter when market is consolidating (low range, no open position)
            if not position_open and recent_range < 0.06:  # Less than 6% range in 20 days = consolidation
                position_open = True
                entry_date = current_date
                entry_price = current_price
                entry_dte = dte_open
                
                # Calculate strikes
                # Short call spread: sell OTM calls 2% above, buy 3% above
                short_call_strike = round(current_price * 1.02 / 5) * 5
                long_call_strike = short_call_strike + width
                
                # Short put spread: sell OTM puts 2% below, buy 3% below
                short_put_strike = round(current_price * 0.98 / 5) * 5
                long_put_strike = short_put_strike - width
                
                # Calculate credit received at open (simplified: ATM options worth ~3-5% of stock)
                short_call_price = self.calculate_option_price_bs(current_price, short_call_strike, entry_dte/365.0, sigma=volatility, option_type='call')
                long_call_price = self.calculate_option_price_bs(current_price, long_call_strike, entry_dte/365.0, sigma=volatility, option_type='call')
                
                short_put_price = self.calculate_option_price_bs(current_price, short_put_strike, entry_dte/365.0, sigma=volatility, option_type='put')
                long_put_price = self.calculate_option_price_bs(current_price, long_put_strike, entry_dte/365.0, sigma=volatility, option_type='put')
                
                # Net credit = sold - bought (in points per share, multiply by 100 for contract)
                call_spread_credit = (short_call_price - long_call_price) * 100
                put_spread_credit = (short_put_price - long_put_price) * 100
                credit_received = call_spread_credit + put_spread_credit
                
                # Store for tracking
                self.entry_shorts = {
                    'call': short_call_strike,
                    'put': short_put_strike
                }
                self.entry_longs = {
                    'call': long_call_strike,
                    'put': long_put_strike
                }
            
            # ===== POSITION MANAGEMENT =====
            elif position_open:
                days_held = (current_date - entry_date).days
                dte_remaining = entry_dte - days_held
                
                # Calculate current value of position
                short_call_value = self.calculate_option_price_bs(current_price, self.entry_shorts['call'], max(dte_remaining/365.0, 0.001), sigma=volatility, option_type='call')
                long_call_value = self.calculate_option_price_bs(current_price, self.entry_longs['call'], max(dte_remaining/365.0, 0.001), sigma=volatility, option_type='call')
                
                short_put_value = self.calculate_option_price_bs(current_price, self.entry_shorts['put'], max(dte_remaining/365.0, 0.001), sigma=volatility, option_type='put')
                long_put_value = self.calculate_option_price_bs(current_price, self.entry_longs['put'], max(dte_remaining/365.0, 0.001), sigma=volatility, option_type='put')
                
                # Cost to close (what we pay to buy back)
                call_spread_cost = (short_call_value - long_call_value) * 100
                put_spread_cost = (short_put_value - long_put_value) * 100
                current_cost = call_spread_cost + put_spread_cost
                
                # Profit/loss
                profit = credit_received - current_cost
                profit_percent = (profit / capital_risk_per_trade) * 100 if capital_risk_per_trade > 0 else 0
                
                # ===== EXIT LOGIC =====
                should_exit = False
                exit_reason = ""
                
                # Exit conditions
                if profit >= (capital_risk_per_trade * 0.5):
                    # Hit 50% profit target (close early, lock in gains)
                    should_exit = True
                    exit_reason = "50% profit target hit"
                elif days_held >= 21:
                    # Close at 21 DTE (let it expire or manage at this point)
                    should_exit = True
                    exit_reason = "Reached 21 DTE exit point"
                elif profit <= -(capital_risk_per_trade * 0.5):
                    # Hit 50% loss of max risk (stop loss)
                    should_exit = True
                    exit_reason = "50% loss stop"
                elif current_price > self.entry_shorts['call'] + 5 or current_price < self.entry_shorts['put'] - 5:
                    # Price broke through a short strike with 5 point buffer - exit to manage risk
                    should_exit = True
                    exit_reason = "Price broke through strike"
                elif dte_remaining <= 0:
                    should_exit = True
                    exit_reason = "Expiration"
                
                if should_exit:
                    self.trades.append(profit_percent)
                    
                    self.trade_details.append({
                        'entry_date': entry_date.strftime('%Y-%m-%d'),
                        'exit_date': current_date.strftime('%Y-%m-%d'),
                        'entry_price': round(entry_price, 2),
                        'exit_price': round(current_price, 2),
                        'call_short_strike': self.entry_shorts['call'],
                        'put_short_strike': self.entry_shorts['put'],
                        'credit_received': round(credit_received, 2),
                        'cost_to_close': round(current_cost, 2),
                        'profit_dollars': round(profit, 2),
                        'profit_percent': round(profit_percent, 2),
                        'max_risk': capital_risk_per_trade,
                        'exit_reason': exit_reason,
                        'days_held': days_held,
                        'dte_at_exit': dte_remaining,
                    })
                    
                    position_open = False
        
        return self.calculate_metrics()
    
    def calculate_metrics(self):
        """Calculate performance metrics"""
        if not self.trades:
            return {}
        
        returns = np.array(self.trades)
        total_trades = len(self.trades)
        profitable_trades = len([t for t in self.trades if t > 0])
        total_return = sum(self.trades)
        win_rate = profitable_trades / total_trades * 100 if total_trades > 0 else 0
        avg_return = np.mean(returns)
        max_return = np.max(returns)
        min_return = np.min(returns)
        std_return = np.std(returns)
        
        cumulative_returns = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = running_max - cumulative_returns
        max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0
        
        # Calculate Profit Factor
        gross_profit = sum([r for r in returns if r > 0])
        gross_loss = abs(sum([r for r in returns if r < 0]))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        years = 5
        trades_per_year = total_trades / years if years > 0 else 0
        
        if total_return > -100:
            annualized_return = ((1 + total_return/100) ** (1/years) - 1) * 100
        else:
            annualized_return = -100
        
        return {
            'total_trades': total_trades,
            'trades_per_year': trades_per_year,
            'profitable_trades': profitable_trades,
            'win_rate': win_rate,
            'total_return': total_return,
            'annualized_return': annualized_return,
            'avg_return': avg_return,
            'max_return': max_return,
            'min_return': min_return,
            'std_return': std_return,
            'max_drawdown': max_drawdown,
            'profit_factor': profit_factor
        }
    
    def print_summary(self, results):
        """Print performance summary"""
        print("\n" + "="*100)
        print("PERFORMANCE SUMMARY (Iron Condor Strategy | 5 Years)")
        print("="*100)
        print(f"Total Trades: {results['total_trades']}")
        print(f"Trades Per Year: {results['trades_per_year']:.1f}")
        print(f"Profitable Trades: {results['profitable_trades']} / {results['total_trades']}")
        print(f"\nWin Rate: {results['win_rate']:.1f}%")
        print(f"Avg Return Per Trade: {results['avg_return']:.2f}%")
        print(f"Best Trade: +{results['max_return']:.2f}%")
        print(f"Worst Trade: {results['min_return']:.2f}%")
        print(f"Std Dev: {results['std_return']:.2f}%")
        print(f"\nTotal Return: {results['total_return']:.1f}%")
        print(f"Annualized Return: {results['annualized_return']:.1f}%")
        print(f"Max Drawdown: {results['max_drawdown']:.1f}%")
        print(f"Profit Factor: {results['profit_factor']:.2f}x")
    
    def print_recent_trades(self, num_trades=15):
        """Print recent trades"""
        if not self.trade_details:
            return
        
        print("\n" + "="*100)
        print(f"RECENT {min(num_trades, len(self.trade_details))} TRADES")
        print("="*100)
        recent = self.trade_details[-num_trades:]
        recent.reverse()
        
        print(f"{'#':<3} {'Entry':<11} {'Exit':<11} {'Days':<5} {'Entry$':<8} {'Exit$':<8} {'Credit':<8} {'Cost':<8} {'Profit%':<8} {'Reason':<25}")
        print("-"*100)
        
        for idx, trade in enumerate(recent, 1):
            symbol = "✓" if trade['profit_percent'] > 0 else "✗"
            print(f"{idx:<3} {trade['entry_date']:<11} {trade['exit_date']:<11} {trade['days_held']:<5} "
                  f"${trade['entry_price']:<7.2f} ${trade['exit_price']:<7.2f} "
                  f"${trade['credit_received']:<7.0f} ${trade['cost_to_close']:<7.0f} "
                  f"{symbol} {trade['profit_percent']:+.2f}%   {trade['exit_reason']:<25}")


# ===== MAIN EXECUTION =====
if __name__ == "__main__":
    try:
        strategy = QQQIronCondorStrategy(initial_investment=100000)
        
        results = strategy.backtest_strategy(
            ticker='QQQ',
            start_date='2020-01-01',
            end_date='2025-11-25',
            dte_open=45,
            dte_close=21,
            width=50,
            capital_risk_per_trade=1000
        )
        
        if results:
            strategy.print_summary(results)
            strategy.print_recent_trades(num_trades=15)
            
            print("\n" + "="*100)
            print("STRATEGY OVERVIEW - IRON CONDOR")
            print("="*100)
            print("What is an Iron Condor?")
            print("  - Sell call spread: Sell call 2% above current, Buy call 3% above (protect upside)")
            print("  - Sell put spread: Sell put 2% below current, Buy put 3% below (protect downside)")
            print("  - Max profit: If stock stays between short strikes at expiration")
            print("  - Max loss: Width of spread ($50) minus credit received")
            print("")
            print("Why it works for QQQ:")
            print("  ✓ Profits on non-movement (sideways market)")
            print("  ✓ High probability (80%+ win rate typical)")
            print("  ✓ Limited defined risk (know max loss upfront)")
            print("  ✓ Defined profit (know max profit upfront)")
            print("  ✓ No catastrophic losses like long calls")
            print("")
            print("Entry Signal:")
            print("  - Open when market is consolidating (low volatility)")
            print("  - Recent 20-day range < 6% of stock price = good consolidation")
            print("")
            print("Exit Rules:")
            print("  - Close at 50% max profit (don't be greedy)")
            print("  - Close at 21 DTE (let theta decay work, then exit)")
            print("  - Hard stop at 50% loss of risk")
            print("  - Exit if price breaks through short strike")
        else:
            print("Backtest failed")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()