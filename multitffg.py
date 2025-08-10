import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

class FearGreedStrategy:
    def __init__(self, fear_greed_file="datasets/fear-greed.csv", price_col="Close"):
        """
        Initialize the Fear & Greed trading strategy.
        
        Parameters:
        -----------
        fear_greed_file : str
            Path to the Fear & Greed index CSV file
        price_col : str
            Price column to use for calculations (e.g., 'Close', 'Adj Close')
        """
        self.fear_greed_file = fear_greed_file
        self.price_col = price_col
        self.fear_greed_data = None
        self.spy_data = None
        self.merged_data = None
        self.results = None
        self.buy_signals = []
        self.sell_signals = []
        
    def load_fear_greed_data(self):
        """Load Fear & Greed index data from CSV file."""
        fear_greed_data = pd.read_csv(self.fear_greed_file)
        fear_greed_data['Date'] = pd.to_datetime(fear_greed_data['Date'])
        self.fear_greed_data = fear_greed_data[['Date', 'fear_greed']]
        return self
        
    def load_price_data(self, ticker='SPY', start_date='2011-01-03', end_date='2020-09-18'):
        """Load price data from Yahoo Finance."""
        print(f"Loading price data for {ticker} from {start_date} to {end_date}")
        
        spy_data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)
        
        # Flatten multi-level columns if they exist
        if spy_data.columns.nlevels > 1:
            spy_data.columns = spy_data.columns.droplevel(1)
        
        spy_data.reset_index(inplace=True)
        self.spy_data = spy_data
        return self
    
    def merge_data(self):
        """Merge price data with Fear & Greed data."""
        if self.fear_greed_data is None:
            self.load_fear_greed_data()
            
        if self.spy_data is None:
            raise ValueError("Price data not loaded. Call load_price_data() first.")
            
        merged_data = pd.merge(self.spy_data, self.fear_greed_data[['Date', 'fear_greed']], 
                              on='Date', how='left')
        merged_data['fear_greed'] = merged_data['fear_greed'].ffill()
        self.merged_data = merged_data
        return self
    
    def run_strategy(self, 
                    lookback_days=3, 
                    momentum_threshold=1.5,
                    use_dynamic_sizing=True, 
                    use_regime_detection=True,
                    use_multi_timeframe=True, 
                    use_volatility_adjustment=True,
                    base_position_size=1.0, 
                    max_position_size=2.0,
                    max_holding_days=8,
                    velocity_threshold=0.3,
                    acceleration_threshold=0.2,
                    zscore_threshold=-1.5,
                    bull_regime_threshold=0.02,
                    bear_regime_threshold=-0.02,
                    high_volatility_threshold=0.5,
                    low_volatility_threshold=0.15,
                    position_sizing_multipliers=None):
        """
        Run the Fear & Greed trading strategy with customizable parameters.
        
        Parameters:
        -----------
        lookback_days : int
            Number of days for short-term moving average
        momentum_threshold : float
            Threshold for considering momentum signals significant
        use_dynamic_sizing : bool
            Whether to adjust position sizes based on signal strength
        use_regime_detection : bool
            Whether to detect and account for market regimes
        use_multi_timeframe : bool
            Whether to use multi-timeframe confirmation
        use_volatility_adjustment : bool
            Whether to adjust for market volatility
        base_position_size : float
            Base position size (1.0 = 100% of capital)
        max_position_size : float
            Maximum allowed position size
        max_holding_days : int
            Maximum number of days to hold a position
        velocity_threshold : float
            Threshold for considering velocity signals significant
        acceleration_threshold : float
            Threshold for considering acceleration signals significant
        zscore_threshold : float
            Z-score threshold for extreme fear/greed
        bull_regime_threshold : float
            Threshold for bull market detection
        bear_regime_threshold : float
            Threshold for bear market detection
        high_volatility_threshold : float
            Threshold for high volatility
        low_volatility_threshold : float
            Threshold for low volatility
        position_sizing_multipliers : dict
            Custom multipliers for position sizing
        
        Returns:
        --------
        self : FearGreedStrategy
            Returns self for method chaining
        """
        if self.merged_data is None:
            self.merge_data()
            
        # Set default position sizing multipliers if not provided
        if position_sizing_multipliers is None:
            position_sizing_multipliers = {
                'strong_momentum': 1.3,
                'strong_velocity': 1.2,
                'bull_market': 1.15,
                'bear_market': 0.8,
                'high_volatility': 0.7,
                'low_volatility': 1.1
            }
        
        data = self.merged_data.copy()
        
        # 1. Multi-timeframe Fear & Greed analysis
        data['fg_change'] = data['fear_greed'].diff()
        data['fg_sma_short'] = data['fear_greed'].rolling(lookback_days).mean()
        data['fg_sma_medium'] = data['fear_greed'].rolling(lookback_days * 2).mean()
        data['fg_sma_long'] = data['fear_greed'].rolling(lookback_days * 4).mean()
        
        # Momentum indicators
        data['fg_momentum_short'] = data['fear_greed'] - data['fg_sma_short']
        data['fg_momentum_medium'] = data['fear_greed'] - data['fg_sma_medium']
        data['fg_momentum_long'] = data['fear_greed'] - data['fg_sma_long']
        data['fg_velocity'] = data['fg_change'].rolling(3).mean()
        data['fg_acceleration'] = data['fg_velocity'].diff()
        
        # 2. Advanced price analysis
        data['price_returns'] = data[self.price_col].pct_change()
        data['volatility'] = data['price_returns'].rolling(20).std() * np.sqrt(252)
        data['price_sma_20'] = data[self.price_col].rolling(20).mean()
        data['price_sma_50'] = data[self.price_col].rolling(50).mean()
        data['price_momentum'] = data[self.price_col] / data['price_sma_20'] - 1
    
        # 3. Regime detection (Bull/Bear/Sideways market)
        if use_regime_detection:
            data['regime_trend'] = data['price_sma_20'] / data['price_sma_50'] - 1
            data['regime'] = np.where(data['regime_trend'] > bull_regime_threshold, 'Bull',
                                     np.where(data['regime_trend'] < bear_regime_threshold, 'Bear', 'Sideways'))
        else:
            data['regime'] = 'Bull'  # Default regime if not using detection
        
        # 4. Correlation analysis
        data['fg_price_corr'] = data['fear_greed'].rolling(30).corr(data[self.price_col])
        
        # 5. Z-score for extreme detection
        data['fg_zscore'] = (data['fear_greed'] - data['fear_greed'].rolling(60).mean()) / data['fear_greed'].rolling(60).std()
        
        # Initialize strategy columns
        data['position'] = 0.0
        data['position_size'] = 0.0
        data['signal'] = 0
        data['signal_strength'] = 0.0
        data['days_in_position'] = 0
        
        # Reset signal arrays
        self.buy_signals = []
        self.sell_signals = []
        
        current_position = 0.0
        days_held = 0
        
        # Define position sizing function
        def calculate_position_size(momentum, velocity, volatility, regime):
            if not use_dynamic_sizing:
                return base_position_size
                
            size = base_position_size
            
            # Increase size for stronger signals
            if abs(momentum) > momentum_threshold * 1.5:
                size *= position_sizing_multipliers['strong_momentum']
            if abs(velocity) > 1.0:
                size *= position_sizing_multipliers['strong_velocity']
                
            # Adjust for market regime
            if regime == 'Bull':
                size *= position_sizing_multipliers['bull_market']
            elif regime == 'Bear':
                size *= position_sizing_multipliers['bear_market']
                
            # Reduce size in high volatility
            if volatility > high_volatility_threshold:
                size *= position_sizing_multipliers['high_volatility']
            elif volatility < low_volatility_threshold:
                size *= position_sizing_multipliers['low_volatility']
                
            return min(size, max_position_size)
        
        # Main trading loop
        for i in range(max(lookback_days * 4, 60), len(data)):
            fg_momentum = data.iloc[i]['fg_momentum_short']
            fg_velocity = data.iloc[i]['fg_velocity']
            fg_acceleration = data.iloc[i]['fg_acceleration']
            volatility = data.iloc[i]['volatility']
            regime = data.iloc[i]['regime']
            price_momentum = data.iloc[i]['price_momentum']
            fg_zscore = data.iloc[i]['fg_zscore']
            
            # Multi-timeframe confirmation
            if use_multi_timeframe:
                mtf_bullish = (data.iloc[i]['fg_momentum_short'] > 0 and 
                              data.iloc[i]['fg_momentum_medium'] > -1 and
                              data.iloc[i]['fg_momentum_long'] > -2)
                mtf_bearish = (data.iloc[i]['fg_momentum_short'] < 0 and 
                              data.iloc[i]['fg_momentum_medium'] < 1 and
                              data.iloc[i]['fg_momentum_long'] < 2)
            else:
                mtf_bullish = mtf_bearish = True
            
            # Increment days held
            if current_position != 0:
                days_held += 1
            
            if pd.notna(fg_momentum) and pd.notna(fg_velocity):
                
                # === ENHANCED BUY CONDITIONS ===
                base_buy_condition = (
                    fg_momentum > momentum_threshold and
                    fg_velocity > velocity_threshold and
                    mtf_bullish
                )
                
                # Strength multipliers
                strength_multipliers = []
                
                # Strong acceleration
                if pd.notna(fg_acceleration) and fg_acceleration > acceleration_threshold:
                    strength_multipliers.append(1.2)
                
                # Extreme fear reversal (contrarian element)
                if fg_zscore < zscore_threshold and fg_velocity > 0:
                    strength_multipliers.append(1.3)
                
                # Price momentum alignment
                if price_momentum > 0:
                    strength_multipliers.append(1.1)
                
                # Bull market bonus
                if regime == 'Bull':
                    strength_multipliers.append(1.1)
                
                signal_strength = np.prod(strength_multipliers) if strength_multipliers else 1.0
                
                # Enhanced buy condition
                buy_condition = (
                    current_position <= 0 and
                    base_buy_condition and
                    signal_strength > 1.0 and
                    (not use_volatility_adjustment or volatility < high_volatility_threshold * 0.8)
                )
                
                # === ENHANCED SELL CONDITIONS ===
                base_sell_condition = (
                    fg_momentum < -momentum_threshold or
                    fg_velocity < -velocity_threshold or
                    days_held >= max_holding_days
                )
                
                sell_condition = (
                    current_position > 0 and (
                        base_sell_condition or
                        (mtf_bearish and fg_velocity < 0) or
                        (regime == 'Bear' and fg_momentum < 0) or
                        (volatility > high_volatility_threshold)  # Exit in extreme volatility
                    )
                )
                
                # === AGGRESSIVE SHORT CONDITIONS ===
                short_condition = (
                    current_position >= 0 and
                    fg_momentum < -momentum_threshold * 1.8 and
                    fg_velocity < -velocity_threshold * 2.5 and
                    mtf_bearish and
                    (not use_volatility_adjustment or volatility < high_volatility_threshold * 1.2)
                )
                
                # Execute trades
                if buy_condition:
                    position_size = calculate_position_size(fg_momentum, fg_velocity, volatility, regime)
                    data.iloc[i, data.columns.get_loc('signal')] = 1
                    data.iloc[i, data.columns.get_loc('signal_strength')] = signal_strength
                    current_position = position_size
                    days_held = 0
                    self.buy_signals.append(i)
                    
                elif sell_condition:
                    data.iloc[i, data.columns.get_loc('signal')] = -1
                    current_position = 0.0
                    days_held = 0
                    self.sell_signals.append(i)
                    
                elif short_condition:
                    position_size = calculate_position_size(fg_momentum, fg_velocity, volatility, regime)
                    data.iloc[i, data.columns.get_loc('signal')] = -2
                    current_position = -position_size
                    days_held = 0
                    self.sell_signals.append(i)
            
            data.iloc[i, data.columns.get_loc('position')] = current_position
            data.iloc[i, data.columns.get_loc('position_size')] = abs(current_position)
            data.iloc[i, data.columns.get_loc('days_in_position')] = days_held
        
        self.results = data
        return self
    
    def calculate_performance_metrics(self, starting_capital=10000):
        """
        Calculate performance metrics for the strategy.
        
        Parameters:
        -----------
        starting_capital : float
            Initial investment amount
            
        Returns:
        --------
        dict
            Dictionary of performance metrics
        """
        if self.results is None:
            raise ValueError("Strategy not run yet. Call run_strategy() first.")
            
        data = self.results.copy()
        
        # Calculate returns with position sizing
        data['returns'] = data[self.price_col].pct_change()
        data['strategy_returns'] = data['returns'] * data['position'].shift(1)
        data['cumulative_strategy_returns'] = (1 + data['strategy_returns']).cumprod()
        data['capital'] = starting_capital * data['cumulative_strategy_returns']
        
        # Calculate buy & hold returns
        data['buy_hold_returns'] = data[self.price_col] / data[self.price_col].iloc[0]
        
        # Strategy return
        strategy_return = data['cumulative_strategy_returns'].iloc[-1] - 1
        
        # Buy & hold return
        buy_hold_return = (data[self.price_col].iloc[-1] / data[self.price_col].iloc[0]) - 1
        
        # Sharpe ratio calculation
        strategy_returns_series = data['strategy_returns'].dropna()
        sharpe_ratio = strategy_returns_series.mean() / strategy_returns_series.std() * np.sqrt(252) if strategy_returns_series.std() > 0 else 0
        
        # Buy & hold Sharpe
        bh_returns = data[self.price_col].pct_change().dropna()
        bh_sharpe = bh_returns.mean() / bh_returns.std() * np.sqrt(252)
        
        # Maximum drawdown
        cumulative = data['cumulative_strategy_returns'].fillna(1)
        running_max = cumulative.expanding().max()
        drawdown = (cumulative / running_max - 1)
        max_drawdown = drawdown.min()
        
        # Win rate calculation
        winning_trades = 0
        if self.buy_signals:
            for signal_idx in self.buy_signals:
                if signal_idx < len(data) - 5:
                    entry_price = data.iloc[signal_idx][self.price_col]
                    future_prices = data.iloc[signal_idx+1:signal_idx+6][self.price_col]
                    if len(future_prices) > 0 and future_prices.max() > entry_price:
                        winning_trades += 1
            win_rate = winning_trades / len(self.buy_signals) if self.buy_signals else 0
        else:
            win_rate = 0
            
        # Total trades
        total_trades = len(self.buy_signals) + len(self.sell_signals)
        
        # Store updated results
        self.results = data
        
        # Return metrics as dictionary
        metrics = {
            'strategy_return': strategy_return,
            'buy_hold_return': buy_hold_return,
            'outperformance': strategy_return - buy_hold_return,
            'sharpe_ratio': sharpe_ratio,
            'buy_hold_sharpe': bh_sharpe,
            'max_drawdown': max_drawdown,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'final_capital': data['capital'].iloc[-1],
            'cumulative_return_factor': data['cumulative_strategy_returns'].iloc[-1]
        }
        
        return metrics
    
    def plot_results(self, figsize=(18, 16)):
        """
        Plot strategy results and analysis charts.
        
        Parameters:
        -----------
        figsize : tuple
            Figure size for the plots
            
        Returns:
        --------
        matplotlib.figure.Figure
            The plot figure
        """
        if self.results is None:
            raise ValueError("Strategy not run yet. Call run_strategy() first.")
            
        data = self.results
        
        # Enhanced visualization
        fig, axes = plt.subplots(4, 1, figsize=figsize)
        
        # Plot 1: Price with position sizing
        axes[0].plot(data['Date'], data[self.price_col], color='blue', 
                    label=f'Price ({self.price_col})', linewidth=1)
        
        if self.buy_signals:
            buy_dates = data.iloc[self.buy_signals]['Date']
            buy_prices = data.iloc[self.buy_signals][self.price_col]
            buy_sizes = data.iloc[self.buy_signals]['position_size']
            
            # Size-based marker scaling
            marker_sizes = 50 + (buy_sizes - 1) * 100
            axes[0].scatter(buy_dates, buy_prices, c=buy_sizes, cmap='Greens', 
                           marker='^', s=marker_sizes, label='Buy Signal (size)', zorder=5, alpha=0.8)
        
        if self.sell_signals:
            sell_dates = data.iloc[self.sell_signals]['Date']
            sell_prices = data.iloc[self.sell_signals][self.price_col]
            axes[0].scatter(sell_dates, sell_prices, color='red', 
                           marker='v', s=100, label='Sell Signal', zorder=5)
        
        axes[0].set_ylabel(f'Price')
        axes[0].set_title(f'Multi-Timeframe Strategy (Position Sizing Shown)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Multi-timeframe Fear & Greed
        axes[1].plot(data['Date'], data['fear_greed'], color='orange', label='F&G Index', linewidth=1)
        axes[1].plot(data['Date'], data['fg_sma_short'], color='green', label='Short MA', alpha=0.7)
        axes[1].plot(data['Date'], data['fg_sma_medium'], color='blue', label='Medium MA', alpha=0.7)
        axes[1].plot(data['Date'], data['fg_sma_long'], color='purple', label='Long MA', alpha=0.7)
        
        axes[1].set_ylabel('Fear & Greed Index')
        axes[1].set_title('Multi-Timeframe Fear & Greed Analysis')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Plot 3: Advanced momentum indicators
        axes[2].plot(data['Date'], data['fg_momentum_short'], color='red', label='Short Momentum', linewidth=1)
        axes[2].plot(data['Date'], data['fg_velocity'], color='blue', label='Velocity', linewidth=1)
        axes[2].plot(data['Date'], data['fg_acceleration'], color='green', label='Acceleration', linewidth=1)
        axes[2].axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        axes[2].set_ylabel('Momentum Indicators')
        axes[2].set_title('Advanced Fear & Greed Momentum Analysis')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        # Plot 4: Strategy performance
        axes[3].plot(data['Date'], data['cumulative_strategy_returns'], 
                    color='green', label='Strategy', linewidth=2)
        axes[3].plot(data['Date'], data['buy_hold_returns'], 
                    color='blue', label='Buy & Hold', linewidth=1, alpha=0.7)
        
        axes[3].set_ylabel('Cumulative Returns')
        axes[3].set_title('Strategy Performance Comparison')
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def print_summary(self, metrics=None):
        """Print a summary of strategy performance metrics."""
        if metrics is None:
            metrics = self.calculate_performance_metrics()
            
        print(f"\n=== MULTI-TIMEFRAME STRATEGY RESULTS ===")
        print(f"Strategy Return: {metrics['strategy_return']:.2%}")
        print(f"Buy & Hold Return: {metrics['buy_hold_return']:.2%}")
        print(f"Outperformance: {metrics['outperformance']:+.2%}")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f} (Buy & Hold: {metrics['buy_hold_sharpe']:.2f})")
        print(f"Maximum Drawdown: {metrics['max_drawdown']:.2%}")
        print(f"Total Trades: {metrics['total_trades']}")
        print(f"Win Rate: {metrics['win_rate']:.1%}")
        print(f"Starting Capital: $10,000.00")
        print(f"Final Capital: ${metrics['final_capital']:,.2f}")
        
        
        return metrics
    
    def optimize(self, param_grid, metric='sharpe_ratio', maximize=True):
        """
        Optimize strategy parameters using grid search.
        
        Parameters:
        -----------
        param_grid : dict
            Dictionary with parameter names as keys and lists of parameter values as values
        metric : str
            Metric to optimize ('sharpe_ratio', 'strategy_return', etc.)
        maximize : bool
            Whether to maximize (True) or minimize (False) the metric
            
        Returns:
        --------
        dict
            Best parameters and their performance
        """
        import itertools
        
        # Generate all combinations of parameters
        param_names = sorted(param_grid)
        param_values = [param_grid[name] for name in param_names]
        param_combinations = list(itertools.product(*param_values))
        
        best_metric_value = -np.inf if maximize else np.inf
        best_params = None
        best_metrics = None
        
        total_combinations = len(param_combinations)
        print(f"Optimizing over {total_combinations} parameter combinations...")
        
        for i, combination in enumerate(param_combinations):
            # Create parameter dictionary
            params = dict(zip(param_names, combination))
            
            # Update with progress
            if i % 10 == 0 or i == total_combinations - 1:
                print(f"Testing combination {i+1}/{total_combinations}...")
            
            # Run strategy with these parameters
            self.run_strategy(**params)
            
            # Calculate metrics
            metrics = self.calculate_performance_metrics()
            
            # Check if this is the best result
            current_metric = metrics[metric]
            is_better = (current_metric > best_metric_value) if maximize else (current_metric < best_metric_value)
            
            if is_better:
                best_metric_value = current_metric
                best_params = params
                best_metrics = metrics
                print(f"New best {metric}: {best_metric_value:.4f} with params: {best_params}")
        
        # Run the strategy with the best parameters
        self.run_strategy(**best_params)
        final_metrics = self.calculate_performance_metrics()
        
        return {
            'best_params': best_params,
            'best_metrics': final_metrics
        }



def multi_timeframe_boolean_optimization(strategy, 
                                         timeframes,
                                         fixed_params,
                                         boolean_params,
                                         metric_weights={'return': 0.5, 'win_rate': 0.4, 'drawdown': 0.1},
                                         results_csv='boolean_optimization_results.csv'):
    """
    Runs optimization on multiple train/test timeframes with fixed params and boolean params.
    Prioritizes return and win rate, penalizes drawdown.
    
    Params:
        strategy: your FearGreedStrategy instance (already loaded with data)
        timeframes: list of dicts with keys 'train_start', 'train_end', 'test_start', 'test_end'
        fixed_params: dict with fixed params (lookback_days, momentum_threshold, use_dynamic_sizing)
        boolean_params: dict with keys and list of boolean values to test
        metric_weights: dict to weight the composite score of return, win rate, and drawdown
        results_csv: output CSV filename
    
    Returns:
        DataFrame with results and summary stats.
    """
    
    results = []
    
    # Create all boolean param combos
    from itertools import product
    keys = list(boolean_params.keys())
    combos = list(product(*[boolean_params[k] for k in keys]))
    
    print(f"Testing {len(timeframes)} timeframes and {len(combos)} boolean combos each...")

    for tf in timeframes:
        train_start, train_end = tf['train_start'], tf['train_end']
        test_start, test_end = tf['test_start'], tf['test_end']

        print(f"\n=== Timeframe Train: {train_start} → {train_end} | Test: {test_start} → {test_end} ===")

        # Load price data for training period
        strategy.load_price_data(ticker='SPY', start_date=train_start, end_date=train_end)

        # Run optimization over boolean combos on training data only
        best_score = -np.inf
        best_params = None
        best_metrics_train = None
        best_metrics_test = None

        for i, combo in enumerate(combos, 1):
            bool_kwargs = dict(zip(keys, combo))
            params = {**fixed_params, **bool_kwargs}

            # Run strategy on training period
            strategy.merge_data()  # assumes fear-greed data already loaded
            strategy.run_strategy(**params)
            train_metrics = strategy.calculate_performance_metrics(starting_capital=10000)

            # Use a composite score prioritizing return and win rate, penalizing drawdown
            score = (metric_weights['return'] * train_metrics['strategy_return'] +
                     metric_weights['win_rate'] * train_metrics['win_rate'] -
                     metric_weights['drawdown'] * abs(train_metrics['max_drawdown']))

            if score > best_score:
                best_score = score
                best_params = params
                best_metrics_train = train_metrics

            if i % 10 == 0 or i == len(combos):
                print(f"Tested {i}/{len(combos)} combos. Current best score: {best_score:.4f}")

        # Now test the best params on the test period
        strategy.load_price_data(ticker='SPY', start_date=test_start, end_date=test_end)
        strategy.merge_data()
        strategy.run_strategy(**best_params)
        test_metrics = strategy.calculate_performance_metrics(starting_capital=10000)

        results.append({
            'Train_Start': train_start,
            'Train_End': train_end,
            'Test_Start': test_start,
            'Test_End': test_end,
            **best_params,
            'Train_Sharpe': best_metrics_train['sharpe_ratio'],
            'Train_Return': best_metrics_train['strategy_return'],
            'Train_Drawdown': best_metrics_train['max_drawdown'],
            'Train_WinRate': best_metrics_train['win_rate'],
            'Test_Sharpe': test_metrics['sharpe_ratio'],
            'Test_Return': test_metrics['strategy_return'],
            'Test_Drawdown': test_metrics['max_drawdown'],
            'Test_WinRate': test_metrics['win_rate'],
        })

    df_results = pd.DataFrame(results)
    df_results.to_csv(results_csv, index=False)
    print(f"\n✅ Boolean optimization results saved to {results_csv}")

    # Calculate summary stats
    print("\n=== SUMMARY STATISTICS ===")
    print(f"Avg Test Return: {df_results['Test_Return'].mean():.4f}")
    print(f"Worst Test Drawdown: {df_results['Test_Drawdown'].min():.4f}")
    print(f"Worst Test Win Rate: {df_results['Test_WinRate'].min():.4f}")

    return df_results

if __name__ == "__main__":
    strategy = FearGreedStrategy(fear_greed_file="datasets/fear-greed.csv", price_col="Close")
    strategy.load_fear_greed_data()
    strategy.load_price_data(ticker='SPY', start_date='2011-01-03', end_date='2020-09-18')
    strategy.merge_data()

    # Fixed params based on previous optimization
    fixed_params = {
        'lookback_days': 3,
        'momentum_threshold': 2.0,
    }

    # Booleans to optimize
    boolean_grid = {
        'use_dynamic_sizing': [True, False],
        'use_multi_timeframe': [True, False],
        'use_regime_detection': [True, False],
        'use_volatility_adjustment': [True, False]
    }

    # Combine fixed params and boolean grid
    from itertools import product

    boolean_keys = list(boolean_grid.keys())
    boolean_combinations = list(product(*boolean_grid.values()))

    results = []
    metric_weights = {'return': 0.5, 'win_rate': 0.4, 'drawdown': 0.1}  # adjust weights as desired

    for combo in boolean_combinations:
        params = fixed_params.copy()
        params.update(dict(zip(boolean_keys, combo)))
        print(f"Testing params: {params}")
        strategy.run_strategy(**params)
        metrics = strategy.calculate_performance_metrics(starting_capital=10000)

        score = (metric_weights['return'] * metrics['strategy_return'] +
                metric_weights['win_rate'] * metrics['win_rate'] -
                metric_weights['drawdown'] * abs(metrics['max_drawdown']))

        results.append({
            **params,
            'Return': metrics['strategy_return'],
            'Drawdown': metrics['max_drawdown'],
            'WinRate': metrics['win_rate'],
            'Score': score
        })

    df_results = pd.DataFrame(results)

# Sort by Score first, then Return and WinRate if you want
    df_results = df_results.sort_values(by=['Score', 'Return', 'WinRate'], ascending=[False, False, False])

    # Sort by Return and WinRate (prioritize return, then win rate)
    df_results = df_results.sort_values(by=['Return', 'WinRate'], ascending=[False, False])

    print("=== BOOLEAN OPTIMIZATION RESULTS ===")
    print(df_results)

    # Calculate summary stats
    avg_return = df_results['Return'].mean()
    worst_drawdown = df_results['Drawdown'].min()
    worst_winrate = df_results['WinRate'].min()

    print("\n=== SUMMARY STATISTICS ===")
    print(f"Avg Return: {avg_return:.4f}")
    print(f"Worst Drawdown: {worst_drawdown:.4f}")
    print(f"Worst Win Rate: {worst_winrate:.4f}")

    # Save results
    df_results.to_csv('boolean_optimization_results.csv', index=False)
    print("\n✅ Boolean optimization results saved to boolean_optimization_results.csv")
