import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import matplotlib.ticker as mticker
from strategy_functions import (
    run_fgi_strategy,
    run_rsi_strategy,
    run_risk_free_curve,
    calculate_drawdown,
    run_buy_and_hold_curve
)

# --- Parameters ---
start_date = '2011-01-03'
end_date   = '2025-08-09'
initial_capital = 10000
base_income = 25000

# --- Tax Brackets (2024, single filer) ---
federal_brackets = [
    (0, 11000,     0.10),
    (11000, 44725, 0.12),
    (44725, 95375, 0.22),
    (95375, 182100, 0.24),
    (182100, 231250, 0.32),
    (231250, 578125, 0.35),
    (578125, float('inf'), 0.37)
]
california_brackets = [
    (0, 10275,     0.01),
    (10275, 24475, 0.02),
    (24475, 37725, 0.04),
    (37725, 52475, 0.06),
    (52475, 66295, 0.08),
    (66295, 338639, 0.093),
    (338639, 406364, 0.103),
    (406364, 677275, 0.113),
    (677275, float('inf'), 0.123)
]

# --- Load Data ---
fear_greed_data = pd.read_csv('datasets/fear_greed_combined_2011_2025.csv')
fear_greed_data['Date'] = pd.to_datetime(fear_greed_data['Date'])

spy_data = yf.download('SPY', start=start_date, end=end_date, auto_adjust=True)
if spy_data.columns.nlevels > 1:
    spy_data.columns = spy_data.columns.droplevel(1)
spy_data.reset_index(inplace=True)
spy_data['Date'] = pd.to_datetime(spy_data['Date'])

merged_data = pd.merge(
    spy_data, fear_greed_data[['Date', 'fear_greed']], on='Date', how='left'
)
merged_data['fear_greed'] = merged_data['fear_greed'].ffill()

# --- Analysis periods ---
periods = [
    ('2011-01-03', '2014-12-31'),
    ('2015-01-01', '2018-12-31'),
    ('2019-01-01', '2022-12-31'),
    ('2023-01-01', '2025-08-09'),
    ('2011-01-03', '2025-08-09')
]

for period_start, period_end in periods:
    # Filter data by period
    p_start = pd.to_datetime(period_start)
    p_end   = pd.to_datetime(period_end)
    period_mask = (merged_data['Date'] >= p_start) & (merged_data['Date'] <= p_end)
    period_data = merged_data.loc[period_mask].reset_index(drop=True)
    period_prices = spy_data.loc[(spy_data['Date'] >= p_start) & (spy_data['Date'] <= p_end)].reset_index(drop=True)
    dates = period_data['Date']

    # --- Compute strategy equity curves ---
    fgi_curve = run_fgi_strategy(
        period_data,
        initial_capital=initial_capital,
        do_tax=True,
        base_income=base_income,
        federal_brackets=federal_brackets,
        state_brackets=california_brackets
    )
    rsi_curve = run_rsi_strategy(
        period_prices,
        initial_capital=initial_capital,
        do_tax=True,
        base_income=base_income,
        federal_brackets=federal_brackets,
        state_brackets=california_brackets
    )
    rf_curve = run_risk_free_curve(dates, initial_capital=initial_capital, rate=0.045)
    bh_curve = run_buy_and_hold_curve(
        period_prices,
        initial_capital=initial_capital,
        do_tax=True,
        base_income=base_income,
        federal_brackets=federal_brackets,
        state_brackets=california_brackets
    )

    # Align curves with dates
    fgi_curve = fgi_curve.reindex(dates, method='ffill')
    rsi_curve = rsi_curve.reindex(dates, method='ffill')
    rf_curve  = rf_curve.reindex(dates, method='ffill')
    bh_curve  = bh_curve.reindex(dates, method='ffill')

    curves = {
        "FGI Strategy (After Tax)": fgi_curve,
        "RSI 30/70 (After Tax)": rsi_curve,
        "Buy & Hold SPY (After Tax)": bh_curve,
        "Risk-Free 4.5%": rf_curve
    }
    drawdowns = {k: calculate_drawdown(v) for k, v in curves.items()}

    # Get final values & max drawdowns for legends
    final_values = {k: v.iloc[-1] for k, v in curves.items()}
    max_dds = {k: drawdowns[k].min() for k in curves.keys()}

    # --- Plot: stacked equity & drawdown curves ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12,10), sharex=True, gridspec_kw={"height_ratios": [2,1]})

    # Equity curves with final values in legend
    for label, curve in curves.items():
        ax1.plot(curve.index, curve.values, label=f"{label} [${curve.iloc[-1]:,.0f}]")
    ax1.set_title(f'Portfolio Value: {period_start} to {period_end}', fontsize=15)
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.legend()
    ax1.grid(True)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))

    # Drawdown curves with max drawdown in legend
    for label, dd in drawdowns.items():
        max_dd = dd.min()
        ax2.plot(dd.index, dd.values, label=f"{label} [Max DD: {max_dd*100:+.1f}%]")
    ax2.set_title(f'Drawdowns: {period_start} to {period_end}', fontsize=15)
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Drawdown (%)')
    ax2.legend()
    ax2.grid(True)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x * 100:.1f}%'))

    plt.tight_layout()
    plt.savefig(f"equity_drawdown_{period_start}_{period_end}.png", dpi=1200)


    # --- Print Summary Table for Interpretation ---
    print("="*70)
    print(f"Period: {period_start} to {period_end}")
    print("{:<30} {:>12} {:>16}".format("Strategy", "Final Value", "Max Drawdown"))
    for k in curves.keys():
        final = final_values[k]
        maxdd = max_dds[k]
        print("{:<30} {:>12,.0f} {:>15.2f}%".format(k, final, maxdd*100))
    print("="*70 + "\n")