import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def analyze_trade_log(csv_file):
    """
    Comprehensive analysis of the trade log CSV with improved safety and flexibility.
    """
    # Load the data
    df = pd.read_csv(csv_file)
    
    # Ensure date columns are datetime
    for col in ['Entry_Date', 'Exit_Date']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # Create Month & Year columns if missing
    if 'Month' not in df.columns and 'Entry_Date' in df.columns:
        df['Month'] = df['Entry_Date'].dt.month
    if 'Year' not in df.columns and 'Entry_Date' in df.columns:
        df['Year'] = df['Entry_Date'].dt.year

    print("=" * 80)
    print("COMPREHENSIVE TRADE LOG ANALYSIS")
    print("=" * 80)

    # Basic Stats
    total_trades = len(df)
    win_trades = df['Win_Loss_Binary'].sum() if 'Win_Loss_Binary' in df else 0
    loss_trades = total_trades - win_trades
    win_rate = win_trades / total_trades if total_trades > 0 else 0
    avg_hold = df['Holding_Period_Days'].mean() if 'Holding_Period_Days' in df else 0
    total_pnl = df['Net_PnL'].sum() if 'Net_PnL' in df else 0
    avg_return = df['Net_Percent_Return'].mean() if 'Net_Percent_Return' in df else 0

    print("\n📊 BASIC TRADE STATISTICS:")
    print("-" * 40)
    print(f"Total Trades: {total_trades}")
    print(f"Winning Trades: {win_trades}")
    print(f"Losing Trades: {loss_trades}")
    print(f"Win Rate: {win_rate:.1%}")
    print(f"Average Holding Period: {avg_hold:.1f} days")
    print(f"Total Net P&L: ${total_pnl:,.2f}")
    print(f"Average Trade Return: {avg_return:.2f}%")

    # Best & Worst Trades
    if total_trades > 0:
        best_trade = df.loc[df['Net_PnL'].idxmax()] if 'Net_PnL' in df else None
        worst_trade = df.loc[df['Net_PnL'].idxmin()] if 'Net_PnL' in df else None

        print(f"\n🏆 BEST & WORST TRADES:")
        print("-" * 30)
        if best_trade is not None:
            print(f"Best Trade:")
            print(f"  Trade #{best_trade['Trade_Number']} | "
                  f"{best_trade['Entry_Date'].strftime('%Y-%m-%d')} → {best_trade['Exit_Date'].strftime('%Y-%m-%d')}")
            print(f"  ${best_trade['Entry_Price']:.2f} → ${best_trade['Exit_Price']:.2f} "
                  f"({best_trade['Percent_Change']:+.1f}%)")
            print(f"  Net P&L: ${best_trade['Net_PnL']:,.2f} "
                  f"({best_trade['Net_Percent_Return']:+.1f}%)")
        
        if worst_trade is not None:
            print(f"\nWorst Trade:")
            print(f"  Trade #{worst_trade['Trade_Number']} | "
                  f"{worst_trade['Entry_Date'].strftime('%Y-%m-%d')} → {worst_trade['Exit_Date'].strftime('%Y-%m-%d')}")
            print(f"  ${worst_trade['Entry_Price']:.2f} → ${worst_trade['Exit_Price']:.2f} "
                  f"({worst_trade['Percent_Change']:+.1f}%)")
            print(f"  Net P&L: ${worst_trade['Net_PnL']:,.2f} "
                  f"({worst_trade['Net_Percent_Return']:+.1f}%)")

    # Holding Period Analysis
    if 'Holding_Period_Days' in df:
        print(f"\n⏱️ HOLDING PERIOD ANALYSIS:")
        print("-" * 35)
        print(f"Min Holding Period: {df['Holding_Period_Days'].min()} days")
        print(f"Max Holding Period: {df['Holding_Period_Days'].max()} days")
        print(f"Most Common: {df['Holding_Period_Days'].mode().iloc[0]} days")
        
        hold_analysis = df.groupby('Holding_Period_Days').agg({
            'Win_Loss_Binary': ['count', 'sum', 'mean'],
            'Net_Percent_Return': 'mean'
        }).round(3)
        hold_analysis.columns = ['Total_Trades', 'Wins', 'Win_Rate', 'Avg_Return']
        print("\nWin Rate by Holding Period (top):")
        print(hold_analysis.sort_values('Total_Trades', ascending=False).head(8))

    # Fear & Greed Analysis
    if 'Entry_Fear_Greed' in df and 'Exit_Fear_Greed' in df:
        print(f"\n😨😍 FEAR & GREED ANALYSIS:")
        print("-" * 35)
        print(f"Entry F&G - Mean: {df['Entry_Fear_Greed'].mean():.1f}, Std: {df['Entry_Fear_Greed'].std():.1f}")
        print(f"Exit F&G - Mean: {df['Exit_Fear_Greed'].mean():.1f}, Std: {df['Exit_Fear_Greed'].std():.1f}")
        if 'Fear_Greed_Change' in df:
            print(f"Average F&G Change: {df['Fear_Greed_Change'].mean():+.1f}")

        df['FG_Entry_Bucket'] = pd.cut(
            df['Entry_Fear_Greed'],
            bins=[0, 25, 45, 55, 75, 100],
            labels=['Extreme Fear (0-25)', 'Fear (25-45)', 'Neutral (45-55)',
                    'Greed (55-75)', 'Extreme Greed (75-100)']
        )
        fg_analysis = df.groupby('FG_Entry_Bucket').agg({
            'Win_Loss_Binary': ['count', 'mean'],
            'Net_Percent_Return': 'mean'
        }).round(3)
        fg_analysis.columns = ['Total_Trades', 'Win_Rate', 'Avg_Return']
        print("\nPerformance by Entry Fear & Greed Level:")
        print(fg_analysis)

    # Monthly Performance
    if 'Month' in df:
        print(f"\n📅 MONTHLY PERFORMANCE:")
        print("-" * 25)
        monthly_stats = df.groupby('Month').agg({
            'Win_Loss_Binary': ['count', 'sum', 'mean'],
            'Net_PnL': 'sum',
            'Net_Percent_Return': 'mean'
        }).round(2)
        monthly_stats.columns = ['Total_Trades', 'Wins', 'Win_Rate', 'Total_PnL', 'Avg_Return']
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        monthly_stats.index = [month_names[i - 1] for i in monthly_stats.index]
        print(monthly_stats)

    # Yearly Performance
    if 'Year' in df:
        print(f"\n📊 YEARLY PERFORMANCE:")
        print("-" * 25)
        yearly_stats = df.groupby('Year').agg({
            'Win_Loss_Binary': ['count', 'sum', 'mean'],
            'Net_PnL': 'sum',
            'Net_Percent_Return': 'mean',
            'Holding_Period_Days': 'mean'
        }).round(2)
        yearly_stats.columns = ['Total_Trades', 'Wins', 'Win_Rate', 'Total_PnL', 'Avg_Return', 'Avg_Hold_Days']
        print(yearly_stats)

    # Transaction Costs
    if 'Gross_PnL' in df and 'Total_Costs' in df:
        print(f"\n💰 TRANSACTION COSTS ANALYSIS:")
        print("-" * 35)
        total_gross_pnl = df['Gross_PnL'].sum()
        total_costs = df['Total_Costs'].sum()
        cost_impact = (total_costs / total_gross_pnl * 100) if total_gross_pnl != 0 else 0
        print(f"Total Gross P&L: ${total_gross_pnl:,.2f}")
        print(f"Total Transaction Costs: ${total_costs:,.2f}")
        print(f"Cost Impact: {cost_impact:.1f}%")
        print(f"Average Cost per Trade: ${df['Total_Costs'].mean():.2f}")

    return df

def create_trade_visualizations(df):
    """
    Create visualizations for trade analysis
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Fear & Greed Strategy - Trade Analysis Dashboard', fontsize=16, fontweight='bold')
    
    # 1. Cumulative P&L over time
    df_sorted = df.sort_values('Entry_Date')
    df_sorted['Cumulative_PnL'] = df_sorted['Net_PnL'].cumsum()
    
    axes[0,0].plot(df_sorted['Entry_Date'], df_sorted['Cumulative_PnL'], linewidth=2, color='green')
    axes[0,0].set_title('Cumulative P&L Over Time')
    axes[0,0].set_ylabel('Cumulative P&L ($)')
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].tick_params(axis='x', rotation=45)
    
    # 2. Trade returns distribution
    axes[0,1].hist(df['Net_Percent_Return'], bins=30, alpha=0.7, color='blue', edgecolor='black')
    axes[0,1].axvline(df['Net_Percent_Return'].mean(), color='red', linestyle='--', label=f'Mean: {df["Net_Percent_Return"].mean():.1f}%')
    axes[0,1].set_title('Distribution of Trade Returns')
    axes[0,1].set_xlabel('Net Return (%)')
    axes[0,1].set_ylabel('Frequency')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # 3. Win rate by holding period
    hold_analysis = df.groupby('Holding_Period_Days')['Win_Loss_Binary'].agg(['count', 'mean']).reset_index()
    hold_analysis = hold_analysis[hold_analysis['count'] >= 3]  # Only periods with 3+ trades
    
    scatter = axes[0,2].scatter(hold_analysis['Holding_Period_Days'], hold_analysis['mean'], 
                               s=hold_analysis['count']*10, alpha=0.6, c=hold_analysis['mean'], cmap='RdYlGn')
    axes[0,2].set_title('Win Rate by Holding Period')
    axes[0,2].set_xlabel('Holding Period (Days)')
    axes[0,2].set_ylabel('Win Rate')
    axes[0,2].grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=axes[0,2], label='Win Rate')
    
    # 4. Monthly performance
    monthly_perf = df.groupby('Month')['Net_PnL'].sum()
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    bars = axes[1,0].bar(range(1, 13), [monthly_perf.get(i, 0) for i in range(1, 13)], 
                        color=['green' if monthly_perf.get(i, 0) > 0 else 'red' for i in range(1, 13)])
    axes[1,0].set_title('Monthly P&L Performance')
    axes[1,0].set_xlabel('Month')
    axes[1,0].set_ylabel('Total P&L ($)')
    axes[1,0].set_xticks(range(1, 13))
    axes[1,0].set_xticklabels(month_names)
    axes[1,0].grid(True, alpha=0.3)
    
    # 5. Fear & Greed vs Performance
    axes[1,1].scatter(df['Entry_Fear_Greed'], df['Net_Percent_Return'], 
                     c=df['Win_Loss_Binary'], cmap='RdYlGn', alpha=0.6)
    axes[1,1].set_title('Entry Fear & Greed vs Trade Performance')
    axes[1,1].set_xlabel('Entry Fear & Greed Index')
    axes[1,1].set_ylabel('Net Return (%)')
    axes[1,1].grid(True, alpha=0.3)
    axes[1,1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    # 6. Yearly win rate trend
    yearly_wr = df.groupby('Year')['Win_Loss_Binary'].mean()
    axes[1,2].plot(yearly_wr.index, yearly_wr.values, marker='o', linewidth=2, markersize=8, color='purple')
    axes[1,2].set_title('Win Rate Trend by Year')
    axes[1,2].set_xlabel('Year')
    axes[1,2].set_ylabel('Win Rate')
    axes[1,2].grid(True, alpha=0.3)
    axes[1,2].set_ylim(0, 1)
    
    # Format y-axis as percentage
    axes[1,2].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    
    plt.tight_layout()
    plt.savefig('trade_analysis_dashboard.png', dpi=300, bbox_inches='tight')
    plt.show()

# Run the analysis
print("Loading and analyzing trade log...")
trade_df = analyze_trade_log("fear_greed_strategy_trades_2019_2025.csv")

print("\nCreating visualizations...")
create_trade_visualizations(trade_df)

print(f"\n✅ Analysis complete!")
print(f"📊 Dashboard saved as: trade_analysis_dashboard.png")
print(f"📈 CSV files ready for further analysis in Excel/Python")