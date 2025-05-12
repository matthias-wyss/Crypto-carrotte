import pandas as pd
import requests
import time
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from pathlib import Path
import pandas as pd
import os
import glob
import re
import matplotlib.pyplot as plt
from scipy import stats
from .get_full_data_coinglass import merge_and_compute_carry_trade, calculate_period_statistics



def merge_and_compute_carry_trade_with_staking(
    spot_df: pd.DataFrame,
    futures_df: pd.DataFrame,
    merge_on: str = 'timestamp',
    is_eth: bool = False,
    staking_rates_df: pd.DataFrame = None,
    default_staking_rate: float = 0.036  # Default 3.6% APY for periods without data
) -> pd.DataFrame:
    """
    Merge spot and futures dataframes and compute delta-neutral carry trade performance,
    including staking returns for ETH using actual time-varying staking rates.
    
    Strategy: 
    - Long spot ETH/BTC
    - Short futures ETH/BTC (equal notional amounts)
    - For ETH: Earn staking rewards on spot position based on actual Lido staking rates
    """
    # First, get the base carry trade calculations
    merged_df = merge_and_compute_carry_trade(spot_df, futures_df, merge_on)
    
    if is_eth:
        # Convert date to datetime if it's not already (without timezone info)
        merged_df['date'] = pd.to_datetime(merged_df['date']).dt.tz_localize(None)
        
        # Calculate default daily staking rate (APY to daily)
        default_daily_staking_rate = (1 + default_staking_rate) ** (1/365) - 1
        
        # Initialize staking rate column with default rate
        merged_df['daily_staking_return'] = default_daily_staking_rate
        merged_df['staking_apy'] = default_staking_rate * 100  # Convert to percentage
        
        # If staking rates dataframe is provided, use actual rates
        if staking_rates_df is not None and not staking_rates_df.empty:
            # Prepare staking rates dataframe
            staking_df = staking_rates_df.copy()
            
            # Convert time column to datetime and remove timezone info
            if 'time' in staking_df.columns:
                staking_df['date'] = pd.to_datetime(staking_df['time']).dt.tz_localize(None)
            elif 'date' in staking_df.columns:
                staking_df['date'] = pd.to_datetime(staking_df['date']).dt.tz_localize(None)
            else:
                raise ValueError("Staking rates dataframe must have 'time' or 'date' column")
            
            # Check which APR column to use - prioritize instant APR
            apr_column = None
            if 'Lido staking APR(instant)' in staking_df.columns:
                apr_column = 'Lido staking APR(instant)'
            elif 'protocol_apr' in staking_df.columns:
                apr_column = 'protocol_apr'
            
            if apr_column is None:
                print("Warning: No valid APR column found in staking_rates_df. Using default rate.")
            else:
                # Convert APR to decimal (assuming it's already in percentage)
                staking_df['staking_rate_decimal'] = staking_df[apr_column] / 100
                
                # Create a simplified dataframe with just date and rate
                rate_lookup = staking_df[['date', 'staking_rate_decimal']].drop_duplicates()
                
                # Create a function to find the closest date if exact match not found
                def find_closest_rate(target_date):
                    # Make sure target_date has no timezone info
                    target_date = pd.to_datetime(target_date).tz_localize(None)
                    
                    # First check if we have an exact date match (comparing naive datetimes)
                    exact_match = rate_lookup[rate_lookup['date'].dt.date == target_date.date()]
                    if not exact_match.empty:
                        return exact_match['staking_rate_decimal'].values[0]
                    
                    # Find closest date before target date
                    before_dates = rate_lookup[rate_lookup['date'].dt.date <= target_date.date()]
                    if not before_dates.empty:
                        closest_before = before_dates.iloc[before_dates['date'].argmax()]
                        return closest_before['staking_rate_decimal']
                    
                    # If no date before, use default
                    return default_staking_rate
                
                # Apply lookup to each date in merged_df
                staking_rates = {}
                cutoff_date = pd.to_datetime('2022-09-01')
                
                for idx, row in merged_df.iterrows():
                    if row['date'] < cutoff_date:
                        # Use default rate for dates before Lido's launch
                        staking_rates[idx] = default_staking_rate
                    else:
                        # Use actual rate or closest available
                        staking_rates[idx] = find_closest_rate(row['date'])
                
                # Add staking rates to merged_df
                merged_df['staking_apy'] = pd.Series(staking_rates) * 100  # Convert to percentage
                merged_df['daily_staking_return'] = ((1 + merged_df['staking_apy']/100) ** (1/365)) - 1
        
        # Calculate staking PnL
        initial_investment = merged_df['SP_close'].iloc[0]
        merged_df['staking_pnl'] = merged_df['daily_staking_return'] * initial_investment
        
        # Calculate cumulative returns
        merged_df['cumulative_staking_return'] = (1 + merged_df['daily_staking_return']).cumprod() - 1
        
        # Combine funding and staking returns
        merged_df['total_daily_return'] = merged_df['daily_funding_return'] + merged_df['daily_staking_return']
        merged_df['total_return'] = (1 + merged_df['total_daily_return']).cumprod() - 1
        
        # Add combined APY metric
        merged_df['total_apy'] = merged_df['FR_annualized'] + merged_df['staking_apy']
    
    return merged_df

def calculate_period_statistics_with_staking(
    df: pd.DataFrame, 
    is_eth: bool = False,
    risk_free_rate: float = 0.0
) -> Dict[str, Any]:
    """
    Calculate comprehensive statistics for the delta-neutral carry trade and underlying asset,
    including staking returns for ETH.
    """
    # Get base statistics
    stats = calculate_period_statistics(df, risk_free_rate)
    
    if is_eth:
        # Add staking-specific statistics
        staking_return = df['cumulative_staking_return'].iloc[-1] * 100
        total_return = df['total_return'].iloc[-1] * 100
        
        years = stats['years']
        annualized_staking_return = ((1 + staking_return/100) ** (1/years) - 1) * 100 if years > 0 else 0
        annualized_total_return = ((1 + total_return/100) ** (1/years) - 1) * 100 if years > 0 else 0
        
        # Calculate Sharpe ratio with total returns
        daily_total_excess_returns = df['total_daily_return'] - risk_free_rate
        total_sharpe = (daily_total_excess_returns.mean() / daily_total_excess_returns.std()) * np.sqrt(252) if daily_total_excess_returns.std() > 0 else 0
        
        # Add staking metrics to stats dictionary
        stats.update({
            'staking_return_pct': staking_return,
            'annualized_staking_return_pct': annualized_staking_return,
            'total_return_pct': total_return,
            'annualized_total_return_pct': annualized_total_return,
            'total_sharpe_ratio': total_sharpe,
            'avg_staking_apy': df['staking_apy'].mean(),
            'avg_total_apy': df['total_apy'].mean(),
            'min_staking_apy': df['staking_apy'].min(),
            'max_staking_apy': df['staking_apy'].max()
        })
        
        # Market regime analysis with staking
        stats.update({
            'bull_market_staking_return': df[df['bull_market'] == 1]['daily_staking_return'].mean() * 100,
            'bear_market_staking_return': df[df['bull_market'] == 0]['daily_staking_return'].mean() * 100,
            'bull_market_total_return': df[df['bull_market'] == 1]['total_daily_return'].mean() * 100,
            'bear_market_total_return': df[df['bull_market'] == 0]['total_daily_return'].mean() * 100
        })
    
    return stats

def plot_carry_trade_performance_with_staking(merged_df: pd.DataFrame, stats: dict, is_eth: bool = True, title: str = None):
    """
    Plot comprehensive performance of the delta-neutral carry trade and underlying asset,
    including staking returns for ETH.
    """
    fig = plt.figure(figsize=(16, 12))
    spec = fig.add_gridspec(3, 2)
    
    # Plot 1: Asset price and market regimes (same as before)
    ax1 = fig.add_subplot(spec[0, :])
    ax1.plot(merged_df['date'], merged_df['SP_close'], label='Asset Price', color='blue')
    
    # Highlight bull markets and corrections
    bull_regions = []
    current_start = None
    for idx, row in merged_df.iterrows():
        if row['bull_market'] == 1 and current_start is None:
            current_start = row['date']
        elif row['bull_market'] == 0 and current_start is not None:
            bull_regions.append((current_start, row['date']))
            current_start = None
    
    if current_start is not None:
        bull_regions.append((current_start, merged_df['date'].iloc[-1]))
        
    for start, end in bull_regions:
        ax1.axvspan(start, end, alpha=0.2, color='green')
        
    # Highlight corrections
    correction_regions = []
    current_start = None
    for idx, row in merged_df.iterrows():
        if row['in_correction'] == 1 and current_start is None:
            current_start = row['date']
        elif row['in_correction'] == 0 and current_start is not None:
            correction_regions.append((current_start, row['date']))
            current_start = None
    
    if current_start is not None:
        correction_regions.append((current_start, merged_df['date'].iloc[-1]))
        
    for start, end in correction_regions:
        ax1.axvspan(start, end, alpha=0.2, color='red')
        
    ax1.set_title('Asset Price with Market Regimes' if title is None else f'{title} Price with Market Regimes', fontsize=14)
    ax1.set_ylabel('Price ($)')
    ax1.legend(['Asset Price', 'Bull Market', 'Market Correction'])
    ax1.grid(True)
    
    # Plot 2: Funding rates and staking APY
    ax2 = fig.add_subplot(spec[1, 0])
    ax2.plot(merged_df['date'], merged_df['FR_annualized'], color='purple', label='Funding APY')
    if is_eth:
        ax2.plot(merged_df['date'], merged_df['staking_apy'], color='green', label='Staking APY')
        ax2.plot(merged_df['date'], merged_df['total_apy'], color='blue', label='Total APY')
    
    # Add horizontal lines for averages
    ax2.axhline(y=stats['annualized_funding_pct'], color='purple', linestyle='--', 
                label=f'Avg Funding: {stats["annualized_funding_pct"]:.2f}%')
    
    if is_eth:
        ax2.axhline(y=stats['avg_staking_apy'], color='green', linestyle='--',
                   label=f'Avg Staking: {stats["avg_staking_apy"]:.2f}%')
        ax2.axhline(y=stats['avg_total_apy'], color='blue', linestyle='--',
                   label=f'Avg Total: {stats["avg_total_apy"]:.2f}%')
    
    ax2.set_title('Annualized Rates', fontsize=12)
    ax2.set_ylabel('Annual Rate (%)')
    ax2.legend()
    ax2.grid(True)
    
    # Plot 3: Cumulative returns comparison
    ax3 = fig.add_subplot(spec[1, 1])
    ax3.plot(merged_df['date'], merged_df['funding_return'] * 100, color='purple', label='Funding Return')
    if is_eth:
        ax3.plot(merged_df['date'], merged_df['cumulative_staking_return'] * 100, color='green', label='Staking Return')
        ax3.plot(merged_df['date'], merged_df['total_return'] * 100, color='blue', label='Total Return')
    
    ax3.set_title('Cumulative Returns', fontsize=12)
    ax3.set_ylabel('Return (%)')
    ax3.legend()
    ax3.grid(True)
    
    # Plot 4: Returns by market regime
    ax4 = fig.add_subplot(spec[2, 0])
    
    # Set width and positions for bars
    x = np.arange(2)
    width = 0.25 if is_eth else 0.4
    
    # Create bars for different return components
    funding_returns = [stats['bull_market_daily_return'], stats['bear_market_daily_return']]
    rects1 = ax4.bar(x - width if is_eth else x, funding_returns, width, label='Funding')
    
    if is_eth:
        staking_returns = [stats['bull_market_staking_return'], stats['bear_market_staking_return']]
        rects2 = ax4.bar(x, staking_returns, width, label='Staking')
        
        total_returns = [stats['bull_market_total_return'], stats['bear_market_total_return']]
        rects3 = ax4.bar(x + width, total_returns, width, label='Total')
    
    ax4.set_ylabel('Daily Return (%)')
    ax4.set_title('Returns by Market Regime')
    ax4.set_xticks(x)
    ax4.set_xticklabels(['Bull Market', 'Bear Market'])
    ax4.legend()
    
    # Add value labels to bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax4.annotate(f'{height:.2f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    autolabel(rects1)
    if is_eth:
        autolabel(rects2)
        autolabel(rects3)
    
    # Plot 5: Strategy drawdowns
    ax5 = fig.add_subplot(spec[2, 1])
    strategy_equity = (1 + merged_df['funding_return'])
    strategy_hwm = strategy_equity.cummax()
    strategy_dd = (strategy_equity - strategy_hwm) / strategy_hwm * 100
    ax5.fill_between(merged_df['date'], strategy_dd, 0, color='purple', alpha=0.3, label='Funding DD')
    
    if is_eth:
        total_equity = (1 + merged_df['total_return'])
        total_hwm = total_equity.cummax()
        total_dd = (total_equity - total_hwm) / total_hwm * 100
        ax5.fill_between(merged_df['date'], total_dd, 0, color='blue', alpha=0.3, label='Total DD')
    
    ax5.set_title('Strategy Drawdowns', fontsize=12)
    ax5.set_ylabel('Drawdown (%)')
    ax5.legend()
    ax5.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return fig

def display_statistics_table_with_staking(btc_stats, eth_stats=None):
    """
    Display statistics in a nicely formatted table, including staking data for ETH
    
    Parameters:
        btc_stats: Dictionary of BTC statistics
        eth_stats: Optional dictionary of ETH statistics for comparison
    """
    import pandas as pd
    from IPython.display import display, HTML
    
    # Define the sections and metrics to include
    sections = {
        'Period Information': [
            ('Start Date', 'start_date'),
            ('End Date', 'end_date'),
            ('Total Days', 'days'),
            ('Years', 'years')
        ],
        'Price Information': [
            ('Entry Price', 'entry_price'),
            ('Exit Price', 'exit_price')
        ],
        'Underlying Asset Performance': [
            ('Total Return', 'spot_return_pct', '%'),
            ('Annualized Return', 'spot_annualized_return_pct', '%'),
            ('Volatility (Annualized)', 'spot_volatility_pct', '%'),
            ('Maximum Drawdown', 'max_spot_drawdown_pct', '%')
        ],
        'Strategy Performance': [
            ('Total Funding Return', 'funding_return_pct', '%'),
            ('Annualized Funding Return', 'annualized_funding_return_pct', '%'),
            ('Funding Sharpe Ratio', 'sharpe_ratio'),
            ('Maximum Funding Strategy Drawdown', 'max_strategy_drawdown_pct', '%')
        ]
    }
    
    # Add ETH staking section if ETH stats are provided
    if eth_stats is not None and 'staking_return_pct' in eth_stats:
        sections['ETH Staking Performance'] = [
            ('Total Staking Return', 'staking_return_pct', '%'),
            ('Annualized Staking Return', 'annualized_staking_return_pct', '%'),
            ('Average Staking APY', 'avg_staking_apy', '%'),
            ('Min Staking APY', 'min_staking_apy', '%'),
            ('Max Staking APY', 'max_staking_apy', '%')
        ]
        sections['Combined Performance (ETH)'] = [
            ('Total Combined Return', 'total_return_pct', '%'),
            ('Annualized Combined Return', 'annualized_total_return_pct', '%'),
            ('Combined Sharpe Ratio', 'total_sharpe_ratio'),
            ('Average Combined APY', 'avg_total_apy', '%')
        ]
    
    sections.update({
        'Funding Rate Analysis': [
            ('Average Daily Funding', 'avg_daily_funding_pct', '%'),
            ('Annualized Average Funding', 'annualized_funding_pct', '%'),
            ('Minimum Daily Funding', 'min_daily_funding_pct', '%'),
            ('Maximum Daily Funding', 'max_daily_funding_pct', '%'),
            ('Annualized Min Funding', 'annualized_min_funding_pct', '%'),
            ('Annualized Max Funding', 'annualized_max_funding_pct', '%'),
            ('Negative Funding Days', 'negative_funding_days', ' days'),
            ('Negative Funding Days %', 'negative_funding_days_pct', '%'),
            ('Average Negative Funding', 'avg_negative_funding_pct', '%'),
            ('Annualized Negative Funding', 'annualized_negative_funding_pct', '%'),
            ('Max Consecutive Negative Days', 'max_consecutive_negative_days', ' days')
        ],
        'Market Regime Analysis': [
            ('Bull Market Days', 'bull_market_days', ' days'),
            ('Bull Market %', 'bull_market_pct', '%'),
            ('Bear Market Days', 'bear_market_days', ' days'),
            ('Bull Market Daily Spot Return', 'bull_market_daily_return', '%'),
            ('Bear Market Daily Spot Return', 'bear_market_daily_return', '%'),
            ('Bull Market Daily Funding', 'bull_market_funding_rate', '%'),
            ('Bear Market Daily Funding', 'bear_market_funding_rate', '%'),
            ('Annualized Bull Market Funding', 'annualized_bull_funding', '%'),
            ('Annualized Bear Market Funding', 'annualized_bear_funding', '%')
        ]
    })
    
    # Add ETH staking regime analysis if ETH stats are provided
    if eth_stats is not None and 'bull_market_staking_return' in eth_stats:
        sections['Market Regime Analysis'].extend([
            ('Bull Market Daily Staking', 'bull_market_staking_return', '%'),
            ('Bear Market Daily Staking', 'bear_market_staking_return', '%'),
            ('Bull Market Daily Total', 'bull_market_total_return', '%'),
            ('Bear Market Daily Total', 'bear_market_total_return', '%')
        ])
    
    # Create a list to store the table data
    table_data = []
    
    # Process each section
    for section, metrics in sections.items():
        # Add section header
        table_data.append([section, '' if eth_stats is None else ''])
        
        # Add each metric in the section
        for metric_info in metrics:
            if len(metric_info) == 2:
                label, key = metric_info
                suffix = ''
            else:
                label, key, suffix = metric_info
            
            btc_value = btc_stats.get(key, 'N/A')
            
            # Format the values for display
            if isinstance(btc_value, (int, float)) and key not in ['days', 'years', 'negative_funding_days', 'max_consecutive_negative_days', 'bull_market_days', 'bear_market_days', 'correction_days']:
                if suffix == '%':
                    btc_formatted = f"{btc_value:.2f}{suffix}"
                else:
                    btc_formatted = f"{btc_value:.2f}{suffix}"
            elif key in ['start_date', 'end_date']:
                btc_formatted = str(btc_value).split()[0]
            else:
                btc_formatted = f"{btc_value}{suffix}"
            
            if eth_stats is not None:
                eth_value = eth_stats.get(key, 'N/A')
                if isinstance(eth_value, (int, float)) and key not in ['days', 'years', 'negative_funding_days', 'max_consecutive_negative_days', 'bull_market_days', 'bear_market_days', 'correction_days']:
                    if suffix == '%':
                        eth_formatted = f"{eth_value:.2f}{suffix}"
                    else:
                        eth_formatted = f"{eth_value:.2f}{suffix}"
                elif key in ['start_date', 'end_date']:
                    eth_formatted = str(eth_value).split()[0]
                else:
                    eth_formatted = f"{eth_value}{suffix}"
                
                table_data.append([f"  {label}", btc_formatted, eth_formatted])
            else:
                table_data.append([f"  {label}", btc_formatted])
    
    # Create DataFrame from the table data
    if eth_stats is None:
        columns = ['Metric', 'BTC Value']
        df = pd.DataFrame(table_data, columns=columns)
    else:
        columns = ['Metric', 'BTC Value', 'ETH Value']
        df = pd.DataFrame(table_data, columns=columns)
    
    # Style the DataFrame
    styled_df = df.style.set_properties(**{'text-align': 'left'})
    
    # Add row and section highlighting
    styled_df = styled_df.apply(lambda x: ['background-color: #f2f2f2' if x.name % 2 == 0 
                                          else 'background-color: white' for i in range(len(x))], axis=1)
    
    # Highlight section headers
    def highlight_sections(x):
        is_header = x["Metric"] in sections.keys()
        return ['font-weight: bold; background-color: #d9e1f2' if is_header else '' for i in range(len(x))]
    
    styled_df = styled_df.apply(highlight_sections, axis=1)
    
    # Display the styled table
    display(HTML(styled_df.to_html()))
    
    return styled_df
