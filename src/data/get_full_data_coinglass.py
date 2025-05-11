import pandas as pd
import requests
import time
from typing import Dict, List, Tuple, Optional
import numpy as np
from pathlib import Path
import pandas as pd
import os
import glob
import re
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt




class CoinGlassAPI:
    def __init__(self, api_key: str):
        self.base_url = "https://open-api-v3.coinglass.com"
        self.headers = {
            "accept": "application/json",
            "CG-API-KEY": api_key
        }

    def get_supported_tickers(
        self,
        base_assets: List[str] = ["BTC", "ETH"],
        exclude_list: List[str] = ["XBT"],
        futures=False,
    ) -> pd.DataFrame:
        """
        Fetches the supported trading pairs for each exchange and returns a mapping of
        exchanges to their BTC and ETH tickers.
        
        Parameters:
            api_key: str - Your CoinGlass API key.
            base_assets: List[str] - The base assets you want to track (default is ['BTC', 'ETH']).
            
        Returns:
            Dict[str, Dict[str, str]] - e.g., {'Binance': {'BTC': 'BTCUSDT', 'ETH': 'ETHUSDT'}}
        """
        if futures: 
            endpoint = f"{self.base_url}/api/futures/supported-exchange-pairs"
        else:
            endpoint = f"{self.base_url}/api/spot/supported-exchange-pairs"
        
        try:
            response = requests.get(endpoint, headers=self.headers)
            response.raise_for_status()
            data = response.json()

            if data["code"] != "0":
                # raise ValueError(f"API Error: {data['msg']}")
                print(f"API Error: {data['msg']}")
                return pd.DataFrame()
            
            
            # Flatten the nested dict into a list of records
            records = []
            for exchange_name, data in data["data"].items():
                for pair in data:
                    if pair['baseAsset'] in base_assets and pair['instrumentId'] not in exclude_list:
                        records.append({
                            'exchange': exchange_name,
                            'baseAsset': pair['baseAsset'],
                            'instrumentId': pair['instrumentId']
                        })

            # Create DataFrame and filter
            df = pd.DataFrame(records)
            return df

        except requests.exceptions.RequestException as e:
            print(f"Request error: {e}")
            return pd.DataFrame()

    def get_full_spot_prices(self, symbol: str, start_str: str, end_str: str, exchange: str, interval: str, limit: int) -> pd.DataFrame:
        """
        Downloads historical OHLC spot price data for a given symbol from CoinGlass API.
        """
        print(f"Downloading spot price data for {symbol} on {exchange}...")
        # Convert dates to timestamps
        start_ts = int(pd.to_datetime(start_str).timestamp()) #-9223372036854776
        end_ts = int(pd.to_datetime(end_str).timestamp()) #9223372036854776

        endpoint = f"{self.base_url}/api/price/ohlc-history?exchange={exchange}&symbol={symbol}&type=spot&interval={interval}&limit={limit}&startTime={start_ts}&endTime={end_ts}"
   
        try:
            response = requests.get(endpoint, headers=self.headers)
            response.raise_for_status()
            data = response.json()
            
            if data['code'] != '0':
                # raise ValueError(f"API Error: {data['msg']}")
                print(f"API Error: {data['msg']}")
                return pd.DataFrame()
                    
            # Process the OHLC data
            ohlc_data = data['data']
            if not ohlc_data:
                print(f"No spot data available for {symbol} on {exchange}")
                return pd.DataFrame()
                
            df = pd.DataFrame(ohlc_data, columns=['t', 'o', 'h', 'l', 'c', 'v'])
            
            # Convert timestamp and format data
            df['date'] = pd.to_datetime(df['t'], unit='s')
            df["timestamp"] = df['t']
            df["SP_open"] = df["o"].astype(float)
            df["SP_close"] = df["c"].astype(float)
            df["SP_high"] = df["h"].astype(float)
            df["SP_low"] = df["l"].astype(float)
            df["SP_vol"] = df["v"].astype(float)
            df = df[['date', 'timestamp', 'SP_open', 'SP_close', 'SP_high', 'SP_low', 'SP_vol']]
            
            return df
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching spot data for {symbol} on {exchange}: {e}")
            return pd.DataFrame()

    def get_full_funding_rates(self, symbol: str, start_str: str, end_str: str, exchange: str, interval: str, limit: int) -> pd.DataFrame:
        """
        Downloads historical OHLC funding rate data for a given symbol from CoinGlass API.
        """
        print(f"Downloading funding history rate data for {symbol} on {exchange}...")
        # Convert dates to timestamps
        start_ts = int(pd.to_datetime(start_str).timestamp()) #-9223372036854776
        end_ts = int(pd.to_datetime(end_str).timestamp()) #9223372036854776

        endpoint = f"{self.base_url}/api/futures/fundingRate/ohlc-history?exchange={exchange}&symbol={symbol}&type=futures&interval={interval}&limit={limit}&startTime={start_ts}&endTime={end_ts}"
   
        try:
            response = requests.get(endpoint, headers=self.headers)
            response.raise_for_status()
            data = response.json()
            
            if data['code'] != '0':
                # raise ValueError(f"API Error: {data['msg']}")
                print(f"API Error: {data['msg']}")
                return pd.DataFrame()
                
            # Process the OHLC data
            ohlc_data = data['data']
            if not ohlc_data:
                print(f"No funding rate data available for {symbol} on {exchange}")
                return pd.DataFrame()
                
            df = pd.DataFrame(ohlc_data, columns=['t', 'o', 'h', 'l', 'c'])
            
            # Convert timestamp and format data
            df['date'] = pd.to_datetime(df['t'], unit='s')
            df["timestamp"] = df['t']
            df["FR_open"] = df["o"].astype(float)
            df["FR_close"] = df["c"].astype(float)
            df["FR_high"] = df["h"].astype(float)
            df["FR_low"] = df["l"].astype(float)
            df = df[['date', 'timestamp', 'FR_open', 'FR_close', 'FR_high', 'FR_low']]
            
            return df
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching funding rate data for {symbol} on {exchange}: {e}")
            return pd.DataFrame()

def load_ticker_mappings() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load ticker mappings from CSV files
    
    Returns:
        Tuple of DataFrames: (spot_tickers, perp_tickers)
    """

    # Path to *this script* (get_data_coinglass.py)
    script_dir = Path(__file__).resolve().parent

    # Paths to CSVs relative to the script
    spot_path = script_dir / 'tickers' / 'tickers_spot.csv'
    perp_path = script_dir / 'tickers' / 'tickers_perps.csv'

    try:
        spot_tickers = pd.read_csv(spot_path)
        perp_tickers = pd.read_csv(perp_path)
        return spot_tickers, perp_tickers
    except Exception as e:
        print(f"Error loading ticker mappings: {e}")
        return pd.DataFrame(), pd.DataFrame()

def load_data_for_exchange_pair(
    api: CoinGlassAPI,
    spot_exchange: str,
    spot_symbol: str,
    perp_exchange: str,
    perp_symbol: str,
    start_str: str,
    end_str: str,
    interval: str,
    limit: int,
    save_files: bool = False
) -> pd.DataFrame:
    """
    Loads spot prices and funding rates for a given exchange pair and merges them.
    
    Parameters:
        api: CoinGlassAPI - API instance
        spot_exchange: str - Exchange for spot data
        spot_symbol: str - Symbol for spot data
        perp_exchange: str - Exchange for perpetual futures data
        perp_symbol: str - Symbol for perpetual futures data
        start_str: str - Start date
        end_str: str - End date
        interval: str - Time interval
        limit: int - Data point limit
        save_files: bool - Whether to save intermediate files
        
    Returns:
        DataFrame with merged spot and funding data
    """
    # Get spot prices
    spot_df = api.get_full_spot_prices(spot_symbol, start_str, end_str, spot_exchange, interval, limit) 
    
    if spot_df.empty:
        print(f"No spot data available for {spot_symbol} on {spot_exchange}")
        return pd.DataFrame()
    
    if save_files:
        sanitized_symbol = spot_symbol.replace('/', '')  # Remove '/' from the symbol
        spot_df.to_csv(f'./data/spot_prices/{spot_exchange}_{sanitized_symbol}_{interval}_spot_prices.csv', index=False)
    
    # Get funding rates
    funding_df = api.get_full_funding_rates(perp_symbol, start_str, end_str, perp_exchange, interval, limit)
    
    if funding_df.empty:
        print(f"No funding rate data available for {perp_symbol} on {perp_exchange}")
        return pd.DataFrame()
    
    if save_files:
        funding_df.to_csv(f'./data/funding_rates/{perp_exchange}_{perp_symbol}_{interval}_funding_rates.csv', index=False)

    # Add exchange identifiers
    spot_df['spot_exchange'] = spot_exchange
    spot_df['spot_symbol'] = spot_symbol
    funding_df['perp_exchange'] = perp_exchange
    funding_df['perp_symbol'] = perp_symbol
    
    # Merge on timestamp (retain all relevant columns)
    merged_df = pd.merge(
        spot_df,
        funding_df,
        on=['date', 'timestamp'],
        how='inner'
    )
    
    if merged_df.empty:
        print(f"No matching timestamps between spot and funding data for {spot_exchange}/{perp_exchange}")
        return pd.DataFrame()
    
    # Include all columns in the merged DataFrame
    merged_df = merged_df[[
        'date', 'timestamp', 'SP_open', 'SP_close', 'SP_high', 'SP_low', 'SP_vol', 'spot_exchange', 'spot_symbol', 
        'FR_open', 'FR_close', 'FR_high', 'FR_low', 'perp_exchange', 'perp_symbol'
    ]]
    
    return merged_df

def load_future_data_for_exchange_pair(
    api: CoinGlassAPI,
    perp_exchange: str,
    perp_symbol: str,
    start_str: str,
    end_str: str,
    interval: str,
    limit: int,
    save_files: bool = False
) -> pd.DataFrame:
    """
    Loads funding rates for a given exchange pair and merges them.
    
    Parameters:
        api: CoinGlassAPI - API instance
        perp_exchange: str - Exchange for perpetual futures data
        perp_symbol: str - Symbol for perpetual futures data
        start_str: str - Start date
        end_str: str - End date
        interval: str - Time interval
        limit: int - Data point limit
        save_files: bool - Whether to save intermediate files
        
    Returns:
        DataFrame with funding data
    """
    
    # Get funding rates
    funding_df = api.get_full_funding_rates(perp_symbol, start_str, end_str, perp_exchange, interval, limit)
    
    if funding_df.empty:
        print(f"No funding rate data available for {perp_symbol} on {perp_exchange}")
        return pd.DataFrame()
    
    # Add exchange identifiers
    funding_df['perp_exchange'] = perp_exchange
    funding_df['perp_symbol'] = perp_symbol

    if save_files:
        funding_df.to_csv(f'./data/funding_rates/{perp_exchange}_{perp_symbol}_{interval}_funding_rates.csv', index=False)
    
    # # Merge on timestamp (retain all relevant columns)
    # merged_df = pd.merge(
    #     spot_df,
    #     funding_df,
    #     on=['date', 'timestamp'],
    #     how='inner'
    # )
    
    # if merged_df.empty:
    #     print(f"No matching timestamps between spot and funding data for {spot_exchange}/{perp_exchange}")
    #     return pd.DataFrame()
    
    # Include all columns in the merged DataFrame
    # merged_df = merged_df[[
    #     'date', 'timestamp', 'FR_open', 'FR_close', 'FR_high', 'FR_low',
    #     'spot_exchange', 'spot_symbol', 'perp_exchange', 'perp_symbol'
    # ]]
    
    # return merged_df

    return funding_df

def load_spot_data_for_exchange_pair(
    api: CoinGlassAPI,
    spot_exchange: str,
    spot_symbol: str,
    start_str: str,
    end_str: str,
    interval: str,
    limit: int,
    save_files: bool = False
) -> pd.DataFrame:
    """
    Loads funding rates for a given exchange pair and merges them.
    
    Parameters:
        api: CoinGlassAPI - API instance
        perp_exchange: str - Exchange for perpetual futures data
        perp_symbol: str - Symbol for perpetual futures data
        start_str: str - Start date
        end_str: str - End date
        interval: str - Time interval
        limit: int - Data point limit
        save_files: bool - Whether to save intermediate files
        
    Returns:
        DataFrame with funding data
    """
    
    # Get spot prices
    spot_df = api.get_full_spot_prices(spot_symbol, start_str, end_str, spot_exchange, interval, limit)
    
    if spot_df.empty:
        print(f"No spot price data available for {spot_symbol} on {spot_exchange}")
        return pd.DataFrame()
    
    # Add exchange identifiers
    spot_df['spot_exchange'] = spot_exchange
    spot_df['spot_symbol'] = spot_symbol

    if save_files:
        sanitized_symbol = spot_symbol.replace('/', '')  # Remove '/' from the symbol
        spot_df.to_csv(f'./data/spot_prices/{spot_exchange}_{sanitized_symbol}_{interval}_spot_prices.csv', index=False)
    
    # # Merge on timestamp (retain all relevant columns)
    # merged_df = pd.merge(
    #     spot_df,
    #     funding_df,
    #     on=['date', 'timestamp'],
    #     how='inner'
    # )
    
    # if merged_df.empty:
    #     print(f"No matching timestamps between spot and funding data for {spot_exchange}/{perp_exchange}")
    #     return pd.DataFrame()
    
    # Include all columns in the merged DataFrame
    # merged_df = merged_df[[
    #     'date', 'timestamp', 'FR_open', 'FR_close', 'FR_high', 'FR_low',
    #     'spot_exchange', 'spot_symbol', 'perp_exchange', 'perp_symbol'
    # ]]
    
    # return merged_df

    return spot_df

def compute_funding_performance_multi_exchange(
    api_key: str,
    start_str: str,
    end_str: str,
    interval: str,
    limit: int,
    position_size: float = 1,
    save_files: bool = False
) -> pd.DataFrame:
    """
    Computes the funding performance across multiple exchanges using ticker mappings from CSV files.
    
    Parameters:
        api_key: str - Your CoinGlass API key
        start_str: str - Start datetime as string
        end_str: str - End datetime as string
        interval: str - Time interval (e.g., '1h')
        limit: int - Number of data points
        position_size: float - Position size for PnL calculation
        save_files: bool - Whether to save intermediate files

    Returns:
        A DataFrame with performance metrics across all exchanges
    """
    api = CoinGlassAPI(api_key)
    
    # Load ticker mappings
    spot_tickers, perp_tickers = load_ticker_mappings()
    
    if spot_tickers.empty or perp_tickers.empty:
        print("Error: Could not load ticker mappings")
        return pd.DataFrame()
    
    results = []
    
    # For each perpetual futures exchange/ticker
    for _, perp_row in perp_tickers.iterrows():
        perp_exchange = perp_row['exchange']
        perp_symbol = perp_row['futuresInstrumentId']
        perp_base_asset = perp_row['baseAsset']
        
        print(f"\nProcessing perpetual futures on {perp_exchange} for {perp_symbol}")
        
        # Get funding rates for this perpetual
        funding_df = api.get_full_funding_rates(
            perp_symbol, 
            start_str, 
            end_str, 
            perp_exchange, 
            interval, 
            limit
        )
        
        if funding_df.empty:
            print(f"No funding rate data available for {perp_symbol} on {perp_exchange}")
            continue
            
        funding_df['perp_exchange'] = perp_exchange
        funding_df['perp_symbol'] = perp_symbol
        funding_df['baseAsset'] = perp_base_asset
        
        # Find spot exchanges with data for the same base asset
        matching_spots = spot_tickers[spot_tickers['baseAsset'] == perp_base_asset]
        
        for _, spot_row in matching_spots.iterrows():
            spot_exchange = spot_row['exchange']
            spot_symbol = spot_row['spotInstrumentId']
            
            print(f"Pairing with spot data from {spot_exchange} for {spot_symbol}")
            
            # Get spot prices
            spot_df = api.get_full_spot_prices(
                spot_symbol, 
                start_str, 
                end_str, 
                spot_exchange, 
                interval, 
                limit
            )
            
            if spot_df.empty:
                print(f"No spot data available for {spot_symbol} on {spot_exchange}")
                continue
                
            spot_df['spot_exchange'] = spot_exchange
            spot_df['spot_symbol'] = spot_symbol
            
            # Merge on timestamp
            merged_df = pd.merge(spot_df, funding_df, on=['date', 'timestamp'], how='inner')
            
            if merged_df.empty:
                print(f"No matching timestamps between spot and funding data for {spot_exchange}/{perp_exchange}")
                continue
                
            # Compute funding PnL
            merged_df['fundingPnL'] = merged_df['fundingRate'] * position_size
            merged_df['cumulativeFundingPnL'] = merged_df['fundingPnL'].cumsum()
            
            # Compute percentage return based on initial spot value
            initial_value = merged_df['SP_close'].iloc[0] * position_size
            merged_df['cumulativeReturnPct'] = 100 * merged_df['cumulativeFundingPnL'] / initial_value
            
            # Add to results
            results.append(merged_df)
            
            if save_files:
                merged_df.to_csv(f'{spot_exchange}_{perp_exchange}_{perp_base_asset}.csv', index=False)
    
    if not results:
        print("No valid data pairs found across exchanges")
        return pd.DataFrame()
        
    # Combine all results
    combined_df = pd.concat(results, ignore_index=True)
    
    return combined_df


def calculate_performance_metrics(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    """
    Calculate performance metrics for each exchange pair
    
    Parameters:
        df: DataFrame - Combined performance data
        
    Returns:
        DataFrame with performance metrics
    """
    if df.empty:
        return pd.DataFrame()
    
    metrics = []
    
    # Group by spot and perp exchange pairs
    for (spot_exchange, spot_symbol, perp_exchange, perp_symbol), group in df.groupby(['spot_exchange', 'spot_symbol', 'perp_exchange', 'perp_symbol']):
        # Calculate metrics
        total_return = group['cumulativeReturnPct'].iloc[-1]
        annualized_return = total_return * (365 / (group['date'].max() - group['date'].min()).days)
        volatility = group['fundingPnL'].std() * np.sqrt(365 / interval_to_days(interval))
        sharpe = annualized_return / volatility if volatility != 0 else 0
        max_drawdown = calculate_max_drawdown(group['cumulativeReturnPct'])
        
        metrics.append({
            'spot_exchange': spot_exchange,
            'spot_symbol': spot_symbol,
            'perp_exchange': perp_exchange,
            'perp_symbol': perp_symbol,
            'total_return_pct': total_return,
            'annualized_return_pct': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe,
            'max_drawdown_pct': max_drawdown,
            'data_points': len(group)
        })
    
    return pd.DataFrame(metrics)

def interval_to_days(interval: str) -> float:
    """Convert interval string to number of days"""
    if interval.endswith('h'):
        return float(interval[:-1]) / 24
    elif interval.endswith('d'):
        return float(interval[:-1])
    elif interval.endswith('m'):
        return float(interval[:-1]) * 30  # Approximate
    else:
        return 1  # Default to daily

def calculate_max_drawdown(returns: pd.Series) -> float:
    """Calculate maximum drawdown from a series of returns"""
    cumulative = returns.cummax()
    drawdown = (returns - cumulative) / cumulative
    return drawdown.min() * 100 if not drawdown.empty else 0

def analyze_venue_switching_strategy(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze if switching venues for the short futures position would be beneficial
    
    Parameters:
        metrics_df: DataFrame - Performance metrics by exchange pair
        
    Returns:
        DataFrame with analysis results
    """
    if metrics_df.empty:
        return pd.DataFrame()
    
    # Group by spot exchange and find best perpetual exchange for each period
    spot_exchanges = metrics_df['spot_exchange'].unique()
    
    results = []
    for spot_ex in spot_exchanges:
        spot_data = metrics_df[metrics_df['spot_exchange'] == spot_ex]
        
        # Find best perpetual exchange based on Sharpe ratio
        best_perp = spot_data.loc[spot_data['sharpe_ratio'].idxmax()]
        
        # Compare with average performance
        avg_performance = spot_data.mean()
        
        results.append({
            'spot_exchange': spot_ex,
            'best_perp_exchange': best_perp['perp_exchange'],
            'best_perp_sharpe': best_perp['sharpe_ratio'],
            'avg_perp_sharpe': avg_performance['sharpe_ratio'],
            'sharpe_improvement': best_perp['sharpe_ratio'] - avg_performance['sharpe_ratio'],
            'best_perp_return': best_perp['annualized_return_pct'],
            'avg_perp_return': avg_performance['annualized_return_pct'],
            'return_improvement': best_perp['annualized_return_pct'] - avg_performance['annualized_return_pct']
        })
    
    return pd.DataFrame(results)



#Added by Loris to compute carry trade performance
def merge_and_compute_carry_trade(
    spot_df: pd.DataFrame,
    futures_df: pd.DataFrame,
    merge_on: str = 'timestamp'
) -> pd.DataFrame:
    """
    Merge spot and futures dataframes and compute delta-neutral carry trade performance.
    Strategy: Long spot, Short futures (equal notional amounts)
    
    Parameters:
        spot_df: pd.DataFrame with columns: date, timestamp, SP_open, SP_close, etc.
        futures_df: pd.DataFrame with columns: date, timestamp, FR_open, FR_close, etc.
        merge_on: str - Column to merge on ('date' or 'timestamp')
            
    Returns:
        pd.DataFrame: Merged DataFrame with carry trade performance metrics
    """
    # Ensure timestamp is numeric in both dataframes
    if merge_on == 'timestamp':
        spot_df['timestamp'] = pd.to_numeric(spot_df['timestamp'], errors='coerce')
        futures_df['timestamp'] = pd.to_numeric(futures_df['timestamp'], errors='coerce')
    
    # Sort and remove duplicates
    spot_df = spot_df.sort_values(by=merge_on).drop_duplicates(subset=[merge_on])
    futures_df = futures_df.sort_values(by=merge_on).drop_duplicates(subset=[merge_on])
    
    # Merge dataframes
    merged_df = pd.merge(spot_df, futures_df, on=merge_on, how='inner')
    
    # Handle date columns
    if 'date_x' in merged_df.columns and 'date_y' in merged_df.columns:
        merged_df['date'] = pd.to_datetime(merged_df['date_x'])
        merged_df = merged_df.drop(columns=['date_x', 'date_y'])
    elif 'date' in merged_df.columns:
        merged_df['date'] = pd.to_datetime(merged_df['date'])
    
    # Initial settings
    position_size = 1.0
    entry_price = merged_df['SP_close'].iloc[0]
    exit_price = merged_df['SP_close'].iloc[-1]
    initial_investment = entry_price * position_size
    
    # Store entry and exit prices
    merged_df['entry_price'] = entry_price
    merged_df['exit_price'] = exit_price
    
    # Track positions - in delta neutral, we have equal but opposite notional values
    merged_df['spot_position_value'] = merged_df['SP_close'] * position_size
    merged_df['perp_position_value'] = -merged_df['SP_close'] * position_size  # Short position
    
    # Calculate daily PnL components:
    # 1. For a delta-neutral strategy, price movements should net to zero 
    #    when perfectly balanced (which we're assuming)
    
    # 2. Funding payments - positive funding means we receive when short
    #    These funding rates are daily percentages (e.g., 0.01 means 1% per day)
    merged_df['FR_close_pct'] = merged_df['FR_close'] / 100  # Convert to decimal (e.g., 0.01 -> 0.0001)
    merged_df['funding_pnl'] = merged_df['FR_close_pct'] * initial_investment
    
    # Calculate returns
    merged_df['funding_cumulative_pnl'] = merged_df['funding_pnl'].cumsum()
    merged_df['funding_return'] = merged_df['funding_cumulative_pnl'] / initial_investment
    
    # In a perfect delta-neutral strategy, the total return equals the funding return
    merged_df['total_return'] = merged_df['funding_return']
    
    # For display purposes, calculate the isolated returns of each leg
    merged_df['spot_pct_change'] = merged_df['SP_close'].pct_change().fillna(0)
    merged_df['spot_return'] = (merged_df['SP_close'] / entry_price) - 1
    merged_df['perp_return'] = -merged_df['spot_return']  # Perfectly offsetting in theory
    
    return merged_df

def calculate_period_statistics(df: pd.DataFrame) -> dict:
    """
    Calculate summary statistics for the delta-neutral carry trade performance.
    
    Parameters:
        df: pd.DataFrame with carry trade performance metrics
            
    Returns:
        dict: Dictionary containing summary statistics
    """
    # Calculate total returns
    total_return = df['total_return'].iloc[-1] * 100
    funding_return = df['funding_return'].iloc[-1] * 100
    spot_return = df['spot_return'].iloc[-1] * 100
    perp_return = df['perp_return'].iloc[-1] * 100
    
    # Get entry and exit prices
    entry_price = df['entry_price'].iloc[0]
    exit_price = df['exit_price'].iloc[-1]
    
    # Calculate time period in years
    days = (df['date'].max() - df['date'].min()).days
    years = days / 365
    
    # Calculate annualized return
    annualized_return = ((1 + total_return/100) ** (1/years) - 1) * 100 if years > 0 else 0
    
    # Calculate volatility of daily funding returns
    daily_funding_returns = df['funding_pnl'].diff().fillna(df['funding_pnl'].iloc[0]) / (entry_price * 1)
    volatility = daily_funding_returns.std() * np.sqrt(252) * 100  # Annualized volatility
    
    # Calculate Sharpe ratio (assuming 0% risk-free rate)
    avg_daily_funding_return = daily_funding_returns.mean()
    annualized_mean_return = avg_daily_funding_return * 252
    sharpe_ratio = annualized_mean_return / (daily_funding_returns.std() * np.sqrt(252)) if daily_funding_returns.std() > 0 else 0
    
    # Calculate maximum drawdown
    cumulative_return_series = 1 + (df['total_return'])
    peak = cumulative_return_series.cummax()
    drawdown = (cumulative_return_series - peak) / peak
    max_drawdown = drawdown.min() * 100 if not drawdown.empty else 0
    
    return {
        'total_return_pct': total_return,
        'annualized_return_pct': annualized_return,
        'volatility_pct': volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown_pct': max_drawdown,
        'funding_return_pct': funding_return,
        'spot_return_pct': spot_return,
        'perp_return_pct': perp_return,
        'entry_price': entry_price,
        'exit_price': exit_price,
        'days': days,
        'start_date': df['date'].min(),
        'end_date': df['date'].max()
    }

def plot_carry_trade_performance(merged_df: pd.DataFrame, stats: dict):
    """
    Plot the carry trade performance metrics.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), height_ratios=[2, 1])
    
    # Plot cumulative returns
    ax1.plot(merged_df['date'], merged_df['total_return'] * 100, 
             label=f'Total Return ({stats["total_return_pct"]:.2f}%)', color='blue', linewidth=2)
    ax1.plot(merged_df['date'], merged_df['funding_return'] * 100, 
             label=f'Funding Return ({stats["funding_return_pct"]:.2f}%)', color='green', alpha=0.7)
    
    # Plot spot and perp returns (for illustrative purposes - they should offset in theory)
    ax1.plot(merged_df['date'], merged_df['spot_return'] * 100, 
             label=f'Spot Return ({stats["spot_return_pct"]:.2f}%)', color='orange', linestyle='--', alpha=0.5)
    ax1.plot(merged_df['date'], merged_df['perp_return'] * 100, 
             label=f'Perp Return ({stats["perp_return_pct"]:.2f}%)', color='red', linestyle='--', alpha=0.5)
    
    ax1.set_title('Delta-Neutral Carry Trade Performance', fontsize=12)
    ax1.set_xlabel('')
    ax1.set_ylabel('Cumulative Return (%)')
    ax1.legend()
    ax1.grid(True)
    
    # Plot funding rates - display as percentages
    ax2.plot(merged_df['date'], merged_df['FR_close'], 
             label='Daily Funding Rate (%)', color='red')
    ax2.set_title('Daily Funding Rates Over Time (%)', fontsize=12)
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Daily Funding Rate (%)')
    ax2.legend()
    ax2.grid(True)
    
    # Add performance metrics as text
    metrics_text = (
        f'Annualized Return: {stats["annualized_return_pct"]:.2f}%\n'
        f'Volatility: {stats["volatility_pct"]:.2f}%\n'
        f'Sharpe Ratio: {stats["sharpe_ratio"]:.2f}\n'
        f'Max Drawdown: {stats["max_drawdown_pct"]:.2f}%\n'
        f'Funding Return: {stats["funding_return_pct"]:.2f}%\n'
        f'Spot Return: {stats["spot_return_pct"]:.2f}%\n'
        f'Perp Return: {stats["perp_return_pct"]:.2f}%\n'
        f'Entry Price: ${stats["entry_price"]:.2f}\n'
        f'Exit Price: ${stats["exit_price"]:.2f}\n'
        f'Period: {stats["start_date"].strftime("%Y-%m-%d")} to {stats["end_date"].strftime("%Y-%m-%d")} ({stats["days"]} days)'
    )
    plt.figtext(0.02, 0.02, metrics_text, fontsize=10, 
                bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()

def separate_crypto_data(df, output_folder='src/data/concat'):
    """
    Separates mixed crypto data into separate BTC and ETH dataframes based on symbol column.
    
    Parameters:
        df: pandas DataFrame containing mixed BTC and ETH data
        output_folder: Root folder to save the separated data
        
    Returns:
        tuple: (btc_df, eth_df) - The separated dataframes
    """
    # Check which symbol column is present
    symbol_column = None
    if 'spot_symbol' in df.columns:
        symbol_column = 'spot_symbol'
    elif 'perp_symbol' in df.columns:
        symbol_column = 'perp_symbol'
    else:
        raise ValueError("No symbol column (spot_symbol or perp_symbol) found in dataframe")
    
    # Create a mask to identify ETH data
    eth_mask = df[symbol_column].str.contains('ETH', case=False)
    
    # Create a mask to identify BTC data - including XBT variations
    btc_mask = (df[symbol_column].str.contains('BTC', case=False) | 
                df[symbol_column].str.contains('XBT', case=False))
    
    # Extract ETH and BTC data
    eth_df = df[eth_mask].copy()
    btc_df = df[btc_mask].copy()
    
    # Create output directories if they don't exist
    os.makedirs(f"{output_folder}/BTC", exist_ok=True)
    os.makedirs(f"{output_folder}/ETH", exist_ok=True)
    
    # Generate filenames based on the type of data
    name_suffix = ''
    if 'SP_close' in df.columns:
        name_suffix = 'spot_data'
    elif 'FR_close' in df.columns:
        name_suffix = 'futures_data'
    else:
        name_suffix = 'data'
    
    # Get date range for filename
    start_date = df['date'].min().split(' ')[0] if isinstance(df['date'].iloc[0], str) else df['date'].min().strftime('%Y-%m-%d')
    end_date = df['date'].max().split(' ')[0] if isinstance(df['date'].iloc[0], str) else df['date'].max().strftime('%Y-%m-%d')
    
    # Save the separated dataframes
    btc_filename = f"{output_folder}/BTC/btc_{name_suffix}_{start_date}_to_{end_date}.csv"
    eth_filename = f"{output_folder}/ETH/eth_{name_suffix}_{start_date}_to_{end_date}.csv"
    
    btc_df.to_csv(btc_filename, index=False)
    eth_df.to_csv(eth_filename, index=False)
    
    print(f"BTC data saved to {btc_filename}")
    print(f"ETH data saved to {eth_filename}")
    
    # Return the separate dataframes
    return btc_df, eth_df

def process_data_files(spot_file_path, futures_file_path):
    """
    Process spot and futures data files, separating them into BTC and ETH datasets
    
    Parameters:
        spot_file_path: Path to the spot data CSV file
        futures_file_path: Path to the futures data CSV file
    """
    # Read the data
    spot_df = pd.read_csv(spot_file_path)
    futures_df = pd.read_csv(futures_file_path)
    
    # Convert date to datetime if not already
    if not pd.api.types.is_datetime64_dtype(spot_df['date']):
        spot_df['date'] = pd.to_datetime(spot_df['date'])
    if not pd.api.types.is_datetime64_dtype(futures_df['date']):
        futures_df['date'] = pd.to_datetime(futures_df['date'])
    
    # Separate the spot and futures data
    print("Separating spot data...")
    btc_spot_df, eth_spot_df = separate_crypto_data(spot_df)
    
    print("\nSeparating futures data...")
    btc_futures_df, eth_futures_df = separate_crypto_data(futures_df)
    
    print("\nData separation complete!")
    
    # Return all the dataframes
    return {
        'btc_spot': btc_spot_df,
        'eth_spot': eth_spot_df,
        'btc_futures': btc_futures_df,
        'eth_futures': eth_futures_df
    }
