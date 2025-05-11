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
    
    # Calculate underlying asset returns and volatility
    merged_df['daily_spot_return'] = merged_df['SP_close'].pct_change().fillna(0)
    
    # Convert funding rates from percentage to decimal
    merged_df['FR_close_pct'] = merged_df['FR_close'] / 100  # Convert from % to decimal
    merged_df['FR_annualized'] = merged_df['FR_close'] * 365  # Annualized for display
    
    # Calculate strategy returns
    merged_df['funding_pnl'] = merged_df['FR_close_pct'] * initial_investment
    merged_df['daily_funding_return'] = merged_df['funding_pnl'] / initial_investment
    merged_df['funding_return'] = (1 + merged_df['daily_funding_return']).cumprod() - 1
    
    # The strategy total return equals the funding return in a perfect delta-neutral setup
    merged_df['total_return'] = merged_df['funding_return']
    
    # For reference: track what would happen to spot and perp positions separately
    merged_df['spot_return'] = (merged_df['SP_close'] / entry_price) - 1
    merged_df['perp_return'] = -merged_df['spot_return']
    
    # Detect market regimes - simple definition based on price trend
    merged_df['price_sma_20'] = merged_df['SP_close'].rolling(window=20).mean()
    merged_df['price_sma_50'] = merged_df['SP_close'].rolling(window=50).mean()
    merged_df['bull_market'] = (merged_df['price_sma_20'] > merged_df['price_sma_50']).astype(int)
    
    # Detect market corrections and recoveries
    high_watermark = merged_df['SP_close'].cummax()
    merged_df['drawdown_pct'] = (merged_df['SP_close'] - high_watermark) / high_watermark * 100
    merged_df['in_correction'] = (merged_df['drawdown_pct'] <= -10).astype(int)  # 10% or more drawdown is correction
    
    # Track when we exit a correction (start of recovery)
    merged_df['recovery_start'] = (merged_df['in_correction'].shift(1) == 1) & (merged_df['in_correction'] == 0)
    
    return merged_df

def calculate_period_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate comprehensive statistics for the delta-neutral carry trade and underlying asset.
    """
    # Calculate time period
    days = (df['date'].max() - df['date'].min()).days
    years = days / 365
    
    # Get entry and exit prices
    entry_price = df['entry_price'].iloc[0]
    exit_price = df['exit_price'].iloc[-1]
    
    # Underlying Asset Statistics
    daily_spot_returns = df['daily_spot_return'] 
    spot_volatility = daily_spot_returns.std() * np.sqrt(252) * 100  # Annualized
    spot_return = df['spot_return'].iloc[-1] * 100
    spot_annualized_return = ((1 + spot_return/100) ** (1/years) - 1) * 100 if years > 0 else 0
    
    # Calculate maximum drawdown of underlying
    high_watermark = df['SP_close'].cummax()
    drawdown_series = (df['SP_close'] - high_watermark) / high_watermark
    max_spot_drawdown = drawdown_series.min() * 100
    
    # Carry Trade Strategy Statistics
    funding_return = df['funding_return'].iloc[-1] * 100
    annualized_funding_return = ((1 + funding_return/100) ** (1/years) - 1) * 100 if years > 0 else 0
    
    # Calculate maximum drawdown of strategy returns
    strategy_equity_curve = (1 + df['funding_return'])
    strategy_hwm = strategy_equity_curve.cummax()
    strategy_drawdown_series = (strategy_equity_curve - strategy_hwm) / strategy_hwm
    max_strategy_drawdown = strategy_drawdown_series.min() * 100
    
    # Funding Rates Statistics - including min/max and negative periods
    avg_daily_funding = df['FR_close'].mean()
    annualized_funding = avg_daily_funding * 365
    min_daily_funding = df['FR_close'].min()
    max_daily_funding = df['FR_close'].max()
    annualized_min_funding = min_daily_funding * 365
    annualized_max_funding = max_daily_funding * 365

    # Count days with negative funding rate
    negative_days = (df['FR_close'] < 0).sum()
    negative_days_pct = (negative_days / len(df)) * 100 if len(df) > 0 else 0
    
    # Average negative funding rate when it's negative
    negative_funding = df[df['FR_close'] < 0]['FR_close'].mean() if (df['FR_close'] < 0).any() else 0
    annualized_negative_funding = negative_funding * 365
    
    # Longest streak of negative funding days
    streak_count = 0
    max_streak = 0
    for rate in df['FR_close']:
        if rate < 0:
            streak_count += 1
            max_streak = max(max_streak, streak_count)
        else:
            streak_count = 0
    
    # Market Regime Analysis
    bull_days = df['bull_market'].sum()
    bear_days = len(df) - bull_days
    bull_market_returns = df[df['bull_market'] == 1]['daily_spot_return'].mean() * 100
    bear_market_returns = df[df['bull_market'] == 0]['daily_spot_return'].mean() * 100
    
    bull_market_funding = df[df['bull_market'] == 1]['FR_close'].mean()
    bear_market_funding = df[df['bull_market'] == 0]['FR_close'].mean()
    
    # Correction and Recovery Analysis
    correction_days = df['in_correction'].sum()
    correction_funding_rate = df[df['in_correction'] == 1]['FR_close'].mean()
    recovery_days = df['recovery_start'].sum()
    
    # Correlation between price changes and funding rates
    price_funding_corr = df['daily_spot_return'].corr(df['FR_close'])
    
    # Sharpe Ratio calculation for the strategy (based on funding returns)
    risk_free_rate = 0.02 / 252  # assuming 2% annual risk-free
    daily_excess_returns = df['daily_funding_return'] - risk_free_rate
    sharpe_ratio = (daily_excess_returns.mean() / daily_excess_returns.std()) * np.sqrt(252) if daily_excess_returns.std() > 0 else 0
    
    return {
        # Period info
        'days': days,
        'years': years,
        'start_date': df['date'].min(),
        'end_date': df['date'].max(),
        
        # Price info
        'entry_price': entry_price,
        'exit_price': exit_price,
        
        # Underlying asset stats
        'spot_return_pct': spot_return,
        'spot_annualized_return_pct': spot_annualized_return,
        'spot_volatility_pct': spot_volatility,
        'max_spot_drawdown_pct': max_spot_drawdown,
        
        # Strategy stats
        'funding_return_pct': funding_return,
        'annualized_funding_return_pct': annualized_funding_return,
        'sharpe_ratio': sharpe_ratio,
        'max_strategy_drawdown_pct': max_strategy_drawdown,
        
       # Funding rates - enhanced
        'avg_daily_funding_pct': avg_daily_funding,
        'annualized_funding_pct': annualized_funding,
        'min_daily_funding_pct': min_daily_funding,
        'max_daily_funding_pct': max_daily_funding,
        'annualized_min_funding_pct': annualized_min_funding,
        'annualized_max_funding_pct': annualized_max_funding,
        'negative_funding_days': negative_days,
        'negative_funding_days_pct': negative_days_pct,
        'avg_negative_funding_pct': negative_funding,
        'annualized_negative_funding_pct': annualized_negative_funding,
        'max_consecutive_negative_days': max_streak,
        
        # Market regime analysis
        'bull_market_days': bull_days,
        'bull_market_pct': (bull_days / len(df)) * 100,
        'bear_market_days': bear_days,
        'bull_market_daily_return': bull_market_returns,
        'bear_market_daily_return': bear_market_returns,
        'bull_market_funding_rate': bull_market_funding,
        'bear_market_funding_rate': bear_market_funding,
        'annualized_bull_funding': bull_market_funding * 365,
        'annualized_bear_funding': bear_market_funding * 365,
        
        # Correction analysis
        'correction_days': correction_days,
        'correction_funding_rate': correction_funding_rate,
        'price_funding_correlation': price_funding_corr,
    }

def plot_carry_trade_performance(merged_df: pd.DataFrame, stats: dict, title: str = None):
    """
    Plot comprehensive performance of the delta-neutral carry trade and underlying asset.
    """
    fig = plt.figure(figsize=(16, 12))
    spec = fig.add_gridspec(3, 2)
    
    # Plot 1: Asset price and market regimes
    ax1 = fig.add_subplot(spec[0, :])
    ax1.plot(merged_df['date'], merged_df['SP_close'], label='Asset Price', color='blue')
    
    # Highlight bull markets
    bull_regions = []
    current_start = None
    for idx, row in merged_df.iterrows():
        if row['bull_market'] == 1 and current_start is None:
            current_start = row['date']
        elif row['bull_market'] == 0 and current_start is not None:
            bull_regions.append((current_start, row['date']))
            current_start = None
    
    if current_start is not None:  # Handle case where we end in bull market
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
    
    if current_start is not None:  # Handle case where we end in a correction
        correction_regions.append((current_start, merged_df['date'].iloc[-1]))
        
    for start, end in correction_regions:
        ax1.axvspan(start, end, alpha=0.2, color='red')
        
    ax1.set_title('Asset Price with Market Regimes' if title is None else f'{title} Price with Market Regimes', fontsize=14)
    ax1.set_ylabel('Price ($)')
    ax1.legend(['Asset Price', 'Bull Market', 'Market Correction'])
    ax1.grid(True)
    
    # Plot 2: Funding rates (annualized)
    ax2 = fig.add_subplot(spec[1, 0])
    ax2.plot(merged_df['date'], merged_df['FR_annualized'], color='purple')
    
    # Highlight negative funding rate periods
    negative_mask = merged_df['FR_close'] < 0
    if negative_mask.any():
        ax2.scatter(merged_df.loc[negative_mask, 'date'], 
                   merged_df.loc[negative_mask, 'FR_annualized'], 
                   color='red', s=20, label='Negative Funding')
    
    ax2.set_title('Annualized Funding Rates', fontsize=12)
    ax2.set_ylabel('Annual Rate (%)')
    ax2.axhline(y=stats['annualized_funding_pct'], color='black', linestyle='-', 
                label=f'Avg: {stats["annualized_funding_pct"]:.2f}%')
    ax2.axhline(y=stats['annualized_bull_funding'], color='g', linestyle='--', 
                label=f'Bull Avg: {stats["annualized_bull_funding"]:.2f}%')
    ax2.axhline(y=stats['annualized_bear_funding'], color='orange', linestyle='--', 
                label=f'Bear Avg: {stats["annualized_bear_funding"]:.2f}%')
    
    # Show min and max funding rates
    ax2.axhline(y=stats['annualized_min_funding_pct'], color='red', linestyle=':', 
                label=f'Min: {stats["annualized_min_funding_pct"]:.2f}%')
    ax2.axhline(y=stats['annualized_max_funding_pct'], color='blue', linestyle=':', 
                label=f'Max: {stats["annualized_max_funding_pct"]:.2f}%')
    
    ax2.legend()
    ax2.grid(True)
    
    # Plot 3: Cumulative funding return
    ax3 = fig.add_subplot(spec[1, 1])
    ax3.plot(merged_df['date'], merged_df['funding_return'] * 100, color='green')
    
    # Highlight periods with negative funding
    negative_regions = []
    current_start = None
    for idx, row in merged_df.iterrows():
        if row['FR_close'] < 0 and current_start is None:
            current_start = row['date']
        elif row['FR_close'] >= 0 and current_start is not None:
            negative_regions.append((current_start, row['date']))
            current_start = None
    
    if current_start is not None:  # Handle case where we end with negative funding
        negative_regions.append((current_start, merged_df['date'].iloc[-1]))
        
    for start, end in negative_regions:
        ax3.axvspan(start, end, alpha=0.2, color='red', label='Negative Funding' if start == negative_regions[0][0] else '')
    
    ax3.set_title('Cumulative Funding Return', fontsize=12)
    ax3.set_ylabel('Return (%)')
    ax3.grid(True)
    if negative_regions:
        ax3.legend()
    
    # Plot 4: Price drawdowns
    ax4 = fig.add_subplot(spec[2, 0])
    ax4.fill_between(merged_df['date'], merged_df['drawdown_pct'], 0, color='red', alpha=0.3)
    ax4.set_title('Underlying Asset Drawdown', fontsize=12)
    ax4.set_ylabel('Drawdown (%)')
    ax4.set_ylim(1.1 * merged_df['drawdown_pct'].min(), 5)  # Give some padding below min drawdown
    ax4.grid(True)
    
    # Plot 5: Strategy drawdowns (based on funding returns)
    ax5 = fig.add_subplot(spec[2, 1])
    strategy_equity = (1 + merged_df['funding_return'])
    strategy_hwm = strategy_equity.cummax()
    strategy_dd = (strategy_equity - strategy_hwm) / strategy_hwm * 100
    ax5.fill_between(merged_df['date'], strategy_dd, 0, color='orange', alpha=0.3)
    ax5.set_title('Strategy Drawdown', fontsize=12)
    ax5.set_ylabel('Drawdown (%)')
    ax5.grid(True)
    
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


def display_statistics_table(btc_stats, eth_stats=None):
    """
    Display statistics in a nicely formatted table
    
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
            ('Sharpe Ratio', 'sharpe_ratio'),
            ('Maximum Strategy Drawdown', 'max_strategy_drawdown_pct', '%')
        ],
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
            ('Bull Market Daily Return', 'bull_market_daily_return', '%'),
            ('Bear Market Daily Return', 'bear_market_daily_return', '%'),
            ('Bull Market Daily Funding', 'bull_market_funding_rate', '%'),
            ('Bear Market Daily Funding', 'bear_market_funding_rate', '%'),
            ('Annualized Bull Market Funding', 'annualized_bull_funding', '%'),
            ('Annualized Bear Market Funding', 'annualized_bear_funding', '%'),
            ('Market Correction Days', 'correction_days', ' days'),
            ('Correction Daily Funding', 'correction_funding_rate', '%'),
            ('Price-Funding Correlation', 'price_funding_correlation')
        ]
    }
    
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
