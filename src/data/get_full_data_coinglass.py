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

def analyze_venue_switching_strategy(
    spot_df: pd.DataFrame,
    futures_dfs: Dict[str, pd.DataFrame], 
    window_size: int = 7,
    maker_fee: float = 0.0,
    taker_fee: float = 0.0,
    use_maker_fee: bool = True,
    position_size: float = 1.0,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Analyzes a dynamic strategy that switches the short futures position between exchanges
    to optimize funding returns, accounting for trading fees.
    
    Parameters:
        spot_df: DataFrame - Spot price data for the base asset
        futures_dfs: Dict[str, DataFrame] - Dictionary mapping exchange names to their futures data
        window_size: int - Number of periods to use for making venue decisions
        maker_fee: float - Maker fee as a percentage (e.g., 0.02 for 0.02%)
        taker_fee: float - Taker fee as a percentage (e.g., 0.05 for 0.05%)
        use_maker_fee: bool - Whether to use maker fee (True) or taker fee (False) for venue switching
        position_size: float - Size of the position (multiplier for returns and fees)
        verbose: bool - Whether to print detailed logs during execution
    
    Returns:
        Dictionary containing performance metrics, exchange switching details, and comparison results
    """
    if len(futures_dfs) < 2:
        raise ValueError("Need at least two exchanges to analyze venue switching")

    # Standardize timestamps and ensure all dataframes use the same date format
    spot_df = spot_df.sort_values('timestamp').copy()
    spot_df['date'] = pd.to_datetime(spot_df['date'])
    exchange_dfs = {}
    
    # Initial setup for each exchange
    for exchange, df in futures_dfs.items():
        df = df.sort_values('timestamp').copy()
        df['date'] = pd.to_datetime(df['date'])
        
        # Merge with spot data to ensure we have the same timestamps
        merged = pd.merge(spot_df, df, on='timestamp', how='inner', suffixes=('', f'_{exchange}'))
        
        # Keep only the necessary columns for this exchange
        cols_to_keep = ['date', 'timestamp', 'SP_close', f'FR_close_{exchange}', f'perp_exchange_{exchange}']
        merged = merged[cols_to_keep]
        
        # Rename the FR_close column for clarity
        merged = merged.rename(columns={f'FR_close_{exchange}': 'FR_close'})
        
        exchange_dfs[exchange] = merged
    
    # Find common timestamps across all exchanges and spot
    common_timestamps = set(spot_df['timestamp'])
    for df in exchange_dfs.values():
        common_timestamps = common_timestamps.intersection(set(df['timestamp']))
    
    common_timestamps = sorted(list(common_timestamps))
    if len(common_timestamps) < window_size * 2:
        raise ValueError(f"Not enough common data points across exchanges (need at least {window_size*2})")
    
    # Prepare results dataframe
    results = pd.DataFrame({'timestamp': common_timestamps})
    results = results.sort_values('timestamp').reset_index(drop=True)
    
    # Add date column
    for exchange, df in exchange_dfs.items():
        date_map = dict(zip(df['timestamp'], df['date']))
        results['date'] = results['timestamp'].map(date_map)
        break  # We only need to do this once
    
    # Add spot price
    for exchange, df in exchange_dfs.items():
        price_map = dict(zip(df['timestamp'], df['SP_close']))
        results['SP_close'] = results['timestamp'].map(price_map)
        break  # All exchanges should have the same spot price
    
    # Add funding rates for each exchange
    for exchange, df in exchange_dfs.items():
        funding_map = dict(zip(df['timestamp'], df['FR_close']))
        results[f'FR_{exchange}'] = results['timestamp'].map(funding_map)
    
    # Initialize columns for strategy
    results['best_exchange'] = ''
    results['switched_exchange'] = False
    results['fee_paid'] = 0.0
    results['current_exchange'] = list(exchange_dfs.keys())[0]  # Start with first exchange
    
    # Calculate funding return for a static strategy on each exchange
    for exchange in exchange_dfs:
        results[f'static_funding_{exchange}'] = results[f'FR_{exchange}'] / 100
        results[f'static_cumulative_{exchange}'] = (1 + results[f'static_funding_{exchange}']).cumprod() - 1
    
    # Initialize strategy metrics
    results['dynamic_funding'] = 0.0
    results['dynamic_funding_gross'] = 0.0
    results['dynamic_funding_fee'] = 0.0
    results['dynamic_cumulative'] = 0.0
    initial_price = results['SP_close'].iloc[0]
    
    # Calculate which fee to use
    fee = maker_fee if use_maker_fee else taker_fee
    
    # Implement the dynamic venue switching strategy
    for i in range(len(results)):
        if i < window_size:
            # Use the first exchange for initial periods
            current_exchange = results['current_exchange'].iloc[i - 1] if i > 0 else list(exchange_dfs.keys())[0]
            results.loc[i, 'best_exchange'] = current_exchange
            results.loc[i, 'current_exchange'] = current_exchange
            results.loc[i, 'dynamic_funding'] = results.loc[i, f'FR_{current_exchange}'] / 100
            results.loc[i, 'dynamic_funding_gross'] = results.loc[i, f'FR_{current_exchange}'] / 100
        else:
            current_exchange = results['current_exchange'].iloc[i - 1]
            
            # Calculate the average funding rate for each exchange over the past window_size periods
            avg_funding = {}
            for exchange in exchange_dfs:
                avg_funding[exchange] = results.loc[i-window_size:i-1, f'FR_{exchange}'].mean()
            
            # Find exchange with the highest average funding rate
            best_exchange = max(avg_funding, key=avg_funding.get)
            results.loc[i, 'best_exchange'] = best_exchange
            
            # Determine if we should switch exchanges based on fees
            should_switch = False
            
            if best_exchange != current_exchange:
                # Calculate the expected funding advantage
                funding_advantage = avg_funding[best_exchange] - avg_funding[current_exchange]
                
                # Calculate the cost of switching as a percentage of position value
                switch_cost_pct = fee * 2  # Fee to close position on current exchange + open on new exchange
                
                # Convert cost to same units as funding rate (daily percentage)
                daily_switch_cost = switch_cost_pct  # Already as a percentage
                
                # Switching is profitable if the advantage over the window exceeds the cost
                if funding_advantage > daily_switch_cost * window_size:
                    should_switch = True
                    
            if should_switch:
                results.loc[i, 'current_exchange'] = best_exchange
                results.loc[i, 'switched_exchange'] = True
                
                # Calculate fee paid as percentage of position
                fee_paid = fee * 2 * position_size * (results.loc[i, 'SP_close'] / initial_price)
                results.loc[i, 'fee_paid'] = fee_paid
                
                # Calculate net funding return (gross - fees)
                results.loc[i, 'dynamic_funding_gross'] = results.loc[i, f'FR_{best_exchange}'] / 100
                results.loc[i, 'dynamic_funding_fee'] = fee_paid / position_size
                results.loc[i, 'dynamic_funding'] = results.loc[i, 'dynamic_funding_gross'] - results.loc[i, 'dynamic_funding_fee']
            else:
                results.loc[i, 'current_exchange'] = current_exchange
                results.loc[i, 'dynamic_funding_gross'] = results.loc[i, f'FR_{current_exchange}'] / 100
                results.loc[i, 'dynamic_funding'] = results.loc[i, 'dynamic_funding_gross']
    
    # Calculate cumulative returns for dynamic strategy
    results['dynamic_cumulative'] = (1 + results['dynamic_funding']).cumprod() - 1
    
    # Calculate total trading fees paid
    total_fees = results['fee_paid'].sum()
    
    # Calculate strategy stats for comparison
    strategy_stats = {}
    
    # Dynamic strategy stats
    dynamic_return = results['dynamic_cumulative'].iloc[-1] * 100
    dynamic_gross_return = ((1 + results['dynamic_funding_gross']).cumprod() - 1).iloc[-1] * 100
    days = (results['date'].max() - results['date'].min()).days
    years = days / 365
    dynamic_annual_return = ((1 + dynamic_return/100) ** (1/years) - 1) * 100 if years > 0 else 0
    dynamic_volatility = results['dynamic_funding'].std() * np.sqrt(252) * 100
    dynamic_sharpe = dynamic_annual_return / dynamic_volatility if dynamic_volatility > 0 else 0
    
    strategy_stats['dynamic'] = {
        'total_return_pct': dynamic_return,
        'gross_return_pct': dynamic_gross_return, 
        'total_fees_pct': (total_fees / position_size) * 100,
        'annualized_return_pct': dynamic_annual_return,
        'volatility_pct': dynamic_volatility,
        'sharpe_ratio': dynamic_sharpe,
        'num_switches': results['switched_exchange'].sum(),
        'avg_holding_days': days / (results['switched_exchange'].sum() + 1) if results['switched_exchange'].sum() > 0 else days
    }
    
    # Static strategy stats for each exchange
    for exchange in exchange_dfs:
        static_return = results[f'static_cumulative_{exchange}'].iloc[-1] * 100
        static_annual_return = ((1 + static_return/100) ** (1/years) - 1) * 100 if years > 0 else 0
        static_volatility = results[f'static_funding_{exchange}'].std() * np.sqrt(252) * 100
        static_sharpe = static_annual_return / static_volatility if static_volatility > 0 else 0
        
        strategy_stats[f'static_{exchange}'] = {
            'total_return_pct': static_return,
            'annualized_return_pct': static_annual_return,
            'volatility_pct': static_volatility,
            'sharpe_ratio': static_sharpe,
            'total_fees_pct': 0.0,  # No switching fees in static strategy
            'num_switches': 0
        }
    
    # Find the best static strategy
    best_static_exchange = max(
        [ex for ex in exchange_dfs.keys()], 
        key=lambda ex: strategy_stats[f'static_{ex}']['total_return_pct']
    )
    best_static = strategy_stats[f'static_{best_static_exchange}']
    
    # Calculate improvement over best static strategy
    improvement_pct = dynamic_return - best_static['total_return_pct']
    improvement_annual_pct = dynamic_annual_return - best_static['annualized_return_pct']
    
    # Analyze the venues used
    venue_usage = results['current_exchange'].value_counts().to_dict()
    venue_usage_pct = {k: (v / len(results) * 100) for k, v in venue_usage.items()}
    
    # Calculate average funding rate by exchange
    avg_funding_by_exchange = {}
    for exchange in exchange_dfs:
        avg_funding_by_exchange[exchange] = results[f'FR_{exchange}'].mean()
    
    # Analyze when switches happened
    if results['switched_exchange'].sum() > 0:
        switch_periods = results[results['switched_exchange']]
        switch_funding_diff = []
        
        for i, row in switch_periods.iterrows():
            old_exchange = results.loc[i-1, 'current_exchange']
            new_exchange = row['current_exchange']
            funding_diff = row[f'FR_{new_exchange}'] - row[f'FR_{old_exchange}']
            switch_funding_diff.append(funding_diff)
            
        avg_funding_improvement = np.mean(switch_funding_diff) if switch_funding_diff else 0
    else:
        avg_funding_improvement = 0
    
    # Create result dictionary
    result = {
        'results_df': results,
        'strategy_stats': strategy_stats,
        'dynamic_vs_best_static': {
            'improvement_pct': improvement_pct,
            'improvement_annual_pct': improvement_annual_pct,
            'best_static_exchange': best_static_exchange
        },
        'venue_usage': venue_usage,
        'venue_usage_pct': venue_usage_pct,
        'avg_funding_by_exchange': avg_funding_by_exchange,
        'avg_funding_improvement_on_switch': avg_funding_improvement,
        'total_days': days,
        'years': years,
        'fee_settings': {
            'maker_fee': maker_fee,
            'taker_fee': taker_fee,
            'fee_used': fee,
            'use_maker_fee': use_maker_fee
        }
    }
    
    return result

def plot_venue_switching_results(result: Dict[str, Any], title: str = None):
    """
    Plot the results of the venue switching analysis
    
    Parameters:
        result: Dict - Result from analyze_venue_switching_strategy
        title: str - Optional title for the plots
    """
    results = result['results_df']
    stats = result['strategy_stats']
    
    fig = plt.figure(figsize=(16, 12))
    spec = fig.add_gridspec(3, 2)
    
    # Plot 1: Asset price
    ax1 = fig.add_subplot(spec[0, :])
    ax1.plot(results['date'], results['SP_close'], label='Asset Price', color='blue')
    
    # Highlight exchange switches
    switch_dates = results.loc[results['switched_exchange'], 'date']
    for switch_date in switch_dates:
        ax1.axvline(x=switch_date, color='gray', linestyle='--', alpha=0.5)
    
    # Add colored regions for current exchange
    current_exchange = results['current_exchange'].iloc[0]
    start_date = results['date'].iloc[0]
    exchange_colors = {
        exchange: plt.cm.tab10(i % 10) 
        for i, exchange in enumerate(result['venue_usage'].keys())
    }
    
    for i in range(1, len(results)):
        if results['current_exchange'].iloc[i] != current_exchange:
            end_date = results['date'].iloc[i]
            ax1.axvspan(start_date, end_date, alpha=0.2, color=exchange_colors[current_exchange])
            
            # Update for next segment
            current_exchange = results['current_exchange'].iloc[i]
            start_date = end_date
    
    # Add the last segment
    ax1.axvspan(start_date, results['date'].iloc[-1], alpha=0.2, color=exchange_colors[current_exchange])
    
    ax1.set_title('Asset Price and Exchange Switching' if title is None else f'{title} - Venue Switching Analysis', fontsize=14)
    ax1.set_ylabel('Price ($)')
    
    # Create legend for exchanges
    legend_handles = [plt.Rectangle((0,0),1,1, color=color, alpha=0.5) for color in exchange_colors.values()]
    legend_labels = list(exchange_colors.keys())
    ax1.legend(legend_handles, legend_labels, title="Active Exchange")
    ax1.grid(True)
    
    # Plot 2: Funding rates by exchange
    ax2 = fig.add_subplot(spec[1, 0])
    for exchange in result['avg_funding_by_exchange'].keys():
        ax2.plot(results['date'], results[f'FR_{exchange}'], label=exchange)
        
    # Add markers for switch events
    for i, row in results[results['switched_exchange']].iterrows():
        ax2.plot(row['date'], row[f'FR_{row["current_exchange"]}'], 'o', color='black', markersize=5)
    
    ax2.set_title('Funding Rates by Exchange', fontsize=12)
    ax2.set_ylabel('Daily Funding Rate (%)')
    ax2.legend()
    ax2.grid(True)
    
    # Plot 3: Cumulative returns comparison
    ax3 = fig.add_subplot(spec[1, 1])
    
    # Plot dynamic strategy
    ax3.plot(results['date'], results['dynamic_cumulative'] * 100, label='Dynamic', color='black', linewidth=2)
    
    # Plot static strategies
    for exchange in result['avg_funding_by_exchange'].keys():
        ax3.plot(results['date'], results[f'static_cumulative_{exchange}'] * 100, 
                 label=f'Static {exchange}', color=exchange_colors.get(exchange, 'gray'), 
                 alpha=0.7, linestyle='--')
    
    ax3.set_title('Cumulative Returns Comparison', fontsize=12)
    ax3.set_ylabel('Return (%)')
    ax3.legend()
    ax3.grid(True)
    
    # Plot 4: Fee impact
    ax4 = fig.add_subplot(spec[2, 0])
    ax4.plot(results['date'], results['dynamic_funding_gross'].cumsum() * 100, label='Gross Returns', color='green')
    ax4.plot(results['date'], results['dynamic_cumulative'] * 100, label='Net Returns', color='blue')
    ax4.fill_between(results['date'], 
                    results['dynamic_funding_gross'].cumsum() * 100, 
                    results['dynamic_cumulative'] * 100, 
                    color='red', alpha=0.3, label='Fee Impact')
    ax4.set_title('Fee Impact on Returns', fontsize=12)
    ax4.set_ylabel('Return (%)')
    ax4.legend()
    ax4.grid(True)
    
    # Plot 5: Exchange usage pie chart
    ax5 = fig.add_subplot(spec[2, 1])
    wedges, texts, autotexts = ax5.pie(
        result['venue_usage'].values(), 
        labels=result['venue_usage'].keys(),
        autopct='%1.1f%%',
        textprops={'fontsize': 10},
        colors=[exchange_colors.get(ex, 'gray') for ex in result['venue_usage'].keys()]
    )
    ax5.set_title('Exchange Usage Distribution', fontsize=12)
    
    # Add stats as text
    plt.tight_layout()
    
    # Format stats for display
    dynamic_stats = stats['dynamic']
    best_static = stats[f"static_{result['dynamic_vs_best_static']['best_static_exchange']}"]
    
    stats_text = (
        f"== Dynamic Strategy ==\n"
        f"Total Return: {dynamic_stats['total_return_pct']:.2f}%\n"
        f"Ann. Return: {dynamic_stats['annualized_return_pct']:.2f}%\n"
        f"Sharpe: {dynamic_stats['sharpe_ratio']:.2f}\n"
        f"Switches: {dynamic_stats['num_switches']}\n"
        f"Fees Paid: {dynamic_stats['total_fees_pct']:.2f}%\n\n"
        f"== Best Static ({result['dynamic_vs_best_static']['best_static_exchange']}) ==\n"
        f"Total Return: {best_static['total_return_pct']:.2f}%\n"
        f"Ann. Return: {best_static['annualized_return_pct']:.2f}%\n"
        f"Sharpe: {best_static['sharpe_ratio']:.2f}\n\n"
        f"== Improvement ==\n"
        f"Return: {result['dynamic_vs_best_static']['improvement_pct']:.2f}%\n"
        f"Ann. Return: {result['dynamic_vs_best_static']['improvement_annual_pct']:.2f}%\n"
        f"Period: {result['total_days']} days"
    )
    plt.figtext(0.01, 0.01, stats_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.show()
    
    return fig

def create_venue_switching_report(result):
    """
    Generate a formatted report on the venue switching analysis
    
    Parameters:
        result: Dict - Result from analyze_venue_switching_strategy
    
    Returns:
        pd.DataFrame - Styled DataFrame with the report
    """
    import pandas as pd
    
    # Get the best static exchange name once to avoid nested f-strings
    best_exchange = result['dynamic_vs_best_static']['best_static_exchange']
    static_key = f'static_{best_exchange}'
    
    # Create sections for the report
    sections = {
        'Strategy Performance': [
            ('Period', f"{result['total_days']} days ({result['years']:.2f} years)"),
            ('Dynamic Strategy Total Return', f"{result['strategy_stats']['dynamic']['total_return_pct']:.2f}%"),
            ('Dynamic Strategy Ann. Return', f"{result['strategy_stats']['dynamic']['annualized_return_pct']:.2f}%"),
            ('Dynamic Strategy Sharpe', f"{result['strategy_stats']['dynamic']['sharpe_ratio']:.2f}"),
            ('Best Static Strategy', best_exchange),
            ('Best Static Return', f"{result['strategy_stats'][static_key]['total_return_pct']:.2f}%"),
            ('Best Static Ann. Return', f"{result['strategy_stats'][static_key]['annualized_return_pct']:.2f}%"),
            ('Return Improvement', f"{result['dynamic_vs_best_static']['improvement_pct']:.2f}%"),
            ('Ann. Return Improvement', f"{result['dynamic_vs_best_static']['improvement_annual_pct']:.2f}%"),
        ],
        'Exchange Switching Details': [
            ('Number of Switches', f"{result['strategy_stats']['dynamic']['num_switches']}"),
            ('Avg Holding Period', f"{result['strategy_stats']['dynamic']['avg_holding_days']:.1f} days"),
            ('Total Fees Paid', f"{result['strategy_stats']['dynamic']['total_fees_pct']:.2f}%"),
            ('Fee Setting Used', f"{'Maker' if result['fee_settings']['use_maker_fee'] else 'Taker'} Fee ({result['fee_settings']['fee_used']:.4f}%)"),
            ('Avg Funding Improvement on Switch', f"{result['avg_funding_improvement_on_switch']:.4f}%")
        ],
        'Exchange Stats': []
    }
    
    # Add stats for each exchange
    for exchange, avg_funding in result['avg_funding_by_exchange'].items():
        usage_pct = result['venue_usage_pct'].get(exchange, 0)
        exchange_key = f'static_{exchange}'
        sections['Exchange Stats'].append(
            (f"{exchange}", f"Avg Funding: {avg_funding:.4f}%, Used: {usage_pct:.1f}% of time, Return: {result['strategy_stats'][exchange_key]['total_return_pct']:.2f}%")
        )
    
    # Create the report table
    table_data = []
    for section, items in sections.items():
        table_data.append([section, ""])
        for label, value in items:
            table_data.append([f"  {label}", value])
    
    # Create DataFrame
    df = pd.DataFrame(table_data, columns=['Metric', 'Value'])
    
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
    
    return styled_df



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

def separate_by_exchange(df):
    """
    Separates a dataframe containing mixed exchange data into separate dataframes by exchange.
    
    Parameters:
        df: DataFrame containing data from multiple exchanges
    
    Returns:
        dict: Dictionary mapping exchange names to their respective dataframes
        
    Example:
        If your dataframe has columns like 'date', 'timestamp', 'FR_open', 'FR_close', 'perp_exchange', 'perp_symbol'
        This function will separate it into multiple dataframes based on the exchange value
    """
    # Check dataframe structure
    print(f"Input dataframe columns: {df.columns.tolist()}")
    print(f"First few rows sample:")
    print(df.head(2))
    
    # Find the exchange column
    exchange_cols = []
    for col in df.columns:
        if 'exchange' in str(col).lower():
            exchange_cols.append(col)
    
    if not exchange_cols:
        raise ValueError("Could not find any column containing 'exchange' in the dataframe")
    
    # If multiple exchange columns are found, try to determine which to use
    exchange_column = None
    if len(exchange_cols) > 1:
        # Prefer perp_exchange for funding rate analysis
        if 'perp_exchange' in exchange_cols:
            exchange_column = 'perp_exchange'
        elif 'exchange' in exchange_cols:
            exchange_column = 'exchange'
        else:
            exchange_column = exchange_cols[0]
        print(f"Multiple exchange columns found: {exchange_cols}. Using '{exchange_column}'")
    else:
        exchange_column = exchange_cols[0]
        print(f"Using exchange column: '{exchange_column}'")
    
    # Get unique exchanges
    exchanges = df[exchange_column].unique()
    print(f"Found {len(exchanges)} exchanges: {exchanges}")
    
    # Create result dictionary
    exchange_dfs = {}
    
    # Split by exchange
    for exchange in exchanges:
        exchange_df = df[df[exchange_column] == exchange].copy().reset_index(drop=True)
        exchange_dfs[exchange] = exchange_df
        print(f"Created dataframe for {exchange} with {len(exchange_df)} rows")
    
    return exchange_dfs

def analyze_multi_exchange_carry_trade(spot_df, futures_df, maker_fee=0.0, taker_fee=0.0):
    """
    Analyzes whether switching venues for the short leg of a carry trade improves performance.
    Works with data structure from the process_data_files function.
    
    Parameters:
        spot_df: DataFrame with spot price data across exchanges
        futures_df: DataFrame with futures funding rate data across exchanges
        maker_fee: Maker fee percentage
        taker_fee: Taker fee percentage
        
    Returns:
        Dictionary with analysis results
    """
    # First, let's identify and separate by exchange
    if 'spot_exchange' in spot_df.columns:
        spot_exchange_col = 'spot_exchange'
    elif 'exchange' in spot_df.columns:
        spot_exchange_col = 'exchange'
    else:
        raise ValueError("Could not identify exchange column in spot dataframe")
        
    if 'perp_exchange' in futures_df.columns:
        futures_exchange_col = 'perp_exchange'
    elif 'exchange' in futures_df.columns:
        futures_exchange_col = 'exchange'
    else:
        raise ValueError("Could not identify exchange column in futures dataframe")
    
    # Get unique exchanges
    spot_exchanges = spot_df[spot_exchange_col].unique()
    futures_exchanges = futures_df[futures_exchange_col].unique()
    
    print(f"Spot exchanges: {spot_exchanges}")
    print(f"Futures exchanges: {futures_exchanges}")
    
    # Separate by exchange
    spot_by_exchange = {}
    for exchange in spot_exchanges:
        spot_by_exchange[exchange] = spot_df[spot_df[spot_exchange_col] == exchange].copy()
        print(f"Created spot dataframe for {exchange} with {len(spot_by_exchange[exchange])} rows")
        
    futures_by_exchange = {}
    for exchange in futures_exchanges:
        futures_by_exchange[exchange] = futures_df[futures_df[futures_exchange_col] == exchange].copy()
        print(f"Created futures dataframe for {exchange} with {len(futures_by_exchange[exchange])} rows")
    
    # Select reference exchange for spot
    reference_exchange = spot_exchanges[0]  # Default to first exchange
    reference_spot_df = spot_by_exchange[reference_exchange]
    print(f"Using {reference_exchange} as reference spot exchange")
    
    # For each date, find the best futures exchange based on funding rate
    # Prepare result dataframe
    results = pd.DataFrame()
    
    # Ensure we have timestamps and date columns
    for ex_df in [reference_spot_df] + list(futures_by_exchange.values()):
        if 'timestamp' not in ex_df.columns:
            if 'date' in ex_df.columns:
                ex_df['timestamp'] = pd.to_datetime(ex_df['date']).astype(int) // 10**9
            else:
                raise ValueError("Neither timestamp nor date column found")
                
        if 'date' not in ex_df.columns:
            ex_df['date'] = pd.to_datetime(ex_df['timestamp'], unit='s')
    
    # Get common dates (timestamps) across all exchanges
    common_timestamps = set(reference_spot_df['timestamp'])
    for ex_df in futures_by_exchange.values():
        common_timestamps = common_timestamps.intersection(set(ex_df['timestamp']))
    
    common_timestamps = sorted(list(common_timestamps))
    print(f"Found {len(common_timestamps)} common timestamps across exchanges")
    
    if len(common_timestamps) == 0:
        raise ValueError("No common timestamps found across exchanges")
    
    # Create results dataframe with common timestamps
    results = pd.DataFrame({'timestamp': common_timestamps})
    results['date'] = pd.to_datetime(results['timestamp'], unit='s')
    
    # Add spot price
    spot_prices = {}
    for ts in common_timestamps:
        row = reference_spot_df[reference_spot_df['timestamp'] == ts]
        if not row.empty:
            spot_prices[ts] = row['SP_close'].values[0]
            
    results['SP_close'] = results['timestamp'].map(spot_prices)
    
    # Add funding rates for each exchange
    for exchange, ex_df in futures_by_exchange.items():
        funding_rates = {}
        for ts in common_timestamps:
            row = ex_df[ex_df['timestamp'] == ts]
            if not row.empty:
                funding_rates[ts] = row['FR_close'].values[0]
                
        results[f'FR_{exchange}'] = results['timestamp'].map(funding_rates)
    
    # Implement dynamic venue switching strategy
    window_size = 7  # Days to look back for signal
    results['best_exchange'] = ''
    results['current_exchange'] = list(futures_by_exchange.keys())[0]
    results['switched_exchange'] = False
    results['fee_paid'] = 0.0
    
    # Calculate funding return for static strategy on each exchange
    for exchange in futures_by_exchange:
        results[f'static_funding_{exchange}'] = results[f'FR_{exchange}'] / 100
        results[f'static_cumulative_{exchange}'] = (1 + results[f'static_funding_{exchange}']).cumprod() - 1
    
    # Initialize dynamic strategy
    results['dynamic_funding'] = 0.0
    results['dynamic_funding_gross'] = 0.0
    results['dynamic_cumulative'] = 0.0
    
    # Calculate which fee to use (maker or taker)
    fee = maker_fee
    
    # Implement dynamic venue switching
    for i in range(len(results)):
        if i < window_size:
            # Use first exchange for initial period
            current_exchange = results['current_exchange'].iloc[i - 1] if i > 0 else list(futures_by_exchange.keys())[0]
            results.loc[i, 'best_exchange'] = current_exchange
            results.loc[i, 'current_exchange'] = current_exchange
            results.loc[i, 'dynamic_funding'] = results.loc[i, f'FR_{current_exchange}'] / 100
            results.loc[i, 'dynamic_funding_gross'] = results.loc[i, f'FR_{current_exchange}'] / 100
        else:
            current_exchange = results['current_exchange'].iloc[i - 1]
            
            # Calculate average funding for past window
            avg_funding = {}
            for exchange in futures_by_exchange:
                avg_funding[exchange] = results.loc[i-window_size:i-1, f'FR_{exchange}'].mean()
                
            # Find best exchange
            best_exchange = max(avg_funding, key=avg_funding.get)
            results.loc[i, 'best_exchange'] = best_exchange
            
            # Determine if we should switch
            should_switch = False
            if best_exchange != current_exchange:
                funding_advantage = avg_funding[best_exchange] - avg_funding[current_exchange]
                switch_cost_pct = fee * 2  # Fee to close current position and open new one
                
                if funding_advantage > switch_cost_pct * window_size:
                    should_switch = True
                    
            if should_switch:
                results.loc[i, 'current_exchange'] = best_exchange
                results.loc[i, 'switched_exchange'] = True
                results.loc[i, 'fee_paid'] = fee * 2
                results.loc[i, 'dynamic_funding_gross'] = results.loc[i, f'FR_{best_exchange}'] / 100
                results.loc[i, 'dynamic_funding'] = results.loc[i, 'dynamic_funding_gross'] - fee * 2 / 100
            else:
                results.loc[i, 'current_exchange'] = current_exchange
                results.loc[i, 'dynamic_funding_gross'] = results.loc[i, f'FR_{current_exchange}'] / 100
                results.loc[i, 'dynamic_funding'] = results.loc[i, 'dynamic_funding_gross']
    
    # Calculate cumulative returns
    results['dynamic_cumulative'] = (1 + results['dynamic_funding']).cumprod() - 1
    
    # Calculate strategy statistics
    days = (results['date'].max() - results['date'].min()).days
    years = days / 365
    
    # Dynamic strategy stats
    dynamic_return = results['dynamic_cumulative'].iloc[-1] * 100
    dynamic_annual_return = ((1 + dynamic_return/100) ** (1/years) - 1) * 100 if years > 0 else 0
    dynamic_volatility = results['dynamic_funding'].std() * np.sqrt(252) * 100
    dynamic_sharpe = dynamic_annual_return / dynamic_volatility if dynamic_volatility > 0 else 0
    
    strategy_stats = {
        'dynamic': {
            'total_return_pct': dynamic_return,
            'annualized_return_pct': dynamic_annual_return,
            'volatility_pct': dynamic_volatility,
            'sharpe_ratio': dynamic_sharpe,
            'num_switches': results['switched_exchange'].sum(),
            'total_fees_pct': results['fee_paid'].sum(),
        }
    }
    
    # Static strategy stats
    for exchange in futures_by_exchange:
        static_return = results[f'static_cumulative_{exchange}'].iloc[-1] * 100
        static_annual_return = ((1 + static_return/100) ** (1/years) - 1) * 100 if years > 0 else 0
        static_volatility = results[f'static_funding_{exchange}'].std() * np.sqrt(252) * 100
        static_sharpe = static_annual_return / static_volatility if static_volatility > 0 else 0
        
        strategy_stats[f'static_{exchange}'] = {
            'total_return_pct': static_return,
            'annualized_return_pct': static_annual_return,
            'volatility_pct': static_volatility,
            'sharpe_ratio': static_sharpe,
            'total_fees_pct': 0.0,
            'num_switches': 0
        }
    
    # Find best static strategy
    best_static_exchange = max(
        [ex for ex in futures_by_exchange.keys()], 
        key=lambda ex: strategy_stats[f'static_{ex}']['total_return_pct']
    )
    
    best_static = strategy_stats[f'static_{best_static_exchange}']
    
    # Calculate improvement over best static
    improvement_pct = dynamic_return - best_static['total_return_pct']
    improvement_annual_pct = dynamic_annual_return - best_static['annualized_return_pct']
    
    # Venue usage analysis
    venue_usage = results['current_exchange'].value_counts().to_dict()
    venue_usage_pct = {k: (v / len(results) * 100) for k, v in venue_usage.items()}
    
    # Average funding by exchange
    avg_funding_by_exchange = {}
    for exchange in futures_by_exchange:
        avg_funding_by_exchange[exchange] = results[f'FR_{exchange}'].mean()
    
    # Collect all results
    result = {
        'results_df': results,
        'strategy_stats': strategy_stats,
        'dynamic_vs_best_static': {
            'improvement_pct': improvement_pct,
            'improvement_annual_pct': improvement_annual_pct,
            'best_static_exchange': best_static_exchange
        },
        'venue_usage': venue_usage,
        'venue_usage_pct': venue_usage_pct,
        'avg_funding_by_exchange': avg_funding_by_exchange,
        'total_days': days,
        'years': years,
        'fee_settings': {
            'maker_fee': maker_fee,
            'taker_fee': taker_fee,
            'fee_used': fee
        }
    }
    
    # Plot results
    plot_venue_switching_results(result)
    
    # Return analysis
    return result

def plot_venue_switching_results(result):
    """
    Plot the results of the venue switching analysis
    """
    results = result['results_df']
    stats = result['strategy_stats']
    
    fig = plt.figure(figsize=(16, 12))
    spec = fig.add_gridspec(3, 2)
    
    # Plot 1: Asset price
    ax1 = fig.add_subplot(spec[0, :])
    ax1.plot(results['date'], results['SP_close'], label='Asset Price', color='blue')
    
    # Highlight exchange switches
    switch_dates = results.loc[results['switched_exchange'], 'date']
    for switch_date in switch_dates:
        ax1.axvline(x=switch_date, color='gray', linestyle='--', alpha=0.5)
    
    # Add colored regions for current exchange
    current_exchange = results['current_exchange'].iloc[0]
    start_date = results['date'].iloc[0]
    
    # Create color map
    exchanges = list(result['venue_usage'].keys())
    colors = plt.cm.tab10(np.linspace(0, 1, len(exchanges)))
    exchange_colors = {ex: colors[i] for i, ex in enumerate(exchanges)}
    
    for i in range(1, len(results)):
        if results['current_exchange'].iloc[i] != current_exchange:
            end_date = results['date'].iloc[i]
            ax1.axvspan(start_date, end_date, alpha=0.2, color=exchange_colors[current_exchange])
            
            # Update for next segment
            current_exchange = results['current_exchange'].iloc[i]
            start_date = end_date
    
    # Add the last segment
    ax1.axvspan(start_date, results['date'].iloc[-1], alpha=0.2, color=exchange_colors[current_exchange])
    
    ax1.set_title('Asset Price and Exchange Switching', fontsize=14)
    ax1.set_ylabel('Price ($)')
    
    # Create legend
    legend_handles = [plt.Rectangle((0,0),1,1, color=color, alpha=0.5) for color in exchange_colors.values()]
    ax1.legend(handles=legend_handles, labels=exchanges, title="Active Exchange")
    ax1.grid(True)
    
    # Plot 2: Funding rates by exchange
    ax2 = fig.add_subplot(spec[1, 0])
    for exchange in exchanges:
        ax2.plot(results['date'], results[f'FR_{exchange}'], label=exchange)
    
    ax2.set_title('Funding Rates by Exchange', fontsize=12)
    ax2.set_ylabel('Daily Funding Rate (%)')
    ax2.legend()
    ax2.grid(True)
    
    # Plot 3: Cumulative returns comparison
    ax3 = fig.add_subplot(spec[1, 1])
    
    # Plot dynamic strategy
    ax3.plot(results['date'], results['dynamic_cumulative'] * 100, label='Dynamic', color='black', linewidth=2)
    
    # Plot static strategies
    for exchange in exchanges:
        ax3.plot(results['date'], results[f'static_cumulative_{exchange}'] * 100, 
                 label=f'Static {exchange}', color=exchange_colors.get(exchange, 'gray'), 
                 alpha=0.7, linestyle='--')
    
    ax3.set_title('Cumulative Returns Comparison', fontsize=12)
    ax3.set_ylabel('Return (%)')
    ax3.legend()
    ax3.grid(True)
    
    # Plot 4: Exchange usage pie chart
    ax4 = fig.add_subplot(spec[2, 0])
    wedges, texts, autotexts = ax4.pie(
        result['venue_usage'].values(), 
        labels=result['venue_usage'].keys(),
        autopct='%1.1f%%',
        textprops={'fontsize': 10},
        colors=[exchange_colors.get(ex, 'gray') for ex in result['venue_usage'].keys()]
    )
    ax4.set_title('Exchange Usage Distribution', fontsize=12)
    
    # Plot 5: Statistics table
    ax5 = fig.add_subplot(spec[2, 1])
    ax5.axis('off')
    
    # Format stats for display
    dynamic_stats = stats['dynamic']
    best_static_exchange = result['dynamic_vs_best_static']['best_static_exchange']
    best_static = stats[f'static_{best_static_exchange}']
    
    stats_text = (
        f"== Dynamic Strategy ==\n"
        f"Total Return: {dynamic_stats['total_return_pct']:.2f}%\n"
        f"Ann. Return: {dynamic_stats['annualized_return_pct']:.2f}%\n"
        f"Sharpe: {dynamic_stats['sharpe_ratio']:.2f}\n"
        f"Switches: {dynamic_stats['num_switches']}\n"
        f"Fees Paid: {dynamic_stats['total_fees_pct']:.2f}%\n\n"
        f"== Best Static ({best_static_exchange}) ==\n"
        f"Total Return: {best_static['total_return_pct']:.2f}%\n"
        f"Ann. Return: {best_static['annualized_return_pct']:.2f}%\n"
        f"Sharpe: {best_static['sharpe_ratio']:.2f}\n\n"
        f"== Improvement ==\n"
        f"Return: {result['dynamic_vs_best_static']['improvement_pct']:.2f}%\n"
        f"Ann. Return: {result['dynamic_vs_best_static']['improvement_annual_pct']:.2f}%\n"
        f"Period: {result['total_days']} days"
    )
    ax5.text(0.05, 0.95, stats_text, verticalalignment='top', fontsize=12)
    
    plt.tight_layout()
    plt.show()
    
    return fig
