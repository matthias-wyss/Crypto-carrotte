import pandas as pd
import requests
import time
from typing import Dict, List, Tuple
import numpy as np
from pathlib import Path


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

    def get_spot_prices(self, symbol: str, start_str: str, end_str: str, exchange: str, interval: str, limit: int) -> pd.DataFrame:
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
            df["closePrice"] = df["c"].astype(float)
            df = df[['date', 'timestamp', 'closePrice']]
            
            return df
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching spot data for {symbol} on {exchange}: {e}")
            return pd.DataFrame()

    def get_funding_rates(self, symbol: str, start_str: str, end_str: str, exchange: str, interval: str, limit: int) -> pd.DataFrame:
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
                
            df = pd.DataFrame(ohlc_data, columns=['t', 'o', 'h', 'l', 'c', 'v'])
            
            # Convert timestamp and format data
            df['date'] = pd.to_datetime(df['t'], unit='s')
            df["timestamp"] = df['t']
            df["fundingRate"] = df["c"].astype(float)
            df = df[['date', 'timestamp', 'fundingRate']]
            
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
    spot_df = api.get_spot_prices(spot_symbol, start_str, end_str, spot_exchange, interval, limit)
    
    if spot_df.empty:
        print(f"No spot data available for {spot_symbol} on {spot_exchange}")
        return pd.DataFrame()
    
    if save_files:
        spot_df.to_csv(f'{spot_exchange}_{spot_symbol}_{interval}_spot_prices.csv', index=False)
    
    # Get funding rates
    funding_df = api.get_funding_rates(perp_symbol, start_str, end_str, perp_exchange, interval, limit)
    
    if funding_df.empty:
        print(f"No funding rate data available for {perp_symbol} on {perp_exchange}")
        return pd.DataFrame()
    
    if save_files:
        funding_df.to_csv(f'{perp_exchange}_{perp_symbol}_{interval}_funding_rates.csv', index=False)

    # Add exchange identifiers
    spot_df['spot_exchange'] = spot_exchange
    spot_df['spot_symbol'] = spot_symbol
    funding_df['perp_exchange'] = perp_exchange
    funding_df['perp_symbol'] = perp_symbol
    
    # Merge on timestamp (might need to adjust based on data availability)
    merged_df = pd.merge(spot_df, funding_df, on=['date', 'timestamp'], how='inner')
    
    if merged_df.empty:
        print(f"No matching timestamps between spot and funding data for {spot_exchange}/{perp_exchange}")
    
    return merged_df

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
        funding_df = api.get_funding_rates(
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
            spot_df = api.get_spot_prices(
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
            initial_value = merged_df['closePrice'].iloc[0] * position_size
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


    
   
