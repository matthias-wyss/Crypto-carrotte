import pandas as pd
import requests
import time
from typing import Dict, List

class CoinGlassAPI:
    def __init__(self, api_key: str):
        self.base_url = "https://open-api-v3.coinglass.com"
        self.headers = {
            "accept": "application/json",
            "CG-API-KEY": api_key
        }
    
    def get_spot_prices(self, symbol: str, start_str: str, end_str: str, exchange:str,interval: str, limit: int ) -> pd.DataFrame:
        """
        Downloads historical OHLC spot price data for a given symbol from CoinGlass API.
        """
        print(f"Downloading spot price data for {symbol}...")
        # Convert dates to timestamps
        start_ts = int(pd.to_datetime(start_str).timestamp()) #-9223372036854776
        end_ts = int(pd.to_datetime(end_str).timestamp()) #9223372036854776

        endpoint = f"{self.base_url}/api/price/ohlc-history?exchange={exchange}&symbol={symbol}&type=futures&interval={interval}&limit={limit}&startTime={start_ts}&endTime={end_ts}"
   
        
        try:
            
            response = requests.get(endpoint, headers=self.headers)
            response.raise_for_status()
            data = response.json()
            
            if data['code'] != '0':
                raise ValueError(f"API Error: {data['msg']}")
                
            # Process the OHLC data
            ohlc_data = data['data']
            df = pd.DataFrame(ohlc_data, columns=['t', 'o', 'h', 'l', 'c', 'v'])
            
            # Convert timestamp and format data
            df['timestamp'] = pd.to_datetime(df['t'], unit='s')
            df['closePrice'] = df['c'].astype(float)
            df = df[['timestamp', 'closePrice']]
            
            return df
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching spot data: {e}")
            return pd.DataFrame()

    def get_funding_rates(self, symbol: str, start_str: str, end_str: str, exchange:str,interval: str, limit: int ) -> pd.DataFrame:
        """
        Downloads historical OHLC funding rate data for a given symbol from CoinGlass API.
        """
        print(f"Downloading funding history rate data for {symbol}...")
        # Convert dates to timestamps
        start_ts = int(pd.to_datetime(start_str).timestamp()) #-9223372036854776
        end_ts = int(pd.to_datetime(end_str).timestamp()) #9223372036854776

        endpoint = f"{self.base_url}/api/futures/fundingRate/ohlc-history?exchange={exchange}&symbol={symbol}&type=futures&interval={interval}&limit={limit}&startTime={start_ts}&endTime={end_ts}"
   
        try:
            response = requests.get(endpoint, headers=self.headers)
            response.raise_for_status()
            data = response.json()
            
            if data['code'] != '0':
                raise ValueError(f"API Error: {data['msg']}")
                
            # Process the OHLC data
            ohlc_data = data['data']
            df = pd.DataFrame(ohlc_data, columns=['t', 'o', 'h', 'l', 'c', 'v'])
            
            # Convert timestamp and format data
            df['timestamp'] = pd.to_datetime(df['t'], unit='s')
            df["fundingRate"] = df["c"].astype(float)
            df = df[['timestamp', 'fundingRate']]
            
            return df
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching spot data: {e}")
            return pd.DataFrame()

def load_data(
    api_key: str,
    symbol: str,
    start_str: str,
    end_str: str,
    exchange: str,
    interval: str,
    limit: int,
    saveFile: bool = False,
) -> Dict[str, pd.DataFrame]:
    """
    Loads both spot prices and funding rates for a given symbol.
    
    Returns:
        Dict containing two DataFrames: 'spot' and 'funding'
    """

    api = CoinGlassAPI(api_key)
    
    # Get spot prices
    spot_df = api.get_spot_prices(symbol, start_str,end_str, exchange, interval, limit)

    spot_df.to_csv(f'{symbol}_{interval}_spot_prices.csv', index=False)
    
    # Get funding rates
    funding_df = api.get_funding_rates(symbol, start_str,end_str, exchange, interval, limit)

    funding_df.to_csv(f'{symbol}_{interval}_funding_rates.csv', index=False)
    
    return {
        'spot': spot_df,
        'funding': funding_df
    }

def compute_funding_performance(spot_df: pd.DataFrame, funding_df: pd.DataFrame, position_size: float = 1) -> pd.DataFrame:
    """
    Computes the funding performance metrics per exchange.
    """
    results = []
    
    for exchange in funding_df['exchange'].unique():
        exchange_data = funding_df[funding_df['exchange'] == exchange].copy()
        
        # Calculate funding PnL
        exchange_data['fundingPnL'] = exchange_data['fundingRate'] * position_size
        exchange_data['cumulativeFundingPnL'] = exchange_data['fundingPnL'].cumsum()
        
        # Get the corresponding spot price
        merged_data = pd.merge_asof(
            exchange_data.sort_values('timestamp'),
            spot_df.sort_values('timestamp'),
            on='timestamp',
            direction='nearest'
        )
        
        # Calculate returns
        initial_value = merged_data['closePrice'].iloc[0] * position_size
        merged_data['cumulativeReturnPct'] = 100 * merged_data['cumulativeFundingPnL'] / initial_value
        
        results.append(merged_data)
    
    return pd.concat(results, ignore_index=True)
