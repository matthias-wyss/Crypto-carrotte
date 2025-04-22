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
                raise ValueError(f"API Error: {data['msg']}")
            
           
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
            return {}

    
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
            df['date'] = pd.to_datetime(df['t'], unit='s')
            df["timestamp"] = df['t']
            df["closePrice"] = df["c"].astype(float)
            df = df[['date', 'timestamp', 'closePrice']]
            
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
            df['date'] = pd.to_datetime(df['t'], unit='s')
            df["timestamp"] = df['t']
            df["fundingRate"] = df["c"].astype(float)
            df = df[['date', 'timestamp', 'fundingRate']]
            
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
        Dataframe containing two DataFrames: 'spot' and 'funding'
    """

    api = CoinGlassAPI(api_key)
    
    # Get spot prices
    spot_df = api.get_spot_prices(symbol, start_str,end_str, exchange, interval, limit)

    spot_df.to_csv(f'{symbol}_{interval}_spot_prices.csv', index=False)
    
    # Get funding rates
    funding_df = api.get_funding_rates(symbol, start_str,end_str, exchange, interval, limit)

    funding_df.to_csv(f'{symbol}_{interval}_funding_rates.csv', index=False)


    merged_df = pd.merge(spot_df, funding_df, on=['date', 'timestamp'])

    return merged_df

def compute_funding_performance(
    api_key: str,
    symbol: str,
    start_str: str,
    end_str: str,
    interval: str,
    limit: int,
    exchanges: List[str],
    position_size: float = 1
) -> pd.DataFrame:
    """
    Computes the funding performance across multiple exchanges.
    
    Parameters:
        api_key: str - Your CoinGlass API key.
        symbol: str - The trading pair symbol (e.g., 'BTCUSDT').
        start_str: str - Start datetime as string.
        end_str: str - End datetime as string.
        interval: str - Time interval (e.g., '1h').
        limit: int - Number of data points.
        exchanges: List[str] - List of exchange names.
        position_size: float - Position size for PnL calculation.

    Returns:
        A DataFrame with performance metrics across all exchanges.
    """
    results = []

    for exchange in exchanges:
        print(f"\nProcessing exchange: {exchange}")

        

        # Load merged spot & funding data for this exchange
        merged_df = load_data(api_key, symbol, start_str, end_str, exchange, interval, limit)

        if merged_df.empty:
            print(f"Warning: No data for exchange {exchange}. Skipping.")
            continue

        # Add exchange identifier
        merged_df['exchange'] = exchange

        # Compute funding PnL
        merged_df['fundingPnL'] = merged_df['fundingRate'] * position_size
        merged_df['cumulativeFundingPnL'] = merged_df['fundingPnL'].cumsum()

        # Compute percentage return based on initial spot value
        initial_value = merged_df['closePrice'].iloc[0] * position_size
        merged_df['cumulativeReturnPct'] = 100 * merged_df['cumulativeFundingPnL'] / initial_value

        results.append(merged_df)

    return pd.concat(results, ignore_index=True)

