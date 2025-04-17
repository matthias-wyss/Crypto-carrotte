import pandas as pd
import requests
import time
from dune_client.client import DuneClient
from dune_client.query import QueryBase
import os

def get_spot_prices(symbol='BTCUSDT', interval='1d', start_str='2021-01-01', end_str='2023-12-31'):
    """
    Downloads historical spot closing prices for a given trading pair from the Binance API.

    Parameters:
        symbol (str): The trading pair symbol (e.g., 'BTCUSDT').
        interval (str): Time interval between data points (default is '1d' for daily data).
        start_str (str): Start date in 'YYYY-MM-DD' format.
        end_str (str): End date in 'YYYY-MM-DD' format.

    Returns:
        pd.DataFrame: A DataFrame containing 'timestamp' and 'closePrice' columns.
    """
    print(f"Downloading spot price data for {symbol}...")
    start_ts = int(pd.to_datetime(start_str).timestamp() * 1000)
    end_ts = int(pd.to_datetime(end_str).timestamp() * 1000)
    url = f"https://api.binance.com/api/v3/klines"
    klines = []
    
    # Using session for better performance
    with requests.Session() as session:
        while start_ts < end_ts:
            params = {
                'symbol': symbol,
                'interval': interval,
                'startTime': start_ts,
                'limit': 1000
            }
            try:
                response = session.get(url, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()
                if not data:
                    break
                klines += data
                start_ts = data[-1][0] + 1
                time.sleep(0.5)  # Adjust based on rate limits
            except requests.exceptions.RequestException as e:
                print(f"Error fetching spot data: {e}")
                break
    
    # Process data
    df = pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'num_trades',
        'taker_buy_base', 'taker_buy_quote', 'ignore'
    ])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['close'] = df['close'].astype(float)
    df = df[['timestamp', 'close']]
    df.rename(columns={'close': f'closePrice'}, inplace=True)
    return df


def fetch_historical_funding_rates(symbol, rows=10000):
    """
    Downloads historical funding rates for a given perpetual futures symbol from Binance.

    Parameters:
        symbol (str): Futures symbol (e.g., 'BTCUSDT', 'ETHUSDT').
        rows (int): Number of rows (data points) to fetch, max is typically 10,000.

    Returns:
        pd.DataFrame: A DataFrame containing 'timestamp' and 'fundingRate' columns.
                      Returns an empty DataFrame on error.
    """
    print(f"Downloading funding rates data for {symbol}...")
    url = "https://www.binance.com/bapi/futures/v1/public/future/common/get-funding-rate-history"
    payload = {
        "symbol": symbol,
        "page": 1,
        "rows": rows
    }
    headers = {
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=10)
        response.raise_for_status()
        raw_data = response.json()
        if 'data' not in raw_data:
            raise ValueError("Unexpected response format:", raw_data)
    except requests.exceptions.RequestException as e:
        print(f"Error fetching funding rates: {e}")
        return pd.DataFrame()  # Return an empty dataframe in case of error

    df = pd.DataFrame(raw_data['data'])
    df['timestamp'] = pd.to_datetime(df['calcTime'], unit='ms')
    df['fundingRate'] = df['lastFundingRate'].astype(float)
    df = df[['timestamp', 'fundingRate']]
    df.sort_values('timestamp', inplace=True)
    return df


def load_data(symbol):
    """
    Loads and merges funding rate data with spot closing prices for a given symbol.

    This function:
    - Fetches historical funding rates.
    - Downloads spot prices using the time range from the funding data.
    - Merges both datasets on the date to provide aligned funding and price information.

    Parameters:
        symbol (str): Trading pair symbol (e.g., 'BTCUSDT').

    Returns:
        pd.DataFrame: A merged DataFrame with columns:
                      'timestamp' (from funding data),
                      'fundingRate',
                      'closePrice' (spot price for the corresponding date).
    """
    # Fetch historical funding rates
    funding = fetch_historical_funding_rates(symbol)
    
    # Download spot prices for the symbol based on funding rate dates
    spot = get_spot_prices(symbol=symbol, start_str=str(funding['timestamp'].min().date()), end_str=str(funding['timestamp'].max().date()))

    # Ensure 'date' columns are of datetime type
    funding['date'] = pd.to_datetime(funding['timestamp'].dt.date)
    
    # Rename 'timestamp' column in spot data to avoid conflicts
    spot.rename(columns={'timestamp': f'{symbol.lower()}_spot_timestamp'}, inplace=True)

    # Merge spot prices with funding data on the 'date' column
    merged_data = pd.merge(funding, spot, how='left', left_on='date', right_on=f'{symbol.lower()}_spot_timestamp')
    
    # Drop unnecessary columns
    merged_data.drop(columns=[f'{symbol.lower()}_spot_timestamp', 'date'], inplace=True)
    
    return merged_data


def compute_funding_performance(df, position_size=1):
    """
    Computes the funding performance metrics of a carry trade strategy 
    based on perpetual futures funding rates.

    Parameters:
        df : pd.DataFrame
            A DataFrame containing at least the following columns:
                - 'fundingRate': the 8-hour funding rate (expressed as a decimal).
                - 'closePrice': the spot price of the asset (e.g., BTC or ETH).
        position_size : float, optional (default=1)
            The size of the position held in units of the asset (e.g., 1 BTC).

    Returns:
        pd.DataFrame
            The input DataFrame augmented with the following columns:
                - 'fundingPnL': profit or loss per period from funding rate.
                - 'cumulativeFundingPnL': cumulative funding PnL over time.
                - 'cumulativeReturnPct': cumulative funding return in percentage,
                normalized by the initial value of the spot position.
    """

    # Calculate the funding profit/loss for each 8-hour period
    df['fundingPnL'] = df['fundingRate'] * position_size

    # Compute the cumulative sum of funding PnL over time
    df['cumulativeFundingPnL'] = df['fundingPnL'].cumsum()

    # Calculate the initial spot value of the position (in USD or stablecoin)
    initial_value = df['closePrice'].iloc[0] * position_size

    # Compute the cumulative return percentage relative to initial value
    df['cumulativeReturnPct'] = 100 * df['cumulativeFundingPnL'] / initial_value

    return df


def save_dune_query_to_csv(api_key: str, query_id: int, filename: str) -> None:
    """
    Execute a Dune query and save the results as a CSV file.

    Parameters:
    - api_key (str): Your Dune API key (get it from https://dune.com/account).
    - query_id (int): The ID of the Dune query to execute.
    - filename (str): The path to the CSV file to save the results to.

    Returns:
    - None
    """
    # Initialize the Dune client
    dune = DuneClient(api_key=api_key)

    # Create a reference to the query (no parameters in this case)
    query = QueryBase(name="AutoSaved Query", query_id=query_id, params=[])

    # Run the query and get the result
    execution_result = dune.run_query(query)

    # Convert the result into a pandas DataFrame
    df = pd.DataFrame(execution_result.result.rows)

    # Save the DataFrame to a CSV file
    df.to_csv(filename, index=False)
    print(f"✅ Results saved to {filename}")


def load_csv_to_df(filename: str) -> pd.DataFrame:
    """
    Load a CSV file into a pandas DataFrame.

    Parameters:
    - filename (str): Path to the CSV file to load.

    Returns:
    - pd.DataFrame: A DataFrame containing the CSV data.
    """
    # Load the CSV into a DataFrame and parse the 'time' column as datetime
    df = pd.read_csv(filename, parse_dates=["time"])
    print(f"📂 Loaded {len(df)} rows from {filename}")
    return df


def merge_funding_rates_and_apr_data(df_eth: pd.DataFrame, df_lido_apr: pd.DataFrame) -> pd.DataFrame:
    """
    Merges the Ethereum funding rate data with Lido APR data based on matching timestamps.
    Keeps only rows where both DataFrames have data for the same day.

    Parameters:
        df_eth (pd.DataFrame): Ethereum funding rate data.
        df_lido_apr (pd.DataFrame): Lido APR data.

    Returns:
        pd.DataFrame: Merged DataFrame with funding and APR data.
    """

    # Convert to datetime and drop timezone
    df_eth['timestamp'] = pd.to_datetime(df_eth['timestamp']).dt.tz_localize(None)
    df_lido_apr['time'] = pd.to_datetime(df_lido_apr['time']).dt.tz_localize(None)

    # Extract date (without time)
    df_eth['date'] = df_eth['timestamp'].dt.date
    df_lido_apr['date'] = df_lido_apr['time'].dt.date

    # Keep only common dates
    common_dates = set(df_eth['date']).intersection(set(df_lido_apr['date']))
    df_eth = df_eth[df_eth['date'].isin(common_dates)].copy()
    df_lido_apr = df_lido_apr[df_lido_apr['date'].isin(common_dates)].copy()

    # Sort for merge_asof
    df_eth = df_eth.sort_values('timestamp')
    df_lido_apr = df_lido_apr.sort_values('time')

    # Merge on nearest timestamp
    merged_df = pd.merge_asof(df_eth, df_lido_apr, left_on='timestamp', right_on='time', direction='nearest')

    # Add PnL from funding + APR
    merged_df['cumulativeFundingAPRPnL'] = merged_df['cumulativeFundingPnL'] + merged_df['Lido staking APR(instant)']

    return merged_df
