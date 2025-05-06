import time
import pandas as pd
from src.data.get_data_coinglass import CoinGlassAPI
from src.data.get_full_data_coinglass import load_future_data_for_exchange_pair


def process_tickers_separately(
    api: CoinGlassAPI,
    ticker_map: pd.DataFrame,
    start_date: str,
    end_date: str,
    interval: str,
    limit: int,
) -> pd.DataFrame:
    """
    Processes either spot or futures data based on the `is_spot` parameter.

    Parameters:
        api: CoinGlassAPI - API instance
        ticker_map: pd.DataFrame - DataFrame containing tickers (spot or futures)
        start_date: str - Start date for data fetching
        end_date: str - End date for data fetching
        interval: str - Time interval for data fetching
        limit: int - Data point limit

    Returns:
        pd.DataFrame - Combined DataFrame of all results
    """

    MAX_CALL_PER_MINUTE = 30  # Has to keep below the 30 calls per minute limit

    results = []
    call_count = 0

    for _, row in ticker_map.iterrows():
        exchange = row['exchange']
        symbol = row['futuresInstrumentId']

        print(f"Processing futures data for {symbol} on {exchange}...")

        data_df = load_future_data_for_exchange_pair(
            api=api,
            perp_exchange=exchange,
            perp_symbol=symbol,
            start_str=start_date,
            end_str=end_date,
            interval=interval,
            limit=limit,
            save_files=True,  # Set to save files
        )

        # Increment call count and check if we need to wait
        call_count += 1
        if call_count >= MAX_CALL_PER_MINUTE:
            print("Reached maximum API calls. Waiting for 1 minute...")
            time.sleep(60)
            call_count = 0  # Reset call count after waiting

        if data_df.empty:
            print(f"No futures data available for {symbol} on {exchange}")
            continue

        results.append(data_df)

    # Combine all results into a single DataFrame
    combined_df = pd.concat(results, ignore_index=True)

    file_type = "futures"
    combined_df.to_csv(f"./data/concat/{file_type}_data_{start_date}_to_{end_date}.csv", index=False)
    return combined_df