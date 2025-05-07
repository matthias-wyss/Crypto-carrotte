import time
import pandas as pd
from src.data.get_data_coinglass import CoinGlassAPI
from src.data.get_full_data_coinglass import load_future_data_for_exchange_pair, load_spot_data_for_exchange_pair


def process_tickers_separately(
    api: CoinGlassAPI,
    ticker_map: pd.DataFrame,
    start_date: str,
    end_date: str,
    interval: str,
    limit: int,
    is_spot: bool = False,
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
        is_spot: bool - If True, process spot data; if False, process futures data

    Returns:
        pd.DataFrame - Combined DataFrame of all results
    """

    MAX_CALL_PER_MINUTE = 25  # Has to keep below the 30 calls per minute limit

    results = []
    call_count = 0

    for _, row in ticker_map.iterrows():
        exchange = row['exchange']
        symbol = row['spotInstrumentId'] if is_spot else row['futuresInstrumentId']
        file_type = "spots" if is_spot else "futures"

        print(f"Processing {file_type} data for {symbol} on {exchange}...")

        # Call the appropriate function based on is_spot
        if is_spot:
            data_df = load_spot_data_for_exchange_pair(
                api=api,
                spot_exchange=exchange,
                spot_symbol=symbol,
                start_str=start_date,
                end_str=end_date,
                interval=interval,
                limit=limit,
                save_files=True,  # Set to save files
            )
        else:
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
            print(f"No {file_type} data available for {symbol} on {exchange}")
            continue

        results.append(data_df)

    # Combine all results into a single DataFrame
    combined_df = pd.concat(results, ignore_index=True)

    combined_df.to_csv(f"./data/concat/{file_type}_data_{start_date}_to_{end_date}.csv", index=False)
    return combined_df
