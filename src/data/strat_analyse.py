import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from src.data.get_data import load_data, compute_funding_performance
from src.utils.plot_data import plot_funding_and_price, plot_cumulative_funding_pnl, plot_funding_rate_distribution, plot_annualized_funding_rate



def merge_spot_funding(
        df_FR : pd.DataFrame, 
        df_SP : pd.DataFrame, 
        ) -> pd.DataFrame:
    """
    Merge spot and funding data on date and timestamp.

    Returns:
        pd.DataFrame: Merged DataFrame with relevant columns.
    """

    # Add exchange identifiers
    # spot_df['spot_exchange'] = spot_exchange
    # spot_df['spot_symbol'] = spot_symbol
    # funding_df['perp_exchange'] = perp_exchange
    # funding_df['perp_symbol'] = perp_symbol

    # Merge on timestamp (retain all relevant columns)
    merged_df = pd.merge(
        df_SP,
        df_FR,
        on=['date', 'timestamp'],
        how='inner'
    )

    # if merged_df.empty:
    #     print(f"No matching timestamps between spot and funding data for {spot_exchange}/{perp_exchange}")
    #     return pd.DataFrame()

    # Include all columns in the merged DataFrame
    merged_df = merged_df[[
        'date', 'timestamp', 'SP_open', 'SP_close', 'SP_high', 'SP_low', 'SP_vol', 'spot_exchange', 'spot_symbol', 
        'FR_open', 'FR_close', 'FR_high', 'FR_low', 'perp_exchange', 'perp_symbol'
    ]]

    return merged_df

def prepare_data_for_analyse(
        df_FR : pd.DataFrame, 
        df_SP : pd.DataFrame, 
        perp_exchange : str = "Binance",
        perp_BTC_symbol : str = "BTCUSDT",
        perp_ETH_symbol : str = "ETHUSDT",
        spot_exchange : str = "Binance",
        spot_BTC_symbol : str = "BTCUSDT",
        spot_ETH_symbol : str = "ETHUSDT",
        funding_col : str = "FR_close",
        price_col : str = "SP_close") : 
    
    df_FR["perp_exchange"] = perp_exchange
    df_FR_btc = df_FR[df_FR['perp_symbol'] == perp_BTC_symbol]
    df_FR_eth = df_FR[df_FR['perp_symbol'] == perp_ETH_symbol]

    df_SP["spot_exchange"] = spot_exchange
    df_SP_btc = df_SP[df_SP['spot_symbol'] == spot_BTC_symbol]
    df_SP_eth = df_SP[df_SP['spot_symbol'] == spot_ETH_symbol]

    # merge funding and spot data
    btc_merged_df = merge_spot_funding(df_FR_btc, df_SP_btc)
    eth_merged_df = merge_spot_funding(df_FR_eth, df_SP_eth)

    # renaming for compatibility with the analysis function
    btc_merged_df['fundingRate'] = btc_merged_df[funding_col]
    btc_merged_df['closePrice'] = btc_merged_df[price_col]
    eth_merged_df['fundingRate'] = eth_merged_df[funding_col]
    eth_merged_df['closePrice'] = eth_merged_df[price_col]

    # make sure each date is unique 
    btc_merged_df = btc_merged_df.drop_duplicates(subset=['timestamp'])
    eth_merged_df = eth_merged_df.drop_duplicates(subset=['timestamp'])

    return btc_merged_df, eth_merged_df

def strat_analyse(
        df_btc: pd.DataFrame,
        df_eth: pd.DataFrame,
        position_size: float = 1.0,
) :
    """
    Analyze the funding performance of BTC and ETH dataframes.

    Args:
        df_btc (pd.DataFrame): DataFrame containing BTC data.
        df_eth (pd.DataFrame): DataFrame containing ETH data.
        position_size (float): Position size for funding performance calculation.

    Returns:
        None
    """

    plot_funding_and_price(df = df_btc, symbol = 'BTC', funding_rate_col = 'FR_close', close_price_col = 'SP_close')
    
    plot_funding_and_price(df = df_eth, symbol = 'ETH', funding_rate_col = 'FR_close', close_price_col = 'SP_close')

    # Define position size (e.g., 1 unit of asset)

    # BTC
    df_btc = compute_funding_performance(df_btc, position_size)

    # ETH
    df_eth = compute_funding_performance(df_eth, position_size)


    dfs = {
        'BTC': df_btc,
        'ETH': df_eth
    }

    plot_cumulative_funding_pnl(dfs)


    plot_funding_rate_distribution(df_btc, 'BTC')

    plot_funding_rate_distribution(df_eth, 'ETH')