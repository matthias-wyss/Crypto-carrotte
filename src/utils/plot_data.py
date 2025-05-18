import matplotlib.pyplot as plt
import os
import seaborn as sns
import pandas as pd
import matplotlib.dates as mdates

def plot_funding_and_price(df, symbol="BTC"):
    """
    Plots two separate graphs: one for funding rate, one for close price.
    Also saves both plots as images in ../../data/images.

    Parameters:
        df (pd.DataFrame): DataFrame with 'timestamp', 'fundingRate', and 'closePrice' columns.
        symbol (str): Asset symbol for labeling purposes (e.g., 'BTC', 'ETH').
    """
    # Ensure 'timestamp' is a datetime object
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Create directory if it doesn't exist
    image_dir = os.path.join("data", "images")
    os.makedirs(image_dir, exist_ok=True)

    # Plot funding rate
    plt.figure(figsize=(14, 6))
    plt.plot(df['timestamp'], df['fundingRate'], color='tab:blue', alpha=0.8)
    plt.title(f'{symbol} - Funding Rate Over Time')
    plt.xlabel('Date')
    plt.ylabel('Funding Rate')
    plt.axhline(0, color='black', linestyle='--', linewidth=2)  # Zero line
    plt.grid(True)
    plt.xticks(rotation=45)  # Rotate x-axis labels for clarity
    plt.tight_layout()
    funding_path = os.path.join(image_dir, f"{symbol}_funding_rate.png")
    plt.savefig(funding_path)
    plt.show()

    # Plot close price
    plt.figure(figsize=(14, 6))
    plt.plot(df['timestamp'], df['closePrice'], color='tab:orange')
    plt.title(f'{symbol} - Close Price Over Time')
    plt.xlabel('Date')
    plt.ylabel(f'{symbol} Price (USDT)')
    plt.grid(True)
    plt.xticks(rotation=45)  # Rotate x-axis labels for clarity
    plt.tight_layout()
    price_path = os.path.join(image_dir, f"{symbol}_close_price.png")
    plt.savefig(price_path)
    plt.show()

    print(f"Saved plots to:\n - {funding_path}\n - {price_path}")


def plot_cumulative_funding_pnl(dfs: dict, filename="cumulative_funding_pnl.png"):
    """
    Plots and saves the cumulative funding PnL for multiple tokens as a percentage of the initial position size.

    Parameters:
        dfs (dict): Dictionary where keys are token symbols (e.g., 'BTC', 'ETH') and
                    values are DataFrames with 'timestamp' and 'cumulativeFundingPnL' columns.
        filename (str): Output filename to save the plot in data/images directory.
    """
    # Create directory if it doesn't exist
    image_dir = os.path.join("data", "images")
    os.makedirs(image_dir, exist_ok=True)
    save_path = os.path.join(image_dir, filename)

    # Plotting
    plt.figure(figsize=(14, 6))
    for symbol, df in dfs.items():
        df['timestamp'] = pd.to_datetime(df['timestamp'])  # Ensure 'timestamp' is datetime
        
        # Calculate the percentage of the initial PnL
        initial_pnl = df['cumulativeFundingPnL'].iloc[0]  # Initial position size PnL
        df['cumulativeFundingPnL_pct'] = (df['cumulativeFundingPnL'] - initial_pnl) / (initial_pnl * 1000)
        
        # Plot the percentage PnL
        plt.plot(df['timestamp'], df['cumulativeFundingPnL_pct'], label=symbol)
    
    plt.title("Cumulative Funding PnL (Percentage of Initial Position Size)")
    plt.xlabel("Date")
    plt.ylabel("PnL (%)")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)  # Rotate x-axis labels for clarity
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

    print(f"Saved cumulative PnL plot to: {save_path}")



def plot_funding_rate_distribution(df, symbol="BTC"):
    """
    Plots a histogram and Kernel Density Estimate (KDE) of funding rates.
    Displays the percentage of funding rates above and below 0, and saves the plot as an image in data/images.

    Parameters:
        df (pd.DataFrame): DataFrame with 'fundingRate' column.
        symbol (str): Asset symbol for labeling purposes (e.g., 'BTC', 'ETH').
    """
    # Ensure 'timestamp' is a datetime object
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Create directory if it doesn't exist
    image_dir = os.path.join("data", "images")
    os.makedirs(image_dir, exist_ok=True)

    # Calculate percentage of funding rates above and below 0
    above_zero_pct = (df['fundingRate'] > 0).mean() * 100
    below_zero_pct = (df['fundingRate'] < 0).mean() * 100

    # Plot histogram and KDE
    plt.figure(figsize=(14, 6))
    sns.histplot(df['fundingRate'], kde=True, color='tab:blue', bins=50, stat="density")
    
    # Title and labels
    plt.title(f'{symbol} - Funding Rate Distribution (Histogram and KDE)')
    plt.xlabel('Funding Rate')
    plt.ylabel('Density (%)')
    plt.grid(True)

    # Set left axis as percentage of total density (i.e., probability density / 100)
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x/100}%'))

    # Display percentage text
    plt.figtext(0.15, 0.85, f'Funding Rate > 0: {above_zero_pct:.2f}%', fontsize=12, color='tab:green')
    plt.figtext(0.15, 0.80, f'Funding Rate < 0: {below_zero_pct:.2f}%', fontsize=12, color='tab:red')

    # Save the plot as an image
    distribution_path = os.path.join(image_dir, f"{symbol}_funding_rate_distribution.png")
    plt.tight_layout()
    plt.savefig(distribution_path)
    plt.show()

    print(f'Funding Rate > 0: {above_zero_pct:.2f}%')
    print(f'Funding Rate < 0: {below_zero_pct:.2f}%')
    print(f"Saved plot to: {distribution_path}")


def plot_annualized_funding_rate(df, symbol="BTC"):
    """
    Plots the annualized funding rate (1095 * mean(fundingRate)) over rolling windows of varying lengths (7 days, 30 days, 3 months, and 6 months).
    Saves the plot as an image in data/images and shows it. Also highlights the zero line.

    Parameters:
        df (pd.DataFrame): DataFrame containing 'fundingRate' and 'timestamp' columns.
        symbol (str): Asset symbol for labeling purposes (e.g., 'BTC', 'ETH').
    """
    # Ensure 'timestamp' is a datetime object
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    
    # Drop rows where 'timestamp' is NaT (Not a Time)
    df = df.dropna(subset=['timestamp'])
    
    # Create the 'images' directory if it doesn't already exist
    image_dir = os.path.join("data", "images")
    os.makedirs(image_dir, exist_ok=True)

    # Calculate rolling means and annualize the funding rate (1095 * mean * 100 to get percentage)
    
    # 7-day window (21 periods for 8-hour data)
    df['rolling_mean_7'] = df['fundingRate'].rolling(window=21).mean()
    df['annualized_funding_rate_7'] = df['rolling_mean_7'] * 1095 * 100  # Annualize for 8-hour data

    # 30-day window (90 periods for 8-hour data)
    df['rolling_mean_30'] = df['fundingRate'].rolling(window=90).mean()
    df['annualized_funding_rate_30'] = df['rolling_mean_30'] * 1095 * 100  # Annualize for 8-hour data

    # 3-month window (approx. 3 months * 30 days = 90 days -> 270 periods for 8-hour data)
    df['rolling_mean_3_months'] = df['fundingRate'].rolling(window=270).mean()
    df['annualized_funding_rate_3_months'] = df['rolling_mean_3_months'] * 1095 * 100  # Annualize for 8-hour data

    # 6-month window (approx. 6 months * 30 days = 180 days -> 540 periods for 8-hour data)
    df['rolling_mean_6_months'] = df['fundingRate'].rolling(window=540).mean()
    df['annualized_funding_rate_6_months'] = df['rolling_mean_6_months'] * 1095 * 100  # Annualize for 8-hour data

    # Drop rows with NaN values in any of the new annualized columns
    df = df.dropna(subset=[
        'annualized_funding_rate_7', 
        'annualized_funding_rate_30', 
        'annualized_funding_rate_3_months', 
        'annualized_funding_rate_6_months'
    ])
    
    # Create the plot
    plt.figure(figsize=(14, 6))

    # Plot the annualized funding rate for different windows
    sns.lineplot(data=df, x='timestamp', y='annualized_funding_rate_7', label="7-Day Window", color='tab:orange')
    sns.lineplot(data=df, x='timestamp', y='annualized_funding_rate_30', label="30-Day Window", color='tab:blue')
    sns.lineplot(data=df, x='timestamp', y='annualized_funding_rate_3_months', label="3-Month Window", color='tab:green')
    sns.lineplot(data=df, x='timestamp', y='annualized_funding_rate_6_months', label="6-Month Window", color='tab:red')

    # Title and labels for the plot
    plt.title(f'{symbol} - Annualized Funding Rate Over Time (Various Rolling Windows)')
    plt.xlabel('Date')
    plt.ylabel('Annualized Funding Rate (%)')
    plt.grid(True, linestyle='--', alpha=0.5)

    # Highlight the zero line
    plt.axhline(y=0, color='black', linestyle='--', linewidth=2)

    # Format the x-axis to show years from the timestamp
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())  # Major ticks every year
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))  # Format as year
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability

    # Save the plot as an image
    annualized_path = os.path.join(image_dir, f"{symbol}_funding_rate_rolling.png")
    plt.tight_layout()
    plt.savefig(annualized_path)
    plt.show()

    # Output the path where the plot was saved
    print(f"Saved plot to: {annualized_path}")


def plot_lido_apr(df):
    # Create the 'images' directory if it doesn't already exist
    image_dir = os.path.join("data", "images")

    plt.figure(figsize=(12, 6))
    plt.plot(df['time'], df['Lido staking APR(instant)'], label='Instant APR', alpha=0.5)
    plt.plot(df['time'], df['Lido staking APR(ma_7)'], label='7-Day MA APR')
    plt.plot(df['time'], df['Lido staking APR(ma_30)'], label='30-Day MA APR')
    plt.plot(df['time'], df['protocol_apr'], label='Protocol APR (ma_7)', linestyle='--', color='grey')
    
    plt.title('Lido Staking APR Over Time')
    plt.xlabel('Time')
    plt.ylabel('APR (%)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    lido_apr_path = os.path.join(image_dir, "lido_stacking_apr.png")
    plt.savefig(lido_apr_path)
    plt.show()


def plot_cumulative_funding_apr_pnl(df: pd.DataFrame, filename="cumulative_funding_apr_pnl.png"):
    """
    Plots and saves the cumulative Funding + APR PnL over time.

    Parameters:
        df (pd.DataFrame): DataFrame containing at least 'timestamp' and 'cumulativeFundingAPRPnL' columns.
        filename (str): Output filename to save the plot in the data/images directory.
    """

    # Create output directory if needed
    image_dir = os.path.join("data", "images")
    os.makedirs(image_dir, exist_ok=True)
    save_path = os.path.join(image_dir, filename)

    # Ensure timestamp is datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Plot
    plt.figure(figsize=(14, 6))
    plt.plot(df['timestamp'], df['cumulativeFundingPnL'], label='Funding PnL', color='orange')
    plt.plot(df['timestamp'], df['Lido staking APR(instant)'], label='APR PnL', color='lightblue')
    plt.plot(df['timestamp'], df['cumulativeFundingAPRPnL'], label='Funding + APR PnL', color='green')
    plt.title("Cumulative Funding + APR PnL Over Time")
    plt.xlabel("Date")
    plt.ylabel("PnL (%)")
    plt.grid(True)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

    print(f"Saved cumulative Funding + APR PnL plot to: {save_path}")


def plot_annualized_return_comparison(df, filename="annualized_return_comparison.png"):
    """
    Plots a bar chart comparing mean annualized return: ETH carry vs. ETH carry + staking.
    
    Parameters:
        df (pd.DataFrame): DataFrame with columns for funding and staking returns.
        filename (str): Filename for saving the plot.
    """
    # Create output directory if needed
    image_dir = os.path.join("data", "images")
    os.makedirs(image_dir, exist_ok=True)
    save_path = os.path.join(image_dir, filename)

    # Estimate annualized funding return (assuming 8-hour funding rate)
    annualized_funding = df['fundingRate'].mean() * 3 * 365 * 100 # 3 cycles per day
    mean_staking_apr = df['Lido staking APR(instant)'].mean()
    combined_return = annualized_funding + mean_staking_apr

    # Plot
    labels = ['Funding Only', 'Funding + Staking']
    values = [annualized_funding, combined_return]

    plt.figure(figsize=(8, 6))
    bars = plt.bar(labels, values, color=['orange', 'green'])
    plt.ylabel("Mean Annualized Return (%)")
    plt.title("ETH Carry Trade: Mean Annualized Return Comparison")

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height * 1.01,
                 f'{height:.2f}%', ha='center', va='bottom', fontsize=11)

    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

    print(f"Saved bar chart comparing annualized returns to: {save_path}")