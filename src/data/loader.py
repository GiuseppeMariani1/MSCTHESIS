
import yfinance as yf
import pandas as pd
import numpy as np
import yaml
import os

def load_config(path="config/config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def download_prices(tickers, start_date, end_date):
    """Download adjusted closing prices from yfinance."""
    print(f"Downloading: {tickers}")
    raw = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)
    prices = raw["Close"]
    prices.dropna(how="all", inplace=True)
    print(f"✅ Downloaded {prices.shape[0]} days x {prices.shape[1]} assets")
    return prices

def compute_log_returns(prices):
    """Compute log returns from price series."""
    log_returns = np.log(prices / prices.shift(1)).dropna()
    print(f"✅ Log returns shape: {log_returns.shape}")
    return log_returns

def save_data(df, path):
    """Save dataframe to parquet."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path)
    print(f"✅ Saved to {path}")

def run(config_path="config/config.yaml"):
    # Load config
    config = load_config(config_path)
    tickers    = config["data"]["tickers"]
    start_date = config["data"]["start_date"]
    end_date   = config["data"]["end_date"]

    # Download
    prices = download_prices(tickers, start_date, end_date)

    # Log returns
    log_returns = compute_log_returns(prices)

    # Save
    paths = config['paths']
    save_data(prices,      paths['prices'])
    save_data(log_returns, paths['log_returns'])

    return prices, log_returns

if __name__ == "__main__":
    run()
