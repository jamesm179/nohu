import pandas as pd
import numpy as np
import talib
from sklearn.preprocessing import MinMaxScaler
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

def add_technical_indicators(df):
    """Adds technical indicators to the DataFrame."""
    try:
        df['RSI'] = talib.RSI(df['close'])
        df['MACD'], df['MACD_signal'], df['MACD_hist'] = talib.MACD(df['close'])
        df['upper_band'], df['middle_band'], df['lower_band'] = talib.BBANDS(df['close'])
        df['ATR'] = talib.ATR(df['high'], df['low'], df['close'])
        # Add more indicators as needed
        logging.info("Technical indicators added.")
        return df
    except Exception as e:
        logging.error(f"Error adding technical indicators: {e}")
        return df

def calculate_pcr(option_chain):
    """Calculates the Put-Call Ratio from the option chain."""
    try:
        total_put_oi = sum(x['put_oi'] for x in option_chain)
        total_call_oi = sum(x['call_oi'] for x in option_chain)

        if total_call_oi == 0:
            return 0 # Avoid division by zero

        return total_put_oi / total_call_oi
    except Exception as e:
        logging.error(f"Error calculating PCR: {e}")
        return 0

def calculate_max_pain(option_chain):
    """Calculates Max Pain from the option chain (simplified placeholder)."""
    # This is a highly simplified version. A real implementation would be more complex.
    if not option_chain:
        return 0

    strikes = sorted(list(set(float(o['strike_price']) for o in option_chain)))
    if not strikes:
        return 0

    # For this placeholder, we'll just return the strike with the highest total OI
    max_oi_strike = 0
    max_oi = 0
    for strike in strikes:
        total_oi = sum(float(o.get('put_oi', 0)) + float(o.get('call_oi', 0)) for o in option_chain if float(o['strike_price']) == strike)
        if total_oi > max_oi:
            max_oi = total_oi
            max_oi_strike = strike
    return max_oi_strike

def calculate_iv_rank(df):
    """Calculates IV Rank (simplified placeholder)."""
    # This requires historical IV data, which we don't have yet.
    # We'll simulate a simple IV rank based on recent price volatility.
    if 'close' in df.columns:
        volatility = df['close'].pct_change().rolling(window=20).std()
        if not volatility.empty:
            current_iv = volatility.iloc[-1]
            min_iv = volatility.min()
            max_iv = volatility.max()
            if max_iv - min_iv > 0:
                return (current_iv - min_iv) / (max_iv - min_iv)
    return 0.5 # Default value

def add_options_features(df, pcr_series=None):
    """Adds options-specific features."""
    if 'oi' in df.columns:
        df['oi_change'] = df['oi'].diff()

    if pcr_series is not None:
        df = df.join(pcr_series, how='left').fillna(method='ffill')

    logging.info("Options-specific features added.")
    return df

def add_time_features(df):
    """Adds time-based features."""
    if isinstance(df.index, pd.DatetimeIndex):
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        # Placeholder for time to expiry (requires expiry date)
        # df['time_to_expiry'] = (df['expiry_date'] - df.index).dt.days
        df['time_to_expiry'] = 10
        logging.info("Time features added.")
    else:
        logging.warning("DataFrame index is not a DatetimeIndex. Cannot add time features.")
    return df

def create_labels(df, future_periods=10, threshold=0.01):
    """Creates buy/sell/hold labels based on future price movement."""
    df['future_return'] = df['close'].pct_change(future_periods).shift(-future_periods)

    df['label'] = 0 # Hold
    df.loc[df['future_return'] > threshold, 'label'] = 1 # Buy
    df.loc[df['future_return'] < -threshold, 'label'] = -1 # Sell

    df = df.drop(columns=['future_return'])
    logging.info("Labels created.")
    return df

def normalize_features(df):
    """Scales features for the ML model."""
    # Exclude non-feature columns
    feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Remove label and any other non-feature columns
    cols_to_exclude = ['label']
    feature_cols = [col for col in feature_cols if col not in cols_to_exclude]

    scaler = MinMaxScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    logging.info("Features normalized.")
    return df, scaler

def create_sequences(data, lookback):
    """Creates sequences for LSTM model input."""
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data.iloc[i:i+lookback].drop('label', axis=1).values)
        y.append(data.iloc[i+lookback]['label'])
    return np.array(X), np.array(y)

if __name__ == '__main__':
    # Create a sample DataFrame for demonstration
    data = {
        'timestamp': pd.to_datetime(pd.date_range('2023-01-01', periods=100, freq='h')),
        'open': np.random.rand(100) * 100,
        'high': np.random.rand(100) * 100,
        'low': np.random.rand(100) * 100,
        'close': np.random.rand(100) * 100,
        'volume': np.random.rand(100) * 1000,
        'oi': np.random.rand(100) * 10000
    }
    df = pd.DataFrame(data).set_index('timestamp')

    # Feature Engineering Pipeline
    df = add_technical_indicators(df)
    df = add_options_features(df)
    df = add_time_features(df)

    # Create Labels
    df = create_labels(df)

    # Drop rows with NaN values created by indicators
    df.dropna(inplace=True)

    # Normalize Features
    df_normalized, scaler = normalize_features(df.copy())

    # Create Sequences for LSTM
    X, y = create_sequences(df_normalized, lookback=10)

    logging.info(f"Original DataFrame shape: {df.shape}")
    logging.info(f"Normalized DataFrame shape: {df_normalized.shape}")
    logging.info(f"LSTM sequences shape X: {X.shape}, y: {y.shape}")
    print(df_normalized.head())
