import tensorflow as tf
import joblib
import numpy as np
import pandas as pd
import logging
from data_collector import get_latest_data, fetch_option_chain, get_ltp
from feature_engineer import add_technical_indicators, add_options_features, add_time_features, calculate_pcr
import yaml

# Configure logging
logging.basicConfig(level=logging.INFO)

def load_model_and_scaler(model_path='models/lstm_model.h5', scaler_path='models/scaler.joblib', columns_path='models/feature_columns.joblib'):
    """Loads the trained model, scaler, and feature columns."""
    try:
        model = tf.keras.models.load_model(model_path)
        scaler = joblib.load(scaler_path)
        feature_columns = joblib.load(columns_path)
        logging.info("Model, scaler, and feature columns loaded successfully.")
        return model, scaler, feature_columns
    except Exception as e:
        logging.error(f"Error loading model, scaler, or feature columns: {e}")
        return None, None, None

def get_live_data(conn, symbol, lookback):
    """Fetches live market data from the database."""
    return get_latest_data(conn, symbol, lookback)

def preprocess_live_data(df, scaler, api, symbol, feature_columns, option_chain):
    """Applies the same transformations as training."""
    if option_chain:
        pcr = calculate_pcr(option_chain)
        df['pcr'] = pcr
    else:
        df['pcr'] = 0 # Default value if option chain is not available

    df = add_technical_indicators(df)
    df = add_options_features(df) # add_options_features will now just use the columns in the df
    df = add_time_features(df)
    df.dropna(inplace=True)

    # Reorder columns to match the training data
    df = df.reindex(columns=feature_columns, fill_value=0)

    df[feature_columns] = scaler.transform(df[feature_columns])

    return df

def generate_signal(model, data, lookback):
    """Generates a trading signal from the model's prediction."""
    if len(data) < lookback:
        return "Not enough data", 0

    sequence = data.iloc[-lookback:].values
    sequence = np.expand_dims(sequence, axis=0)

    prediction = model.predict(sequence)
    signal_index = np.argmax(prediction, axis=1)[0]
    confidence = np.max(prediction)

    # Map index back to signal: 0 -> Sell, 1 -> Hold, 2 -> Buy
    signal_map = {-1: "Sell", 0: "Hold", 1: "Buy"}
    signal = signal_map[signal_index - 1]

    return signal, confidence

def filter_signals(signal, confidence, confidence_threshold=0.7):
    """Filters signals based on a confidence threshold."""
    if confidence >= confidence_threshold:
        return signal
    return "Hold" # Default to Hold if confidence is too low

def identify_opportunities(signal, option_chain, spot_price):
    """Scans the option chain for tradeable setups."""
    if not option_chain:
        logging.warning("Option chain is empty. Cannot identify opportunities.")
        return None

    try:
        if signal == "Buy": # Bullish signal, look for OTM calls
            # Find the strike price just above the spot price
            otm_calls = [o for o in option_chain if float(o['strike_price']) > spot_price]
            if not otm_calls:
                return None
            # Sort by strike price to find the nearest OTM
            best_option = min(otm_calls, key=lambda x: float(x['strike_price']))
            return best_option['symbol']

        elif signal == "Sell": # Bearish signal, look for OTM puts
            # Find the strike price just below the spot price
            otm_puts = [o for o in option_chain if float(o['strike_price']) < spot_price]
            if not otm_puts:
                return None
            # Sort by strike price to find the nearest OTM
            best_option = max(otm_puts, key=lambda x: float(x['strike_price']))
            return best_option['symbol']

        return None # No opportunity for "Hold" signal
    except Exception as e:
        logging.error(f"Error identifying opportunities: {e}")
        return None

def calculate_risk_amount(capital, risk_percent):
    """Calculates the amount of capital to risk on a single trade."""
    return capital * risk_percent

if __name__ == '__main__':
    from data_collector import connect_db, connect_shoonya

    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    model, scaler, feature_columns = load_model_and_scaler()
    conn = connect_db()
    api = connect_shoonya()

    if model and scaler and feature_columns is not None and conn and api:
        lookback = config['model']['lookback_period']
        symbol = 'NIFTY' # Example with NIFTY
        live_data = get_live_data(conn, symbol, lookback)

        # Reindex live data to match the training feature columns
        live_data_reindexed = live_data.reindex(columns=feature_columns, fill_value=0)

        option_chain = fetch_option_chain(api, symbol)
        processed_data = preprocess_live_data(live_data_reindexed, scaler, api, symbol, feature_columns, option_chain)

        signal, confidence = generate_signal(model, processed_data, lookback)
        logging.info(f"Generated Signal: {signal} with confidence {confidence:.2f}")

        filtered_signal = filter_signals(signal, confidence, config['trading']['confidence_threshold'])
        logging.info(f"Filtered Signal: {filtered_signal}")

        if filtered_signal != "Hold":
            spot_price = get_ltp(api, 'NSE', symbol)
            option_chain = fetch_option_chain(api, symbol)

            if spot_price and option_chain:
                opportunity = identify_opportunities(filtered_signal, option_chain, spot_price)
                if opportunity:
                    risk_amount = calculate_risk_amount(config['trading']['capital'], config['trading']['risk_per_trade'])
                    logging.info(f"Tradeable opportunity: {opportunity}, Risk Amount: {risk_amount}")
                else:
                    logging.info("No suitable opportunity found.")
