import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import numpy as np
import pandas as pd
import logging
import os
import joblib
from data_collector import connect_db, get_latest_data
from feature_engineer import add_technical_indicators, add_options_features, add_time_features, create_labels, normalize_features, create_sequences

# Configure logging
logging.basicConfig(level=logging.INFO)

def build_lstm_model(input_shape, units=50, dropout=0.2):
    """Builds the LSTM model architecture."""
    model = Sequential()
    model.add(LSTM(units=units, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(dropout))
    model.add(LSTM(units=units))
    model.add(Dropout(dropout))
    model.add(Dense(units=3, activation='softmax')) # 3 classes: Buy, Sell, Hold

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    logging.info("LSTM model built.")
    return model

def prepare_data(df, lookback, test_size=0.15, val_size=0.15):
    """Prepares the data for training, validation, and testing."""

    # Normalize features
    df_normalized, scaler = normalize_features(df.copy())
    feature_columns = df_normalized.columns.drop('label')

    # Create sequences
    X, y = create_sequences(df_normalized, lookback)

    # Adjust y labels to be non-negative for sparse_categorical_crossentropy
    y = y + 1  # -1, 0, 1 -> 0, 1, 2

    # Split into training, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=(test_size + val_size), shuffle=False)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp,
                                                    test_size=(test_size / (test_size + val_size)), shuffle=False)

    logging.info(f"Data prepared: X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")
    return X_train, y_train, X_val, y_val, X_test, y_test, scaler, feature_columns


def train_model(model, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
    """Trains the model."""
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val), verbose=1)
    logging.info("Model training complete.")
    return model, history

def evaluate_model(model, X_test, y_test):
    """Evaluates the model's performance."""
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')

    logging.info(f"Model Evaluation - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}")
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall}

def save_model(model, path='models', filename='lstm_model.h5'):
    """Saves the trained model."""
    if not os.path.exists(path):
        os.makedirs(path)
    model.save(os.path.join(path, filename))
    logging.info(f"Model saved to {os.path.join(path, filename)}")

def save_scaler(scaler, path='models', filename='scaler.joblib'):
    """Saves the scaler object."""
    if not os.path.exists(path):
        os.makedirs(path)
    joblib.dump(scaler, os.path.join(path, filename))
    logging.info(f"Scaler saved to {os.path.join(path, filename)}")

def save_feature_columns(columns, path='models', filename='feature_columns.joblib'):
    """Saves the feature columns."""
    if not os.path.exists(path):
        os.makedirs(path)
    joblib.dump(columns, os.path.join(path, filename))
    logging.info(f"Feature columns saved to {os.path.join(path, filename)}")

def hyperparameter_tune():
    """Placeholder for hyperparameter tuning."""
    logging.info("Hyperparameter tuning not implemented yet.")
    pass

if __name__ == '__main__':
    conn = connect_db()
    if conn:
        # Fetch underlying and options data from the database
        underlying_df = pd.read_sql("SELECT * FROM underlying_data WHERE symbol = 'NIFTY'", conn, index_col='timestamp')
        options_df = pd.read_sql("SELECT * FROM options_data WHERE symbol = 'NIFTY'", conn, index_col='timestamp')
        conn.close()

        if underlying_df is not None and not underlying_df.empty:
            # Calculate PCR from options data
            if not options_df.empty:
                pcr_series = (options_df['put_oi'].sum(axis=1) / options_df['call_oi'].sum(axis=1)).rename('pcr')
                df = underlying_df.join(pcr_series, how='left').fillna(method='ffill')
            else:
                df = underlying_df
                df['pcr'] = 0

            # Feature Engineering
            df = add_technical_indicators(df)
            df = add_options_features(df) # PCR is now in the df
            df = add_time_features(df)
            df = create_labels(df)
            df.dropna(inplace=True)

            # Prepare Data
            lookback = 60
            X_train, y_train, X_val, y_val, X_test, y_test, scaler, feature_columns = prepare_data(df, lookback)

            # Build and Train Model
            if X_train.shape[0] > 0 and X_test.shape[0] > 0:
                input_shape = (X_train.shape[1], X_train.shape[2])
                model = build_lstm_model(input_shape)
                model, history = train_model(model, X_train, y_train, X_val, y_val, epochs=10)  # Using fewer epochs for demonstration

                # Evaluate Model
                evaluate_model(model, X_test, y_test)

                # Save Model, Scaler, and Feature Columns
                save_model(model)
                save_scaler(scaler)
                save_feature_columns(feature_columns)
            else:
                logging.error("Not enough data to train the model after processing.")
        else:
            logging.error("No data retrieved from the database. Cannot train model.")
