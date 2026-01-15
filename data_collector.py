import yaml
import pyotp
from NorenRestApiPy.NorenApi import NorenApi
import pandas as pd
import psycopg2
from apscheduler.schedulers.blocking import BlockingScheduler
import logging
from py_vollib.black_scholes import delta, gamma, theta, vega
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)

class ShoonyaApiPy(NorenApi):
    def __init__(self):
        NorenApi.__init__(self, host='https://api.shoonya.com/NorenWClientTP/',
                          websocket='wss://api.shoonya.com/NorenWSTP/')

def connect_shoonya():
    """
    Connects to the Shoonya API using credentials from config.yaml.
    """
    try:
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)

        api = ShoonyaApiPy()
        # Make sure to replace these with your actual credentials
        ret = api.login(userid=config['shoonya']['user_id'],
                        password=config['shoonya']['password'],
                        twoFA=pyotp.TOTP(config['shoonya']['totp_key']).now(),
                        vendor_code=config['shoonya']['vendor_code'],
                        api_secret=config['shoonya']['api_secret'],
                        imei=config['shoonya']['imei'])

        if ret and ret['stat'] == 'Ok':
            logging.info("Successfully connected to Shoonya API.")
            return api
        else:
            logging.error(f"Failed to connect to Shoonya API: {ret}")
            return None

    except FileNotFoundError:
        logging.error("config.yaml not found. Please create it with your API credentials.")
        return None
    except Exception as e:
        logging.error(f"An error occurred during Shoonya API connection: {e}")
        return None

def fetch_historical_data(api, symbol, from_date, to_date, interval=1):
    """
    Fetches historical data for a given symbol.
    """
    try:
        ret = api.get_time_price_series(exchange='NFO', token=symbol, starttime=from_date.timestamp(), endtime=to_date.timestamp(), interval=interval)
        if ret:
            df = pd.DataFrame(ret)
            df['time'] = pd.to_datetime(df['time'], format='%d-%m-%Y %H:%M:%S')
            df = df.set_index('time')
            return df
        else:
            logging.error(f"Failed to fetch historical data for {symbol}")
            return None
    except Exception as e:
        logging.error(f"An error occurred during historical data fetching: {e}")
        return None

def fetch_option_chain(api, symbol):
    """
    Fetches the option chain for a given symbol.
    """
    try:
        ret = api.get_option_chain(exchange='NFO', search=symbol, strikeprice=0, count=100)
        if ret:
            return ret
        else:
            logging.error(f"Failed to fetch option chain for {symbol}")
            return None
    except Exception as e:
        logging.error(f"An error occurred during option chain fetching: {e}")
        return None

def connect_db():
    """Connects to the PostgreSQL database."""
    try:
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)['database']

        conn = psycopg2.connect(
            host=config['host'],
            port=config['port'],
            user=config['user'],
            password=config['password'],
            dbname=config['name']
        )
        logging.info("Successfully connected to the database.")
        return conn
    except Exception as e:
        logging.error(f"Error connecting to the database: {e}")
        return None

def create_tables(conn):
    """Creates the necessary tables in the database."""
    try:
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS options_data (
                timestamp TIMESTAMP,
                symbol VARCHAR(255),
                strike REAL,
                expiry TIMESTAMP,
                call_premium REAL,
                call_oi BIGINT,
                call_volume BIGINT,
                put_premium REAL,
                put_oi BIGINT,
                put_volume BIGINT,
                iv REAL,
                call_delta REAL,
                call_gamma REAL,
                call_theta REAL,
                call_vega REAL,
                put_delta REAL,
                put_gamma REAL,
                put_theta REAL,
                put_vega REAL,
                PRIMARY KEY (timestamp, symbol, strike, expiry)
            );
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS underlying_data (
                timestamp TIMESTAMP,
                symbol VARCHAR(255),
                ltp REAL,
                high REAL,
                low REAL,
                open REAL,
                close REAL,
                volume BIGINT,
                PRIMARY KEY (timestamp, symbol)
            );
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS trades_log (
                trade_id SERIAL PRIMARY KEY,
                entry_time TIMESTAMP,
                exit_time TIMESTAMP,
                symbol VARCHAR(255),
                strike REAL,
                type VARCHAR(50),
                entry_price REAL,
                exit_price REAL,
                pnl REAL
            );
        """)
        conn.commit()
        cur.close()
        logging.info("Tables created successfully.")
    except Exception as e:
        logging.error(f"Error creating tables: {e}")

def get_india_vix(api):
    """Fetches the current India VIX value."""
    try:
        # The token for India VIX might vary. 'INDIAVIX' is a common representation.
        ret = api.get_quotes(exchange='NSE', token='INDIAVIX')
        if ret and ret['stat'] == 'Ok':
            return float(ret['lp']) / 100 # VIX is in percentage
        else:
            logging.error(f"Could not get quotes for India VIX: {ret}")
            return None # Default to a reasonable IV if fetch fails
    except Exception as e:
        logging.error(f"Error getting India VIX: {e}")
        return None

def calculate_greeks(spot, strike, expiry, iv, flag, rate=0.05):
    """
    Calculates option Greeks using py_vollib.
    flag: 'c' for call, 'p' for put
    """
    try:
        t = (expiry - datetime.now()).days / 365.25
        if t < 0: t = 0 # Handle expired options

        d = delta(flag, spot, strike, t, rate, iv)
        g = gamma(flag, spot, strike, t, rate, iv)
        th = theta(flag, spot, strike, t, rate, iv)
        v = vega(flag, spot, strike, t, rate, iv)

        return {'delta': d, 'gamma': g, 'theta': th, 'vega': v}
    except Exception as e:
        logging.error(f"Error calculating Greeks for strike {strike}: {e}")
        return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0}

import os
import pickle

def get_instrument_map(api, exchange='NSE'):
    """Downloads the instrument master and creates a symbol-to-token map."""
    map_file = f"{exchange}_instrument_map.pickle"
    if os.path.exists(map_file):
        with open(map_file, 'rb') as f:
            return pickle.load(f)

    try:
        instrument_list = api.get_master(exchange=exchange)
        instrument_map = {item['tsym']: item['token'] for item in instrument_list}

        with open(map_file, 'wb') as f:
            pickle.dump(instrument_map, f)

        logging.info(f"Instrument map for {exchange} created and cached.")
        return instrument_map
    except Exception as e:
        logging.error(f"Error creating instrument map for {exchange}: {e}")
        return {}

def get_ltp(api, exchange, token):
    """Gets the last traded price for a given symbol."""
    try:
        ret = api.get_quotes(exchange=exchange, token=token)
        if ret and ret['stat'] == 'Ok':
            return float(ret['lp'])
        else:
            logging.error(f"Could not get quotes for {token}: {ret}")
            return None
    except Exception as e:
        logging.error(f"Error getting LTP for {token}: {e}")
        return None

def get_latest_data(conn, symbol, lookback):
    """Retrieves the latest 'lookback' candles from the database."""
    if conn is None:
        logging.error("Database connection is not available.")
        return None

    try:
        query = "SELECT * FROM underlying_data WHERE symbol = %s ORDER BY timestamp DESC LIMIT %s"
        df = pd.read_sql(query, conn, params=(symbol, lookback), index_col='timestamp')
        df = df.sort_index()  # Sort back to chronological order
        logging.info(f"Retrieved latest {len(df)} data points for {symbol}.")
        return df
    except Exception as e:
        logging.error(f"Error retrieving latest data: {e}")
        return None

def store_data_db(conn, data, table_name):
    """Stores data in the database."""
    if conn is None:
        logging.error("Database connection is not available.")
        return

    try:
        if isinstance(data, pd.DataFrame):
            # For DataFrame, use to_sql for efficient bulk insertion
            from sqlalchemy import create_engine
            with open('config.yaml', 'r') as f:
                db_config = yaml.safe_load(f)['database']
            engine_str = f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['name']}"
            engine = create_engine(engine_str)
            data.to_sql(table_name, engine, if_exists='append', index=False)
            logging.info(f"Successfully stored DataFrame in {table_name}.")
        else:
            # Handle other data types (e.g., dicts for option chain)
            # This part needs to be adapted based on the actual data structure
            logging.warning("store_data_db for non-DataFrame is not fully implemented.")

    except Exception as e:
        logging.error(f"Error storing data in {table_name}: {e}")

def schedule_data_updates():
    """
    Schedules data updates during market hours.
    """
    scheduler = BlockingScheduler()
    # Example: run every 5 minutes from 9:15 to 15:30 on weekdays
    scheduler.add_job(main, 'cron', day_of_week='mon-fri', hour='9-15', minute='*/5')
    scheduler.start()

def main():
    """
    Main function to run the data collection process.
    """
    api = connect_shoonya()
    conn = connect_db()

    if conn:
        create_tables(conn)

    if api and conn:
        # Get instrument maps
        nse_map = get_instrument_map(api, 'NSE')
        nfo_map = get_instrument_map(api, 'NFO')

        # Example usage
        from datetime import datetime, timedelta
        symbols = ['NIFTY 50', 'NIFTY BANK'] # Using more specific names
        to_date = datetime.now()
        from_date = to_date - timedelta(days=30)

        for symbol_name in symbols:
            token = nse_map.get(symbol_name)
            if not token:
                logging.warning(f"Could not find token for {symbol_name}. Skipping.")
                continue

            # Fetch and store underlying data
            underlying_data = fetch_historical_data(api, token, from_date, to_date)
            if underlying_data is not None:
                # Add symbol column before storing
                underlying_data['symbol'] = symbol_name
                store_data_db(conn, underlying_data.reset_index(), 'underlying_data')

            # Fetch and store option chain data
            option_chain = fetch_option_chain(api, symbol_name)
            if option_chain is not None:
                spot_price = get_ltp(api, 'NSE', token)
                iv = get_india_vix(api) or 0.15 # Use default if fetch fails

                if spot_price:
                    options_data = []
                    for option in option_chain:
                        try:
                            expiry = datetime.strptime(option['expiry_date'], '%d-%b-%Y')
                            strike = float(option['strike_price'])

                            # Calculate greeks for call and put
                            call_greeks = calculate_greeks(spot_price, strike, expiry, iv, 'c')
                            put_greeks = calculate_greeks(spot_price, strike, expiry, iv, 'p')

                            options_data.append({
                                'timestamp': datetime.now(), 'symbol': symbol_name, 'strike': strike, 'expiry': expiry,
                                'call_premium': float(option.get('call_ltp', 0)),
                                'call_oi': int(option.get('call_oi', 0)), 'call_volume': 0,
                                'put_premium': float(option.get('put_ltp', 0)),
                                'put_oi': int(option.get('put_oi', 0)), 'put_volume': 0,
                                'iv': iv,
                                **{'call_' + k: v for k, v in call_greeks.items()},
                                **{'put_' + k: v for k, v in put_greeks.items()}
                            })
                        except Exception as e:
                            logging.error(f"Error processing option {option.get('symbol')}: {e}")

                    if options_data:
                        df_options = pd.DataFrame(options_data)
                        store_data_db(conn, df_options, 'options_data')

        conn.close()
        logging.info("Database connection closed.")


if __name__ == '__main__':
    main()
    # schedule_data_updates() # Uncomment to run scheduled updates
