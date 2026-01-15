import yaml
import pyotp
from NorenRestApiPy.NorenApi import NorenApi
import pandas as pd
import psycopg2
from apscheduler.schedulers.blocking import BlockingScheduler
import logging

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
                premium REAL,
                oi BIGINT,
                volume BIGINT,
                iv REAL,
                delta REAL,
                gamma REAL,
                theta REAL,
                vega REAL,
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

def calculate_greeks(spot, strike, expiry, volatility, rate):
    """
    Calculates option Greeks.
    (Placeholder for now)
    """
    # This will be implemented in a later step
    return {'delta': 0.5, 'gamma': 0.02, 'theta': 0.1, 'vega': 0.2}

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
        # Example usage
        from datetime import datetime, timedelta
        symbols = ['NIFTY', 'BANKNIFTY']
        to_date = datetime.now()
        from_date = to_date - timedelta(days=30)

        for symbol in symbols:
            # Fetch and store underlying data
            underlying_data = fetch_historical_data(api, symbol, from_date, to_date)
            if underlying_data is not None:
                # Add symbol column before storing
                underlying_data['symbol'] = symbol
                store_data_db(conn, underlying_data.reset_index(), 'underlying_data')

            # Fetch and store option chain data
            option_chain = fetch_option_chain(api, symbol)
            if option_chain is not None:
                # This part needs to be adapted to the actual structure of the option chain data
                # For now, we'll just log a warning
                logging.warning("Option chain data storage is not fully implemented.")
                # store_data_db(conn, option_chain, 'options_data')

        conn.close()
        logging.info("Database connection closed.")


if __name__ == '__main__':
    main()
    # schedule_data_updates() # Uncomment to run scheduled updates
