import time
import datetime
import yaml
import logging
from data_collector import connect_shoonya, connect_db, get_ltp, fetch_option_chain, get_instrument_map
from signal_generator import load_model_and_scaler, preprocess_live_data, generate_signal, filter_signals, identify_opportunities, calculate_risk_amount, get_live_data
from paper_trader import PaperTradingAccount

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def is_market_open():
    """Checks if the market is open."""
    now = datetime.datetime.now().time()
    market_open = datetime.time(9, 15)
    market_close = datetime.time(15, 30)
    return market_open <= now <= market_close

def main():
    """Main auto-trading orchestrator loop."""
    try:
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        logging.error("config.yaml not found.")
        return

    # Initialize components
    api = connect_shoonya()
    conn = connect_db()
    if not api or not conn:
        logging.error("Failed to connect to Shoonya API or database. Exiting.")
        return

    model, scaler, feature_columns = load_model_and_scaler()
    if not model or not scaler or feature_columns is None:
        logging.error("Failed to load model, scaler, or feature columns. Exiting.")
        return

    account = PaperTradingAccount(config['trading']['capital'])
    lookback = config['model']['lookback_period']

    # Get instrument maps
    nse_map = get_instrument_map(api, 'NSE')
    nfo_map = get_instrument_map(api, 'NFO')

    logging.info("Auto-trading system initialized. Starting main loop.")

    while True:
        if is_market_open():
            logging.info("Market is open. Starting trading cycle.")

            # 1. Fetch latest data for each symbol
            for symbol_name in config['trading']['symbols']:
                token = nse_map.get(symbol_name)
                if not token:
                    logging.warning(f"Could not find token for {symbol_name}. Skipping.")
                    continue

                live_data = get_live_data(conn, symbol_name, lookback)
                if live_data is None or len(live_data) < lookback:
                    logging.warning(f"Not enough data for {symbol_name} to generate a signal.")
                    continue

                # 2. Generate signals
                option_chain = fetch_option_chain(api, symbol_name)
                processed_data = preprocess_live_data(live_data.copy(), scaler, api, symbol_name, feature_columns, option_chain)
                signal, confidence = generate_signal(model, processed_data, lookback)

                logging.info(f"Generated Signal for {symbol_name}: {signal} with confidence {confidence:.2f}")

                # 3. Filter signals
                filtered_signal = filter_signals(signal, confidence, config['trading']['confidence_threshold'])
                logging.info(f"Filtered Signal for {symbol_name}: {filtered_signal}")

                # 4. Risk checks and trade execution
                if filtered_signal != "Hold":
                    if len(account.positions) < config['trading']['max_positions']:
                        spot_price = get_ltp(api, 'NSE', token) # Underlying spot price

                        if spot_price and option_chain:
                            opportunity_symbol = identify_opportunities(filtered_signal, option_chain, spot_price)
                            if opportunity_symbol:
                                opportunity_token = nfo_map.get(opportunity_symbol)
                                if not opportunity_token:
                                    logging.warning(f"Could not find token for option {opportunity_symbol}. Skipping.")
                                    continue

                                # Get current price for the selected option
                                current_price = get_ltp(api, 'NFO', opportunity_token)
                                if current_price:
                                    risk_amount = calculate_risk_amount(account.capital, config['trading']['risk_per_trade'])

                                    # Define stop-loss and take-profit from config
                                    sl_pct = config['trading']['stop_loss_pct']
                                    tp_pct = config['trading']['take_profit_pct']
                                    stop_loss_price = current_price * (1 - sl_pct)
                                    take_profit_price = current_price * (1 + tp_pct)
                                    risk_per_share = current_price - stop_loss_price

                                    if risk_per_share > 0:
                                        quantity = int(risk_amount / risk_per_share)
                                        if quantity > 0:
                                            order_type = 'BUY' # For both calls and puts
                                            logging.info(f"Executing {order_type} order for {opportunity_symbol} of {quantity} contracts.")
                                            account.place_order(opportunity_symbol, 0, order_type, quantity, current_price, stop_loss_price, take_profit_price)
                                        else:
                                            logging.warning("Calculated quantity is zero. Skipping trade.")
                                    else:
                                        logging.warning("Risk per share is zero or negative. Skipping trade.")
                    else:
                        logging.warning("Max positions reached. No new trades will be placed.")

            # 5. Update stop-loss/take-profit (outside the symbol loop)
            market_prices = {}
            for pos_symbol in list(account.positions.keys()):
                token = nfo_map.get(pos_symbol)
                if token:
                    price = get_ltp(api, 'NFO', token)
                    if price:
                        market_prices[pos_symbol] = price

            if market_prices:
                account.check_stop_loss_take_profit(market_prices)

            # 6. Log activities
            account.generate_report()

            logging.info("Trading cycle finished. Waiting for next iteration.")
            time.sleep(300) # Wait for 5 minutes
        else:
            logging.info("Market is closed. Sleeping.")
            time.sleep(60)

if __name__ == '__main__':
    main()
