import pandas as pd
import logging
import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)

class PaperTradingAccount:
    def __init__(self, initial_capital=100000):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.positions = {}
        self.trade_history = []
        logging.info(f"Paper trading account created with initial capital: {self.initial_capital}")

    def place_order(self, symbol, strike, order_type, quantity, price, stop_loss=None, take_profit=None):
        """Places a new order."""
        order = {
            'symbol': symbol,
            'strike': strike,
            'order_type': order_type,
            'quantity': quantity,
            'price': price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'timestamp': datetime.datetime.now()
        }
        self.execute_order(order)

    def execute_order(self, order):
        """Simulates order execution with slippage and costs."""
        # Simplified slippage and cost model
        slippage = 0.001 * order['price']
        transaction_cost = 20 # Flat fee per trade

        if order['order_type'] == 'BUY':
            execution_price = order['price'] + slippage
            cost = (execution_price * order['quantity']) + transaction_cost
            if self.capital >= cost:
                self.capital -= cost
                if order['symbol'] not in self.positions:
                    self.positions[order['symbol']] = {'quantity': 0, 'entry_price': 0, 'stop_loss': order.get('stop_loss'), 'take_profit': order.get('take_profit')}

                # Update position with weighted average price
                current_quantity = self.positions[order['symbol']]['quantity']
                current_price = self.positions[order['symbol']]['entry_price']
                new_quantity = current_quantity + order['quantity']
                self.positions[order['symbol']]['entry_price'] = ((current_price * current_quantity) + (execution_price * order['quantity'])) / new_quantity
                self.positions[order['symbol']]['quantity'] = new_quantity

                self.log_trade('BUY', order, execution_price)
            else:
                logging.warning("Insufficient capital to execute buy order.")

        elif order['order_type'] == 'SELL':
            execution_price = order['price'] - slippage
            proceeds = (execution_price * order['quantity']) - transaction_cost
            if self.positions.get(order['symbol'], {}).get('quantity', 0) >= order['quantity']:
                self.capital += proceeds
                self.positions[order['symbol']]['quantity'] -= order['quantity']
                if self.positions[order['symbol']]['quantity'] == 0:
                    del self.positions[order['symbol']]
                self.log_trade('SELL', order, execution_price)
            else:
                logging.warning("Insufficient position to execute sell order.")

    def log_trade(self, trade_type, order, execution_price):
        """Logs a trade to the trade history."""
        trade_log = {
            'timestamp': order['timestamp'],
            'symbol': order['symbol'],
            'type': trade_type,
            'quantity': order['quantity'],
            'price': execution_price,
            'pnl': 0 # PnL is calculated on closing the position
        }
        self.trade_history.append(trade_log)
        logging.info(f"Trade executed: {trade_log}")

    def update_positions(self, market_data):
        """Marks-to-market all open positions."""
        # This requires a live feed of market prices
        pass

    def check_stop_loss_take_profit(self, market_prices):
        """Checks and executes stop-loss or take-profit orders."""
        positions_to_close = []
        for symbol, position in self.positions.items():
            current_price = market_prices.get(symbol)
            if current_price is None:
                continue

            # Check for stop-loss
            if position.get('stop_loss') and current_price <= position['stop_loss']:
                logging.info(f"Stop-loss triggered for {symbol} at {current_price}")
                positions_to_close.append((symbol, position['quantity'], current_price))
                continue # Skip take-profit check if stop-loss is hit

            # Check for take-profit
            if position.get('take_profit') and current_price >= position['take_profit']:
                logging.info(f"Take-profit triggered for {symbol} at {current_price}")
                positions_to_close.append((symbol, position['quantity'], current_price))

        for symbol, quantity, price in positions_to_close:
            self.place_order(symbol, 0, 'SELL', quantity, price)

    def calculate_pnl(self):
        """Calculates total P&L."""
        # Simplified PnL calculation
        return self.capital - self.initial_capital

    def generate_report(self):
        """Generates a performance report."""
        report = {
            'Initial Capital': self.initial_capital,
            'Current Capital': self.capital,
            'Total P&L': self.calculate_pnl(),
            'Number of Trades': len(self.trade_history),
            'Positions': self.positions
        }
        logging.info("--- Performance Report ---")
        for key, value in report.items():
            logging.info(f"{key}: {value}")
        logging.info("--------------------------")
        return report

if __name__ == '__main__':
    account = PaperTradingAccount(initial_capital=100000)

    # Simulate a few trades with stop-loss and take-profit
    account.place_order('NIFTY_CE', 18000, 'BUY', 50, 150.0, stop_loss=140.0, take_profit=160.0)

    # Simulate market price changes
    market_prices = {'NIFTY_CE': 165.0}
    account.check_stop_loss_take_profit(market_prices) # Should trigger take-profit

    account.place_order('BANKNIFTY_PE', 42000, 'BUY', 25, 200.0, stop_loss=190.0, take_profit=210.0)

    market_prices = {'BANKNIFTY_PE': 185.0}
    account.check_stop_loss_take_profit(market_prices) # Should trigger stop-loss

    # Generate a report
    account.generate_report()

    # Display trade history
    trade_history_df = pd.DataFrame(account.trade_history)
    print("\nTrade History:")
    print(trade_history_df)
