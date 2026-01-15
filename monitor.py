import logging
import pandas as pd
import matplotlib.pyplot as plt
from paper_trader import PaperTradingAccount

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def log_system_status(status):
    """Logs a system heartbeat or status message."""
    logging.info(f"SYSTEM STATUS: {status}")

def alert_on_error(error_msg):
    """Sends a notification on error (placeholder)."""
    # In a real system, this could send an email, SMS, or push notification
    logging.error(f"ALERT: {error_msg}")
    print(f"ALERT: {error_msg}")

def generate_daily_report(account):
    """Generates a daily performance summary."""
    report = account.generate_report()
    log_system_status("Daily report generated.")
    return report

def plot_equity_curve(trade_history, initial_capital, save_path='reports/equity_curve.png'):
    """Plots and saves the equity curve."""
    if not trade_history:
        logging.warning("No trades to plot.")
        return

    df = pd.DataFrame(trade_history)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')

    # Calculate cumulative PnL
    df['pnl'] = df.apply(lambda row: (row['price'] * row['quantity']) if row['type'] == 'SELL' else -(row['price'] * row['quantity']), axis=1)
    df['cumulative_pnl'] = df['pnl'].cumsum()
    df['equity'] = initial_capital + df['cumulative_pnl']

    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df['equity'])
    plt.title('Equity Curve')
    plt.xlabel('Date')
    plt.ylabel('Equity')
    plt.grid(True)

    # Ensure the reports directory exists
    import os
    if not os.path.exists('reports'):
        os.makedirs('reports')

    plt.savefig(save_path)
    logging.info(f"Equity curve saved to {save_path}")
    plt.close()

def analyze_trades(trade_history):
    """Analyzes trade performance metrics."""
    if not trade_history:
        return {}

    df = pd.DataFrame(trade_history)
    wins = df[df['pnl'] > 0]
    losses = df[df['pnl'] < 0]

    win_rate = len(wins) / len(df) if len(df) > 0 else 0
    gross_profit = wins['pnl'].sum()
    gross_loss = losses['pnl'].sum()
    profit_factor = abs(gross_profit / gross_loss) if gross_loss != 0 else float('inf')

    analysis = {
        'Total Trades': len(df),
        'Win Rate': f"{win_rate:.2%}",
        'Profit Factor': f"{profit_factor:.2f}",
        'Gross Profit': f"{gross_profit:.2f}",
        'Gross Loss': f"{gross_loss:.2f}"
    }

    logging.info("--- Trade Analysis ---")
    for key, value in analysis.items():
        logging.info(f"{key}: {value}")
    logging.info("----------------------")
    return analysis

if __name__ == '__main__':
    # Create a sample paper trading account for demonstration
    account = PaperTradingAccount(initial_capital=100000)

    # Simulate trades
    account.place_order('NIFTY_20230125_18000_CE', 18000, 'BUY', 50, 150.0)
    account.place_order('NIFTY_20230125_18000_CE', 18000, 'SELL', 50, 160.0)
    account.place_order('BANKNIFTY_20230125_42000_PE', 42000, 'BUY', 25, 200.0)
    account.place_order('BANKNIFTY_20230125_42000_PE', 42000, 'SELL', 25, 190.0)

    # Monitoring and Reporting
    log_system_status("Running monitoring and reporting demo.")

    # Generate daily report
    daily_report = generate_daily_report(account)

    # Analyze trades (Note: PnL calculation in paper_trader is simplified)
    # For a more accurate analysis, PnL should be calculated per trade
    trade_df = pd.DataFrame(account.trade_history)
    # This is a simplified PnL calculation for demonstration
    buy_trades = trade_df[trade_df['type'] == 'BUY'].set_index('symbol')
    sell_trades = trade_df[trade_df['type'] == 'SELL'].set_index('symbol')
    pnl = (sell_trades['price'] - buy_trades['price']) * sell_trades['quantity']
    trade_df.loc[trade_df['type'] == 'SELL', 'pnl'] = pnl.values

    trade_analysis = analyze_trades(trade_df.to_dict('records'))

    # Plot equity curve
    plot_equity_curve(trade_df.to_dict('records'), account.initial_capital)

    # Simulate an error
    alert_on_error("This is a test error message.")
