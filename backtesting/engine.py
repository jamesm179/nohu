class BacktestingEngine:
    """
    Handles the execution of backtests for trading strategies.

    This engine will simulate the behavior of the trading bot on historical
    market data to evaluate the performance of the strategies.
    """
    def __init__(self, strategy_engine, data_loader):
        self.strategy_engine = strategy_engine
        self.data_loader = data_loader
        print("Initializing BacktestingEngine...")

    def run(self):
        """
        Runs the backtest.
        """
        print("Starting backtest...")
        # 1. Load historical data using the data_loader
        # 2. Loop through the historical data tick by tick
        # 3. For each tick, pass the data to the strategy_engine
        # 4. Collect all generated trades and signals
        # 5. Calculate performance metrics
        print("Backtest finished.")
        # return performance_results
