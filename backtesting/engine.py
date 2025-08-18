import asyncio
from src.strategy.strategy_engine import StrategyEngine
from backtesting.data_loader import DataLoader

class MockMessage:
    """A simple mock to wrap our tick data like a Kafka message."""
    def __init__(self, value):
        self.value = value

class BacktestingEngine:
    """
    Handles the execution of backtests for trading strategies.
    """
    def __init__(self, strategy_engine: StrategyEngine, data_loader: DataLoader):
        """
        Initializes the BacktestingEngine.

        Args:
            strategy_engine: An instance of the StrategyEngine to be tested.
            data_loader: An instance of the DataLoader to provide historical data.
        """
        self.strategy_engine = strategy_engine
        self.data_loader = data_loader
        print("Initializing BacktestingEngine...")

    async def run(self):
        """
        Runs the backtest by feeding historical data to the strategy engine.
        """
        print("--- Starting Backtest Run ---")

        # In a backtest, we don't start the strategy engine's consumer.
        # Instead, we feed data to it directly.

        # Process each tick of historical data
        for tick_data in self.data_loader.load_data():
            mock_message = MockMessage(tick_data)
            # We directly call the message processor, simulating a message
            # coming from the (mock) Kafka consumer.
            await self.strategy_engine._process_trade_message(mock_message)

        print("--- Backtest Run Finished ---")
