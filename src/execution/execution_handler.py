from src.models import Order
from typing import List

class ExecutionHandler:
    """
    Handles the execution of trade orders.
    Can operate in live or backtest mode.
    """
    def __init__(self, backtest: bool = False):
        """
        Initializes the ExecutionHandler.

        Args:
            backtest (bool): If True, handler will run in backtest mode,
                             storing trades instead of executing them.
        """
        self.backtest = backtest
        self.executed_trades: List[Order] = []

        mode = "Backtest" if self.backtest else "Live"
        print(f"Initializing ExecutionHandler ({mode} Mode)...")

    async def execute_order(self, order: Order) -> bool:
        """
        Executes an order. In live mode, this would send the order to an
        exchange. In backtest mode, it records the trade for analysis.
        """
        if self.backtest:
            self.executed_trades.append(order)
            print(f"[ExecutionHandler] [Backtest] Logged trade: {order}")
            return True
        else:
            # Live execution logic would go here.
            print(f"[ExecutionHandler] [Live] ===> EXECUTING ORDER: {order}")
            return True

    async def start(self):
        """Starts the execution handler."""
        print("ExecutionHandler started.")

    async def stop(self):
        """Stops the execution handler."""
        print("ExecutionHandler stopped.")
