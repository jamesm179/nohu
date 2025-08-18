from typing import List
from src.models import Order, OrderSide

class PerformanceCalculator:
    """
    Calculates and reports performance metrics for a backtest.
    """
    def __init__(self, executed_trades: List[Order]):
        """
        Initializes the PerformanceCalculator.

        Args:
            executed_trades: A list of Order objects from a backtest run.
        """
        self.trades = executed_trades
        print(f"Initializing PerformanceCalculator with {len(self.trades)} trades.")

    def calculate_metrics(self):
        """
        Calculates and displays basic performance metrics.

        This implementation uses a simple one-position-at-a-time logic.
        """
        print("\n--- Backtest Performance ---")
        if not self.trades:
            print("No trades were executed.")
            return

        wins = 0
        losses = 0
        total_pnl = 0.0
        open_position = None

        for trade in self.trades:
            if trade.side == OrderSide.BUY:
                # If we are already long, we ignore this signal for now.
                if open_position is None:
                    open_position = trade

            elif trade.side == OrderSide.SELL:
                # We can only sell if we have an open long position.
                if open_position is not None:
                    pnl = (trade.price - open_position.price) * open_position.size
                    total_pnl += pnl
                    if pnl > 0:
                        wins += 1
                    else:
                        losses += 1

                    print(f"Closed trade: BOUGHT @ {open_position.price}, SOLD @ {trade.price}, P&L: {pnl:.2f}")
                    open_position = None # Close the position

        total_trades = wins + losses
        win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0

        print("\n--- Summary ---")
        print(f"Total Closed Trades: {total_trades}")
        print(f"Winning Trades:    {wins}")
        print(f"Losing Trades:     {losses}")
        print(f"Win Rate:          {win_rate:.2f}%")
        print(f"Total P&L:         ${total_pnl:.2f}")
        print("--------------------------\n")

        return {
            "total_trades": total_trades,
            "wins": wins,
            "losses": losses,
            "win_rate": win_rate,
            "total_pnl": total_pnl,
        }
