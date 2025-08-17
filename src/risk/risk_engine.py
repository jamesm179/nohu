from src.models import Order

class RiskEngine:
    """
    Monitors and manages trading risk in real-time.
    - Performs pre-trade risk checks against defined limits.
    - (Future) Monitors real-time P&L and portfolio exposure.
    - (Future) Conducts stress tests and scenario analysis.
    """
    def __init__(self):
        print("Initializing RiskEngine...")
        # Simple risk configuration: max order size per product.
        # In a real system, this would be loaded from a config file.
        self.risk_limits = {
            'BTC-USD': 1.0,   # Max order size of 1 BTC
            'ETH-USD': 10.0,  # Max order size of 10 ETH
        }
        print(f"Risk limits loaded: {self.risk_limits}")

    def check_pre_trade_risk(self, order: Order) -> bool:
        """
        Checks if a proposed order violates any risk limits.
        Currently checks for maximum order size.
        Returns True if the order is within limits, False otherwise.
        """
        max_size = self.risk_limits.get(order.product_id)

        if max_size is None:
            print(f"[RiskEngine] REJECTED: No risk limit configured for {order.product_id}. "
                  f"Order: {order}")
            return False

        if order.size <= 0:
            print(f"[RiskEngine] REJECTED: Order size must be positive. Order: {order}")
            return False

        if order.size > max_size:
            print(f"[RiskEngine] REJECTED: Order size {order.size} for {order.product_id} "
                  f"exceeds max limit of {max_size}. Order: {order}")
            return False

        print(f"[RiskEngine] APPROVED: Order for {order.size} {order.product_id} is within risk limits.")
        return True

    async def start(self):
        """Starts the risk engine."""
        print("RiskEngine started.")

    async def stop(self):
        """Stops the risk engine."""
        print("RiskEngine stopped.")
