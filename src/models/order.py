from dataclasses import dataclass
from enum import Enum

class OrderSide(Enum):
    """Enumeration for the side of an order."""
    BUY = 'BUY'
    SELL = 'SELL'

@dataclass(frozen=True)
class Order:
    """
    Represents a trade order to be sent to the exchange.

    This is an immutable data class, ensuring that order objects
    cannot be accidentally modified after creation.
    """
    product_id: str
    side: OrderSide
    size: float
    price: float
    order_type: str = 'LIMIT'
