import pytest
from src.risk.risk_engine import RiskEngine
from src.models import Order, OrderSide

@pytest.fixture
def risk_engine():
    """Fixture to create a RiskEngine instance for testing."""
    return RiskEngine()

def test_approve_order_within_limits(risk_engine):
    """Tests that a valid order within the size limit is approved."""
    order = Order(product_id='BTC-USD', side=OrderSide.BUY, size=0.5, price=50000.0)
    assert risk_engine.check_pre_trade_risk(order) is True

def test_reject_order_exceeding_limits(risk_engine):
    """Tests that an order exceeding the size limit is rejected."""
    order = Order(product_id='BTC-USD', side=OrderSide.BUY, size=1.5, price=50000.0)
    assert risk_engine.check_pre_trade_risk(order) is False

def test_reject_order_with_no_config(risk_engine):
    """Tests that an order for an unconfigured product is rejected."""
    order = Order(product_id='DOGE-USD', side=OrderSide.BUY, size=100.0, price=0.5)
    assert risk_engine.check_pre_trade_risk(order) is False

def test_reject_order_with_zero_size(risk_engine):
    """Tests that an order with a size of zero is rejected."""
    order = Order(product_id='BTC-USD', side=OrderSide.BUY, size=0.0, price=50000.0)
    assert risk_engine.check_pre_trade_risk(order) is False

def test_reject_order_with_negative_size(risk_engine):
    """Tests that an order with a negative size is rejected."""
    order = Order(product_id='BTC-USD', side=OrderSide.BUY, size=-1.0, price=50000.0)
    assert risk_engine.check_pre_trade_risk(order) is False
