import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock
import pandas as pd
import collections

from src.strategy.strategy_engine import StrategyEngine
from src.risk.risk_engine import RiskEngine
from src.execution import ExecutionHandler
from src.models import Order, OrderSide

@pytest.fixture
def mock_dependencies():
    """Provides a dictionary of mocked dependencies for the StrategyEngine."""
    mock_risk_engine = MagicMock(spec=RiskEngine)
    mock_risk_engine.check_pre_trade_risk.return_value = True

    mock_execution_handler = MagicMock(spec=ExecutionHandler)
    mock_execution_handler.execute_order = AsyncMock(return_value=True)

    return {
        "risk_engine": mock_risk_engine,
        "execution_handler": mock_execution_handler
    }

class MockMessage:
    def __init__(self, value):
        self.value = value

@pytest.mark.asyncio
async def test_sma_crossover_strategy(mock_dependencies, capsys):
    """Tests that the SMA crossover strategy generates correct signals."""
    engine = StrategyEngine(None, **mock_dependencies)
    engine._initialize_product_state('BTC-USD') # Initialize state

    # Arrange
    base_price = 100
    cfg = engine.strategies['sma_crossover']
    prices = [float(base_price)] * cfg['long_window']
    prices.extend([float(base_price + 10)] * cfg['short_window'])
    prices.extend([float(base_price - 20)] * cfg['short_window'])

    # Act
    for price in prices:
        await engine._process_sma_crossover('BTC-USD', price)

    # Assert
    captured = capsys.readouterr()
    assert "BUY SIGNAL (SMA Crossover)" in captured.out
    assert "SELL SIGNAL (SMA Crossover)" in captured.out
    assert mock_dependencies["execution_handler"].execute_order.call_count == 2

@pytest.mark.asyncio
async def test_rsi_strategy(mock_dependencies, capsys):
    """Tests that the RSI strategy generates correct signals."""
    engine = StrategyEngine(None, **mock_dependencies)
    engine._initialize_product_state('BTC-USD') # Initialize state

    # Arrange
    base_price = 100
    prices = [float(base_price)] * 20
    prices.extend([95, 94, 93, 92, 91, 90]) # Oversold
    prices.extend([105, 106, 107, 108, 109, 110]) # Overbought

    # Act
    for price in prices:
        await engine._process_rsi_strategy('BTC-USD', price)

    # Assert
    captured = capsys.readouterr()
    assert "BUY SIGNAL (RSI)" in captured.out
    assert "SELL SIGNAL (RSI)" in captured.out
    assert mock_dependencies["execution_handler"].execute_order.call_count == 2
