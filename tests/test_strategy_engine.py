import pytest
from src.strategy.strategy_engine import StrategyEngine

@pytest.fixture
def strategy_engine():
    """Fixture to create a StrategyEngine instance for testing."""
    return StrategyEngine(
        market_data_pipeline=None,
        risk_engine=None,
        short_window=5,
        long_window=10
    )

def test_sma_crossover_strategy_generates_correct_signals(strategy_engine, capsys):
    """
    Tests that the SMA crossover strategy correctly identifies and logs
    a Golden Cross (BUY) and a Death Cross (SELL) in the correct order.
    """
    # 1. Arrange - Create a predictable stream of mock trade messages
    class MockMessage:
        def __init__(self, value):
            self.value = value

    base_price = 100
    # Phase 1: Long period of low, stable prices to establish a baseline
    prices = [float(base_price)] * strategy_engine.long_window
    # Phase 2: A smaller, sharp rise to trigger a Golden Cross
    prices.extend([float(base_price + 5)] * strategy_engine.short_window)
    # Phase 3: A sharp fall to trigger a Death Cross
    prices.extend([float(base_price - 5)] * strategy_engine.short_window)

    mock_messages = []
    for i, price in enumerate(prices):
        trade = {
            'timestamp': f'2025-01-01T12:00:{i:02d}Z',
            'product_id': 'BTC-USD',
            'price': price,
            'size': 1.0,
            'side': 'BUY'
        }
        mock_messages.append(MockMessage(trade))

    # 2. Act - Process the messages one by one
    for msg in mock_messages:
        strategy_engine._process_trade_message(msg)

    # 3. Assert - Check the captured output for signals
    captured = capsys.readouterr()
    output = captured.out

    assert "--- BUY SIGNAL (Golden Cross) ---" in output
    assert "--- SELL SIGNAL (Death Cross) ---" in output

    # Check that the signals appeared in the correct order
    buy_signal_index = output.find("BUY SIGNAL")
    sell_signal_index = output.find("SELL SIGNAL")

    assert buy_signal_index != -1, "BUY Signal was not found in the output"
    assert sell_signal_index != -1, "SELL Signal was not found in the output"
    assert buy_signal_index < sell_signal_index, "BUY Signal did not appear before SELL Signal"
