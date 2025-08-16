import pytest

def test_smoke_run():
    """
    A simple smoke test to verify that the test suite is running.
    """
    assert True, "Smoke test failed. The test suite is not running correctly."

def test_import_core_components():
    """
    Tests that the core application components can be imported without errors,
    which confirms that the project structure and Python paths are set up correctly.
    """
    try:
        from src.data.market_data_pipeline import MarketDataPipeline
        from src.strategy.strategy_engine import StrategyEngine
        from src.risk.risk_engine import RiskEngine
        import main
    except ImportError as e:
        pytest.fail(f"Failed to import one or more core components: {e}")
