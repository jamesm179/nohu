import asyncio
from backtesting.engine import BacktestingEngine
from backtesting.data_loader import DataLoader
from backtesting.performance import PerformanceCalculator
from src.strategy.strategy_engine import StrategyEngine
from src.risk.risk_engine import RiskEngine
from src.execution import ExecutionHandler

async def main():
    """
    Sets up and runs a backtest session, then calculates performance.
    """
    print("Setting up backtest...")

    # 1. Initialize components
    data_loader = DataLoader(csv_path='data/btc_usd_test_data.csv')
    risk_engine = RiskEngine()
    execution_handler = ExecutionHandler(backtest=True)
    strategy_engine = StrategyEngine(
        market_data_pipeline=None,
        risk_engine=risk_engine,
        execution_handler=execution_handler
    )
    backtester = BacktestingEngine(
        strategy_engine=strategy_engine,
        data_loader=data_loader
    )

    # 2. Run the backtest
    await backtester.run()

    # 3. Calculate and display performance
    performance_calculator = PerformanceCalculator(
        executed_trades=execution_handler.executed_trades
    )
    performance_calculator.calculate_metrics()

    print("Backtest setup and run complete.")

if __name__ == "__main__":
    asyncio.run(main())
